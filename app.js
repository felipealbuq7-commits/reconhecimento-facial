/**
 * app.js — Sistema de Reconhecimento Facial com MediaPipe FaceMesh
 * =================================================================
 * Como funciona a comparação:
 *   1. Ao cadastrar, os 468 landmarks são normalizados (translação +
 *      escala, usando apenas X e Y — Z é ruidoso) e salvos.
 *   2. Ao verificar, calcula-se a distância euclidiana 2D média
 *      contra cada cadastro.
 *   3. Se a menor distância for ≤ THRESHOLD → Acesso Permitido.
 *
 * Calibração do THRESHOLD:
 *   - Mesmo rosto, variações normais de ângulo/luz: dist ≈ 0.03–0.07
 *   - Rostos diferentes:                            dist ≈ 0.10–0.20+
 *   - Valor seguro: 0.08 (permissivo) a 0.06 (mais estrito)
 */

'use strict';

// ──────────────────────────────────────────────────────────────
// CONSTANTES
// ──────────────────────────────────────────────────────────────
const THRESHOLD   = 0.072;   // distância 2D máxima para "mesmo rosto"
const LOCK_MS     = 3000;    // tempo (ms) que o resultado fica na tela
const MAX_RECORDS = 100;     // limite de cadastros

// ──────────────────────────────────────────────────────────────
// ELEMENTOS DO DOM
// ──────────────────────────────────────────────────────────────
const video         = document.getElementById('video');
const overlay       = document.getElementById('overlay');
const ctx           = overlay.getContext('2d');
const videoWrapper  = document.getElementById('video-wrapper');
const accessOverlay = document.getElementById('access-overlay');
const statusBadge   = document.getElementById('status-badge');

const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const btnCheck      = document.getElementById('btn-check');
const btnCapture    = document.getElementById('btn-capture');
const btnClearAll   = document.getElementById('btn-clear-all');

const tabVerify     = document.getElementById('tab-verify');
const tabRegister   = document.getElementById('tab-register');
const panelVerify   = document.getElementById('panel-verify');
const panelRegister = document.getElementById('panel-register');
const verifyResult  = document.getElementById('verify-result');
const debugInfo     = document.getElementById('debug-info');

const registryList  = document.getElementById('registry-list');
const countBadge    = document.getElementById('count-badge');

// Modal de cadastro
const modal         = document.getElementById('modal');
const modalPreview  = document.getElementById('modal-preview');
const modalInfo     = document.getElementById('modal-info');
const inputName     = document.getElementById('input-name');
const modalConfirm  = document.getElementById('modal-confirm');
const modalCancel   = document.getElementById('modal-cancel');

// Modal de resultado de verificação
const modalResult   = document.getElementById('modal-result');
const resultIcon    = document.getElementById('result-icon');
const resultTitle   = document.getElementById('result-title');
const resultPreview = document.getElementById('result-preview');
const resultInfo    = document.getElementById('result-info');
const resultClose   = document.getElementById('result-close');

// ──────────────────────────────────────────────────────────────
// ESTADO
// ──────────────────────────────────────────────────────────────
let faceMesh        = null;
let camUtil         = null;
let stream          = null;
let faceDetected    = false;
let lastResults     = null;
let currentMode     = 'verify';   // 'verify' | 'register'
let locked          = false;      // trava durante exibição de resultado
let pendingSnap     = null;       // { thumb, fullImg, landmarks }

// ──────────────────────────────────────────────────────────────
// REGISTROS — carrega do localStorage
// ──────────────────────────────────────────────────────────────
let records = [];
try {
  records = JSON.parse(localStorage.getItem('facialRecords') || '[]');
} catch (_) { records = []; }

renderRegistry();

// ──────────────────────────────────────────────────────────────
// ABAS
// ──────────────────────────────────────────────────────────────
tabVerify.addEventListener('click', () => switchTab('verify'));
tabRegister.addEventListener('click', () => switchTab('register'));

function switchTab(mode) {
  currentMode = mode;
  tabVerify.classList.toggle('active', mode === 'verify');
  tabRegister.classList.toggle('active', mode === 'register');
  panelVerify.classList.toggle('hidden', mode !== 'verify');
  panelRegister.classList.toggle('hidden', mode !== 'register');

  // Limpa resultado anterior ao trocar de aba
  verifyResult.classList.add('hidden');
  hideAccessOverlay();
}

// ──────────────────────────────────────────────────────────────
// MEDIAPIPE FACE MESH
// ──────────────────────────────────────────────────────────────
function initFaceMesh() {
  faceMesh = new FaceMesh({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });

  faceMesh.onResults(onFaceMeshResults);
}

// ──────────────────────────────────────────────────────────────
// CALLBACK PRINCIPAL — chamado a cada frame
// ──────────────────────────────────────────────────────────────
function onFaceMeshResults(results) {
  lastResults = results;

  // Sincroniza canvas
  overlay.width  = video.videoWidth  || overlay.offsetWidth;
  overlay.height = video.videoHeight || overlay.offsetHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const faces = (results.multiFaceLandmarks || []);
  faceDetected = faces.length > 0;

  // Cor dos landmarks depende do modo
  const meshColor = currentMode === 'verify'
    ? 'rgba(108,99,255,0.22)'
    : 'rgba(34,197,94,0.22)';
  const lineColor = currentMode === 'verify' ? '#6c63ff' : '#22c55e';

  faces.forEach((lm) => {
    drawConnectors(ctx, lm, FACEMESH_TESSELATION,
      { color: meshColor, lineWidth: 0.5 });
    drawConnectors(ctx, lm, FACEMESH_FACE_OVAL,
      { color: lineColor, lineWidth: 2 });
    drawConnectors(ctx, lm, FACEMESH_LEFT_EYE,
      { color: '#38bdf8', lineWidth: 1.5 });
    drawConnectors(ctx, lm, FACEMESH_RIGHT_EYE,
      { color: '#38bdf8', lineWidth: 1.5 });
    if (FACEMESH_LEFT_IRIS)
      drawConnectors(ctx, lm, FACEMESH_LEFT_IRIS,  { color: '#a78bfa', lineWidth: 1.5 });
    if (FACEMESH_RIGHT_IRIS)
      drawConnectors(ctx, lm, FACEMESH_RIGHT_IRIS, { color: '#a78bfa', lineWidth: 1.5 });
    drawConnectors(ctx, lm, FACEMESH_LIPS,
      { color: '#f472b6', lineWidth: 1.5 });
  });

  // Atualiza badge de detecção e botões
  updateStatusBadge(faces.length);
  btnCheck.disabled   = !faceDetected;
  btnCapture.disabled = !faceDetected;
}

// ──────────────────────────────────────────────────────────────
// STATUS BADGE
// ──────────────────────────────────────────────────────────────
function updateStatusBadge(count) {
  if (locked) return;
  statusBadge.className = 'status-badge';
  if (count === 0) {
    statusBadge.classList.add('status-searching');
    statusBadge.textContent = '🔍 Procurando rosto...';
  } else {
    statusBadge.classList.add('status-detected');
    statusBadge.textContent = `✅ Rosto detectado`;
  }
}

// ──────────────────────────────────────────────────────────────
// INICIAR CÂMERA
// ──────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  try {
    btnStart.disabled = true;
    btnStart.textContent = '⏳ Aguarde...';

    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    if (!faceMesh) initFaceMesh();

    camUtil = new Camera(video, {
      onFrame: async () => { await faceMesh.send({ image: video }); },
      width: 1280, height: 720,
    });
    camUtil.start();

    btnStart.classList.add('hidden');
    btnStop.disabled = false;
    statusBadge.className = 'status-badge status-searching';
    statusBadge.textContent = '🔍 Procurando rosto...';

  } catch (err) {
    btnStart.disabled = false;
    btnStart.textContent = '▶ Iniciar Câmera';
    showToast('❌ Câmera: ' + err.message, 'error');
  }
});

// ──────────────────────────────────────────────────────────────
// PARAR CÂMERA
// ──────────────────────────────────────────────────────────────
btnStop.addEventListener('click', stopCamera);

function stopCamera() {
  if (camUtil) { camUtil.stop(); camUtil = null; }
  if (stream)  { stream.getTracks().forEach(t => t.stop()); stream = null; }

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  faceDetected = false;

  btnStart.classList.remove('hidden');
  btnStart.disabled = false;
  btnStart.textContent = '▶ Iniciar Câmera';
  btnStop.disabled = true;
  btnCheck.disabled = true;
  btnCapture.disabled = true;

  statusBadge.className = 'status-badge status-idle';
  statusBadge.textContent = '⏸ Câmera desligada';
  hideAccessOverlay();
  verifyResult.classList.add('hidden');
}

// ──────────────────────────────────────────────────────────────
// VERIFICAR ACESSO
// ──────────────────────────────────────────────────────────────
btnCheck.addEventListener('click', () => {
  if (!faceDetected || locked) return;

  const currentLM = lastResults?.multiFaceLandmarks?.[0];
  if (!currentLM) return;

  // Captura thumb para exibição
  const snap   = captureFrame();
  const thumb  = cropFace(snap, currentLM);

  // Compara com cada cadastro
  const { match, dist } = findBestMatch(normalizeLandmarks(currentLM));

  // Mostra diagnóstico
  showDebug(dist, match);

  if (match) {
    showResult('allowed', match, thumb);
  } else {
    showResult('denied', null, thumb);
  }
});

// ──────────────────────────────────────────────────────────────
// MOSTRAR RESULTADO DE VERIFICAÇÃO
// ──────────────────────────────────────────────────────────────
function showResult(type, matchRecord, thumbDataUrl) {
  locked = true;

  // Overlay na câmera
  showAccessOverlay(type, matchRecord);

  // Borda do wrapper
  videoWrapper.classList.remove('allowed', 'denied');
  videoWrapper.classList.add(type);

  // Painel inline
  verifyResult.className = `verify-result ${type}`;
  verifyResult.classList.remove('hidden');

  if (type === 'allowed') {
    verifyResult.innerHTML = `
      <span class="vr-icon">✅</span>
      <div class="vr-text">
        ACESSO PERMITIDO
        <span class="vr-name">👤 ${escapeHtml(matchRecord.name)}</span>
      </div>`;
  } else {
    verifyResult.innerHTML = `
      <span class="vr-icon">🚫</span>
      <div class="vr-text">
        ACESSO NEGADO
        <span class="vr-name">Rosto não cadastrado no sistema</span>
      </div>`;
  }

  // Modal com detalhes
  resultPreview.src = thumbDataUrl;
  if (type === 'allowed') {
    resultIcon.textContent  = '✅';
    resultTitle.textContent = 'ACESSO PERMITIDO';
    resultTitle.style.color = '#4ade80';
    resultInfo.innerHTML    = `
      Bem-vindo(a), <strong>${escapeHtml(matchRecord.name)}</strong>!<br/>
      Cadastrado em: ${formatDate(matchRecord.timestamp)}
    `;
    resultPreview.style.borderColor = 'var(--success)';
  } else {
    resultIcon.textContent  = '🚫';
    resultTitle.textContent = 'ACESSO NEGADO';
    resultTitle.style.color = '#f87171';
    resultInfo.innerHTML    = `Rosto <strong>não encontrado</strong> no banco de cadastros.<br/>Solicite o cadastro ao administrador.`;
    resultPreview.style.borderColor = 'var(--danger)';
  }
  modalResult.classList.remove('hidden');

  // Libera o lock após LOCK_MS
  setTimeout(() => {
    locked = false;
    hideAccessOverlay();
    videoWrapper.classList.remove('allowed', 'denied');
  }, LOCK_MS);
}

// ──────────────────────────────────────────────────────────────
// OVERLAY NA CÂMERA
// ──────────────────────────────────────────────────────────────
function showAccessOverlay(type, matchRecord) {
  accessOverlay.className = `access-overlay ${type}`;
  accessOverlay.classList.remove('hidden');

  if (type === 'allowed') {
    accessOverlay.innerHTML = `
      <span class="access-emoji">✅</span>
      <span class="access-label">ACESSO PERMITIDO</span>
      <span class="access-name">👤 ${escapeHtml(matchRecord.name)}</span>
    `;
  } else {
    accessOverlay.innerHTML = `
      <span class="access-emoji">🚫</span>
      <span class="access-label">ACESSO NEGADO</span>
      <span class="access-name">Rosto não cadastrado</span>
    `;
  }
}

function hideAccessOverlay() {
  accessOverlay.classList.add('hidden');
  accessOverlay.className = 'access-overlay hidden';
}

// Fecha modal de resultado
resultClose.addEventListener('click', () => modalResult.classList.add('hidden'));
modalResult.addEventListener('click', (e) => {
  if (e.target === modalResult) modalResult.classList.add('hidden');
});

// ──────────────────────────────────────────────────────────────
// CAPTURAR ROSTO PARA CADASTRO
// ──────────────────────────────────────────────────────────────
btnCapture.addEventListener('click', () => {
  if (!faceDetected) return;

  const currentLM = lastResults?.multiFaceLandmarks?.[0];
  if (!currentLM) return;

  const snap     = captureFrame();
  const thumb    = cropFace(snap, currentLM);
  const normLM   = normalizeLandmarks(currentLM);

  pendingSnap = {
    thumb,
    fullImg: snap.toDataURL('image/jpeg', 0.9),
    landmarks: normLM,
  };

  // Pré-preenche modal
  modalPreview.src    = thumb;
  modalInfo.textContent = `${new Date().toLocaleString('pt-BR')}`;
  inputName.value     = '';
  modal.classList.remove('hidden');
  setTimeout(() => inputName.focus(), 80);
});

// ──────────────────────────────────────────────────────────────
// CONFIRMAR CADASTRO
// ──────────────────────────────────────────────────────────────
modalConfirm.addEventListener('click', () => {
  const name = (inputName.value || '').trim() || 'Anônimo';
  if (!pendingSnap) return;

  const record = {
    id:        Date.now(),
    name,
    thumb:     pendingSnap.thumb,
    fullImg:   pendingSnap.fullImg,
    landmarks: pendingSnap.landmarks,
    timestamp: new Date().toISOString(),
    verifyCount: 0,
  };

  records.unshift(record);
  if (records.length > MAX_RECORDS) records.pop(); // mantém limite
  persistRecords();
  renderRegistry();

  modal.classList.add('hidden');
  pendingSnap = null;
  showToast(`✅ "${name}" cadastrado com sucesso!`);
});

modalCancel.addEventListener('click', () => {
  modal.classList.add('hidden');
  pendingSnap = null;
});

inputName.addEventListener('keydown', (e) => {
  if (e.key === 'Enter')  modalConfirm.click();
  if (e.key === 'Escape') modalCancel.click();
});

modal.addEventListener('click', (e) => {
  if (e.target === modal) modalCancel.click();
});

// ──────────────────────────────────────────────────────────────
// CAPTURAR FRAME DO VÍDEO
// ──────────────────────────────────────────────────────────────
function captureFrame() {
  const snap    = document.createElement('canvas');
  snap.width    = video.videoWidth;
  snap.height   = video.videoHeight;
  const snapCtx = snap.getContext('2d');
  // Des-espelha a imagem capturada
  snapCtx.translate(snap.width, 0);
  snapCtx.scale(-1, 1);
  snapCtx.drawImage(video, 0, 0);
  return snap;
}

// ──────────────────────────────────────────────────────────────
// RECORTAR ROSTO DO CANVAS
// ──────────────────────────────────────────────────────────────
function cropFace(sourceCanvas, landmarks) {
  if (!landmarks || landmarks.length === 0)
    return sourceCanvas.toDataURL('image/jpeg', 0.85);

  const W = sourceCanvas.width;
  const H = sourceCanvas.height;

  // Nota: como des-espelhamos o canvas, precisamos inverter X
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  landmarks.forEach(({ x, y }) => {
    const rx = (1 - x) * W; // inverte X (des-espelhado)
    const ry = y * H;
    if (rx < minX) minX = rx;
    if (ry < minY) minY = ry;
    if (rx > maxX) maxX = rx;
    if (ry > maxY) maxY = ry;
  });

  const padX = (maxX - minX) * 0.28;
  const padY = (maxY - minY) * 0.28;
  const cx   = (minX + maxX) / 2;
  const cy   = (minY + maxY) / 2;
  const size = Math.max((maxX - minX) + padX * 2, (maxY - minY) + padY * 2);
  const sx   = Math.max(0, cx - size / 2);
  const sy   = Math.max(0, cy - size / 2);
  const sw   = Math.min(W - sx, size);
  const sh   = Math.min(H - sy, size);

  const thumb    = document.createElement('canvas');
  thumb.width    = 200;
  thumb.height   = 200;
  thumb.getContext('2d').drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, 200, 200);
  return thumb.toDataURL('image/jpeg', 0.88);
}

// ──────────────────────────────────────────────────────────────
// NORMALIZAR LANDMARKS  (apenas X e Y — Z é ruidoso no FaceMesh)
//  Retorna array plano [x0,y0, x1,y1, …] normalizado:
//  - translação: subtrai centróide 2D
//  - escala: distância média dos pontos ao centróide
// ──────────────────────────────────────────────────────────────
function normalizeLandmarks(landmarks) {
  const n = landmarks.length;
  let cx = 0, cy = 0;

  for (const { x, y } of landmarks) { cx += x; cy += y; }
  cx /= n; cy /= n;

  let scale = 0;
  for (const { x, y } of landmarks) {
    scale += Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
  }
  scale = (scale / n) || 1;

  // Array plano 2D
  const out = new Float32Array(n * 2);
  landmarks.forEach(({ x, y }, i) => {
    out[i * 2]     = (x - cx) / scale;
    out[i * 2 + 1] = (y - cy) / scale;
  });
  return out;
}

// ──────────────────────────────────────────────────────────────
// COMPARAR DOIS VETORES 2D DE LANDMARKS
//  Retorna distância euclidiana média por ponto (0 = idêntico)
// ──────────────────────────────────────────────────────────────
function compareLandmarks(a, b) {
  if (a.length !== b.length) return Infinity;
  const n = a.length / 2;
  let sum = 0;
  for (let i = 0; i < a.length; i += 2) {
    const dx = a[i]     - b[i];
    const dy = a[i + 1] - b[i + 1];
    sum += Math.sqrt(dx * dx + dy * dy);
  }
  return sum / n;
}

// ──────────────────────────────────────────────────────────────
// ENCONTRAR MELHOR MATCH NA BASE DE CADASTROS
// Retorna { match: record|null, dist: number }
// ──────────────────────────────────────────────────────────────
function findBestMatch(normLM) {
  if (records.length === 0) return { match: null, dist: Infinity };

  let bestRecord = null;
  let bestDist   = Infinity;

  for (const rec of records) {
    if (!rec.landmarks || rec.landmarks.length === 0) continue;

    // Após persistência, landmarks vem como Array JS normal (JSON não tem Float32Array).
    // Garante Float32Array independente do formato salvo.
    let stored;
    if (rec.landmarks instanceof Float32Array) {
      stored = rec.landmarks;
    } else if (Array.isArray(rec.landmarks)) {
      stored = new Float32Array(rec.landmarks);
    } else {
      // fallback para objeto com chaves numéricas (nunca deveria ocorrer)
      stored = new Float32Array(Object.keys(rec.landmarks).sort((a,b)=>a-b).map(k => rec.landmarks[k]));
    }

    // Tamanhos incompatíveis — registro antigo com Z (3 coords) vs novo (2 coords)
    // Nesse caso pula em vez de retornar Infinity
    if (stored.length !== normLM.length) {
      console.warn(`[FaceMatch] Cadastro "${rec.name}" tem ${stored.length} coords, esperado ${normLM.length}. Recadastre.`);
      continue;
    }

    const dist = compareLandmarks(normLM, stored);
    if (dist < bestDist) {
      bestDist   = dist;
      bestRecord = rec;
    }
  }

  console.debug(`[FaceMatch] melhor dist=${bestDist.toFixed(5)}  threshold=${THRESHOLD}  pessoa="${bestRecord?.name}"`);

  if (bestDist <= THRESHOLD) {
    bestRecord.verifyCount = (bestRecord.verifyCount || 0) + 1;
    persistRecords();
    renderRegistry();
    return { match: bestRecord, dist: bestDist };
  }
  return { match: null, dist: bestDist };
}

// ──────────────────────────────────────────────────────────────
// DIAGNÓSTICO — mostra a distância calculada ao usuário
// ──────────────────────────────────────────────────────────────
function showDebug(dist, matchRecord) {
  if (!debugInfo) return;
  const isFinite_ = isFinite(dist);
  const pct       = isFinite_ ? Math.min(100, (dist / 0.20) * 100) : 100;
  const color     = dist <= THRESHOLD ? '#22c55e'
                  : dist <= THRESHOLD * 1.5 ? '#f59e0b'
                  : '#ef4444';
  const label     = matchRecord ? `"${escapeHtml(matchRecord.name)}"` : 'nenhum';

  debugInfo.classList.remove('hidden');
  debugInfo.innerHTML = `
    Distância: <strong>${isFinite_ ? dist.toFixed(5) : '∞'}</strong>
    &nbsp;|&nbsp; Limite: ${THRESHOLD}
    &nbsp;|&nbsp; Mais próximo: ${label}
    <div class="dist-bar">
      <div class="dist-fill" style="width:${pct}%;background:${color}"></div>
    </div>
  `;
}


function persistRecords() {
  try {
    // Float32Array não serializa direto — converte para Array normal
    const toSave = records.map(r => ({
      ...r,
      landmarks: r.landmarks ? Array.from(r.landmarks) : [],
    }));
    localStorage.setItem('facialRecords', JSON.stringify(toSave));
  } catch (e) {
    console.warn('localStorage cheio:', e);
    showToast('⚠ Armazenamento cheio. Exclua cadastros antigos.', 'warning');
  }
}

function renderRegistry() {
  countBadge.textContent = records.length;

  if (records.length === 0) {
    registryList.innerHTML = `
      <div class="empty-state">
        <span>📭</span>
        <p>Nenhum rosto cadastrado.</p>
        <small>Use a aba "Registrar Rosto" para começar.</small>
      </div>`;
    return;
  }

  registryList.innerHTML = records.map((r) => `
    <div class="face-card" data-id="${r.id}">
      <img
        class="face-thumb"
        src="${r.thumb}"
        alt="${escapeHtml(r.name)}"
        title="Ver foto"
        onclick="viewRecord(${r.id})"
      />
      <div class="face-info">
        <div class="face-name">${escapeHtml(r.name)}</div>
        <div class="face-time">📅 ${formatDate(r.timestamp)}</div>
        <div class="face-count">🔓 ${r.verifyCount || 0} verificação(ões)</div>
      </div>
      <div class="face-actions">
        <button class="btn-icon" title="Download" onclick="downloadRecord(${r.id})">⬇</button>
        <button class="btn-icon" title="Remover"  onclick="deleteRecord(${r.id})">🗑</button>
      </div>
    </div>
  `).join('');
}

// ──────────────────────────────────────────────────────────────
// LIMPAR TODOS
// ──────────────────────────────────────────────────────────────
btnClearAll.addEventListener('click', () => {
  if (records.length === 0) return;
  if (!confirm(`Remover todos os ${records.length} cadastro(s)?`)) return;
  records = [];
  persistRecords();
  renderRegistry();
});

// ──────────────────────────────────────────────────────────────
// VER / DOWNLOAD / DELETAR REGISTRO
// ──────────────────────────────────────────────────────────────
function viewRecord(id) {
  const r = records.find(x => x.id === id);
  if (!r) return;
  resultIcon.textContent  = '👤';
  resultTitle.textContent = escapeHtml(r.name);
  resultTitle.style.color = 'var(--text)';
  resultPreview.src       = r.thumb;
  resultPreview.style.borderColor = 'var(--primary)';
  resultInfo.innerHTML    = `
    Cadastrado em: ${formatDate(r.timestamp)}<br/>
    Verificações: ${r.verifyCount || 0}
  `;
  modalResult.classList.remove('hidden');
}

function downloadRecord(id) {
  const r = records.find(x => x.id === id);
  if (!r) return;
  const a    = document.createElement('a');
  a.href     = r.fullImg || r.thumb;
  a.download = `${r.name.replace(/\s+/g, '_')}_${r.id}.jpg`;
  a.click();
}

function deleteRecord(id) {
  if (!confirm('Remover este cadastro?')) return;
  records = records.filter(x => x.id !== id);
  persistRecords();
  renderRegistry();
}

// ──────────────────────────────────────────────────────────────
// UTILITÁRIOS
// ──────────────────────────────────────────────────────────────
function formatDate(iso) {
  return new Date(iso).toLocaleString('pt-BR', {
    day:'2-digit', month:'2-digit', year:'numeric',
    hour:'2-digit', minute:'2-digit',
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showToast(message, type = 'success') {
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  const colors = { success:'#22c55e', error:'#ef4444', warning:'#f59e0b' };
  toast.style.cssText = `
    position:fixed; bottom:28px; left:50%;
    transform:translateX(-50%);
    background:${colors[type]||colors.success};
    color:#fff; padding:12px 24px;
    border-radius:999px; font-weight:600; font-size:0.88rem;
    box-shadow:0 4px 20px rgba(0,0,0,.4);
    z-index:9999; white-space:nowrap; max-width:90vw;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

// ──────────────────────────────────────────────────────────────
// Expõe funções usadas em handlers inline
// ──────────────────────────────────────────────────────────────
window.viewRecord     = viewRecord;
window.downloadRecord = downloadRecord;
window.deleteRecord   = deleteRecord;
