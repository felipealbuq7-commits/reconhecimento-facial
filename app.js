/**
 * app.js — Sistema de Reconhecimento Facial com MediaPipe FaceMesh
 * =================================================================
 * Persistência: API REST em http://localhost:3001/api/faces  (server.js → db.json)
 *
 * Como funciona a comparação:
 *   1. Ao cadastrar, os 468 landmarks são normalizados (translação +
 *      escala, usando apenas X e Y — Z é ruidoso) e enviados ao servidor.
 *   2. Ao verificar, o mesmo processo é feito no rosto atual e
 *      calcula-se a distância euclidiana 2D média contra cada cadastro.
 *   3. Se a menor distância for ≤ THRESHOLD → Acesso Permitido.
 *
 * Calibração do THRESHOLD:
 *   Mesmo rosto, variações normais de ângulo/luz : dist ≈ 0.03 – 0.07
 *   Rostos diferentes                            : dist ≈ 0.10 – 0.20+
 */

'use strict';

// ──────────────────────────────────────────────────────────────
// CONSTANTES
// ──────────────────────────────────────────────────────────────
const API_BASE  = 'http://localhost:3001/api';
const THRESHOLD = 0.072;
const LOCK_MS   = 3000;

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

const modal         = document.getElementById('modal');
const modalPreview  = document.getElementById('modal-preview');
const modalInfo     = document.getElementById('modal-info');
const inputName     = document.getElementById('input-name');
const modalConfirm  = document.getElementById('modal-confirm');
const modalCancel   = document.getElementById('modal-cancel');

const modalResult   = document.getElementById('modal-result');
const resultIcon    = document.getElementById('result-icon');
const resultTitle   = document.getElementById('result-title');
const resultPreview = document.getElementById('result-preview');
const resultInfo    = document.getElementById('result-info');
const resultClose   = document.getElementById('result-close');

// ──────────────────────────────────────────────────────────────
// ESTADO
// ──────────────────────────────────────────────────────────────
let faceMesh     = null;
let camUtil      = null;
let stream       = null;
let faceDetected = false;
let lastResults  = null;
let currentMode  = 'verify';
let locked       = false;
let pendingSnap  = null;   // { thumb, landmarks }
let records      = [];     // cache em memória dos cadastros do servidor

// ──────────────────────────────────────────────────────────────
// INICIALIZAÇÃO — carrega cadastros do servidor
// ──────────────────────────────────────────────────────────────
(async function init() {
  await loadFaces();
})();

// ──────────────────────────────────────────────────────────────
// API — funções de comunicação com o servidor
// ──────────────────────────────────────────────────────────────

/** Busca todos os cadastros e atualiza o cache + lista */
async function loadFaces() {
  try {
    const res  = await fetch(`${API_BASE}/faces`);
    const data = await res.json();
    if (data.ok) {
      records = data.faces;
      renderRegistry();
    }
  } catch (err) {
    showToast('⚠ Servidor offline. Verifique se node server.js está rodando.', 'warning');
    console.error('[API] loadFaces:', err);
  }
}

/** Envia novo cadastro ao servidor */
async function apiSaveFace(name, thumb, landmarks) {
  const res = await fetch(`${API_BASE}/faces`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ name, thumb, landmarks }),
  });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || 'Erro ao salvar.');
  return data.face;
}

/** Incrementa contador de verificações */
async function apiIncrementVerify(id) {
  try {
    await fetch(`${API_BASE}/faces/${id}/verify`, { method: 'PATCH' });
  } catch (_) { /* silencia — não é crítico */ }
}

/** Remove um cadastro */
async function apiDeleteFace(id) {
  const res  = await fetch(`${API_BASE}/faces/${id}`, { method: 'DELETE' });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || 'Erro ao remover.');
}

/** Remove todos os cadastros */
async function apiDeleteAll() {
  const res  = await fetch(`${API_BASE}/faces`, { method: 'DELETE' });
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || 'Erro ao limpar.');
}

// ──────────────────────────────────────────────────────────────
// ABAS
// ──────────────────────────────────────────────────────────────
tabVerify.addEventListener('click',   () => switchTab('verify'));
tabRegister.addEventListener('click', () => switchTab('register'));

function switchTab(mode) {
  currentMode = mode;
  tabVerify.classList.toggle('active',   mode === 'verify');
  tabRegister.classList.toggle('active', mode === 'register');
  panelVerify.classList.toggle('hidden',   mode !== 'verify');
  panelRegister.classList.toggle('hidden', mode !== 'register');
  verifyResult.classList.add('hidden');
  if (debugInfo) debugInfo.classList.add('hidden');
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
    minTrackingConfidence:  0.55,
  });
  faceMesh.onResults(onFaceMeshResults);
}

// ──────────────────────────────────────────────────────────────
// CALLBACK PRINCIPAL — chamado a cada frame
// ──────────────────────────────────────────────────────────────
function onFaceMeshResults(results) {
  lastResults = results;

  overlay.width  = video.videoWidth  || overlay.offsetWidth;
  overlay.height = video.videoHeight || overlay.offsetHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const faces = results.multiFaceLandmarks || [];
  faceDetected = faces.length > 0;

  const meshColor = currentMode === 'verify' ? 'rgba(108,99,255,0.22)' : 'rgba(34,197,94,0.22)';
  const lineColor = currentMode === 'verify' ? '#6c63ff' : '#22c55e';

  faces.forEach((lm) => {
    drawConnectors(ctx, lm, FACEMESH_TESSELATION, { color: meshColor, lineWidth: 0.5 });
    drawConnectors(ctx, lm, FACEMESH_FACE_OVAL,   { color: lineColor, lineWidth: 2 });
    drawConnectors(ctx, lm, FACEMESH_LEFT_EYE,    { color: '#38bdf8', lineWidth: 1.5 });
    drawConnectors(ctx, lm, FACEMESH_RIGHT_EYE,   { color: '#38bdf8', lineWidth: 1.5 });
    if (FACEMESH_LEFT_IRIS)  drawConnectors(ctx, lm, FACEMESH_LEFT_IRIS,  { color: '#a78bfa', lineWidth: 1.5 });
    if (FACEMESH_RIGHT_IRIS) drawConnectors(ctx, lm, FACEMESH_RIGHT_IRIS, { color: '#a78bfa', lineWidth: 1.5 });
    drawConnectors(ctx, lm, FACEMESH_LIPS, { color: '#f472b6', lineWidth: 1.5 });
  });

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
    statusBadge.textContent = '✅ Rosto detectado';
  }
}

// ──────────────────────────────────────────────────────────────
// INICIAR CÂMERA
// ──────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  try {
    btnStart.disabled    = true;
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
    btnStart.disabled    = false;
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
  btnStart.disabled    = false;
  btnStart.textContent = '▶ Iniciar Câmera';
  btnStop.disabled     = true;
  btnCheck.disabled    = true;
  btnCapture.disabled  = true;

  statusBadge.className   = 'status-badge status-idle';
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

  const snap  = captureFrame();
  const thumb = cropFace(snap, currentLM);

  const { match, dist } = findBestMatch(normalizeLandmarks(currentLM));

  showDebug(dist, match);

  if (match) {
    // Incrementa contador no servidor (assíncrono, não bloqueia)
    apiIncrementVerify(match.id).then(() => {
      const rec = records.find(r => r.id === match.id);
      if (rec) { rec.verifyCount = (rec.verifyCount || 0) + 1; renderRegistry(); }
    });
    showResult('allowed', match, thumb);
  } else {
    showResult('denied', null, thumb);
  }
});

// ──────────────────────────────────────────────────────────────
// MOSTRAR RESULTADO
// ──────────────────────────────────────────────────────────────
function showResult(type, matchRecord, thumbDataUrl) {
  locked = true;

  showAccessOverlay(type, matchRecord);
  videoWrapper.classList.remove('allowed', 'denied');
  videoWrapper.classList.add(type);

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

  resultPreview.src = thumbDataUrl;
  if (type === 'allowed') {
    resultIcon.textContent  = '✅';
    resultTitle.textContent = 'ACESSO PERMITIDO';
    resultTitle.style.color = '#4ade80';
    resultInfo.innerHTML    = `Bem-vindo(a), <strong>${escapeHtml(matchRecord.name)}</strong>!<br/>
      Cadastrado em: ${formatDate(matchRecord.timestamp)}`;
    resultPreview.style.borderColor = 'var(--success)';
  } else {
    resultIcon.textContent  = '🚫';
    resultTitle.textContent = 'ACESSO NEGADO';
    resultTitle.style.color = '#f87171';
    resultInfo.innerHTML    = `Rosto <strong>não encontrado</strong> no banco de dados.<br/>Solicite o cadastro ao administrador.`;
    resultPreview.style.borderColor = 'var(--danger)';
  }
  modalResult.classList.remove('hidden');

  setTimeout(() => {
    locked = false;
    hideAccessOverlay();
    videoWrapper.classList.remove('allowed', 'denied');
  }, LOCK_MS);
}

// ──────────────────────────────────────────────────────────────
// OVERLAYS
// ──────────────────────────────────────────────────────────────
function showAccessOverlay(type, matchRecord) {
  accessOverlay.className = `access-overlay ${type}`;
  accessOverlay.classList.remove('hidden');
  if (type === 'allowed') {
    accessOverlay.innerHTML = `
      <span class="access-emoji">✅</span>
      <span class="access-label">ACESSO PERMITIDO</span>
      <span class="access-name">👤 ${escapeHtml(matchRecord.name)}</span>`;
  } else {
    accessOverlay.innerHTML = `
      <span class="access-emoji">🚫</span>
      <span class="access-label">ACESSO NEGADO</span>
      <span class="access-name">Rosto não cadastrado</span>`;
  }
}

function hideAccessOverlay() {
  accessOverlay.classList.add('hidden');
  accessOverlay.className = 'access-overlay hidden';
}

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

  const snap   = captureFrame();
  const thumb  = cropFace(snap, currentLM);
  const normLM = normalizeLandmarks(currentLM);

  pendingSnap = { thumb, landmarks: Array.from(normLM) };

  modalPreview.src      = thumb;
  modalInfo.textContent = new Date().toLocaleString('pt-BR');
  inputName.value       = '';
  modal.classList.remove('hidden');
  setTimeout(() => inputName.focus(), 80);
});

// ──────────────────────────────────────────────────────────────
// CONFIRMAR CADASTRO → SALVA NO SERVIDOR (db.json)
// ──────────────────────────────────────────────────────────────
modalConfirm.addEventListener('click', async () => {
  const name = (inputName.value || '').trim() || 'Anônimo';
  if (!pendingSnap) return;

  modalConfirm.disabled    = true;
  modalConfirm.textContent = '⏳ Salvando...';

  try {
    const saved = await apiSaveFace(name, pendingSnap.thumb, pendingSnap.landmarks);
    records.unshift(saved);
    renderRegistry();
    modal.classList.add('hidden');
    pendingSnap = null;
    showToast(`✅ "${name}" cadastrado e salvo no banco de dados!`);
  } catch (err) {
    showToast('❌ Erro ao salvar: ' + err.message, 'error');
  } finally {
    modalConfirm.disabled    = false;
    modalConfirm.textContent = '✔ Salvar';
  }
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
  const sCtx    = snap.getContext('2d');
  sCtx.translate(snap.width, 0);
  sCtx.scale(-1, 1);
  sCtx.drawImage(video, 0, 0);
  return snap;
}

// ──────────────────────────────────────────────────────────────
// RECORTAR ROSTO
// ──────────────────────────────────────────────────────────────
function cropFace(sourceCanvas, landmarks) {
  if (!landmarks || landmarks.length === 0)
    return sourceCanvas.toDataURL('image/jpeg', 0.85);

  const W = sourceCanvas.width;
  const H = sourceCanvas.height;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  landmarks.forEach(({ x, y }) => {
    const rx = (1 - x) * W;
    const ry = y * H;
    if (rx < minX) minX = rx;
    if (ry < minY) minY = ry;
    if (rx > maxX) maxX = rx;
    if (ry > maxY) maxY = ry;
  });

  const cx   = (minX + maxX) / 2;
  const cy   = (minY + maxY) / 2;
  const size = Math.max((maxX - minX) * 1.56, (maxY - minY) * 1.56);
  const sx   = Math.max(0, cx - size / 2);
  const sy   = Math.max(0, cy - size / 2);
  const sw   = Math.min(W - sx, size);
  const sh   = Math.min(H - sy, size);

  const thumb  = document.createElement('canvas');
  thumb.width  = 200;
  thumb.height = 200;
  thumb.getContext('2d').drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, 200, 200);
  return thumb.toDataURL('image/jpeg', 0.88);
}

// ──────────────────────────────────────────────────────────────
// NORMALIZAR LANDMARKS (X e Y apenas — Z é ruidoso)
// ──────────────────────────────────────────────────────────────
function normalizeLandmarks(landmarks) {
  const n = landmarks.length;
  let cx = 0, cy = 0;
  for (const { x, y } of landmarks) { cx += x; cy += y; }
  cx /= n; cy /= n;

  let scale = 0;
  for (const { x, y } of landmarks)
    scale += Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
  scale = (scale / n) || 1;

  const out = new Float32Array(n * 2);
  landmarks.forEach(({ x, y }, i) => {
    out[i * 2]     = (x - cx) / scale;
    out[i * 2 + 1] = (y - cy) / scale;
  });
  return out;
}

// ──────────────────────────────────────────────────────────────
// COMPARAR DOIS VETORES 2D (distância euclidiana média)
// ──────────────────────────────────────────────────────────────
function compareLandmarks(a, b) {
  if (a.length !== b.length) return Infinity;
  const n = a.length / 2;
  let sum = 0;
  for (let i = 0; i < a.length; i += 2) {
    const dx = a[i] - b[i];
    const dy = a[i + 1] - b[i + 1];
    sum += Math.sqrt(dx * dx + dy * dy);
  }
  return sum / n;
}

// ──────────────────────────────────────────────────────────────
// ENCONTRAR MELHOR MATCH (contra cache em memória)
// ──────────────────────────────────────────────────────────────
function findBestMatch(normLM) {
  if (records.length === 0) return { match: null, dist: Infinity };

  let bestRecord = null;
  let bestDist   = Infinity;

  for (const rec of records) {
    if (!rec.landmarks || rec.landmarks.length === 0) continue;

    // Converte array JSON → Float32Array para cálculo
    const stored = new Float32Array(rec.landmarks);

    if (stored.length !== normLM.length) {
      console.warn(`[Match] "${rec.name}": ${stored.length} coords ≠ ${normLM.length}. Recadastre.`);
      continue;
    }

    const dist = compareLandmarks(normLM, stored);
    if (dist < bestDist) { bestDist = dist; bestRecord = rec; }
  }

  console.debug(`[Match] dist=${bestDist.toFixed(5)} | threshold=${THRESHOLD} | pessoa="${bestRecord?.name}"`);

  return bestDist <= THRESHOLD
    ? { match: bestRecord, dist: bestDist }
    : { match: null,       dist: bestDist };
}

// ──────────────────────────────────────────────────────────────
// DIAGNÓSTICO
// ──────────────────────────────────────────────────────────────
function showDebug(dist, matchRecord) {
  if (!debugInfo) return;
  const finite = isFinite(dist);
  const pct    = finite ? Math.min(100, (dist / 0.20) * 100) : 100;
  const color  = dist <= THRESHOLD         ? '#22c55e'
               : dist <= THRESHOLD * 1.5   ? '#f59e0b'
               :                             '#ef4444';
  const label  = matchRecord ? `"${escapeHtml(matchRecord.name)}"` : 'nenhum';

  debugInfo.classList.remove('hidden');
  debugInfo.innerHTML = `
    Distância: <strong>${finite ? dist.toFixed(5) : '∞'}</strong>
    &nbsp;|&nbsp; Limite: ${THRESHOLD}
    &nbsp;|&nbsp; Mais próximo: ${label}
    <div class="dist-bar"><div class="dist-fill" style="width:${pct}%;background:${color}"></div></div>
  `;
}

// ──────────────────────────────────────────────────────────────
// RENDERIZAR LISTA
// ──────────────────────────────────────────────────────────────
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
      <img class="face-thumb" src="${r.thumb || ''}" alt="${escapeHtml(r.name)}"
           onclick="viewRecord(${r.id})" title="Ver detalhes" />
      <div class="face-info">
        <div class="face-name">${escapeHtml(r.name)}</div>
        <div class="face-time">📅 ${formatDate(r.timestamp)}</div>
        <div class="face-count">🔓 ${r.verifyCount || 0} verificação(ões)</div>
      </div>
      <div class="face-actions">
        <button class="btn-icon" title="Remover" onclick="deleteRecord(${r.id})">🗑</button>
      </div>
    </div>
  `).join('');
}

// ──────────────────────────────────────────────────────────────
// LIMPAR TODOS
// ──────────────────────────────────────────────────────────────
btnClearAll.addEventListener('click', async () => {
  if (records.length === 0) return;
  if (!confirm(`Remover todos os ${records.length} cadastro(s) do banco de dados?`)) return;
  try {
    await apiDeleteAll();
    records = [];
    renderRegistry();
    showToast('🗑 Todos os cadastros foram removidos.');
  } catch (err) {
    showToast('❌ ' + err.message, 'error');
  }
});

// ──────────────────────────────────────────────────────────────
// VER / DELETAR CADASTRO
// ──────────────────────────────────────────────────────────────
function viewRecord(id) {
  const r = records.find(x => x.id === id);
  if (!r) return;
  resultIcon.textContent  = '👤';
  resultTitle.textContent = escapeHtml(r.name);
  resultTitle.style.color = 'var(--text)';
  resultPreview.src       = r.thumb || '';
  resultPreview.style.borderColor = 'var(--primary)';
  resultInfo.innerHTML    = `
    Cadastrado em: ${formatDate(r.timestamp)}<br/>
    Verificações: <strong>${r.verifyCount || 0}</strong><br/>
    Landmarks: <strong>${r.landmarks?.length || 0}</strong> valores
  `;
  modalResult.classList.remove('hidden');
}

async function deleteRecord(id) {
  if (!confirm('Remover este cadastro do banco de dados?')) return;
  try {
    await apiDeleteFace(id);
    records = records.filter(x => x.id !== id);
    renderRegistry();
    showToast('🗑 Cadastro removido.');
  } catch (err) {
    showToast('❌ ' + err.message, 'error');
  }
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
    position:fixed;bottom:28px;left:50%;transform:translateX(-50%);
    background:${colors[type]||colors.success};color:#fff;
    padding:12px 24px;border-radius:999px;font-weight:600;font-size:.88rem;
    box-shadow:0 4px 20px rgba(0,0,0,.4);z-index:9999;
    white-space:nowrap;max-width:90vw;
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

// Expõe para handlers inline
window.viewRecord   = viewRecord;
window.deleteRecord = deleteRecord;
