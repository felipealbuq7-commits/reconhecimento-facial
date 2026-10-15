/**
 * server.js — API REST para o Sistema de Reconhecimento Facial
 * =============================================================
 * Endpoints:
 *   GET    /api/faces           → lista todos os cadastros
 *   POST   /api/faces           → cria novo cadastro
 *   DELETE /api/faces/:id       → remove cadastro
 *   DELETE /api/faces           → remove todos
 *   PATCH  /api/faces/:id/verify → incrementa contador de verificações
 *
 * Banco de dados: db.json (arquivo local no mesmo diretório)
 * Estrutura do db.json:
 * {
 *   "faces": [
 *     {
 *       "id": 1234567890,
 *       "name": "João Silva",
 *       "thumb": "data:image/jpeg;base64,...",
 *       "landmarks": [0.123, -0.456, ...],   ← array plano 2D normalizado (468×2 = 936 números)
 *       "timestamp": "2026-10-15T12:00:00.000Z",
 *       "verifyCount": 3
 *     }
 *   ]
 * }
 */

'use strict';

const express = require('express');
const cors    = require('cors');
const fs      = require('fs');
const path    = require('path');

const app  = express();
const PORT = 3001;
const DB_PATH = path.join(__dirname, 'db.json');

// ── Middlewares ──────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '10mb' }));   // landmarks + thumb podem ser grandes

// Serve os arquivos estáticos do frontend (index.html, app.js, style.css)
app.use(express.static(__dirname));

// ── Helpers do banco de dados ────────────────────────────────

/** Lê o db.json e retorna o objeto { faces: [] } */
function readDB() {
  try {
    const raw = fs.readFileSync(DB_PATH, 'utf8');
    return JSON.parse(raw);
  } catch (_) {
    // Arquivo não existe ainda → cria estrutura vazia
    return { faces: [] };
  }
}

/** Salva o objeto { faces: [] } no db.json de forma legível */
function writeDB(data) {
  fs.writeFileSync(DB_PATH, JSON.stringify(data, null, 2), 'utf8');
}

// Garante que o arquivo existe ao iniciar
if (!fs.existsSync(DB_PATH)) {
  writeDB({ faces: [] });
  console.log('✅ db.json criado.');
}

// ── ROTAS ────────────────────────────────────────────────────

/**
 * GET /api/faces
 * Lista todos os cadastros.
 * Os landmarks são enviados normalmente (array de números).
 */
app.get('/api/faces', (req, res) => {
  try {
    const db = readDB();
    res.json({ ok: true, faces: db.faces });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

/**
 * POST /api/faces
 * Cria um novo cadastro facial.
 * Body esperado:
 * {
 *   "name": "string",
 *   "thumb": "data:image/jpeg;base64,...",
 *   "landmarks": [number, ...]    ← array plano 2D (936 elementos)
 * }
 */
app.post('/api/faces', (req, res) => {
  try {
    const { name, thumb, landmarks } = req.body;

    if (!name || typeof name !== 'string') {
      return res.status(400).json({ ok: false, error: 'Campo "name" obrigatório.' });
    }
    if (!Array.isArray(landmarks) || landmarks.length === 0) {
      return res.status(400).json({ ok: false, error: 'Campo "landmarks" obrigatório (array).' });
    }

    const db = readDB();

    const record = {
      id:          Date.now(),
      name:        name.trim().slice(0, 100),
      thumb:       thumb || null,
      landmarks:   landmarks,          // array plano de números
      timestamp:   new Date().toISOString(),
      verifyCount: 0,
    };

    db.faces.unshift(record);          // mais recente no topo
    writeDB(db);

    console.log(`[DB] Cadastro salvo: "${record.name}" (id=${record.id}, ${landmarks.length} coords)`);
    res.status(201).json({ ok: true, face: record });

  } catch (err) {
    console.error('[DB] Erro ao salvar:', err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

/**
 * PATCH /api/faces/:id/verify
 * Incrementa o contador de verificações de um cadastro.
 */
app.patch('/api/faces/:id/verify', (req, res) => {
  try {
    const id = Number(req.params.id);
    const db = readDB();
    const face = db.faces.find(f => f.id === id);

    if (!face) {
      return res.status(404).json({ ok: false, error: 'Cadastro não encontrado.' });
    }

    face.verifyCount = (face.verifyCount || 0) + 1;
    face.lastVerified = new Date().toISOString();
    writeDB(db);

    res.json({ ok: true, verifyCount: face.verifyCount });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

/**
 * DELETE /api/faces/:id
 * Remove um cadastro pelo ID.
 */
app.delete('/api/faces/:id', (req, res) => {
  try {
    const id = Number(req.params.id);
    const db = readDB();
    const before = db.faces.length;
    db.faces = db.faces.filter(f => f.id !== id);

    if (db.faces.length === before) {
      return res.status(404).json({ ok: false, error: 'Cadastro não encontrado.' });
    }

    writeDB(db);
    console.log(`[DB] Cadastro removido: id=${id}`);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

/**
 * DELETE /api/faces
 * Remove TODOS os cadastros.
 */
app.delete('/api/faces', (req, res) => {
  try {
    writeDB({ faces: [] });
    console.log('[DB] Todos os cadastros removidos.');
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

// ── Inicia o servidor ─────────────────────────────────────────
app.listen(PORT, () => {
  console.log('');
  console.log('╔══════════════════════════════════════════════╗');
  console.log('║   🛡  Sistema de Reconhecimento Facial       ║');
  console.log('╠══════════════════════════════════════════════╣');
  console.log(`║  Servidor rodando em http://localhost:${PORT}   ║`);
  console.log(`║  Banco de dados:  db.json                    ║`);
  console.log('║                                              ║');
  console.log('║  Abra o navegador em:                        ║');
  console.log(`║  http://localhost:${PORT}                       ║`);
  console.log('╚══════════════════════════════════════════════╝');
  console.log('');
});
