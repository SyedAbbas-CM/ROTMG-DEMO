import express from 'express';
import fs from 'fs';
import path from 'path';

const router = express.Router();

const LOG_FILE = path.join(process.cwd(), 'logs', 'boss_llm.jsonl');
function readLines(n = 50) {
  if (!fs.existsSync(LOG_FILE)) return [];
  const lines = fs.readFileSync(LOG_FILE, 'utf8').trim().split('\n');
  return lines.slice(-n).map(l => {
    try { return JSON.parse(l); } catch(_) { return null; }
  }).filter(Boolean);
}

// GET /api/llm/history → last 50 log entries
router.get('/history', (req, res) => {
  res.json(readLines(50));
});

// POST /api/llm/rate { planId:string, rating:number }
router.post('/rate', (req, res) => {
  const { planId, rating } = req.body || {};
  if (!planId || typeof rating !== 'number') {
    return res.status(400).json({ error: 'planId and rating required' });
  }
  const entry = { type: 'rating_manual', planId, rating, ts: Date.now() };
  try {
    fs.appendFileSync(LOG_FILE, JSON.stringify(entry) + '\n');
    res.json({ ok: true });
  } catch (err) {
    console.error('[LLMRoutes] Failed to append rating', err);
    res.status(500).json({ error: 'failed' });
  }
});

export default router; 