import fs from 'fs';
import path from 'path';

const LOG_DIR = path.join(process.cwd(), 'logs');
const LOG_FILE = path.join(LOG_DIR, 'boss_llm.jsonl');

function ensureDir() {
  if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
}

export function logLLM(entry) {
  try {
    const line = JSON.stringify({...entry,ts:Date.now()})+'\n';
    fs.appendFileSync(LOG_FILE,line,'utf8');
  }catch(err){ /* ignore */ }
}

/**
 * Helper to log a rating event separate from generation event.
 * @param {string} planId
 * @param {number} rating
 */
export function logLLMRating(planId, rating) {
  try {
    ensureDir();
    fs.appendFileSync(LOG_FILE, JSON.stringify({ type: 'rating', planId, rating, ts: Date.now() }) + '\n');
  } catch (err) {
    console.warn('[LLMLogger] failed to write rating', err);
  }
} 