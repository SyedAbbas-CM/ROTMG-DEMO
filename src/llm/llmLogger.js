import fs from 'fs';
import path from 'path';

const LOG_DIR = path.join(process.cwd(), 'logs');
const LOG_FILE = path.join(LOG_DIR, 'boss_llm.jsonl');

function ensureDir() {
  if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
}

export function logLLM(entryObj) {
  try {
    ensureDir();
    fs.appendFileSync(LOG_FILE, JSON.stringify(entryObj) + '\n');
  } catch (err) {
    console.warn('[LLMLogger] failed to write', err);
  }
} 