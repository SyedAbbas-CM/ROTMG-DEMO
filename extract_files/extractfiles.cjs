/*  extractfiles.cjs  ────────────────────────────────────────────────
 *  - Case-insensitive file matching
 *  - Section parsing  (  SECTION: core | all  +  [core] …  )
 *  - Writes chunks into   ./chunks/<section>/chunk-N.txt
 *  --------------------------------------------------------------- */

const fs   = require('fs');
const path = require('path');

// ── CONFIG ─────────────────────────────────────────────────────────
const MAX_CHARS   = 24000;               // GPT-4o safe
const INPUT_FILE  = 'input.txt';         // file list + sections
const SEARCH_ROOT = path.resolve(__dirname, '..'); // project root
const CHUNK_DIR   = 'chunks';            // base output folder
// ──────────────────────────────────────────────────────────────────

// Small colour helpers for console output
const col = {
  green : (s) => `\x1b[32m${s}\x1b[0m`,
  red   : (s) => `\x1b[31m${s}\x1b[0m`,
  yellow: (s) => `\x1b[33m${s}\x1b[0m`
};

/* ------------------------------------------------------------------
   findFileInsensitive(relPath)
   Walks the filesystem segment-by-segment, matching each part
   case-insensitively, and returns the first absolute path found.
------------------------------------------------------------------ */
function findFileInsensitive(relPath) {
  const parts = relPath.split(/[\\/]/).filter(Boolean);
  let current = SEARCH_ROOT;
  for (const part of parts) {
    const entries = fs.readdirSync(current);
    const hit = entries.find(e => e.toLowerCase() === part.toLowerCase());
    if (!hit) return null;
    current = path.join(current, hit);
  }
  return current;
}

/* ------------------------------------------------------------------
   parseInputFile() → { sections: {name: [paths]}, target }
   - Recognises SECTION: xyz  (first non-blank line)
   - Recognises [xyz] headers
------------------------------------------------------------------ */
function parseInputFile(txt) {
  const lines = txt.split(/\r?\n/);
  const sections = {};
  let current = 'default';

  // Look for directive SECTION: NAME
  let targetSection = null;
  const firstRealLine = lines.find(l => l.trim().length);
  const matchDirective = firstRealLine && firstRealLine.match(/^SECTION\s*:\s*(.+)$/i);
  if (matchDirective) {
    targetSection = matchDirective[1].trim().toLowerCase();
  }

  for (let line of lines) {
    line = line.trim();
    if (!line || /^SECTION\s*:/i.test(line)) continue;          // skip blank / directive
    const header = line.match(/^\[(.+?)\]$/);                   // [section]
    if (header) {
      current = header[1].trim().toLowerCase();
      continue;
    }
    // strip trailing description after '–' or '--'
    const clean = line.split(/–|--/)[0].trim();
    if (clean.match(/[\w./-]+\.\w+$/)) {
      if (!sections[current]) sections[current] = [];
      sections[current].push(clean.replace(/\\/g, '/'));
    }
  }
  return { sections, targetSection };
}

/* ------------------------------------------------------------------ */
function chunkText(text, max) {
  const out = [];
  let buf = '';
  for (const ln of text.split('\n')) {
    if (buf.length + ln.length + 1 > max) { out.push(buf); buf = ''; }
    buf += ln + '\n';
  }
  if (buf.trim()) out.push(buf);
  return out;
}

/* ------------------------------------------------------------------ */
function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
}

/* ------------------------------------------------------------------ */
function deleteOldChunks(dir) {
  if (!fs.existsSync(dir)) return;
  for (const f of fs.readdirSync(dir)) {
    if (f.startsWith('chunk-') && f.endsWith('.txt')) {
      fs.unlinkSync(path.join(dir, f));
    }
  }
}

/* ------------------------------------------------------------------ */
function run() {
  if (!fs.existsSync(INPUT_FILE)) {
    console.error(col.red(`❌ ${INPUT_FILE} not found`));
    return;
  }

  const { sections, targetSection } = parseInputFile(
    fs.readFileSync(INPUT_FILE, 'utf8')
  );

  const sectionsToProcess =
    !targetSection || targetSection === 'all' || targetSection === '*'
      ? Object.keys(sections)
      : [targetSection];

  if (sectionsToProcess.length === 0) {
    console.error(col.red('❌ No matching section to process.'));
    return;
  }

  for (const sec of sectionsToProcess) {
    const list = sections[sec];
    if (!list || list.length === 0) {
      console.warn(col.yellow(`⚠️  Section “[${sec}]” has no files.`));
      continue;
    }

    console.log(col.yellow(`\n=== SECTION [${sec}] ===`));
    console.log(col.yellow(`📄 ${list.length} requested paths`));

    const found = [];
    const missing = [];
    const blobs = [];

    for (const rel of list) {
      const abs = findFileInsensitive(rel);
      if (abs && fs.existsSync(abs)) {
        found.push(rel);
        blobs.push(`// FILE: ${rel}\n${fs.readFileSync(abs, 'utf8')}`);
      } else {
        missing.push(rel);
        blobs.push(`// FILE: ${rel}\n// ❌ Not found\n`);
      }
    }

    console.log(col.green(`✅ Found ${found.length}`));
    if (missing.length)
      console.log(col.red(`⚠️  Missing ${missing.length}`));

    // Write chunks ---------------------------------------------------
    const outDir = path.join(CHUNK_DIR, sec);
    ensureDir(outDir);
    deleteOldChunks(outDir);

    const chunks = chunkText(blobs.join('\n\n'), MAX_CHARS);
    chunks.forEach((chunk, i) => {
      const fname = path.join(outDir, `chunk-${i + 1}.txt`);
      fs.writeFileSync(fname, chunk, 'utf8');
      console.log(col.green(`📦  ${fname}`));
    });
    console.log(col.yellow(`🧩 Total chunks for [${sec}]: ${chunks.length}`));
  }
}

run();
