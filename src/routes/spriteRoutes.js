// server/src/routes/spriteSheetRoutes.js
import express from 'express';
import fs      from 'fs';
import path    from 'path';

const router = express.Router();

/** absolute path to /public/assets/images (png sheets live here) */
const IMAGES_DIR = path.join(process.cwd(), 'public', 'assets', 'images');

/* ――― helpers ―――─────────────────────────────────────────────── */
/* walk the directory tree and collect every “*.png” – returns paths
   relative to IMAGES_DIR and always with forward‑slashes */
function findPngFiles(dir, base = '') {
  const list = fs.readdirSync(dir, { withFileTypes: true });
  let out    = [];

  for (const entry of list) {
    const rel  = path.join(base, entry.name);
    const full = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      out = out.concat(findPngFiles(full, rel));
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.png')) {
      out.push(rel.replace(/\\/g, '/'));
    }
  }
  return out;
}

/* ――― routes ―――──────────────────────────────────────────────── */

/** GET /api/spritesheets           → [ "hero.png", "folder/ships.png", … ] */
router.get('/', (req, res) => {
  try {
    const files = findPngFiles(IMAGES_DIR);
    res.json(files);
  } catch (e) {
    console.error('[spriteSheetRoutes] list error:', e);
    res.status(500).json({ error: 'fs_error' });
  }
});

/** POST /api/spritesheets/<encodedPath>.png.json  
    Body = JSON produced by the sprite‑editor. */
router.post('/:encodedPath', (req, res) => {
  try {
    const rel = decodeURIComponent(req.params.encodedPath);  // ex: "folder/tank.png.json"

    /* basic safety */
    if (!rel.endsWith('.json') || rel.includes('..')) {
      return res.status(400).json({ error: 'bad_path' });
    }

    const fullPath = path.join(IMAGES_DIR, rel);
    fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    fs.writeFileSync(fullPath, JSON.stringify(req.body, null, 2));

    res.json({ success: true });
  } catch (e) {
    console.error('[spriteSheetRoutes] save error:', e);
    res.status(500).json({ error: 'fs_error' });
  }
});

export default router;
