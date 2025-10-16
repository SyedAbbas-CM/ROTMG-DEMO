import express from 'express';
import fs from 'fs';
import path from 'path';

export default function createAssetsRoutes(baseDir) {
  const router = express.Router();

  const imagesDirBase = path.join(baseDir, 'public', 'assets', 'images');
  const atlasesDirBase = path.join(baseDir, 'public', 'assets', 'atlases');

  // GET /api/assets/images – flat list of image paths relative to public/
  router.get('/images', (_req, res) => {
    const images = [];
    const walk = (dir, base = '') => {
      fs.readdirSync(dir, { withFileTypes: true }).forEach((ent) => {
        const full = path.join(dir, ent.name);
        const rel = path.posix.join(base, ent.name);
        if (ent.isDirectory()) return walk(full, rel);
        if (/\.(png|jpe?g|gif)$/i.test(ent.name)) {
          images.push('assets/images/' + rel.replace(/\\/g, '/'));
        }
      });
    };
    try {
      walk(imagesDirBase);
      res.json({ images });
    } catch (err) {
      console.error('[ASSETS] Error generating image list', err);
      res.status(500).json({ error: 'Failed to list images' });
    }
  });

  // GET /api/assets/images/tree – nested folder tree structure
  router.get('/images/tree', (_req, res) => {
    const buildNode = (dir) => {
      const node = { name: path.basename(dir), type: 'folder', children: [] };
      fs.readdirSync(dir, { withFileTypes: true }).forEach((ent) => {
        const full = path.join(dir, ent.name);
        if (ent.isDirectory()) {
          node.children.push(buildNode(full));
        } else if (/\.(png|jpe?g|gif)$/i.test(ent.name)) {
          node.children.push({
            name: ent.name,
            type: 'image',
            path: 'assets/images/' + path.relative(imagesDirBase, full).replace(/\\/g, '/'),
          });
        }
      });
      return node;
    };
    try {
      res.json(buildNode(imagesDirBase));
    } catch (err) {
      console.error('[ASSETS] Error building image tree', err);
      res.status(500).json({ error: 'Failed to build tree' });
    }
  });

  // GET /api/assets/atlases – list of atlas JSON files
  router.get('/atlases', (_req, res) => {
    try {
      const atlases = fs
        .readdirSync(atlasesDirBase)
        .filter((f) => f.endsWith('.json'))
        .map((f) => '/assets/atlases/' + f);
      res.json({ atlases });
    } catch (err) {
      console.error('[ASSETS] Error listing atlases', err);
      res.status(500).json({ error: 'Failed to list atlases' });
    }
  });

  // GET /api/assets/atlas/:file – fetch a single atlas JSON by filename
  router.get('/atlas/:file', (req, res) => {
    const filename = req.params.file;
    // Allow only simple filenames like "chars2.json" – prevents path traversal
    if (!/^[\w-]+\.json$/.test(filename)) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    const atlasPath = path.join(atlasesDirBase, filename);
    if (!fs.existsSync(atlasPath)) {
      return res.status(404).json({ error: 'Atlas not found' });
    }
    try {
      const data = JSON.parse(fs.readFileSync(atlasPath, 'utf8'));
      res.json(data);
    } catch (err) {
      console.error('[ASSETS] Error reading atlas', err);
      res.status(500).json({ error: 'Failed to read atlas' });
    }
  });

  // POST /api/assets/atlases/save – persist atlas JSON sent from the editor
  router.post('/atlases/save', (req, res) => {
    const { filename, data } = req.body || {};
    if (!filename || !data) {
      return res.status(400).json({ error: 'filename and data required' });
    }
    if (!/^[\w-]+\.json$/.test(filename)) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    const atlasPath = path.join(atlasesDirBase, filename);
    try {
      fs.writeFileSync(atlasPath, JSON.stringify(data, null, 2));
      res.json({ success: true, path: '/assets/atlases/' + filename });
    } catch (err) {
      console.error('[ASSETS] Error saving atlas', err);
      res.status(500).json({ error: 'Failed to save atlas' });
    }
  });

  // POST /api/assets/images/save – save a base64-encoded PNG image to public/assets/images
  router.post('/images/save', (req, res) => {
    const { path: relPath, data } = req.body || {};
    if (!relPath || !data || !data.startsWith('data:image/png;base64,')) {
      return res.status(400).json({ error: 'path and base64 data required' });
    }
    // Prevent path traversal, force .png only
    if (relPath.includes('..') || !relPath.toLowerCase().endsWith('.png')) {
      return res.status(400).json({ error: 'Invalid path' });
    }
    const abs = path.join(baseDir, 'public', relPath);
    try {
      const pngBuf = Buffer.from(data.split(',')[1], 'base64');
      fs.writeFileSync(abs, pngBuf);
      res.json({ success: true });
    } catch (err) {
      console.error('[ASSETS] Error saving image', err);
      res.status(500).json({ error: 'Failed to save image' });
    }
  });

  return router;
}


