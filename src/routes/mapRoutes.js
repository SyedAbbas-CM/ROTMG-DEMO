import express from 'express';

function requireAdminTokenIfConfigured(req, res, next) {
  const expected = process.env.ADMIN_TOKEN;
  if (!expected) return next();
  const provided = req.headers['x-admin-token'];
  if (provided && provided === expected) return next();
  return res.status(403).json({ error: 'forbidden' });
}

export default function createMapRoutes({ mapManager, bootstrap, storedMapsRef }) {
  const router = express.Router();

  // GET /api/maps/:id/meta – return map metadata
  router.get('/:id/meta', (req, res) => {
    const mapId = req.params.id;
    const meta = mapManager.getMapMetadata(mapId);
    if (!meta) return res.status(404).json({ error: 'map not found' });
    res.json(meta);
  });

  // POST /api/maps/entrypoints/set { mapId, points:[{x,y}] }
  router.post('/entrypoints/set', requireAdminTokenIfConfigured, (req, res) => {
    const { mapId, points } = req.body || {};
    if (!mapId || !Array.isArray(points)) {
      return res.status(400).json({ error: 'mapId and points[] required' });
    }
    const meta = mapManager.getMapMetadata(mapId);
    if (!meta) return res.status(404).json({ error: 'map not found' });
    meta.entryPoints = points.map(p => ({ x: Math.floor(p.x), y: Math.floor(p.y) }));
    return res.json({ ok: true, entryPoints: meta.entryPoints });
  });

  // POST /api/maps/reload – reapply bootstrap config (idempotent)
  router.post('/reload', requireAdminTokenIfConfigured, async (_req, res) => {
    try {
      const cfg = bootstrap.loadConfig('src/config/game.config.json');
      await bootstrap.apply(cfg);
      // Rebuild stored maps index
      storedMapsRef.clear();
      for (const map of mapManager.maps?.values?.() || []) {
        storedMapsRef.set(map.id, mapManager.getMapMetadata(map.id));
      }
      res.json({ ok: true, maps: Array.from(storedMapsRef.keys()) });
    } catch (err) {
      console.error('[Maps] reload failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // POST /api/maps/save-active { mapId, filename? }
  router.post('/save-active', requireAdminTokenIfConfigured, async (req, res) => {
    try {
      const { mapId, filename } = req.body || {};
      if (!mapId) return res.status(400).json({ error: 'mapId required' });
      const meta = mapManager.getMapMetadata(mapId);
      if (!meta) return res.status(404).json({ error: 'map not found' });

      // Build a minimal persistable JSON including layers if available
      const out = {
        id: meta.id,
        name: meta.name,
        width: meta.width,
        height: meta.height,
        tileSize: meta.tileSize,
        chunkSize: meta.chunkSize,
        objects: meta.objects || [],
        enemySpawns: meta.enemySpawns || [],
        entryPoints: meta.entryPoints || [],
        portals: meta.objects?.filter?.(o => o?.type === 'portal') || meta.portals || []
      };

      // Use MapManager.saveMap (simple metadata) to maps directory
      const fname = filename || `${mapId}.json`;
      const ok = await mapManager.saveMap(mapId, fname);
      if (!ok) return res.status(500).json({ error: 'save failed' });
      res.json({ ok: true, file: fname });
    } catch (err) {
      console.error('[Maps] save-active failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  return router;
}

// File: /src/routes/mapRoutes.js
// NOTE: This route file was using CommonJS in an ESM project. It appears unused by the server.
// Converting to ESM and keeping minimal functionality in case the editor or tools reference it.
import express from 'express';
const router = express.Router();

// Example tile-based map
const mapSize = 100*100;
let map = new Array(mapSize).fill(0);

router.get('/', (req, res) => {
  res.json({ map });
});

router.post('/change', (req, res) => {
  const { location, block } = req.body;
  if (location >= 0 && location < map.length) {
    map[location] = block;
    return res.json({ success: true });
  }
  res.json({ success: false });
});

export default router;
