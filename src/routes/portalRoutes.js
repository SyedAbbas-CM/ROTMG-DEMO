import express from 'express';

// Factory: create a router bound to a specific mapManager
function requireAdminTokenIfConfigured(req, res, next) {
  const expected = process.env.ADMIN_TOKEN;
  if (!expected) return next();
  const provided = req.headers['x-admin-token'];
  if (provided && provided === expected) return next();
  return res.status(403).json({ error: 'forbidden' });
}

function createLimiter(limit = 20, windowMs = 10000) {
  const hits = new Map(); // ip -> {count, ts}
  return function limitWrites(req, res, next) {
    const now = Date.now();
    const ip = req.ip || req.headers['x-forwarded-for'] || 'local';
    const rec = hits.get(ip) || { count: 0, ts: now };
    if (now - rec.ts > windowMs) { rec.count = 0; rec.ts = now; }
    rec.count++;
    hits.set(ip, rec);
    if (rec.count > limit) return res.status(429).json({ error: 'rate_limited' });
    next();
  };
}

const limitWrites = createLimiter();

export default function createPortalRoutes(mapManager) {
  const router = express.Router();

  function getPortalsForMap(mapId) {
    const meta = mapManager.getMapMetadata(mapId);
    if (!meta) return [];
    const objects = meta.objects || [];
    return objects.filter(o => o && o.type === 'portal' && o.destMap);
  }

  // GET /api/portals/list?mapId=...
  router.get('/list', (req, res) => {
    try {
      const mapId = req.query.mapId;
      if (mapId) {
        return res.json({ mapId, portals: getPortalsForMap(mapId) });
      }
      const result = {};
      for (const map of mapManager.maps?.values?.() || []) {
        result[map.id] = getPortalsForMap(map.id);
      }
      res.json(result);
    } catch (err) {
      console.error('[Portals] list failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // POST /api/portals/add { mapId, x, y, destMap, sprite? }
  router.post('/add', requireAdminTokenIfConfigured, limitWrites, (req, res) => {
    try {
      const { mapId, x, y, destMap, sprite } = req.body || {};
      if (!mapId || !destMap || !Number.isFinite(x) || !Number.isFinite(y)) {
        return res.status(400).json({ error: 'mapId, destMap, x, y required' });
      }
      const meta = mapManager.getMapMetadata(mapId);
      if (!meta) return res.status(404).json({ error: 'map not found' });
      if (!meta.objects) meta.objects = [];
      const portalObj = {
        id: `portal_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
        type: 'portal',
        sprite: sprite || 'portal',
        x: Math.floor(x),
        y: Math.floor(y),
        destMap
      };
      meta.objects.push(portalObj);
      res.json({ ok: true, portal: portalObj });
    } catch (err) {
      console.error('[Portals] add failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // POST /api/portals/remove { mapId, x, y, destMap }
  router.post('/remove', requireAdminTokenIfConfigured, limitWrites, (req, res) => {
    try {
      const { mapId, x, y, destMap } = req.body || {};
      if (!mapId || !Number.isFinite(x) || !Number.isFinite(y)) {
        return res.status(400).json({ error: 'mapId, x, y required' });
      }
      const meta = mapManager.getMapMetadata(mapId);
      if (!meta) return res.status(404).json({ error: 'map not found' });
      if (!Array.isArray(meta.objects)) meta.objects = [];
      const initial = meta.objects.length;
      meta.objects = meta.objects.filter(o => {
        if (!o || o.type !== 'portal') return true;
        const matchPos = o.x === Math.floor(x) && o.y === Math.floor(y);
        const matchDest = destMap ? (o.destMap === destMap) : true;
        return !(matchPos && matchDest);
      });
      const removed = initial - meta.objects.length;
      res.json({ ok: true, removed });
    } catch (err) {
      console.error('[Portals] remove failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // POST /api/portals/link-both { from, to, x, y, backX?, backY? }
  router.post('/link-both', requireAdminTokenIfConfigured, limitWrites, (req, res) => {
    try {
      const { from, to, x, y, backX = 2, backY = 2 } = req.body || {};
      if (!from || !to || !Number.isFinite(x) || !Number.isFinite(y)) {
        return res.status(400).json({ error: 'from, to, x, y required' });
      }
      const fromMeta = mapManager.getMapMetadata(from);
      const toMeta = mapManager.getMapMetadata(to);
      if (!fromMeta || !toMeta) return res.status(404).json({ error: 'map(s) not found' });
      if (!fromMeta.objects) fromMeta.objects = [];
      if (!toMeta.objects) toMeta.objects = [];
      const a = { id: `portal_${Date.now()}a`, type: 'portal', sprite: 'portal', x: Math.floor(x), y: Math.floor(y), destMap: to };
      const b = { id: `portal_${Date.now()}b`, type: 'portal', sprite: 'portal', x: Math.floor(backX), y: Math.floor(backY), destMap: from };
      fromMeta.objects.push(a);
      toMeta.objects.push(b);
      res.json({ ok: true, created: { a, b } });
    } catch (err) {
      console.error('[Portals] link-both failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  return router;
}


