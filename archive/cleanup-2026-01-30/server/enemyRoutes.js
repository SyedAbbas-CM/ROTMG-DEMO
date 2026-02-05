import express from 'express';

function requireAdminTokenIfConfigured(req, res, next) {
  const expected = process.env.ADMIN_TOKEN;
  if (!expected) return next();
  const provided = req.headers['x-admin-token'];
  if (provided && provided === expected) return next();
  return res.status(403).json({ error: 'forbidden' });
}

function createLimiter(limit = 30, windowMs = 10000) {
  const hits = new Map();
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

export default function createEnemyRoutes({ mapManager, getWorldCtx }) {
  const router = express.Router();

  // List enemies for a map
  router.get('/list', (req, res) => {
    try {
      const mapId = req.query.mapId;
      if (!mapId) return res.status(400).json({ error: 'mapId required' });
      const ctx = getWorldCtx(mapId);
      const enemies = ctx.enemyMgr.getEnemiesData(mapId);
      res.json({ enemies });
    } catch (err) {
      console.error('[Enemies] list failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // Spawn enemy immediately in running world
  router.post('/spawn', requireAdminTokenIfConfigured, limitWrites, (req, res) => {
    try {
      const { mapId, id, type, x, y } = req.body || {};
      if (!mapId || (!id && (type === undefined || type === null)) || !Number.isFinite(x) || !Number.isFinite(y)) {
        return res.status(400).json({ error: 'mapId, (id or type), x, y required' });
      }
      const ctx = getWorldCtx(mapId);
      let enemyId;
      if (id !== undefined && id !== null) enemyId = ctx.enemyMgr.spawnEnemyById(id, x, y, mapId);
      else enemyId = ctx.enemyMgr.spawnEnemy(Number(type) || 0, x, y, mapId);
      return res.json({ ok: true, enemyId });
    } catch (err) {
      console.error('[Enemies] spawn failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // Remove enemy by id
  router.post('/remove', requireAdminTokenIfConfigured, limitWrites, (req, res) => {
    try {
      const { mapId, enemyId } = req.body || {};
      if (!mapId || !enemyId) return res.status(400).json({ error: 'mapId and enemyId required' });
      const ctx = getWorldCtx(mapId);
      const index = ctx.enemyMgr.findIndexById(enemyId);
      if (index === -1) return res.status(404).json({ error: 'enemy not found' });
      ctx.enemyMgr.removeEnemy(index);
      return res.json({ ok: true });
    } catch (err) {
      console.error('[Enemies] remove failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  // Remove enemy at tile position (nearest match within epsilon)
  router.post('/remove-at', requireAdminTokenIfConfigured, limitWrites, (req, res) => {
    try {
      const { mapId, x, y, eps = 0.6 } = req.body || {};
      if (!mapId || !Number.isFinite(x) || !Number.isFinite(y)) {
        return res.status(400).json({ error: 'mapId, x, y required' });
      }
      const ctx = getWorldCtx(mapId);
      const list = ctx.enemyMgr.getEnemiesData(mapId);
      const found = list.find(e => Math.abs(e.x - x) <= eps && Math.abs(e.y - y) <= eps);
      if (!found) return res.status(404).json({ error: 'enemy not found near coords' });
      const idx = ctx.enemyMgr.findIndexById(found.id);
      if (idx === -1) return res.status(404).json({ error: 'enemy not found' });
      ctx.enemyMgr.removeEnemy(idx);
      return res.json({ ok: true, removedId: found.id });
    } catch (err) {
      console.error('[Enemies] remove-at failed', err);
      res.status(500).json({ error: 'failed' });
    }
  });

  return router;
}


