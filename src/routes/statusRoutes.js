import express from 'express';

export default function createStatusRoutes({ getWorldCtx, mapManager, getClients }) {
  const router = express.Router();

  router.get('/', (_req, res) => {
    const clients = getClients();
    const worlds = Array.from(mapManager.maps?.keys?.() || []);
    const worldSummaries = worlds.map(mapId => {
      const ctx = getWorldCtx(mapId);
      return {
        mapId,
        enemies: ctx.enemyMgr.enemyCount,
        bullets: ctx.bulletMgr.bulletCount,
        units: ctx.soldierMgr?.count || 0
      };
    });
    res.json({
      time: Date.now(),
      clientCount: clients.size,
      worlds: worldSummaries
    });
  });

  return router;
}


