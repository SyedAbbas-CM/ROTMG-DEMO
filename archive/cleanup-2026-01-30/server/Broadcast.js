import { MessageType } from '../../public/src/shared/messages.js';
import { NETWORK_SETTINGS } from '../../public/src/constants/constants.js';

export function createBroadcaster({ mapManager, getWorldCtx, getClients, sendToClient }) {
  function broadcastWorldUpdates() {
    const now = Date.now();
    const UPDATE_RADIUS = NETWORK_SETTINGS.UPDATE_RADIUS_TILES;
    const UPDATE_RADIUS_SQ = UPDATE_RADIUS * UPDATE_RADIUS;

    // Group clients by map
    const clients = getClients();
    const clientsByMap = new Map();
    clients.forEach((client, id) => {
      const m = client.mapId;
      if (!clientsByMap.has(m)) clientsByMap.set(m, new Set());
      clientsByMap.get(m).add(id);
    });

    clientsByMap.forEach((idSet, mapId) => {
      const players = {};
      idSet.forEach(cid => { players[cid] = clients.get(cid).player; });

      const ctx = getWorldCtx(mapId);
      const enemies = ctx.enemyMgr.getEnemiesData(mapId);
      const bullets = ctx.bulletMgr.getBulletsData(mapId);
      const units = ctx.soldierMgr ? ctx.soldierMgr.getSoldiersData() : [];

      const meta = mapManager.getMapMetadata(mapId) || { width: 0, height: 0 };
      const clamp = (arr) => arr.filter(o => o.x >= 0 && o.y >= 0 && o.x < meta.width && o.y < meta.height);
      const enemiesClamped = clamp(enemies);
      const bulletsClamped = clamp(bullets);
      const unitsClamped = clamp(units);
      const bags = clamp(ctx.bagMgr.getBagsData(mapId));
      const objects = mapManager.getObjects(mapId);

      idSet.forEach(cid => {
        const c = clients.get(cid);
        if (!c) return;
        const px = c.player.x;
        const py = c.player.y;

        const within = (arr) => arr.filter(e => { const dx = e.x - px; const dy = e.y - py; return (dx*dx + dy*dy) <= UPDATE_RADIUS_SQ; });

        const payload = {
          players,
          enemies: within(enemiesClamped).slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
          bullets: within(bulletsClamped).slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
          units: within(unitsClamped).slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
          bags: within(bags),
          objects,
          timestamp: now
        };

        if (ctx.bulletMgr.stats) {
          payload.bulletStats = { ...ctx.bulletMgr.stats };
        }

        sendToClient(c.socket, MessageType.WORLD_UPDATE, payload);
      });

      // Send player list
      idSet.forEach(cid => {
        const c = clients.get(cid);
        if (c) sendToClient(c.socket, MessageType.PLAYER_LIST, players);
      });

      // Reset per-frame bullet stats
      if (ctx.bulletMgr.stats) {
        ctx.bulletMgr.stats.wallHit = 0;
        ctx.bulletMgr.stats.entityHit = 0;
        ctx.bulletMgr.stats.created = 0;
      }
    });
  }

  return { broadcastWorldUpdates };
}


