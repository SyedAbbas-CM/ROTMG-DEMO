/**
 * WorldContext manager: owns per-world managers and broadcasting helpers.
 */
export function createWorldContextManager({
  wss,
  mapManager,
  itemManager,
  logger,
  sendToClient,
  getClients, // () => Map<clientId, { socket, player, mapId }>
  classes,
  pvpEnabled = false
}) {
  const {
    BulletManager,
    EnemyManager,
    CollisionManager,
    BagManager,
    SoldierManager,
    UnitSystems,
    UnitNetworkAdapter
  } = classes;

  // Per-world manager containers: mapId → { bulletMgr, enemyMgr, collMgr, bagMgr, soldierMgr, unitSystems, unitNetAdapter }
  const worldContexts = new Map();

  function getWorldCtx(mapId) {
    if (!worldContexts.has(mapId)) {
      const bulletMgr = new BulletManager(10000);
      const enemyMgr  = new EnemyManager(1000, itemManager);
      const collMgr   = new CollisionManager(bulletMgr, enemyMgr, mapManager, null, null, { pvpEnabled });
      const bagMgr    = new BagManager(500);

      // Initialize Unit Systems
      const soldierMgr = new SoldierManager(2000);
      const unitSystems = new UnitSystems(soldierMgr, mapManager);
      const unitNetAdapter = new UnitNetworkAdapter(wss, soldierMgr, unitSystems);

      enemyMgr._bagManager = bagMgr; // inject for drops

      logger.info('worldCtx',`Created managers for world ${mapId} including unit systems`);
      worldContexts.set(mapId, { bulletMgr, enemyMgr, collMgr, bagMgr, soldierMgr, unitSystems, unitNetAdapter });
    }
    return worldContexts.get(mapId);
  }

  // Helper to broadcast to all clients in a given world
  function broadcastToWorld(mapId, type, payload){
    const clients = getClients();
    clients.forEach((c)=>{
      if(c.mapId===mapId){
        sendToClient(c.socket, type, payload);
      }
    });
  }

  function cleanupAll() {
    worldContexts.forEach((ctx) => {
      if (ctx.collMgr?.cleanup) ctx.collMgr.cleanup();
      if (ctx.enemyMgr?.cleanup) ctx.enemyMgr.cleanup();
      if (ctx.bulletMgr?.cleanup) ctx.bulletMgr.cleanup();
    });
  }

  return { getWorldCtx, broadcastToWorld, cleanupAll, worldContexts };
}


