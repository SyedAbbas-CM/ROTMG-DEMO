// public/src/shared/messages.js

// Single source of truth for message types and light helpers used by both
// client (browser) and server (Node) via direct ES imports.

export const MessageType = {
  // System messages
  HEARTBEAT: 0,

  // Connection messages
  HANDSHAKE: 1,
  HANDSHAKE_ACK: 2,
  PING: 3,
  PONG: 4,

  // Game state messages
  PLAYER_JOIN: 10,
  PLAYER_LEAVE: 11,
  PLAYER_UPDATE: 12,
  PLAYER_LIST: 13,
  PLAYER_DEATH: 14,
  PLAYER_RESPAWN: 15,
  CHARACTER_SELECT: 16,  // Server->Client: prompt class selection (all chars dead)

  // Entity messages
  ENEMY_LIST: 20,
  ENEMY_UPDATE: 21,
  ENEMY_DEATH: 22,

  // Bullet messages
  BULLET_CREATE: 30,
  BULLET_LIST: 31,
  BULLET_REMOVE: 32,

  // Loot bags
  BAG_LIST: 33,
  // Loot interaction
  PICKUP_ITEM: 34,
  INVENTORY_UPDATE: 35,
  BAG_REMOVE: 36,
  PICKUP_DENIED: 37,
  MOVE_ITEM: 38,
  MOVE_DENIED: 39,
  EQUIP_ITEM: 42,      // Client->Server: equip item from inventory slot
  UNEQUIP_ITEM: 43,    // Client->Server: unequip item to inventory
  EQUIPMENT_UPDATE: 44, // Server->Client: equipment changed

  // Collision messages
  COLLISION: 40,
  COLLISION_RESULT: 41,

  // Map messages
  MAP_INFO: 50,
  CHUNK_REQUEST: 51,
  CHUNK_DATA: 52,
  CHUNK_NOT_FOUND: 53,

  // Portal/world
  PORTAL_ENTER: 54,
  WORLD_SWITCH: 55,

  // World update
  WORLD_UPDATE: 60,

  // Map request
  MAP_REQUEST: 70,

  // Player list request
  PLAYER_LIST_REQUEST: 80,

  // Chat and text
  CHAT_MESSAGE: 90,
  SPEECH: 91,
  PLAYER_TEXT: 92,

  // Unit control
  UNIT_COMMAND: 100,
  UNIT_UPDATE: 101,
  UNIT_SPAWN: 102
};

// Helpers to build/validate message payloads (lightweight – no runtime deps)
export function makeChunkDataPayload(chunkX, chunkY, data){
  return { chunkX, chunkY, data };
}

export function makeWorldSwitchPayload(meta, spawnX, spawnY){
  return {
    mapId: meta.mapId,
    spawnX,
    spawnY,
    width: meta.width,
    height: meta.height,
    tileSize: meta.tileSize,
    chunkSize: meta.chunkSize,
    procedural: meta.procedural,
    seed: meta.seed,
    objects: meta.objects || [],
    enemySpawns: meta.enemySpawns || [],
    timestamp: Date.now()
  };
}

// In the browser, expose MessageType on window for legacy modules
if (typeof window !== 'undefined') {
  window.MessageType = MessageType;
}


