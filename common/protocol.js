// common/protocol.js
// Single source of truth for binary framing and message type IDs

export class BinaryPacket {
  static encode(type, data) {
    const jsonStr = JSON.stringify(data ?? {});
    const jsonBytes = new TextEncoder().encode(jsonStr);
    const packet = new ArrayBuffer(5 + jsonBytes.byteLength);
    const view = new DataView(packet);
    view.setUint8(0, type);
    view.setUint32(1, jsonBytes.byteLength, true);
    new Uint8Array(packet, 5).set(jsonBytes);
    return packet;
  }

  static decode(packet) {
    const view = new DataView(packet);
    const type = view.getUint8(0);
    const length = view.getUint32(1, true);
    const jsonBytes = new Uint8Array(packet, 5, length);
    const jsonStr = new TextDecoder().decode(jsonBytes);
    try {
      const data = JSON.parse(jsonStr);
      return { type, data };
    } catch {
      return { type, data: {} };
    }
  }
}

export const MessageType = {
  // System / heartbeat
  HEARTBEAT: 0,

  // Connection
  HANDSHAKE: 1,
  HANDSHAKE_ACK: 2,
  PING: 3,
  PONG: 4,

  // Players
  PLAYER_JOIN: 10,
  PLAYER_LEAVE: 11,
  PLAYER_UPDATE: 12,
  PLAYER_LIST: 13,

  // Enemies
  ENEMY_LIST: 20,
  ENEMY_UPDATE: 21,
  ENEMY_DEATH: 22,

  // Bullets
  BULLET_CREATE: 30,
  BULLET_LIST: 31,
  BULLET_REMOVE: 32,

  // Loot / inventory
  BAG_LIST: 33,
  PICKUP_ITEM: 34,
  INVENTORY_UPDATE: 35,
  BAG_REMOVE: 36,
  PICKUP_DENIED: 37,
  MOVE_ITEM: 38,
  MOVE_DENIED: 39,

  // Collisions
  COLLISION: 40,
  COLLISION_RESULT: 41,

  // Map & world
  MAP_INFO: 50,
  CHUNK_REQUEST: 51,
  CHUNK_DATA: 52,
  CHUNK_NOT_FOUND: 53,
  PORTAL_ENTER: 54,
  WORLD_SWITCH: 55,
  WORLD_UPDATE: 60,
  MAP_REQUEST: 70,

  // Diagnostics
  PLAYER_LIST_REQUEST: 80,

  // Chat / speech
  CHAT_MESSAGE: 90,
  SPEECH: 91
};


