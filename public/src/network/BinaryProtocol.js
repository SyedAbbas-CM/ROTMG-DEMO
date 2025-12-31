// public/src/network/BinaryProtocol.js
// Client-side binary protocol decoder for high-performance game updates
// Mirrors server-side common/BinaryProtocol.js

// Sprite registry (synchronized from server on handshake)
const spriteById = [];
const spriteRegistry = new Map();

// Entity ID registry (synchronized from server)
const entityById = [];
const entityRegistry = new Map();

export function registerSprite(id, name) {
  spriteById[id] = name;
  spriteRegistry.set(name, id);
}

export function getSpriteName(id) {
  return spriteById[id] || null;
}

export function registerEntity(id, stringId) {
  entityById[id] = stringId;
  entityRegistry.set(stringId, id);
}

export function getEntityStringId(numId) {
  return entityById[numId] || `entity_${numId}`;
}

// Fixed-point conversion (0.01 precision)
const FIXED_POINT_SCALE = 100;

function fromFixedPoint(value) {
  return value / FIXED_POINT_SCALE;
}

function fromVelocity(value) {
  return value / 10;
}

/**
 * Binary packet types
 */
export const BinaryPacketType = {
  FULL_SYNC: 0x01,
  DELTA_UPDATE: 0x02,
  ENTITY_CREATE: 0x03,
  ENTITY_REMOVE: 0x04,
  WORLD_DELTA: 0x10,
};

/**
 * Delta flags
 */
export const DeltaFlags = {
  POSITION: 0x01,
  VELOCITY: 0x02,
  HEALTH: 0x04,
  STATE: 0x08,
  ALL: 0xFF,
};

/**
 * BinaryReader - efficient buffer reading
 */
export class BinaryReader {
  constructor(buffer) {
    this.buffer = buffer instanceof ArrayBuffer ? buffer : buffer.buffer;
    this.view = new DataView(this.buffer);
    this.offset = 0;
  }

  readUint8() {
    return this.view.getUint8(this.offset++);
  }

  readInt8() {
    return this.view.getInt8(this.offset++);
  }

  readUint16() {
    const val = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readInt16() {
    const val = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readUint32() {
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readFloat32() {
    const val = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readPosition() {
    return {
      x: fromFixedPoint(this.readInt16()),
      y: fromFixedPoint(this.readInt16())
    };
  }

  readVelocity() {
    return {
      vx: fromVelocity(this.readInt8()),
      vy: fromVelocity(this.readInt8())
    };
  }

  readEntityId() {
    return getEntityStringId(this.readUint16());
  }

  readSprite() {
    return getSpriteName(this.readUint8());
  }

  hasMore() {
    return this.offset < this.buffer.byteLength;
  }
}

/**
 * Decode a bullet from binary
 */
export function decodeBullet(reader) {
  const id = reader.readEntityId();
  const deltaFlags = reader.readUint8();

  const bullet = { id };

  if (deltaFlags & DeltaFlags.POSITION) {
    const pos = reader.readPosition();
    bullet.x = pos.x;
    bullet.y = pos.y;
  }

  if (deltaFlags & DeltaFlags.VELOCITY) {
    const vel = reader.readVelocity();
    bullet.vx = vel.vx;
    bullet.vy = vel.vy;
  }

  if (deltaFlags & DeltaFlags.STATE) {
    bullet.ownerId = reader.readEntityId();
    bullet.spriteName = reader.readSprite();
    bullet.damage = reader.readUint8();
    bullet.faction = reader.readUint8();
    bullet.life = reader.readUint8() / 10;
  }

  return bullet;
}

/**
 * Decode an enemy from binary
 */
export function decodeEnemy(reader) {
  const id = reader.readEntityId();
  const deltaFlags = reader.readUint8();

  const enemy = { id };

  if (deltaFlags & DeltaFlags.POSITION) {
    const pos = reader.readPosition();
    enemy.x = pos.x;
    enemy.y = pos.y;
  }

  if (deltaFlags & DeltaFlags.HEALTH) {
    enemy.health = reader.readUint16();
    enemy.maxHealth = reader.readUint16();
  }

  if (deltaFlags & DeltaFlags.STATE) {
    enemy.spriteName = reader.readSprite();
    enemy.type = getSpriteName(reader.readUint8());
    const stateFlags = reader.readUint8();
    enemy.isDying = !!(stateFlags & 0x01);
    enemy.isFlashing = !!(stateFlags & 0x02);
  }

  return enemy;
}

/**
 * Decode a player from binary
 */
export function decodePlayer(reader) {
  const id = reader.readEntityId();
  const deltaFlags = reader.readUint8();

  const player = { id };

  if (deltaFlags & DeltaFlags.POSITION) {
    const pos = reader.readPosition();
    player.x = pos.x;
    player.y = pos.y;
  }

  if (deltaFlags & DeltaFlags.VELOCITY) {
    const vel = reader.readVelocity();
    player.vx = vel.vx;
    player.vy = vel.vy;
  }

  if (deltaFlags & DeltaFlags.HEALTH) {
    player.health = reader.readUint16();
    player.maxHealth = reader.readUint16();
  }

  if (deltaFlags & DeltaFlags.STATE) {
    const stateFlags = reader.readUint8();
    player.isDead = !!(stateFlags & 0x01);
    player.level = reader.readUint8();
  }

  return player;
}

/**
 * Decode a complete world delta update
 */
export function decodeWorldDelta(buffer) {
  const reader = new BinaryReader(buffer);

  const type = reader.readUint8();
  if (type !== BinaryPacketType.WORLD_DELTA) {
    console.error(`[BinaryProtocol] Invalid packet type: ${type}, expected ${BinaryPacketType.WORLD_DELTA}`);
    return null;
  }

  const timestamp = reader.readUint32();

  // Removed entities
  const removedCount = reader.readUint16();
  const removed = [];
  for (let i = 0; i < removedCount; i++) {
    removed.push(reader.readEntityId());
  }

  // Players
  const playerCount = reader.readUint16();
  const players = {};
  for (let i = 0; i < playerCount; i++) {
    const player = decodePlayer(reader);
    if (player.id) {
      players[player.id] = player;
    }
  }

  // Enemies
  const enemyCount = reader.readUint16();
  const enemies = [];
  for (let i = 0; i < enemyCount; i++) {
    enemies.push(decodeEnemy(reader));
  }

  // Bullets
  const bulletCount = reader.readUint16();
  const bullets = [];
  for (let i = 0; i < bulletCount; i++) {
    bullets.push(decodeBullet(reader));
  }

  return { timestamp, removed, players, enemies, bullets };
}

/**
 * Handle binary world delta on client
 * Merges delta data with existing game state
 */
export function applyWorldDelta(gameState, delta) {
  if (!delta) return;

  // Remove entities that server says are gone
  for (const id of delta.removed) {
    if (gameState.bullets) {
      gameState.bullets = gameState.bullets.filter(b => b.id !== id);
    }
    if (gameState.enemies) {
      gameState.enemies = gameState.enemies.filter(e => e.id !== id);
    }
  }

  // Merge player updates
  if (delta.players) {
    for (const [id, playerDelta] of Object.entries(delta.players)) {
      if (gameState.players && gameState.players[id]) {
        Object.assign(gameState.players[id], playerDelta);
      } else if (gameState.players) {
        gameState.players[id] = playerDelta;
      }
    }
  }

  // Merge enemy updates
  if (delta.enemies) {
    for (const enemyDelta of delta.enemies) {
      const existing = gameState.enemies?.find(e => e.id === enemyDelta.id);
      if (existing) {
        Object.assign(existing, enemyDelta);
      } else if (gameState.enemies) {
        gameState.enemies.push(enemyDelta);
      }
    }
  }

  // Merge bullet updates
  if (delta.bullets) {
    for (const bulletDelta of delta.bullets) {
      const existing = gameState.bullets?.find(b => b.id === bulletDelta.id);
      if (existing) {
        Object.assign(existing, bulletDelta);
      } else if (gameState.bullets) {
        gameState.bullets.push(bulletDelta);
      }
    }
  }
}

export default {
  BinaryReader,
  BinaryPacketType,
  DeltaFlags,
  decodeWorldDelta,
  applyWorldDelta,
  decodeBullet,
  decodeEnemy,
  decodePlayer,
  registerSprite,
  registerEntity,
  getSpriteName,
  getEntityStringId,
};
