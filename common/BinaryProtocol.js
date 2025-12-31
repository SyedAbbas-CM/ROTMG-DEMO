// common/BinaryProtocol.js
// High-performance binary protocol for game entity serialization
// Reduces bandwidth by 5-10x compared to JSON

/**
 * Binary encoding utilities
 * - Fixed-point: floats as int16 (0.01 precision, -327.67 to 327.67 range)
 * - Entity IDs: uint16 (65535 max entities)
 * - Velocities: int8 (-127 to 127 tiles/sec)
 */

// Sprite name -> uint8 registry (populated at runtime)
const spriteRegistry = new Map();
const spriteById = [];
let nextSpriteId = 0;

export function registerSprite(name) {
  if (!spriteRegistry.has(name)) {
    spriteRegistry.set(name, nextSpriteId);
    spriteById[nextSpriteId] = name;
    nextSpriteId++;
  }
  return spriteRegistry.get(name);
}

export function getSpriteId(name) {
  if (!name) return 0;
  if (!spriteRegistry.has(name)) {
    registerSprite(name);
  }
  return spriteRegistry.get(name);
}

export function getSpriteName(id) {
  return spriteById[id] || null;
}

// Entity ID registry (string -> uint16)
const entityRegistry = new Map();
const entityById = [];
let nextEntityId = 1; // 0 reserved for "no entity"

export function getEntityId(stringId) {
  if (!stringId) return 0;
  if (!entityRegistry.has(stringId)) {
    entityRegistry.set(stringId, nextEntityId);
    entityById[nextEntityId] = stringId;
    nextEntityId++;
    // Wrap around at 65535
    if (nextEntityId > 65535) nextEntityId = 1;
  }
  return entityRegistry.get(stringId);
}

export function getEntityStringId(numId) {
  return entityById[numId] || null;
}

// Fixed-point conversion (0.01 precision)
const FIXED_POINT_SCALE = 100;

export function toFixedPoint(value) {
  return Math.round(value * FIXED_POINT_SCALE);
}

export function fromFixedPoint(value) {
  return value / FIXED_POINT_SCALE;
}

// Velocity conversion (0.1 precision, int8)
export function toVelocity(value) {
  return Math.max(-127, Math.min(127, Math.round(value * 10)));
}

export function fromVelocity(value) {
  return value / 10;
}

/**
 * Binary packet types for delta updates
 */
export const BinaryPacketType = {
  // Full state sync (on join or reconnect)
  FULL_SYNC: 0x01,

  // Delta updates
  DELTA_UPDATE: 0x02,

  // Entity lifecycle
  ENTITY_CREATE: 0x03,
  ENTITY_REMOVE: 0x04,

  // Optimized world update (replaces WORLD_UPDATE)
  WORLD_DELTA: 0x10,
};

/**
 * Entity type flags
 */
export const EntityType = {
  PLAYER: 0x01,
  ENEMY: 0x02,
  BULLET: 0x03,
  BAG: 0x04,
  OBJECT: 0x05,
  UNIT: 0x06,
};

/**
 * Delta flags - which fields changed
 */
export const DeltaFlags = {
  POSITION: 0x01,
  VELOCITY: 0x02,
  HEALTH: 0x04,
  STATE: 0x08,    // isDead, isFlashing, etc
  ALL: 0xFF,
};

/**
 * BinaryWriter - efficient buffer writing
 */
export class BinaryWriter {
  constructor(initialSize = 4096) {
    this.buffer = new ArrayBuffer(initialSize);
    this.view = new DataView(this.buffer);
    this.offset = 0;
  }

  ensureCapacity(bytes) {
    if (this.offset + bytes > this.buffer.byteLength) {
      const newSize = Math.max(this.buffer.byteLength * 2, this.offset + bytes);
      const newBuffer = new ArrayBuffer(newSize);
      new Uint8Array(newBuffer).set(new Uint8Array(this.buffer));
      this.buffer = newBuffer;
      this.view = new DataView(this.buffer);
    }
  }

  writeUint8(value) {
    this.ensureCapacity(1);
    this.view.setUint8(this.offset++, value);
  }

  writeInt8(value) {
    this.ensureCapacity(1);
    this.view.setInt8(this.offset++, value);
  }

  writeUint16(value) {
    this.ensureCapacity(2);
    this.view.setUint16(this.offset, value, true);
    this.offset += 2;
  }

  writeInt16(value) {
    this.ensureCapacity(2);
    this.view.setInt16(this.offset, value, true);
    this.offset += 2;
  }

  writeUint32(value) {
    this.ensureCapacity(4);
    this.view.setUint32(this.offset, value, true);
    this.offset += 4;
  }

  writeFloat32(value) {
    this.ensureCapacity(4);
    this.view.setFloat32(this.offset, value, true);
    this.offset += 4;
  }

  // Write position as fixed-point int16 (0.01 precision)
  writePosition(x, y) {
    this.writeInt16(toFixedPoint(x));
    this.writeInt16(toFixedPoint(y));
  }

  // Write velocity as int8 (0.1 precision)
  writeVelocity(vx, vy) {
    this.writeInt8(toVelocity(vx));
    this.writeInt8(toVelocity(vy));
  }

  // Write entity ID (string -> uint16)
  writeEntityId(stringId) {
    this.writeUint16(getEntityId(stringId));
  }

  // Write sprite (string -> uint8)
  writeSprite(spriteName) {
    this.writeUint8(getSpriteId(spriteName));
  }

  getBuffer() {
    return this.buffer.slice(0, this.offset);
  }

  getSize() {
    return this.offset;
  }
}

/**
 * BinaryReader - efficient buffer reading
 */
export class BinaryReader {
  constructor(buffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
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
 * Encode a bullet for binary transmission
 * Full: 14 bytes (on create)
 * Delta: 6 bytes (position only)
 */
export function encodeBullet(writer, bullet, deltaFlags = DeltaFlags.ALL) {
  writer.writeEntityId(bullet.id);
  writer.writeUint8(deltaFlags);

  if (deltaFlags & DeltaFlags.POSITION) {
    writer.writePosition(bullet.x, bullet.y);
  }

  if (deltaFlags & DeltaFlags.VELOCITY) {
    writer.writeVelocity(bullet.vx, bullet.vy);
  }

  if (deltaFlags & DeltaFlags.STATE) {
    writer.writeEntityId(bullet.ownerId);
    writer.writeSprite(bullet.spriteName);
    writer.writeUint8(bullet.damage || 10);
    writer.writeUint8(bullet.faction || 0);
    writer.writeUint8(Math.round((bullet.life || 1) * 10)); // 0.1s precision
  }
}

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
 * Encode an enemy for binary transmission
 * Full: 18 bytes (on create)
 * Delta: 8 bytes (position + health)
 */
export function encodeEnemy(writer, enemy, deltaFlags = DeltaFlags.ALL) {
  writer.writeEntityId(enemy.id);
  writer.writeUint8(deltaFlags);

  if (deltaFlags & DeltaFlags.POSITION) {
    writer.writePosition(enemy.x, enemy.y);
  }

  if (deltaFlags & DeltaFlags.HEALTH) {
    writer.writeUint16(enemy.health || 0);
    writer.writeUint16(enemy.maxHealth || 100);
  }

  if (deltaFlags & DeltaFlags.STATE) {
    writer.writeSprite(enemy.spriteName);
    writer.writeUint8(enemy.type ? getSpriteId(enemy.type) : 0);
    writer.writeUint8(
      (enemy.isDying ? 0x01 : 0) |
      (enemy.isFlashing ? 0x02 : 0)
    );
  }
}

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
 * Encode a player for binary transmission
 * Full: 20 bytes
 * Delta: 6 bytes (position only)
 */
export function encodePlayer(writer, player, deltaFlags = DeltaFlags.ALL) {
  writer.writeEntityId(player.id);
  writer.writeUint8(deltaFlags);

  if (deltaFlags & DeltaFlags.POSITION) {
    writer.writePosition(player.x, player.y);
  }

  if (deltaFlags & DeltaFlags.VELOCITY) {
    writer.writeVelocity(player.vx || 0, player.vy || 0);
  }

  if (deltaFlags & DeltaFlags.HEALTH) {
    writer.writeUint16(player.health || 0);
    writer.writeUint16(player.maxHealth || 100);
  }

  if (deltaFlags & DeltaFlags.STATE) {
    writer.writeUint8(
      (player.isDead ? 0x01 : 0)
    );
    writer.writeUint8(player.level || 1);
  }
}

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
 * Encode a complete world delta update
 * Header: 9 bytes
 * Per bullet: 6-14 bytes
 * Per enemy: 8-18 bytes
 * Per player: 6-20 bytes
 */
// Debug counter for encoding
let encodeWorldDeltaDebugCount = 0;

export function encodeWorldDelta(players, enemies, bullets, removedIds, timestamp) {
  const writer = new BinaryWriter(8192);

  // Header
  writer.writeUint8(BinaryPacketType.WORLD_DELTA);
  writer.writeUint32(timestamp);

  // Removed entities
  writer.writeUint16(removedIds.length);
  for (const id of removedIds) {
    writer.writeEntityId(id);
  }

  // Players (always full update for now)
  const playerList = Object.values(players);
  writer.writeUint16(playerList.length);
  for (const player of playerList) {
    encodePlayer(writer, player, DeltaFlags.POSITION | DeltaFlags.HEALTH);
  }

  // Enemies (position + health delta)
  writer.writeUint16(enemies.length);
  for (const enemy of enemies) {
    encodeEnemy(writer, enemy, DeltaFlags.POSITION | DeltaFlags.HEALTH);
  }

  // Debug: Log bullet encoding details
  encodeWorldDeltaDebugCount++;
  const shouldLog = encodeWorldDeltaDebugCount <= 3 || encodeWorldDeltaDebugCount % 100 === 0;
  const bulletStartOffset = writer.getSize();

  // Bullets (position + velocity for smooth client-side prediction)
  writer.writeUint16(bullets.length);
  for (let i = 0; i < bullets.length; i++) {
    const bullet = bullets[i];
    // Debug: Log first few bullets
    if (shouldLog && i < 2 && bullets.length > 0) {
      console.log(`[BINARY-TX] Bullet #${i}: id=${bullet.id}, x=${bullet.x?.toFixed(2)}, y=${bullet.y?.toFixed(2)}, vx=${bullet.vx?.toFixed(2)}, vy=${bullet.vy?.toFixed(2)}`);
    }
    encodeBullet(writer, bullet, DeltaFlags.POSITION | DeltaFlags.VELOCITY);
  }

  const buffer = writer.getBuffer();

  // Debug: Log raw bytes for bullets section
  if (shouldLog && bullets.length > 0) {
    const bytes = new Uint8Array(buffer);
    const bulletSectionStart = bulletStartOffset;
    const bulletSectionEnd = Math.min(bulletSectionStart + 30, bytes.length);
    const hexBytes = Array.from(bytes.slice(bulletSectionStart, bulletSectionEnd)).map(b => b.toString(16).padStart(2, '0')).join(' ');
    console.log(`[BINARY-TX] #${encodeWorldDeltaDebugCount} Encoded ${bullets.length} bullets, buffer size ${buffer.byteLength}, bullet section (offset ${bulletStartOffset}): ${hexBytes}`);
  }

  return buffer;
}

export function decodeWorldDelta(buffer) {
  const reader = new BinaryReader(buffer);

  const type = reader.readUint8();
  if (type !== BinaryPacketType.WORLD_DELTA) {
    throw new Error(`Invalid packet type: ${type}`);
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
    players[player.id] = player;
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
 * Delta tracker - tracks entity state changes
 */
export class DeltaTracker {
  constructor() {
    this.lastState = new Map(); // entityId -> { x, y, health, ... }
    this.newEntities = new Set();
    this.removedEntities = new Set();
  }

  /**
   * Check if entity changed, return delta flags
   */
  getEntityDelta(id, current) {
    const last = this.lastState.get(id);

    if (!last) {
      this.newEntities.add(id);
      this.lastState.set(id, { ...current });
      return DeltaFlags.ALL; // New entity, send everything
    }

    let flags = 0;

    // Position changed? (threshold: 0.05 tiles)
    const dx = Math.abs(current.x - last.x);
    const dy = Math.abs(current.y - last.y);
    if (dx > 0.05 || dy > 0.05) {
      flags |= DeltaFlags.POSITION;
      last.x = current.x;
      last.y = current.y;
    }

    // Velocity changed?
    if (current.vx !== undefined && current.vy !== undefined) {
      if (Math.abs((current.vx || 0) - (last.vx || 0)) > 0.1 ||
          Math.abs((current.vy || 0) - (last.vy || 0)) > 0.1) {
        flags |= DeltaFlags.VELOCITY;
        last.vx = current.vx;
        last.vy = current.vy;
      }
    }

    // Health changed?
    if (current.health !== undefined && current.health !== last.health) {
      flags |= DeltaFlags.HEALTH;
      last.health = current.health;
    }

    // State changed?
    if (current.isDead !== last.isDead ||
        current.isDying !== last.isDying ||
        current.isFlashing !== last.isFlashing) {
      flags |= DeltaFlags.STATE;
      last.isDead = current.isDead;
      last.isDying = current.isDying;
      last.isFlashing = current.isFlashing;
    }

    return flags;
  }

  /**
   * Mark entity as removed
   */
  removeEntity(id) {
    this.lastState.delete(id);
    this.removedEntities.add(id);
    this.newEntities.delete(id);
  }

  /**
   * Get and clear removed entities list
   */
  getAndClearRemoved() {
    const removed = [...this.removedEntities];
    this.removedEntities.clear();
    return removed;
  }

  /**
   * Check if entity is new (just created)
   */
  isNew(id) {
    return this.newEntities.has(id);
  }

  /**
   * Clear new entity flags after sending create packets
   */
  clearNewFlags() {
    this.newEntities.clear();
  }
}

// ==============================================================================
// CLIENT → SERVER BINARY PROTOCOL (Input messages)
// ==============================================================================

/**
 * Client binary message types (first byte of payload)
 */
export const ClientBinaryType = {
  PLAYER_UPDATE: 0x01,    // Position + velocity + angle
  BULLET_CREATE: 0x02,    // Fire bullet
  PING: 0x03,             // Latency check
  USE_ABILITY: 0x04,      // Ability usage
};

/**
 * Angle conversion (0-65535 = 0-2*PI)
 */
function fromAngle(value) {
  return (value / 65535) * Math.PI * 2;
}

/**
 * Decode client PLAYER_UPDATE
 * Format: [type:1][x:2][y:2][vx:1][vy:1][angle:2] = 9 bytes
 */
export function decodeClientPlayerUpdate(reader) {
  const pos = reader.readPosition();
  const vel = reader.readVelocity();
  const angle = fromAngle(reader.readUint16());

  return {
    x: pos.x,
    y: pos.y,
    vx: vel.vx,
    vy: vel.vy,
    angle: angle,
    rotation: angle // Alias for compatibility
  };
}

/**
 * Decode client BULLET_CREATE
 * Format: [type:1][x:2][y:2][angle:2][speed:1][damage:1] = 9 bytes
 */
export function decodeClientBulletCreate(reader) {
  const pos = reader.readPosition();
  const angle = fromAngle(reader.readUint16());
  const speed = reader.readUint8() / 10; // 0.1 precision
  const damage = reader.readUint8();

  return {
    x: pos.x,
    y: pos.y,
    angle: angle,
    speed: speed,
    damage: damage
  };
}

/**
 * Decode client PING
 * Format: [type:1][timestamp:4] = 5 bytes
 */
export function decodeClientPing(reader) {
  const timestamp = reader.readUint32();
  return { time: timestamp };
}

/**
 * Decode client USE_ABILITY
 * Format: [type:1][abilityId:1][targetX:2][targetY:2] = 6 bytes
 */
export function decodeClientUseAbility(reader) {
  const abilityId = reader.readUint8();
  const pos = reader.readPosition();

  return {
    abilityId: abilityId,
    targetX: pos.x,
    targetY: pos.y
  };
}

/**
 * Decode a client binary message
 * @param {ArrayBuffer} buffer - Raw binary data from client
 * @returns {{ type: number, data: Object } | null}
 */
export function decodeClientBinaryMessage(buffer) {
  if (!buffer || buffer.byteLength < 1) {
    return null;
  }

  const reader = new BinaryReader(buffer);
  const type = reader.readUint8();

  try {
    switch (type) {
      case ClientBinaryType.PLAYER_UPDATE:
        return { type, data: decodeClientPlayerUpdate(reader) };

      case ClientBinaryType.BULLET_CREATE:
        return { type, data: decodeClientBulletCreate(reader) };

      case ClientBinaryType.PING:
        return { type, data: decodeClientPing(reader) };

      case ClientBinaryType.USE_ABILITY:
        return { type, data: decodeClientUseAbility(reader) };

      default:
        console.warn(`[BinaryProtocol] Unknown client binary type: ${type}`);
        return null;
    }
  } catch (error) {
    console.error(`[BinaryProtocol] Decode error for type ${type}:`, error);
    return null;
  }
}

/**
 * Check if buffer looks like a client binary message
 * (First byte is a valid ClientBinaryType)
 */
export function isClientBinaryMessage(buffer) {
  if (!buffer || buffer.byteLength < 1) return false;
  const view = new DataView(buffer);
  const firstByte = view.getUint8(0);
  return firstByte >= 0x01 && firstByte <= 0x04;
}

/**
 * Calculate size comparison between JSON and binary
 */
export function calculateSavings(jsonSize, binarySize) {
  const savings = jsonSize - binarySize;
  const percentage = ((savings / jsonSize) * 100).toFixed(1);
  return { jsonSize, binarySize, savings, percentage };
}

export default {
  BinaryWriter,
  BinaryReader,
  DeltaTracker,
  encodeBullet,
  decodeBullet,
  encodeEnemy,
  decodeEnemy,
  encodePlayer,
  decodePlayer,
  encodeWorldDelta,
  decodeWorldDelta,
  registerSprite,
  getSpriteId,
  getSpriteName,
  getEntityId,
  getEntityStringId,
  BinaryPacketType,
  EntityType,
  DeltaFlags,
  calculateSavings,
  // Client binary protocol
  ClientBinaryType,
  decodeClientBinaryMessage,
  isClientBinaryMessage,
  decodeClientPlayerUpdate,
  decodeClientBulletCreate,
  decodeClientPing,
  decodeClientUseAbility,
};
