// public/src/network/BinaryProtocol.js
// Client-side binary protocol for high-performance game communication
// Handles both encoding (client→server) and decoding (server→client)

// Sprite registry (synchronized from server on handshake)
const spriteById = [];
const spriteRegistry = new Map();

// Pre-register all known enemy sprites in deterministic order
// MUST match the order in common/BinaryProtocol.js on the server
const KNOWN_SPRITES = [
  'Light_Infantry',
  'Archer',
  'Light_Cavalry',
  'Heavy_Cavalry',
  'Heavy_Infantry',
  'Mega_Cavalry',
  // Add more as needed - MUST match server list
];

// Entity ID registry (synchronized from server)
const entityById = [];
const entityRegistry = new Map();

export function registerSprite(id, name) {
  spriteById[id] = name;
  spriteRegistry.set(name, id);
}

// Initialize known sprites on module load (matching server IDs)
KNOWN_SPRITES.forEach((name, id) => registerSprite(id, name));

export function getSpriteName(id) {
  return spriteById[id] || null;
}

export function getSpriteId(name) {
  return spriteRegistry.get(name) || 0;
}

export function registerEntity(id, stringId) {
  entityById[id] = stringId;
  entityRegistry.set(stringId, id);
}

export function getEntityStringId(numId) {
  return entityById[numId] || `entity_${numId}`;
}

export function getEntityNumId(stringId) {
  if (!entityRegistry.has(stringId)) {
    const newId = entityById.length || 1;
    entityById[newId] = stringId;
    entityRegistry.set(stringId, newId);
  }
  return entityRegistry.get(stringId);
}

// Fixed-point conversion (0.01 precision)
const FIXED_POINT_SCALE = 100;

function toFixedPoint(value) {
  return Math.round(value * FIXED_POINT_SCALE);
}

function fromFixedPoint(value) {
  return value / FIXED_POINT_SCALE;
}

function toVelocity(value) {
  return Math.max(-127, Math.min(127, Math.round(value * 10)));
}

function fromVelocity(value) {
  return value / 10;
}

// Angle conversion (0-65535 = 0-2*PI)
function toAngle(radians) {
  const normalized = ((radians % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
  return Math.round((normalized / (Math.PI * 2)) * 65535);
}

function fromAngle(value) {
  return (value / 65535) * Math.PI * 2;
}

/**
 * BinaryWriter - efficient buffer writing for client→server messages
 */
export class BinaryWriter {
  constructor(initialSize = 64) {
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

  writePosition(x, y) {
    this.writeInt16(toFixedPoint(x));
    this.writeInt16(toFixedPoint(y));
  }

  writeVelocity(vx, vy) {
    this.writeInt8(toVelocity(vx));
    this.writeInt8(toVelocity(vy));
  }

  writeAngle(radians) {
    this.writeUint16(toAngle(radians));
  }

  getBuffer() {
    return this.buffer.slice(0, this.offset);
  }

  getUint8Array() {
    return new Uint8Array(this.buffer, 0, this.offset);
  }
}

/**
 * Client input message types (client→server binary)
 */
export const ClientBinaryType = {
  PLAYER_UPDATE: 0x01,    // Position + velocity + angle
  BULLET_CREATE: 0x02,    // Fire bullet
  PING: 0x03,             // Latency check
  USE_ABILITY: 0x04,      // Ability usage
};

/**
 * Encode PLAYER_UPDATE for binary transmission
 * Format: [type:1][x:2][y:2][vx:1][vy:1][angle:2] = 9 bytes
 * vs JSON: ~60 bytes
 */
export function encodePlayerUpdate(x, y, vx, vy, angle) {
  const writer = new BinaryWriter(16);
  writer.writeUint8(ClientBinaryType.PLAYER_UPDATE);
  writer.writePosition(x, y);
  writer.writeVelocity(vx || 0, vy || 0);
  writer.writeAngle(angle || 0);
  return writer.getBuffer();
}

/**
 * Encode BULLET_CREATE for binary transmission
 * Format: [type:1][x:2][y:2][angle:2][speed:1][damage:1] = 9 bytes
 * vs JSON: ~80 bytes
 */
export function encodeBulletCreate(x, y, angle, speed, damage) {
  const writer = new BinaryWriter(16);
  writer.writeUint8(ClientBinaryType.BULLET_CREATE);
  writer.writePosition(x, y);
  writer.writeAngle(angle);
  writer.writeUint8(Math.min(255, Math.round(speed * 10))); // 0.1 precision, max 25.5
  writer.writeUint8(damage || 10);
  return writer.getBuffer();
}

/**
 * Encode PING for binary transmission
 * Format: [type:1][timestamp:4] = 5 bytes
 */
export function encodePing(timestamp) {
  const writer = new BinaryWriter(8);
  writer.writeUint8(ClientBinaryType.PING);
  writer.writeUint32(timestamp);
  return writer.getBuffer();
}

/**
 * Encode USE_ABILITY for binary transmission
 * Format: [type:1][abilityId:1][targetX:2][targetY:2] = 6 bytes
 */
export function encodeUseAbility(abilityId, targetX, targetY) {
  const writer = new BinaryWriter(8);
  writer.writeUint8(ClientBinaryType.USE_ABILITY);
  writer.writeUint8(abilityId);
  writer.writePosition(targetX, targetY);
  return writer.getBuffer();
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
    // Validate buffer
    if (!buffer) {
      console.error('[BinaryReader] FATAL: null/undefined buffer passed to constructor');
      this.buffer = new ArrayBuffer(0);
    } else if (buffer instanceof ArrayBuffer) {
      this.buffer = buffer;
    } else if (buffer.buffer instanceof ArrayBuffer) {
      // It's a TypedArray view - need to handle byteOffset!
      const typedArray = buffer;
      // Create a new ArrayBuffer with just our data (not the whole underlying buffer)
      this.buffer = typedArray.buffer.slice(typedArray.byteOffset, typedArray.byteOffset + typedArray.byteLength);
      console.warn(`[BinaryReader] Received TypedArray instead of ArrayBuffer, sliced to ${this.buffer.byteLength} bytes`);
    } else {
      console.error('[BinaryReader] FATAL: Invalid buffer type:', typeof buffer, buffer);
      this.buffer = new ArrayBuffer(0);
    }

    this.view = new DataView(this.buffer);
    this.offset = 0;
    this.bufferSize = this.buffer.byteLength;
  }

  // Validate read before attempting
  _validateRead(bytes, operation) {
    if (this.offset + bytes > this.bufferSize) {
      console.error(`[BinaryReader] OVERFLOW: ${operation} needs ${bytes} bytes at offset ${this.offset}, but buffer only has ${this.bufferSize} bytes`);
      return false;
    }
    return true;
  }

  readUint8() {
    if (!this._validateRead(1, 'readUint8')) return 0;
    return this.view.getUint8(this.offset++);
  }

  readInt8() {
    if (!this._validateRead(1, 'readInt8')) return 0;
    return this.view.getInt8(this.offset++);
  }

  readUint16() {
    if (!this._validateRead(2, 'readUint16')) return 0;
    const val = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readInt16() {
    if (!this._validateRead(2, 'readInt16')) return 0;
    const val = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return val;
  }

  readUint32() {
    if (!this._validateRead(4, 'readUint32')) return 0;
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readFloat32() {
    if (!this._validateRead(4, 'readFloat32')) return 0;
    const val = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readPosition() {
    const rawX = this.readInt16();
    const rawY = this.readInt16();
    const x = fromFixedPoint(rawX);
    const y = fromFixedPoint(rawY);

    // Validate result - return safe defaults if NaN
    if (!isFinite(x) || !isFinite(y)) {
      console.error(`[BinaryReader] BAD POSITION: rawX=${rawX}, rawY=${rawY}, x=${x}, y=${y}, offset=${this.offset - 4}`);
      return { x: 0, y: 0, invalid: true };
    }

    return { x, y };
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
    if (pos.invalid) {
      bullet._invalid = true; // Mark for filtering
    }
    bullet.x = pos.x;
    bullet.y = pos.y;
  }

  if (deltaFlags & DeltaFlags.VELOCITY) {
    const vel = reader.readVelocity();
    // Validate velocity
    if (!isFinite(vel.vx) || !isFinite(vel.vy)) {
      console.error(`[BinaryReader] BAD VELOCITY for bullet ${id}: vx=${vel.vx}, vy=${vel.vy}`);
      bullet.vx = 0;
      bullet.vy = 0;
    } else {
      bullet.vx = vel.vx;
      bullet.vy = vel.vy;
    }
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
    if (pos.invalid) {
      enemy._invalid = true; // Mark for filtering
    }
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
    enemy.renderScale = reader.readUint8() || 2;
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
    if (pos.invalid) {
      player._invalid = true;
    }
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

  // Debug: Log raw buffer info
  if (!decodeWorldDelta._bufferLogged) {
    decodeWorldDelta._bufferLogged = true;
    const bytes = new Uint8Array(buffer);
    const hexFirst20 = Array.from(bytes.slice(0, 20)).map(b => b.toString(16).padStart(2, '0')).join(' ');
    console.log(`[BINARY] Buffer info: byteLength=${buffer.byteLength}, first 20 bytes: ${hexFirst20}`);
  }

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
    // Skip invalid players
    if (player._invalid || !isFinite(player.x) || !isFinite(player.y)) {
      console.error(`[BINARY] SKIPPING INVALID PLAYER: id=${player.id}, x=${player.x}, y=${player.y}`);
      continue;
    }
    if (player.id) {
      players[player.id] = player;
    }
  }

  // Enemies
  const enemyCount = reader.readUint16();
  const enemies = [];
  for (let i = 0; i < enemyCount; i++) {
    const enemy = decodeEnemy(reader);
    // Filter out invalid enemies (NaN positions or marked invalid)
    if (enemy._invalid || !isFinite(enemy.x) || !isFinite(enemy.y)) {
      console.error(`[BINARY] SKIPPING INVALID ENEMY: id=${enemy.id}, x=${enemy.x}, y=${enemy.y}`);
      continue;
    }
    enemies.push(enemy);
  }

  // Debug: Log offset before bullets
  const bulletStartOffset = reader.offset;

  // Bullets
  const bulletCount = reader.readUint16();
  const bullets = [];

  // Debug: Log detailed info about bullet decoding
  if (!decodeWorldDelta._bulletDebugCount) decodeWorldDelta._bulletDebugCount = 0;
  decodeWorldDelta._bulletDebugCount++;
  const shouldLogBullets = decodeWorldDelta._bulletDebugCount <= 3 || decodeWorldDelta._bulletDebugCount % 50 === 0;

  if (shouldLogBullets && bulletCount > 0) {
    console.log(`[BINARY] #${decodeWorldDelta._bulletDebugCount} Decoding ${bulletCount} bullets at offset ${bulletStartOffset}, buffer size ${buffer.byteLength}`);
  }

  for (let i = 0; i < bulletCount; i++) {
    const bulletStartOff = reader.offset;
    const bullet = decodeBullet(reader);

    // Filter out invalid bullets (NaN positions or marked invalid)
    if (bullet._invalid || isNaN(bullet.x) || isNaN(bullet.y) || !isFinite(bullet.x) || !isFinite(bullet.y)) {
      // Log raw bytes around this bullet (throttled)
      if (!decodeWorldDelta._invalidCount) decodeWorldDelta._invalidCount = 0;
      decodeWorldDelta._invalidCount++;
      if (decodeWorldDelta._invalidCount <= 5 || decodeWorldDelta._invalidCount % 100 === 0) {
        const bytes = new Uint8Array(buffer);
        const start = Math.max(0, bulletStartOff - 4);
        const end = Math.min(bytes.length, reader.offset + 4);
        const rawBytes = Array.from(bytes.slice(start, end)).map(b => b.toString(16).padStart(2, '0')).join(' ');
        console.error(`[BINARY] SKIPPING INVALID BULLET #${decodeWorldDelta._invalidCount}: x=${bullet.x}, y=${bullet.y}, id=${bullet.id}`);
      }
      continue; // Skip this bullet - don't add to array
    }
    bullets.push(bullet);
  }

  // Debug first decode
  if (!decodeWorldDelta._debugged && bullets.length > 0) {
    decodeWorldDelta._debugged = true;
    console.log(`[BINARY] First bullet decoded: x=${bullets[0].x}, y=${bullets[0].y}, id=${bullets[0].id}`);
    console.log(`[BINARY] Final reader offset: ${reader.offset}, buffer size: ${buffer.byteLength}`);
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
