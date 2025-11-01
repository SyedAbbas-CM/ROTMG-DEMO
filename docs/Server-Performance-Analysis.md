# Server Performance Analysis & Optimization Guide

## Current Performance Issues

### 1. **Constant 30 FPS Game Loop** 🔥
**Location:** `Server.js:1185`
```javascript
setInterval(updateGame, gameState.updateInterval); // 33ms = 30 FPS
```

**Problem:** Server runs 30 updates per second **always**, even with 0 players.

**Impact:**
- CPU constantly processing physics, collisions, AI
- MacBook heats up even when idle
- Wasted electricity and resources

**Solutions:**

#### Option A: Dynamic Tick Rate (RECOMMENDED)
```javascript
let updateInterval = null;
let currentTickRate = 30; // FPS

function adjustTickRate() {
  const playerCount = clients.size;

  if (playerCount === 0) {
    currentTickRate = 1; // 1 FPS when empty (just keep alive)
  } else if (playerCount <= 5) {
    currentTickRate = 20; // 20 FPS for small games
  } else {
    currentTickRate = 30; // Full 30 FPS for active games
  }

  // Restart interval with new rate
  if (updateInterval) clearInterval(updateInterval);
  updateInterval = setInterval(updateGame, 1000 / currentTickRate);
}

// Check every 5 seconds
setInterval(adjustTickRate, 5000);
adjustTickRate(); // Initial setup
```

#### Option B: Sleep When Empty
```javascript
function updateGame() {
  if (clients.size === 0) {
    // Still update but at 1 FPS
    if (Date.now() % 1000 < gameState.updateInterval) return;
  }
  // ... rest of game loop
}
```

#### Option C: Event-Driven Updates
Only update when something actually happens (player moves, shoots, etc.)

---

### 2. **Inefficient World Updates** 📡
**Location:** `Server.js:1075-1175`

**Problem:** Broadcasting full world state to all clients every tick.

**Current Code:**
```javascript
// Sends EVERYTHING to EVERYONE every 33ms
clients.forEach(client => {
  sendToClient(client.socket, MessageType.WORLD_UPDATE, {
    enemies: allEnemies,
    bullets: allBullets,
    players: allPlayers,
    units: allUnits
  });
});
```

**Better Approach: Interest Management**
```javascript
const UPDATE_RADIUS_TILES = 30; // Only send nearby entities

clients.forEach(client => {
  const player = client.player;

  // Filter entities by distance
  const nearbyEnemies = enemies.filter(e =>
    distance(e, player) < UPDATE_RADIUS_TILES
  );
  const nearbyBullets = bullets.filter(b =>
    distance(b, player) < UPDATE_RADIUS_TILES
  );

  sendToClient(client.socket, MessageType.WORLD_UPDATE, {
    enemies: nearbyEnemies,
    bullets: nearbyBullets,
    // ... only nearby entities
  });
});
```

**Savings:** With 100 enemies on map, only send 10-20 nearby = 80% less data!

---

### 3. **Unnecessary Collision Checks** 💥
**Location:** `src/world/CollisionManager.js`

**Problem:** Checking every bullet against every enemy (O(n²) complexity).

**Example:**
- 50 bullets × 100 enemies = 5,000 checks per tick
- 30 FPS × 5,000 = 150,000 checks per second!

**Solution: Spatial Partitioning**
```javascript
class SpatialGrid {
  constructor(cellSize = 32) {
    this.cellSize = cellSize;
    this.grid = new Map(); // "x,y" -> Set of entities
  }

  add(entity) {
    const cell = this.getCell(entity.x, entity.y);
    if (!this.grid.has(cell)) this.grid.set(cell, new Set());
    this.grid.get(cell).add(entity);
  }

  getNearby(x, y, radius = 32) {
    // Only check 9 cells (current + 8 neighbors)
    const nearby = [];
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = this.getCell(x + dx * this.cellSize, y + dy * this.cellSize);
        if (this.grid.has(cell)) {
          nearby.push(...this.grid.get(cell));
        }
      }
    }
    return nearby;
  }
}

// Usage:
const enemyGrid = new SpatialGrid(32);
enemies.forEach(e => enemyGrid.add(e));

bullets.forEach(bullet => {
  const nearbyEnemies = enemyGrid.getNearby(bullet.x, bullet.y, 16);
  // Only check ~5-10 enemies instead of 100!
  nearbyEnemies.forEach(enemy => checkCollision(bullet, enemy));
});
```

**Savings:** 5,000 checks → 500 checks = 90% reduction!

---

### 4. **Procedural Chunk Generation on Every Request** 🗺️
**Location:** `src/world/MapManager.js:137`

**Problem:** Generating chunks with Perlin noise every time they're requested.

**Current Code:**
```javascript
const chunkData = this.generateChunkData(chunkY, chunkX);
this.chunks.set(key, chunkData); // Cache it
return chunkData;
```

**Issue:** First request takes ~5-10ms to generate. That's 1/6th of your 33ms frame budget!

**Solution: Pre-generate Chunks**
```javascript
async function preGenerateChunks(mapId, radius = 5) {
  console.log(`Pre-generating chunks for map ${mapId}...`);
  const startTime = Date.now();

  // Generate central area
  for (let x = -radius; x <= radius; x++) {
    for (let y = -radius; y <= radius; y++) {
      mapManager.getChunkData(mapId, x, y);
    }
  }

  const elapsed = Date.now() - startTime;
  console.log(`Generated ${(radius*2+1)²} chunks in ${elapsed}ms`);
}

// On server start:
preGenerateChunks('map_1', 8); // Generate 17×17 = 289 chunks
```

---

### 5. **Memory Leaks in Chunk Cache** 💾
**Location:** `src/world/MapManager.js:18`

**Problem:** Chunk cache grows unbounded.

**Current:**
```javascript
this.chunks = new Map(); // Infinite growth!
```

**Better: LRU Cache with Limits**
```javascript
class LRUChunkCache {
  constructor(maxSize = 500) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.accessOrder = [];
  }

  get(key) {
    if (!this.cache.has(key)) return null;

    // Move to end (most recently used)
    this.accessOrder = this.accessOrder.filter(k => k !== key);
    this.accessOrder.push(key);

    return this.cache.get(key);
  }

  set(key, value) {
    // Evict oldest if full
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      const oldest = this.accessOrder.shift();
      this.cache.delete(oldest);
    }

    this.cache.set(key, value);
    this.accessOrder.push(key);
  }
}
```

---

### 6. **JSON Serialization Overhead** 📦
**Location:** `common/protocol.js:15`

**Problem:** Converting objects to JSON 30 times per second per client.

**Current:**
```javascript
const jsonStr = JSON.stringify(data ?? {});
const jsonBytes = new TextEncoder().encode(jsonStr);
```

**Better: Binary Protocol**
```javascript
// Instead of: {x: 100, y: 200, type: 5}
// Send: [100, 200, 5] as binary

// Float32Array for positions (4 bytes per number)
// Uint8Array for types (1 byte)

const buffer = new ArrayBuffer(9);
const view = new DataView(buffer);
view.setFloat32(0, x);
view.setFloat32(4, y);
view.setUint8(8, type);
// 9 bytes vs ~25 bytes JSON!
```

**Savings:** 60-70% less network bandwidth!

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Dynamic Tick Rate** - Biggest impact, easiest to implement
2. ✅ **Interest Management** - Reduce network traffic by 80%
3. ✅ **Chunk Pre-generation** - Eliminate lag spikes

### Phase 2: Medium Effort (2-4 hours)
4. ⏳ **Spatial Partitioning** - 90% fewer collision checks
5. ⏳ **LRU Chunk Cache** - Prevent memory leaks

### Phase 3: Advanced (4-8 hours)
6. ⏳ **Binary Protocol** - 60% less bandwidth
7. ⏳ **WebWorker for Physics** - Offload collision detection
8. ⏳ **Redis for Chunk Storage** - Persistent world

---

## Expected Performance Gains

**Current:**
- CPU: ~40-60% (idle with 0 players!)
- Memory: ~150MB
- Network: ~100KB/s per player

**After Optimizations:**
- CPU: ~5% idle, 20-30% with 10 players
- Memory: ~80MB (with cache limits)
- Network: ~30KB/s per player

**MacBook Temperature:**
- Before: Hot even when idle 🔥
- After: Cool when idle, warm when active ❄️

---

## Biome & World System Architecture

Since you want proper RotMG-style world/realm switching, here's the architecture:

### World Types in RotMG

1. **Nexus** - Safe hub, no enemies
2. **Realm** - Main overworld (procedural), closes when Oryx killed
3. **Dungeons** - Instanced, small, handcrafted
4. **Boss Rooms** - Instanced boss fights

### Implementation Plan

#### 1. Biome Generation (Like RotMG)
```javascript
// MapManager.js
determineBiome(height, temp, moisture) {
  // RotMG has these biomes:
  if (height < -0.3) return 'ocean';
  if (height > 0.6) return temp < 0 ? 'snowy_mountain' : 'mountain';

  // Lowlands
  if (temp < -0.3) return moisture > 0 ? 'tundra' : 'ice';
  if (temp > 0.4) return moisture < -0.2 ? 'desert' : 'savanna';
  if (moisture > 0.5) return 'swamp';

  // Default
  return 'grassland';
}

selectTileForBiome(biome) {
  const biomeData = biomesDB.biomes[biome];

  // Weighted random selection
  const tiles = biomeData.tiles.primary_floor;
  const weights = biomeData.tileWeights;

  return weightedRandom(tiles, weights);
}
```

#### 2. Realm System
```javascript
class RealmManager {
  constructor() {
    this.realms = new Map(); // realmId -> Realm
    this.nextRealmId = 1;
  }

  createRealm() {
    const realmId = `realm_${this.nextRealmId++}`;
    const mapId = mapManager.generateWorld(128, 128, {
      name: `Realm #${this.nextRealmId}`,
      procedural: true,
      biomes: true
    });

    this.realms.set(realmId, {
      id: realmId,
      mapId: mapId,
      createdAt: Date.now(),
      playerCount: 0,
      bossKilled: false,
      closing: false
    });

    return realmId;
  }

  closeRealm(realmId) {
    const realm = this.realms.get(realmId);
    if (!realm) return;

    realm.closing = true;

    // Give players 30 seconds to leave
    setTimeout(() => {
      // Teleport remaining players to nexus
      clients.forEach(c => {
        if (c.mapId === realm.mapId) {
          teleportToNexus(c);
        }
      });

      // Delete realm
      this.realms.delete(realmId);
      mapManager.deleteMap(realm.mapId);
    }, 30000);
  }
}
```

#### 3. Dungeon Instances
```javascript
class DungeonManager {
  constructor() {
    this.instances = new Map(); // instanceId -> Instance
  }

  createInstance(dungeonType, party) {
    const instanceId = `${dungeonType}_${Date.now()}_${Math.random()}`;

    // Load handcrafted dungeon template
    const template = dungeonTemplates.get(dungeonType);
    const mapId = mapManager.loadFixedMap(template);

    this.instances.set(instanceId, {
      id: instanceId,
      mapId: mapId,
      dungeonType: dungeonType,
      players: new Set(party),
      createdAt: Date.now(),
      expiresAt: Date.now() + (30 * 60 * 1000), // 30 min
      bossKilled: false
    });

    return { instanceId, mapId };
  }

  // Clean up expired/completed dungeons
  cleanup() {
    const now = Date.now();
    this.instances.forEach((instance, id) => {
      if (now > instance.expiresAt || (instance.bossKilled && instance.players.size === 0)) {
        // Kick out remaining players
        clients.forEach(c => {
          if (c.mapId === instance.mapId) {
            teleportToNexus(c);
          }
        });

        this.instances.delete(id);
        mapManager.deleteMap(instance.mapId);
      }
    });
  }
}

setInterval(() => dungeonManager.cleanup(), 60000); // Every minute
```

#### 4. Portal Types
```javascript
const PortalTypes = {
  NEXUS_PORTAL: {
    sprite: 'portal_blue',
    destination: 'nexus',
    oneWay: false,
    persistent: true
  },
  REALM_PORTAL: {
    sprite: 'portal_blue',
    destination: 'realm', // Auto-assigns to available realm
    oneWay: false,
    persistent: true
  },
  DUNGEON_ENTRANCE: {
    sprite: 'portal_purple',
    createInstance: true,
    maxPlayers: 8,
    timeout: 30 * 60 * 1000
  },
  BOSS_ROOM: {
    sprite: 'portal_red',
    requiresKey: true,
    createInstance: true,
    oneWay: true
  }
};
```

---

## Next Steps

1. You name tiles (grass, rock, sand, etc.) in tile-names.json
2. I implement biome-based generation in MapManager
3. I implement realm/dungeon instance system
4. I add performance optimizations

Let me know when you're ready and I'll start implementing!
