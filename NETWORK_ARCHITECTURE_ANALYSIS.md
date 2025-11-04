# ROTMG-DEMO Network Architecture Analysis & Lag Compensation Strategy

**Date**: 2025-11-02
**Issue**: Game unplayable at 200ms ping (chunks don't load, shooting doesn't work)
**Target**: Support up to 300ms cross-continent latency
**Approach**: Moderate hybrid changes with protocol modifications

---

## Table of Contents

1. [Current Architecture Overview](#current-architecture-overview)
2. [Critical Problems at 200ms Ping](#critical-problems-at-200ms-ping)
3. [Architectural Decisions & Trade-offs](#architectural-decisions--trade-offs)
4. [Proposed Changes by System](#proposed-changes-by-system)
5. [Implementation Phases](#implementation-phases)
6. [File-by-File Change Summary](#file-by-file-change-summary)

---

## Current Architecture Overview

### Architecture Pattern
**Server-Authoritative with Client-Side Prediction**

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

CLIENT                              SERVER
──────────────────────────────────────────
├─ Local Prediction                 ├─ Authoritative State
│  └─ Bullets (immediate)           │  └─ All entities
│  └─ Movement (60 FPS)             │  └─ Collision detection
│  └─ Collision detection           │  └─ Damage calculation
│                                    │
├─ Send Updates                     ├─ Broadcast State
│  └─ PLAYER_UPDATE (60 FPS)        │  └─ WORLD_UPDATE (~10 FPS)
│  └─ BULLET_CREATE (on shoot)      │  └─ Contains all entities
│  └─ COLLISION (on hit)            │  └─ No delta compression
│                                    │
└─ Network Manager                  └─ Network Manager
   └─ Binary protocol (type+JSON)      └─ Binary protocol
   └─ WebSocket connection             └─ WebSocket server
   └─ No retry/acknowledgment          └─ Broadcast to all clients
```

### Current Network Protocol

**Binary Packet Format** (`common/protocol.js`):
```
[Type: 1 byte][Length: 4 bytes, little-endian][JSON Payload: variable]
Example: [30, 150, 0, 0, 0, {"x":10,"y":20,...}]
```

**Key Message Types**:
- `HANDSHAKE (1)` / `HANDSHAKE_ACK (2)` - Connection setup
- `PLAYER_UPDATE (12)` - Client → Server position updates (60 FPS)
- `BULLET_CREATE (30)` - Client → Server shooting
- `COLLISION (40)` - Client → Server hit detection
- `CHUNK_REQUEST (51)` / `CHUNK_DATA (52)` - Map streaming
- `WORLD_UPDATE (60)` - Server → Client game state (~10 FPS)

### Current Data Flow

#### Shooting Flow
```
t=0ms:    Client: Player shoots
          ├─ ClientBulletManager.addBullet() creates local bullet
          │  └─ ID: "local_<timestamp>_<index>"
          ├─ Bullet renders immediately (prediction)
          └─ networkManager.sendShoot() sends BULLET_CREATE

t=200ms:  Server: Receives BULLET_CREATE
          ├─ BulletManager.addBullet() creates server bullet
          │  └─ ID: "bullet_<counter>"
          │  └─ Requires worldId (rejects if missing!)
          └─ Waits for next broadcast tick

t=300ms:  Server: WORLD_UPDATE broadcast tick
          └─ Sends all bullets to all clients

t=500ms:  Client: Receives WORLD_UPDATE
          ├─ ClientBulletManager.updateBullets()
          └─ Reconciles local bullet with server bullet
             └─ Matches by position proximity (< 1 tile)
             └─ Replaces local ID with server ID

TOTAL LATENCY: 500ms from trigger to confirmation
```

#### Chunk Loading Flow
```
t=0ms:    Client: Player moves to new area
          ├─ ClientMapManager.updateVisibleChunks()
          ├─ Detects missing chunks
          └─ networkManager.requestChunk(x, y)
             └─ CHUNK_REQUEST sent

t=200ms:  Server: Receives CHUNK_REQUEST
          ├─ MapManager.getChunk(x, y)
          │  └─ Generates if not cached (procedural)
          └─ Sends CHUNK_DATA response

t=400ms:  Client: Receives CHUNK_DATA
          ├─ ClientMapManager.setChunkData()
          ├─ Processes tiles
          ├─ Extracts objects
          └─ Updates collision system

TOTAL LATENCY: 400ms round-trip
THROTTLE: 1500ms between identical requests (too long!)
```

---

## Critical Problems at 200ms Ping

### Problem 1: Shooting System Fails

**Location**: `public/src/network/ClientNetworkManager.js:1168-1180`

```javascript
sendShoot(bulletData) {
    // PROBLEM: Fire-and-forget, no acknowledgment!
    return this.send(MessageType.BULLET_CREATE, bulletData);
}
```

**Why It Fails**:
1. **No acknowledgment** - Client doesn't know if server received it
2. **No retry mechanism** - Lost packets = lost bullets
3. **No timeout detection** - Can't tell if bullet was rejected
4. **Silent failures** - `worldId` mismatch causes rejection (see logs)

**Evidence from `server_diagnostic.log:193-225`**:
```
[SERVER BULLET CREATE] ID: bullet_1762080459040_1_0.5813594161194409,
  ClientPos: (13.1681, 35.0000), BulletPos: (12.7753, 34.9242)

[Multiple WORLD_UPDATE sends...]

[SERVER BULLET] ID: bullet_1762080459040_1_0.5813594161194409,
  Pos: (-0.6568, 32.3320), Reason: OUT_OF_BOUNDS
```
Bullet created, flew off screen, removed - but if packet was lost, would fail silently.

**Impact at 200ms Ping**:
- 5-10% packet loss = 5-10% of shots don't register
- No visual feedback that shot was rejected
- Players think game is broken

---

### Problem 2: Chunk Loading Blocks Movement

**Location**: `public/src/map/ClientMapManager.js:886-946`

```javascript
isWallOrObstacle(x, y, isBullet = false) {
    const tileX = Math.floor(x);
    const tileY = Math.floor(y);

    const tile = this.getTile(tileX, tileY);

    if (!tile) {
        if (isBullet) {
            return false; // Bullets pass through unloaded chunks
        }
        return true; // ❌ PLAYERS ARE BLOCKED!
    }
    // ... rest of collision logic
}
```

**Location**: `public/src/map/ClientMapManager.js:588-603`

```javascript
// Throttle identical requests
const now = Date.now();
const lastReq = this.lastChunkRequestTime.get(key) || 0;
if (now - lastReq >= this.requestThrottleMs) { // ❌ 1500ms throttle!
    this.pendingChunks.add(key);
    this.lastChunkRequestTime.set(key, now);
    try {
        this.networkManager.requestChunk(chunkX, chunkY);
        chunksRequested.push(`(${chunkX},${chunkY})`);
    } catch (error) {
        console.error(`Error requesting chunk (${chunkX}, ${chunkY}):`, error);
        this.pendingChunks.delete(key);
    }
}
```

**Why It Fails**:
1. **Blocking on unloaded chunks** - Player can't move into unexplored areas
2. **1500ms throttle** - At 200ms ping, 400ms round-trip, but can't retry for 1.5s
3. **No timeout detection** - If chunk request is lost, no retry until player re-enters area
4. **No priority system** - All chunks requested with same urgency

**Impact at 200ms Ping**:
```
Player moves forward → Enters unloaded chunk boundary →
Movement blocked → Stutters/stops → Waits 400ms for chunk →
Chunk arrives → Can move again → Repeats every few seconds

Result: Constant stuttering, feels like invisible walls everywhere
```

---

### Problem 3: No Entity Interpolation

**Current**: Entities render at exact positions from `WORLD_UPDATE`

```javascript
// public/src/game/game.js - updateWorld()
updateWorld(enemies, bullets, players, objects) {
    // Updates entities to exact server positions
    this.enemyManager.updateEnemies(enemies);
    this.bulletManager.updateBullets(bullets);
    // No interpolation between updates!
}
```

**Why It Fails**:
- Server sends updates at ~10 FPS (100ms intervals)
- At 200ms ping, updates arrive 200-300ms late
- Client renders entities at 60 FPS but only has 10 FPS position data
- **Result**: Entities "teleport" every 100ms instead of smooth movement

**Impact**:
- Other players appear to jump around
- Enemies jitter across the screen
- Bullets teleport instead of flying smoothly

---

### Problem 4: No Lag Compensation for Hit Detection

**Current**: Server validates collisions at current positions

```javascript
// src/entities/CollisionManager.js:203-220
validateCollision(bulletId, enemyId, bulletPos, enemyPos, clientId, timestamp) {
    // Find entities
    const bi = this.bulletManager.findIndexById(bulletId);
    const ei = this.enemyManager.findIndexById(enemyId);

    // Check collision with CURRENT positions (not client's perceived time!)
    if (this.checkAABBCollision(
        this.bulletManager.x[bi], this.bulletManager.y[bi], // Current position
        // ... vs ...
        this.enemyManager.x[ei], this.enemyManager.y[ei]    // Current position
    )) {
        return this.processCollision(bi, ei, clientId);
    }
}
```

**Why It Fails**:
- Client sees enemy at position A, shoots
- 200ms later, server receives collision report
- Enemy has moved to position B
- Server checks collision at position B (not A where client saw it)
- **Collision rejected even though client's shot was accurate!**

**Impact**:
- Shots that look like hits don't register
- Fast-moving enemies impossible to hit
- Players feel like hit detection is broken

---

### Problem 5: High Bandwidth Usage

**Current**:
- Client sends `PLAYER_UPDATE` at **60 FPS** = 60 messages/second
- Server sends `WORLD_UPDATE` at **~10 FPS** = 10 messages/second
- Each `WORLD_UPDATE` contains **ALL entities** (no delta compression)

**Bandwidth Per Player**:
```
Outgoing:
  PLAYER_UPDATE: ~100 bytes × 60 FPS = 6 KB/s

Incoming:
  WORLD_UPDATE: ~50 KB × 10 FPS = 500 KB/s
  (10 enemies × 40 bytes + 50 bullets × 30 bytes + 10 players × 40 bytes)

Total: ~506 KB/s per player
```

**At 200ms ping with 10% packet loss**:
- High send rate increases chance of packet loss
- Large packets more likely to be corrupted
- No recovery mechanism for lost updates

---

## Architectural Decisions & Trade-offs

### Decision 1: Reliable Message Delivery

**Options**:

| Option | Pros | Cons | Recommended? |
|--------|------|------|--------------|
| **TCP-only (current)** | Simple, WebSocket handles reliability | Still can have delays, no application-level retry | ❌ Current (insufficient) |
| **Add ACK/NACK system** | Application-level confirmation, can retry critical messages | Adds complexity, more messages | ✅ **YES** - Best for moderate approach |
| **Switch to UDP + reliability layer** | Fine-grained control, can mix reliable/unreliable | Major rewrite, complex | ❌ Too aggressive |

**Decision**: **Implement acknowledgment system for critical messages (shooting, chunk requests)**

---

### Decision 2: Chunk Loading Strategy

**Options**:

| Option | Pros | Cons | Recommended? |
|--------|------|------|--------------|
| **Block on unloaded (current)** | Safe, no visual glitches | Unplayable at high latency | ❌ Current (fails) |
| **Allow movement, stream chunks** | Smooth movement, better UX | Might see "pop-in" | ✅ **YES** - Industry standard |
| **Preload entire map** | No streaming needed | Huge memory usage, slow initial load | ❌ Not scalable |
| **Predictive loading** | Load ahead of player | Complex, can waste bandwidth | ✅ **YES** - Add on top of streaming |

**Decision**: **Allow movement through unloaded chunks + predictive loading based on velocity**

---

### Decision 3: Entity Synchronization

**Options**:

| Option | Pros | Cons | Recommended? |
|--------|------|------|--------------|
| **Lockstep (deterministic)** | Perfect sync, low bandwidth | Requires waiting for slowest client | ❌ Wrong for action game |
| **Snapshot interpolation** | Smooth, industry standard | Adds latency (render in past) | ✅ **YES** - Best for 300ms target |
| **Extrapolation** | No added latency | Can predict wrong, looks janky | ⚠️ Maybe for very fast entities |
| **Dead reckoning** | Smooth between updates | Complex state management | ⚠️ Partial - for players only |

**Decision**: **Client-side interpolation with 100-150ms buffer for other players/enemies**

---

### Decision 4: Hit Detection

**Options**:

| Option | Pros | Cons | Recommended? |
|--------|------|------|--------------|
| **Server validates current position (current)** | Simple, authoritative | Unfair at high latency | ❌ Current (unfair) |
| **Server rewinds to client time** | Fair for shooter, standard | Can feel unfair for victim | ✅ **YES** - Industry standard |
| **Client-authoritative** | No latency issues | Cheating possible | ❌ Not secure |
| **Hybrid with trust score** | Adaptive anti-cheat | Very complex | ❌ Too aggressive |

**Decision**: **Server-side lag compensation (rewind entity positions to client's perceived time)**

---

### Decision 5: Update Rate

**Options**:

| Option | Pros | Cons | Recommended? |
|--------|------|------|--------------|
| **Fixed 60 FPS updates (current)** | Simple, consistent | Wasteful at high latency | ❌ Current (wasteful) |
| **Adaptive rate based on RTT** | Efficient, reduces packet loss | Variable network load | ✅ **YES** - Best for 300ms target |
| **Event-driven updates** | Minimal bandwidth | Complex state sync | ⚠️ Partial - for some systems |
| **Delta compression** | Reduced bandwidth | More CPU for encoding/decoding | ✅ **YES** - Add on top of adaptive |

**Decision**: **Adaptive update rate (16-100ms based on RTT) + delta compression for world updates**

---

## Proposed Changes by System

### System 1: Network Protocol Layer

**File**: `common/protocol.js`

**Current State**:
```javascript
// Simple binary packet: [type][length][json]
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
```

**Proposed Changes**:

1. **Add sequence numbers** for ordering and duplicate detection
2. **Add acknowledgment message types**
3. **Add request IDs** for matching responses

**New Protocol Format**:
```
[Type: 1 byte]
[Flags: 1 byte]        // NEW: bit 0 = needs ACK, bit 1 = is ACK, etc.
[Sequence: 2 bytes]    // NEW: message sequence number
[Request ID: 4 bytes]  // NEW: for request/response matching
[Length: 4 bytes]
[JSON Payload: variable]

Total header: 12 bytes (was 5 bytes)
```

**New Message Types to Add**:
```javascript
// Acknowledgment messages
ACK: 200,              // Generic acknowledgment
NACK: 201,             // Negative acknowledgment (reject)
BULLET_ACK: 202,       // Bullet creation confirmed
CHUNK_ACK: 203,        // Chunk request acknowledged
TIMEOUT: 204,          // Server telling client a request timed out
```

**Trade-offs**:
- ✅ Enables reliable delivery for critical messages
- ✅ Can detect packet loss at application level
- ✅ Can implement selective retransmission
- ❌ 7 bytes overhead per packet (~5-10% bandwidth increase)
- ⚠️ Requires protocol version upgrade (backward compatibility needed)

---

### System 2: Shooting System

**Files**:
- `public/src/network/ClientNetworkManager.js`
- `public/src/game/ClientBulletManager.js`
- `Server.js`
- `src/entities/BulletManager.js`

**Current Flow**:
```
Client shoots → Send BULLET_CREATE → Hope it works → See in WORLD_UPDATE 500ms later
```

**Proposed Flow**:
```
Client shoots →
  ├─ Create local bullet immediately
  ├─ Generate requestId
  ├─ Send BULLET_CREATE with requestId
  ├─ Store in pendingBullets Map
  └─ Start timeout timer (RTT × 3)

Server receives →
  ├─ Validate bullet (worldId, position, rate limit)
  ├─ Create server bullet
  └─ Send BULLET_ACK with requestId + serverId

Client receives ACK →
  ├─ Remove from pendingBullets
  ├─ Reconcile local bullet with server ID
  └─ Cancel timeout

On timeout →
  ├─ Retry up to 3 times
  ├─ Exponential backoff (RTT × 2, RTT × 4, RTT × 6)
  └─ After 3 fails: remove local bullet, show error
```

**New Client Code Structure**:
```javascript
class ReliableNetworkManager extends ClientNetworkManager {
    constructor() {
        super();
        this.pendingBullets = new Map(); // requestId → {data, sentAt, retries, timeoutId}
        this.nextRequestId = 1;
        this.measuredRTT = 100; // Start with 100ms estimate
    }

    sendShoot(bulletData) {
        const requestId = this.nextRequestId++;
        const localBulletId = `local_${Date.now()}_${requestId}`;

        // Add to pending tracking
        const request = {
            data: { ...bulletData, requestId, localBulletId },
            sentAt: Date.now(),
            retries: 0,
            timeoutId: null
        };

        this.pendingBullets.set(requestId, request);

        // Send to server
        this.send(MessageType.BULLET_CREATE, request.data);

        // Set timeout for retry
        request.timeoutId = setTimeout(
            () => this.retryBulletIfNeeded(requestId),
            this.measuredRTT * 3
        );

        return localBulletId;
    }

    handleBulletAck(ackData) {
        const { requestId, serverId, accepted, reason } = ackData;
        const request = this.pendingBullets.get(requestId);

        if (!request) return; // Already processed or timed out

        clearTimeout(request.timeoutId);
        this.pendingBullets.delete(requestId);

        if (accepted) {
            // Success! Reconcile local bullet with server ID
            this.game.bulletManager.reconcileBullet(
                request.data.localBulletId,
                serverId
            );
        } else {
            // Rejected! Remove local bullet and notify player
            this.game.bulletManager.removeBullet(request.data.localBulletId);
            console.error(`Bullet rejected: ${reason}`);
            this.game.showError(`Shot rejected: ${reason}`);
        }
    }

    retryBulletIfNeeded(requestId) {
        const request = this.pendingBullets.get(requestId);
        if (!request) return;

        request.retries++;

        if (request.retries >= 3) {
            // Give up after 3 retries
            this.pendingBullets.delete(requestId);
            this.game.bulletManager.removeBullet(request.data.localBulletId);
            console.error(`Bullet request ${requestId} failed after 3 retries`);
            this.game.showError('Connection unstable - shot failed');
            return;
        }

        // Retry with exponential backoff
        console.warn(`Retrying bullet request ${requestId} (attempt ${request.retries + 1})`);
        this.send(MessageType.BULLET_CREATE, request.data);

        const backoff = this.measuredRTT * (2 ** request.retries) * 3;
        request.timeoutId = setTimeout(
            () => this.retryBulletIfNeeded(requestId),
            backoff
        );
    }
}
```

**New Server Code Structure**:
```javascript
// In Server.js BULLET_CREATE handler
case MessageType.BULLET_CREATE:
    const { requestId, localBulletId, ...bulletData } = packet.data;

    // Validate bullet
    const validation = validateBulletRequest(bulletData, clientId);

    if (!validation.valid) {
        // Reject and send NACK
        sendToClient(c.socket, MessageType.BULLET_ACK, {
            requestId,
            accepted: false,
            reason: validation.reason
        });
        console.warn(`[BULLET REJECTED] Client ${clientId}: ${validation.reason}`);
        break;
    }

    // Create bullet
    const serverId = `bullet_${Date.now()}_${clientId}_${Math.random()}`;
    bulletManager.addBullet({
        ...bulletData,
        id: serverId,
        ownerId: clientId
    });

    // Send ACK immediately (don't wait for WORLD_UPDATE)
    sendToClient(c.socket, MessageType.BULLET_ACK, {
        requestId,
        serverId,
        accepted: true
    });

    console.log(`[BULLET CREATED] ${serverId} for client ${clientId}`);
    break;
```

**Validation Function**:
```javascript
function validateBulletRequest(bulletData, clientId) {
    // Check worldId
    if (!bulletData.worldId) {
        return { valid: false, reason: 'missing_worldId' };
    }

    // Check player exists in that world
    const player = clients.find(c => c.id === clientId);
    if (!player || player.worldId !== bulletData.worldId) {
        return { valid: false, reason: 'worldId_mismatch' };
    }

    // Check rate limiting
    const lastShot = playerLastShot.get(clientId) || 0;
    const timeSinceLastShot = Date.now() - lastShot;
    const minFireInterval = 100; // 10 shots/second max

    if (timeSinceLastShot < minFireInterval) {
        return { valid: false, reason: 'rate_limit' };
    }

    playerLastShot.set(clientId, Date.now());

    // Check position validity (basic sanity check)
    if (Math.abs(bulletData.x - player.x) > 5 || Math.abs(bulletData.y - player.y) > 5) {
        return { valid: false, reason: 'position_mismatch' };
    }

    return { valid: true };
}
```

**Expected Impact**:
- ✅ 0% shot loss (with retries)
- ✅ Immediate feedback if shot rejected
- ✅ 200-300ms confirmation (vs 500ms current)
- ✅ Rate limiting prevents spam
- ⚠️ 3× more messages for bullet creation (BULLET_CREATE + BULLET_ACK + potentially retries)

---

### System 3: Chunk Loading System

**Files**:
- `public/src/map/ClientMapManager.js`
- `src/world/MapManager.js`

**Current Problems**:
1. Blocks player movement on unloaded chunks (line 946)
2. 1500ms fixed throttle (line 592)
3. No timeout/retry mechanism
4. No priority system
5. No predictive loading

**Proposed Changes**:

#### Change 3.1: Allow Movement Through Unloaded Chunks

**Current** (`ClientMapManager.js:940-946`):
```javascript
if (!tile) {
    if (isBullet) {
        return false; // Bullets pass through
    }
    return true; // ❌ BLOCKS PLAYER
}
```

**Proposed**:
```javascript
if (!tile) {
    // Request chunk with high priority if not already pending
    const chunkX = Math.floor(tileX / this.chunkSize);
    const chunkY = Math.floor(tileY / this.chunkSize);
    this.requestChunkUrgent(chunkX, chunkY);

    // Allow movement through unloaded areas
    // Player will see ground tiles streaming in
    return false; // ✅ DON'T BLOCK
}
```

#### Change 3.2: Adaptive Throttling Based on RTT

**Current** (`ClientMapManager.js:592`):
```javascript
if (now - lastReq >= this.requestThrottleMs) { // Fixed 1500ms
    this.networkManager.requestChunk(chunkX, chunkY);
}
```

**Proposed**:
```javascript
// Adaptive throttle: 2× RTT minimum, max 1000ms
const adaptiveThrottle = Math.min(
    Math.max(this.networkManager.measuredRTT * 2, 500),
    1000
);

if (now - lastReq >= adaptiveThrottle) {
    this.requestChunkWithPriority(chunkX, chunkY, priority);
}
```

**RTT-based throttle examples**:
- 50ms RTT: 500ms throttle (use minimum)
- 150ms RTT: 300ms throttle
- 300ms RTT: 600ms throttle
- 500ms RTT: 1000ms throttle (use maximum)

#### Change 3.3: Chunk Request Timeout & Retry

**New System**:
```javascript
class ChunkRequestManager {
    constructor(networkManager) {
        this.networkManager = networkManager;
        this.pendingRequests = new Map(); // "x,y" → {requestId, sentAt, priority, retries}
        this.nextRequestId = 1;
    }

    requestChunk(chunkX, chunkY, priority = 'NORMAL') {
        const key = `${chunkX},${chunkY}`;

        // Check if already pending
        const existing = this.pendingRequests.get(key);
        if (existing) {
            // Upgrade priority if needed
            if (priority === 'URGENT' && existing.priority === 'NORMAL') {
                existing.priority = 'URGENT';
                console.log(`[CHUNK] Upgraded ${key} to URGENT`);
            }
            return;
        }

        // Create new request
        const requestId = this.nextRequestId++;
        const request = {
            requestId,
            chunkX,
            chunkY,
            priority,
            sentAt: Date.now(),
            retries: 0,
            timeoutId: null
        };

        this.pendingRequests.set(key, request);

        // Send request
        this.networkManager.send(MessageType.CHUNK_REQUEST, {
            requestId,
            chunkX,
            chunkY,
            priority
        });

        // Set timeout
        const timeout = this.getTimeout(priority);
        request.timeoutId = setTimeout(
            () => this.handleChunkTimeout(key),
            timeout
        );

        console.log(`[CHUNK] Requested ${key} (priority: ${priority}, timeout: ${timeout}ms)`);
    }

    handleChunkResponse(chunkData) {
        const key = `${chunkData.x},${chunkData.y}`;
        const request = this.pendingRequests.get(key);

        if (!request) {
            console.warn(`[CHUNK] Received unexpected chunk ${key}`);
            return;
        }

        clearTimeout(request.timeoutId);
        this.pendingRequests.delete(key);

        const latency = Date.now() - request.sentAt;
        console.log(`[CHUNK] Received ${key} in ${latency}ms (retries: ${request.retries})`);
    }

    handleChunkTimeout(key) {
        const request = this.pendingRequests.get(key);
        if (!request) return;

        request.retries++;

        // Give up after 5 retries
        if (request.retries >= 5) {
            this.pendingRequests.delete(key);
            console.error(`[CHUNK] Failed to load ${key} after 5 retries`);
            // Could show error to player or mark chunk as unavailable
            return;
        }

        // Retry with increased timeout
        console.warn(`[CHUNK] Timeout for ${key}, retrying (attempt ${request.retries + 1})`);

        request.sentAt = Date.now();
        this.networkManager.send(MessageType.CHUNK_REQUEST, {
            requestId: request.requestId,
            chunkX: request.chunkX,
            chunkY: request.chunkY,
            priority: request.priority
        });

        const timeout = this.getTimeout(request.priority) * (1.5 ** request.retries);
        request.timeoutId = setTimeout(
            () => this.handleChunkTimeout(key),
            timeout
        );
    }

    getTimeout(priority) {
        const baseTimeout = this.networkManager.measuredRTT * 3;
        return priority === 'URGENT' ? baseTimeout : baseTimeout * 2;
    }
}
```

#### Change 3.4: Predictive Chunk Loading

**Load chunks ahead of player movement**:

```javascript
class PredictiveChunkLoader {
    constructor(mapManager, networkManager) {
        this.mapManager = mapManager;
        this.networkManager = networkManager;
        this.playerVelocityHistory = [];
        this.maxHistoryLength = 10; // Track last 10 frames
    }

    updatePredictiveLoading(playerX, playerY, velocityX, velocityY) {
        // Track velocity over time
        this.playerVelocityHistory.push({ vx: velocityX, vy: velocityY, time: Date.now() });
        if (this.playerVelocityHistory.length > this.maxHistoryLength) {
            this.playerVelocityHistory.shift();
        }

        // Calculate average velocity direction
        const avgVelocity = this.calculateAverageVelocity();

        if (avgVelocity.speed < 0.1) {
            // Player is stationary or moving slowly, no prediction needed
            return;
        }

        // Predict position 1-2 seconds ahead based on latency
        const lookaheadTime = Math.max(1, this.networkManager.measuredRTT / 100); // 1-3 seconds
        const predictedX = playerX + avgVelocity.vx * lookaheadTime;
        const predictedY = playerY + avgVelocity.vy * lookaheadTime;

        // Request chunks around predicted position
        const predictedChunkX = Math.floor(predictedX / this.mapManager.chunkSize);
        const predictedChunkY = Math.floor(predictedY / this.mapManager.chunkSize);

        // Request in cone ahead of player movement
        const direction = Math.atan2(avgVelocity.vy, avgVelocity.vx);
        const coneAngle = Math.PI / 3; // 60 degree cone

        for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
                const chunkX = predictedChunkX + dx;
                const chunkY = predictedChunkY + dy;

                // Check if chunk is in direction cone
                const angleToChunk = Math.atan2(dy, dx);
                const angleDiff = Math.abs(angleToChunk - direction);

                if (angleDiff < coneAngle) {
                    const key = `${chunkX},${chunkY}`;
                    if (!this.mapManager.chunks.has(key)) {
                        this.mapManager.requestChunkWithPriority(chunkX, chunkY, 'NORMAL');
                    }
                }
            }
        }
    }

    calculateAverageVelocity() {
        if (this.playerVelocityHistory.length === 0) {
            return { vx: 0, vy: 0, speed: 0 };
        }

        let sumVx = 0, sumVy = 0;
        for (const v of this.playerVelocityHistory) {
            sumVx += v.vx;
            sumVy += v.vy;
        }

        const vx = sumVx / this.playerVelocityHistory.length;
        const vy = sumVy / this.playerVelocityHistory.length;
        const speed = Math.sqrt(vx * vx + vy * vy);

        return { vx, vy, speed };
    }
}
```

**Expected Impact**:
- ✅ No more movement blocking
- ✅ Chunks load ahead of player (smoother exploration)
- ✅ Automatic retry on failure
- ✅ Responsive throttling (300ms at 150ms ping vs 1500ms current)
- ⚠️ Slight increase in bandwidth (predictive loading)
- ⚠️ Might see chunk "pop-in" at edges (acceptable trade-off)

---

### System 4: Entity Interpolation

**Files**:
- `public/src/game/game.js`
- New file: `public/src/network/EntityInterpolator.js`

**Current State**: Entities rendered at exact server positions (causing teleporting/jittering)

**Proposed**: Snapshot Interpolation

**Concept**:
```
Server sends updates at 10 FPS (every 100ms):
  t=0:   Enemy at (10, 5)
  t=100: Enemy at (12, 6)
  t=200: Enemy at (14, 7)

Client receives at 200ms latency:
  t=200: Receives position (10, 5) from t=0
  t=300: Receives position (12, 6) from t=100
  t=400: Receives position (14, 7) from t=200

Client renders 150ms in the past:
  t=200: Render at (10, 5) [from t=0]
  t=250: Render interpolated (11, 5.5) [between t=0 and t=100]
  t=300: Render interpolated (12, 6) [from t=100]
  t=350: Render interpolated (13, 6.5) [between t=100 and t=200]

Result: Smooth 60 FPS rendering from 10 FPS updates!
```

**Implementation**:

```javascript
// public/src/network/EntityInterpolator.js

class EntityInterpolator {
    constructor(bufferTimeMs = 150) {
        this.bufferTimeMs = bufferTimeMs; // Render 150ms in past
        this.snapshots = []; // Array of {time, entities}
        this.maxSnapshotAge = 1000; // Keep 1 second of history
    }

    /**
     * Add a snapshot from server
     * @param {number} serverTime - Server timestamp
     * @param {Object} entities - Entity data from WORLD_UPDATE
     */
    addSnapshot(serverTime, entities) {
        this.snapshots.push({
            time: serverTime,
            entities: JSON.parse(JSON.stringify(entities)) // Deep copy
        });

        // Remove old snapshots
        const cutoffTime = serverTime - this.maxSnapshotAge;
        this.snapshots = this.snapshots.filter(s => s.time >= cutoffTime);

        // Sort by time (just in case packets arrive out of order)
        this.snapshots.sort((a, b) => a.time - b.time);
    }

    /**
     * Get interpolated state for rendering
     * @param {number} renderTime - Current client time
     * @returns {Object} Interpolated entity state
     */
    getInterpolatedState(renderTime) {
        if (this.snapshots.length === 0) {
            return null;
        }

        // Render in the past to have buffer for interpolation
        const targetTime = renderTime - this.bufferTimeMs;

        // Find two snapshots to interpolate between
        let snapshotBefore = null;
        let snapshotAfter = null;

        for (let i = 0; i < this.snapshots.length - 1; i++) {
            if (this.snapshots[i].time <= targetTime && this.snapshots[i + 1].time >= targetTime) {
                snapshotBefore = this.snapshots[i];
                snapshotAfter = this.snapshots[i + 1];
                break;
            }
        }

        // If no bracketing snapshots, use latest available
        if (!snapshotBefore) {
            if (this.snapshots[this.snapshots.length - 1].time < targetTime) {
                // Target time is ahead of all snapshots (extrapolate)
                return this.extrapolate(targetTime);
            } else {
                // Target time is before all snapshots (use first)
                return this.snapshots[0].entities;
            }
        }

        if (!snapshotAfter) {
            return snapshotBefore.entities;
        }

        // Interpolate between snapshots
        const timeDelta = snapshotAfter.time - snapshotBefore.time;
        const t = (targetTime - snapshotBefore.time) / timeDelta;

        return this.interpolateEntities(snapshotBefore.entities, snapshotAfter.entities, t);
    }

    /**
     * Interpolate between two entity states
     */
    interpolateEntities(state1, state2, t) {
        const result = {
            enemies: [],
            bullets: [],
            players: {}
        };

        // Interpolate enemies
        if (state1.enemies && state2.enemies) {
            const enemyMap = new Map();
            state2.enemies.forEach(e => enemyMap.set(e.id, e));

            state1.enemies.forEach(e1 => {
                const e2 = enemyMap.get(e1.id);
                if (e2) {
                    result.enemies.push({
                        ...e2,
                        x: this.lerp(e1.x, e2.x, t),
                        y: this.lerp(e1.y, e2.y, t),
                        rotation: this.lerpAngle(e1.rotation, e2.rotation, t)
                    });
                } else {
                    // Enemy was removed, fade out or use state1
                    result.enemies.push(e1);
                }
            });

            // Add new enemies from state2
            state2.enemies.forEach(e2 => {
                if (!state1.enemies.find(e1 => e1.id === e2.id)) {
                    result.enemies.push(e2);
                }
            });
        }

        // Interpolate bullets (similar to enemies)
        if (state1.bullets && state2.bullets) {
            const bulletMap = new Map();
            state2.bullets.forEach(b => bulletMap.set(b.id, b));

            state1.bullets.forEach(b1 => {
                const b2 = bulletMap.get(b1.id);
                if (b2) {
                    result.bullets.push({
                        ...b2,
                        x: this.lerp(b1.x, b2.x, t),
                        y: this.lerp(b1.y, b2.y, t)
                    });
                }
            });

            state2.bullets.forEach(b2 => {
                if (!state1.bullets.find(b1 => b1.id === b2.id)) {
                    result.bullets.push(b2);
                }
            });
        }

        // Interpolate players (skip local player - use client prediction)
        if (state1.players && state2.players) {
            const localPlayerId = window.gameState?.character?.id;

            for (const id in state2.players) {
                if (id === localPlayerId) {
                    // Don't interpolate local player (use client prediction)
                    continue;
                }

                const p1 = state1.players[id];
                const p2 = state2.players[id];

                if (p1 && p2) {
                    result.players[id] = {
                        ...p2,
                        x: this.lerp(p1.x, p2.x, t),
                        y: this.lerp(p1.y, p2.y, t),
                        rotation: this.lerpAngle(p1.rotation, p2.rotation, t)
                    };
                } else if (p2) {
                    result.players[id] = p2;
                }
            }
        }

        return result;
    }

    /**
     * Extrapolate state when target time is ahead of latest snapshot
     */
    extrapolate(targetTime) {
        if (this.snapshots.length < 2) {
            return this.snapshots[this.snapshots.length - 1]?.entities || null;
        }

        const latest = this.snapshots[this.snapshots.length - 1];
        const previous = this.snapshots[this.snapshots.length - 2];

        const timeDelta = latest.time - previous.time;
        const extrapolateTime = targetTime - latest.time;
        const t = 1 + (extrapolateTime / timeDelta); // t > 1 means extrapolation

        // Limit extrapolation to prevent wild predictions
        const clampedT = Math.min(t, 1.2); // Max 20% extrapolation

        return this.interpolateEntities(previous.entities, latest.entities, clampedT);
    }

    /**
     * Linear interpolation
     */
    lerp(a, b, t) {
        return a + (b - a) * t;
    }

    /**
     * Angular interpolation (handles wraparound)
     */
    lerpAngle(a, b, t) {
        // Normalize angles to [0, 2π]
        a = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
        b = ((b % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);

        // Take shortest path
        let diff = b - a;
        if (diff > Math.PI) {
            diff -= 2 * Math.PI;
        } else if (diff < -Math.PI) {
            diff += 2 * Math.PI;
        }

        return a + diff * t;
    }
}

export default EntityInterpolator;
```

**Integration into game loop**:

```javascript
// public/src/game/game.js

import EntityInterpolator from '../network/EntityInterpolator.js';

class Game {
    constructor() {
        // ... existing code ...
        this.interpolator = new EntityInterpolator(150); // 150ms buffer
        this.serverTimeOffset = 0; // Will be calibrated during handshake
    }

    // Called when WORLD_UPDATE received
    updateWorld(enemies, bullets, players, objects) {
        // Get server timestamp from WORLD_UPDATE
        const serverTime = Date.now() - this.serverTimeOffset; // Approximate server time

        // Add snapshot to interpolator
        this.interpolator.addSnapshot(serverTime, {
            enemies,
            bullets,
            players,
            objects
        });

        // Don't update entities directly anymore - let interpolator handle it
    }

    // Called every render frame (60 FPS)
    render(deltaTime) {
        // Get interpolated state
        const renderTime = Date.now() - this.serverTimeOffset;
        const interpolatedState = this.interpolator.getInterpolatedState(renderTime);

        if (interpolatedState) {
            // Render interpolated entities
            this.renderEnemies(interpolatedState.enemies);
            this.renderBullets(interpolatedState.bullets);
            this.renderOtherPlayers(interpolatedState.players);
        }

        // Render local player using client prediction (not interpolated)
        this.renderLocalPlayer();

        // ... rest of rendering ...
    }
}
```

**Expected Impact**:
- ✅ Smooth 60 FPS rendering of entities
- ✅ No more teleporting/jittering
- ✅ Works even with variable latency
- ⚠️ Entities rendered 150ms in the past (acceptable for non-local entities)
- ⚠️ Adds complexity to rendering loop

---

### System 5: Lag Compensation for Hit Detection

**Files**:
- `src/entities/CollisionManager.js`
- New file: `src/network/LagCompensator.js`

**Current Problem**: Server validates collisions at current time, not when client saw enemy

**Proposed Solution**: Rewind entity positions to client's perceived time

**Implementation**:

```javascript
// src/network/LagCompensator.js

class LagCompensator {
    constructor(enemyManager) {
        this.enemyManager = enemyManager;
        this.entityHistory = new Map(); // entityId → [{time, x, y, width, height}, ...]
        this.historyDuration = 1000; // Keep 1 second of history
        this.snapshotInterval = 50; // Snapshot every 50ms
        this.lastSnapshotTime = 0;
    }

    /**
     * Record entity positions every frame
     * Call this in server's main update loop
     */
    recordSnapshot(currentTime) {
        if (currentTime - this.lastSnapshotTime < this.snapshotInterval) {
            return; // Don't snapshot too frequently
        }

        this.lastSnapshotTime = currentTime;

        // Record all enemy positions
        for (let i = 0; i < this.enemyManager.enemyCount; i++) {
            const id = this.enemyManager.id[i];

            if (!this.entityHistory.has(id)) {
                this.entityHistory.set(id, []);
            }

            const history = this.entityHistory.get(id);
            history.push({
                time: currentTime,
                x: this.enemyManager.x[i],
                y: this.enemyManager.y[i],
                width: this.enemyManager.width[i],
                height: this.enemyManager.height[i]
            });

            // Trim old history
            const cutoffTime = currentTime - this.historyDuration;
            this.entityHistory.set(id, history.filter(h => h.time >= cutoffTime));
        }

        // Clean up history for removed entities
        const activeIds = new Set();
        for (let i = 0; i < this.enemyManager.enemyCount; i++) {
            activeIds.add(this.enemyManager.id[i]);
        }

        for (const id of this.entityHistory.keys()) {
            if (!activeIds.has(id)) {
                this.entityHistory.delete(id);
            }
        }
    }

    /**
     * Rewind entity to its position at a specific time
     * @param {string} entityId - Enemy ID
     * @param {number} targetTime - Time to rewind to
     * @returns {Object|null} {x, y, width, height} or null if not found
     */
    rewindEntity(entityId, targetTime) {
        const history = this.entityHistory.get(entityId);
        if (!history || history.length === 0) {
            return null;
        }

        // Find two snapshots to interpolate between
        let before = null;
        let after = null;

        for (let i = 0; i < history.length - 1; i++) {
            if (history[i].time <= targetTime && history[i + 1].time >= targetTime) {
                before = history[i];
                after = history[i + 1];
                break;
            }
        }

        // If target time is before all history, use oldest snapshot
        if (!before && history[0].time > targetTime) {
            return history[0];
        }

        // If target time is after all history, use newest snapshot
        if (!before && history[history.length - 1].time < targetTime) {
            return history[history.length - 1];
        }

        // Exact match
        if (before && !after) {
            return before;
        }

        // Interpolate between two snapshots
        const timeDelta = after.time - before.time;
        const t = (targetTime - before.time) / timeDelta;

        return {
            x: before.x + (after.x - before.x) * t,
            y: before.y + (after.y - before.y) * t,
            width: before.width, // Don't interpolate size
            height: before.height
        };
    }

    /**
     * Validate collision with lag compensation
     * @param {Object} collisionData - From client: {bulletId, enemyId, timestamp, bulletPos, enemyPos}
     * @param {number} clientLatency - Measured RTT / 2
     * @returns {boolean} Whether collision is valid
     */
    validateCollisionWithCompensation(collisionData, clientLatency) {
        const { bulletPos, enemyId, timestamp } = collisionData;

        // Calculate when client saw this collision
        const clientPerceivedTime = timestamp; // Client's timestamp
        const serverCurrentTime = Date.now();

        // Rewind to when client saw the enemy
        // Use one-way latency (RTT / 2) as approximation
        const rewindTime = serverCurrentTime - clientLatency;

        // Get enemy's position at that time
        const enemyAtClientTime = this.rewindEntity(enemyId, rewindTime);

        if (!enemyAtClientTime) {
            console.warn(`[LAG COMP] No history for enemy ${enemyId} at time ${rewindTime}`);
            return false;
        }

        // Check collision using rewound position
        const collision = this.checkAABBCollision(
            bulletPos.x, bulletPos.y, collisionData.bulletWidth, collisionData.bulletHeight,
            enemyAtClientTime.x, enemyAtClientTime.y,
            enemyAtClientTime.width, enemyAtClientTime.height
        );

        if (collision) {
            console.log(`[LAG COMP] Valid hit on ${enemyId}:
                Client saw at: (${collisionData.enemyPos.x.toFixed(2)}, ${collisionData.enemyPos.y.toFixed(2)})
                Rewound to: (${enemyAtClientTime.x.toFixed(2)}, ${enemyAtClientTime.y.toFixed(2)})
                Latency: ${clientLatency}ms`);
        }

        return collision;
    }

    checkAABBCollision(ax, ay, aw, ah, bx, by, bw, bh) {
        // Same AABB logic as CollisionManager
        const acx = ax + aw / 2;
        const acy = ay + ah / 2;
        const bcx = bx + bw / 2;
        const bcy = by + bh / 2;

        const a_left = acx - aw / 2;
        const a_right = acx + aw / 2;
        const a_top = acy - ah / 2;
        const a_bottom = acy + ah / 2;

        const b_left = bcx - bw / 2;
        const b_right = bcx + bw / 2;
        const b_top = bcy - bh / 2;
        const b_bottom = bcy + bh / 2;

        return a_right >= b_left && a_left <= b_right &&
               a_bottom >= b_top && a_top <= b_bottom;
    }
}

module.exports = LagCompensator;
```

**Integration into Server.js**:

```javascript
// Server.js

const LagCompensator = require('./src/network/LagCompensator');

// Initialize per world
const lagCompensator = new LagCompensator(enemyManager);

// In main update loop (every tick)
function update(deltaTime) {
    // ... existing update code ...

    // Record entity positions for lag compensation
    lagCompensator.recordSnapshot(Date.now());
}

// In COLLISION message handler
case MessageType.COLLISION:
    const clientLatency = getClientLatency(c.id); // From ping/pong measurements

    const isValid = lagCompensator.validateCollisionWithCompensation(
        packet.data,
        clientLatency
    );

    if (isValid) {
        // Process collision
        const result = collisionManager.processCollision(...);
        sendToClient(c.socket, MessageType.COLLISION_RESULT, {
            valid: true,
            ...result
        });
    } else {
        sendToClient(c.socket, MessageType.COLLISION_RESULT, {
            valid: false,
            reason: 'position_mismatch'
        });
    }
    break;
```

**Expected Impact**:
- ✅ Fair hit detection at high latency
- ✅ Shots that look like hits will register
- ✅ Standard technique used by CS:GO, Overwatch, etc.
- ⚠️ Can feel unfair for enemy (might die after taking cover)
- ⚠️ Memory overhead for position history (~50 KB per 100 enemies)
- ⚠️ CPU overhead for rewinding (minimal, ~0.1ms per collision)

---

### System 6: RTT Measurement & Adaptive Rates

**Files**:
- `public/src/network/ClientNetworkManager.js`
- `Server.js`

**Proposed Changes**:

#### Change 6.1: Ping/Pong System for RTT Measurement

**Client Side**:
```javascript
class ClientNetworkManager {
    constructor() {
        // ... existing code ...
        this.measuredRTT = 100; // Initial estimate
        this.rttSamples = []; // Last 10 samples
        this.maxRTTSamples = 10;
        this.pendingPings = new Map(); // pingId → sentTime
        this.pingInterval = null;
    }

    connect() {
        // ... existing connection code ...

        // Start ping interval after connection
        this.socket.addEventListener('open', () => {
            this.startPingInterval();
        });
    }

    startPingInterval() {
        // Send ping every 5 seconds
        this.pingInterval = setInterval(() => {
            this.sendPing();
        }, 5000);
    }

    sendPing() {
        const pingId = Date.now();
        this.pendingPings.set(pingId, Date.now());

        this.send(MessageType.PING, {
            id: pingId,
            clientTime: pingId
        });

        // Clean up old pings (timeout after 10 seconds)
        const now = Date.now();
        for (const [id, sentTime] of this.pendingPings) {
            if (now - sentTime > 10000) {
                this.pendingPings.delete(id);
            }
        }
    }

    handlePong(data) {
        const sentTime = this.pendingPings.get(data.id);
        if (!sentTime) {
            console.warn('[PING] Received pong for unknown ping:', data.id);
            return;
        }

        const rtt = Date.now() - sentTime;
        this.updateRTT(rtt);
        this.pendingPings.delete(data.id);

        console.log(`[PING] RTT: ${rtt}ms (avg: ${this.measuredRTT.toFixed(0)}ms)`);
    }

    updateRTT(newRTT) {
        // Add to samples
        this.rttSamples.push(newRTT);
        if (this.rttSamples.length > this.maxRTTSamples) {
            this.rttSamples.shift();
        }

        // Calculate exponential moving average
        // 90% old value, 10% new value (smooth out spikes)
        this.measuredRTT = this.measuredRTT * 0.9 + newRTT * 0.1;

        // Also track min/max for display
        this.minRTT = Math.min(...this.rttSamples);
        this.maxRTT = Math.max(...this.rttSamples);

        // Update UI if available
        if (window.gameState) {
            window.gameState.networkStats = {
                rtt: Math.round(this.measuredRTT),
                minRTT: this.minRTT,
                maxRTT: this.maxRTT,
                jitter: this.maxRTT - this.minRTT
            };
        }
    }

    disconnect() {
        // ... existing disconnect code ...

        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
}
```

**Server Side**:
```javascript
// Server.js - In message handler

case MessageType.PING:
    // Simply echo back the ping
    sendToClient(c.socket, MessageType.PONG, {
        id: packet.data.id,
        clientTime: packet.data.clientTime,
        serverTime: Date.now()
    });
    break;
```

#### Change 6.2: Adaptive Player Update Rate

**Current**: Send updates at 60 FPS regardless of latency

**Proposed**: Adjust send rate based on RTT

```javascript
class AdaptiveUpdateSender {
    constructor(networkManager) {
        this.networkManager = networkManager;
        this.lastUpdateSent = 0;
        this.minInterval = 16; // 60 FPS max
        this.maxInterval = 100; // 10 FPS min
    }

    shouldSendUpdate() {
        const now = Date.now();
        const interval = this.getUpdateInterval();

        if (now - this.lastUpdateSent >= interval) {
            this.lastUpdateSent = now;
            return true;
        }

        return false;
    }

    getUpdateInterval() {
        const rtt = this.networkManager.measuredRTT;

        // At low latency (< 50ms): send at 60 FPS (16ms interval)
        // At medium latency (50-150ms): send at 30 FPS (33ms interval)
        // At high latency (150-300ms): send at 15 FPS (66ms interval)
        // At very high latency (> 300ms): send at 10 FPS (100ms interval)

        if (rtt < 50) {
            return 16; // 60 FPS
        } else if (rtt < 150) {
            return 33; // 30 FPS
        } else if (rtt < 300) {
            return 66; // 15 FPS
        } else {
            return 100; // 10 FPS
        }
    }

    forceUpdate() {
        // Allow immediate update for critical actions (shooting, etc.)
        this.lastUpdateSent = 0;
    }
}

// In game loop
if (updateSender.shouldSendUpdate()) {
    networkManager.sendPlayerUpdate(playerData);
}
```

**Expected Impact**:
- ✅ Reduced bandwidth at high latency (50% reduction at 200ms ping)
- ✅ Lower packet loss rate
- ✅ More stable connection
- ⚠️ Slightly lower position update rate (acceptable trade-off)

---

### System 7: Delta Compression for WORLD_UPDATE

**Files**:
- `Server.js`
- `public/src/network/ClientNetworkManager.js`

**Current Problem**: Server sends ALL entities in every WORLD_UPDATE

**Proposed Solution**: Only send changed entities

```javascript
// Server.js

class DeltaCompressor {
    constructor() {
        this.clientLastStates = new Map(); // clientId → lastState
    }

    /**
     * Create delta update for a client
     * @param {string} clientId
     * @param {Object} currentState - {enemies, bullets, players, objects}
     * @returns {Object} Delta update
     */
    createDeltaUpdate(clientId, currentState) {
        const lastState = this.clientLastStates.get(clientId) || {};

        const delta = {
            enemies: this.getDeltaEntities(lastState.enemies, currentState.enemies),
            bullets: this.getDeltaEntities(lastState.bullets, currentState.bullets),
            players: this.getDeltaPlayers(lastState.players, currentState.players),
            objects: this.getDeltaEntities(lastState.objects, currentState.objects),
            removedEnemies: this.getRemovedIds(lastState.enemies, currentState.enemies),
            removedBullets: this.getRemovedIds(lastState.bullets, currentState.bullets)
        };

        // Update last state
        this.clientLastStates.set(clientId, JSON.parse(JSON.stringify(currentState)));

        return delta;
    }

    getDeltaEntities(oldList = [], newList = []) {
        const changes = [];
        const oldMap = new Map(oldList.map(e => [e.id, e]));

        for (const entity of newList) {
            const old = oldMap.get(entity.id);

            if (!old) {
                // New entity
                changes.push({ ...entity, _new: true });
            } else if (this.hasChanged(old, entity)) {
                // Changed entity
                changes.push(this.getChangedFields(old, entity));
            }
        }

        return changes;
    }

    getDeltaPlayers(oldPlayers = {}, newPlayers = {}) {
        const changes = {};

        for (const id in newPlayers) {
            const old = oldPlayers[id];
            const current = newPlayers[id];

            if (!old) {
                changes[id] = { ...current, _new: true };
            } else if (this.hasChanged(old, current)) {
                changes[id] = this.getChangedFields(old, current);
            }
        }

        return changes;
    }

    hasChanged(old, current) {
        // Check position (threshold 0.01 tiles)
        if (Math.abs(old.x - current.x) > 0.01) return true;
        if (Math.abs(old.y - current.y) > 0.01) return true;

        // Check other important fields
        if (old.health !== current.health) return true;
        if (old.rotation !== current.rotation) return true;

        return false;
    }

    getChangedFields(old, current) {
        const changes = { id: current.id };

        // Only include changed fields
        if (Math.abs(old.x - current.x) > 0.01) changes.x = current.x;
        if (Math.abs(old.y - current.y) > 0.01) changes.y = current.y;
        if (old.health !== current.health) changes.health = current.health;
        if (old.rotation !== current.rotation) changes.rotation = current.rotation;

        return changes;
    }

    getRemovedIds(oldList = [], newList = []) {
        const newIds = new Set(newList.map(e => e.id));
        return oldList
            .filter(e => !newIds.has(e.id))
            .map(e => e.id);
    }

    clearClientState(clientId) {
        this.clientLastStates.delete(clientId);
    }
}

// Usage in Server.js broadcast
const deltaCompressor = new DeltaCompressor();

function broadcastWorldUpdate() {
    const currentState = {
        enemies: enemyManager.getEnemiesData(mapId),
        bullets: bulletManager.getBulletsData(mapId),
        players: getPlayersData(),
        objects: mapManager.getObjects(mapId)
    };

    clients.forEach(c => {
        const delta = deltaCompressor.createDeltaUpdate(c.id, currentState);
        sendToClient(c.socket, MessageType.WORLD_UPDATE, delta);
    });
}
```

**Client Side** - Apply delta updates:
```javascript
// public/src/network/ClientNetworkManager.js

handlers[MessageType.WORLD_UPDATE] = (data) => {
    // Apply delta updates
    if (data.enemies) {
        data.enemies.forEach(enemy => {
            if (enemy._new) {
                // Add new enemy
                game.enemyManager.addEnemy(enemy);
            } else {
                // Update existing enemy (partial data)
                game.enemyManager.updateEnemy(enemy);
            }
        });
    }

    // Remove deleted enemies
    if (data.removedEnemies) {
        data.removedEnemies.forEach(id => {
            game.enemyManager.removeEnemy(id);
        });
    }

    // Similar for bullets, players, objects...

    // Feed to interpolator
    game.interpolator.addSnapshot(Date.now(), data);
};
```

**Expected Impact**:
- ✅ 50-80% bandwidth reduction for WORLD_UPDATE
- ✅ Smaller packets = lower chance of corruption
- ✅ More stable at high latency
- ⚠️ More complex message handling
- ⚠️ Slightly higher CPU usage for delta calculation

---

## Implementation Phases

### Phase 1: Critical Fixes (Week 1)
**Goal**: Make game playable at 200ms ping

**Tasks**:
1. **RTT Measurement** (2 hours)
   - Add PING/PONG messages
   - Implement measuredRTT tracking
   - Display in UI

2. **Fix Chunk Loading** (4 hours)
   - Remove movement blocking (line 946)
   - Add adaptive throttling based on RTT
   - Add chunk request timeout & retry
   - Priority-based chunk requests

3. **Fix Shooting** (6 hours)
   - Add BULLET_ACK message type
   - Implement acknowledgment tracking
   - Add retry logic with exponential backoff
   - Add bullet validation on server
   - Better error messages

**Expected Outcome**: Game functional at 200ms ping (chunks load, shots register)

---

### Phase 2: Smoothness Improvements (Week 2)
**Goal**: Make game feel smooth at 200ms ping

**Tasks**:
1. **Entity Interpolation** (8 hours)
   - Create EntityInterpolator class
   - Integrate into game render loop
   - Handle edge cases (new entities, removed entities)
   - Tune buffer time

2. **Adaptive Update Rates** (3 hours)
   - Implement AdaptiveUpdateSender
   - Adjust client send rate based on RTT
   - Critical action overrides (shooting)

3. **UI Improvements** (2 hours)
   - Network stats display (RTT, jitter, packet loss)
   - Loading indicators for chunks
   - "Shot rejected" notifications

**Expected Outcome**: Smooth gameplay with minimal jitter

---

### Phase 3: Advanced Optimization (Week 3)
**Goal**: Optimize for 300ms ping and reduce bandwidth

**Tasks**:
1. **Lag Compensation** (10 hours)
   - Create LagCompensator class
   - Record entity position history
   - Rewind positions for collision validation
   - Tune history duration

2. **Delta Compression** (6 hours)
   - Create DeltaCompressor
   - Modify WORLD_UPDATE to send deltas
   - Update client to apply deltas
   - Handle full sync for new clients

3. **Predictive Chunk Loading** (4 hours)
   - Create PredictiveChunkLoader
   - Calculate player velocity
   - Request chunks ahead of movement
   - Cone-based prediction

**Expected Outcome**: Playable at 300ms ping, reduced bandwidth

---

### Phase 4: Polish & Testing (Week 4)
**Goal**: Production-ready lag compensation

**Tasks**:
1. **Testing** (8 hours)
   - Simulate various latencies (50ms, 150ms, 300ms)
   - Test packet loss scenarios (1%, 5%, 10%)
   - Test jitter scenarios
   - Edge case testing (connection drops, etc.)

2. **Monitoring & Metrics** (4 hours)
   - Add telemetry for network performance
   - Track success rates (bullets, chunks, collisions)
   - Alert on anomalies

3. **Documentation** (3 hours)
   - Document new network protocol
   - Update architecture diagrams
   - Write troubleshooting guide

**Expected Outcome**: Production-ready system with monitoring

---

## File-by-File Change Summary

### Files to Modify

| File | Changes | Priority | Estimated Time |
|------|---------|----------|----------------|
| `common/protocol.js` | Add new message types (ACK, NACK), optional: add sequence numbers | P1 | 1 hour |
| `public/src/network/ClientNetworkManager.js` | Add RTT measurement, acknowledgment tracking, retry logic | P1 | 4 hours |
| `public/src/map/ClientMapManager.js` | Remove blocking, adaptive throttling, timeout/retry | P1 | 3 hours |
| `Server.js` | Add BULLET_ACK, PONG handlers, validation | P1 | 3 hours |
| `src/entities/BulletManager.js` | Add validation logic | P1 | 1 hour |
| `public/src/game/game.js` | Integrate interpolator, adaptive updates | P2 | 4 hours |
| `src/entities/CollisionManager.js` | Integrate lag compensator | P3 | 3 hours |

### New Files to Create

| File | Purpose | Priority | Estimated Time |
|------|---------|----------|----------------|
| `public/src/network/EntityInterpolator.js` | Client-side interpolation | P2 | 6 hours |
| `public/src/network/ChunkRequestManager.js` | Chunk request tracking & retry | P1 | 3 hours |
| `public/src/network/PredictiveChunkLoader.js` | Predictive chunk loading | P3 | 3 hours |
| `src/network/LagCompensator.js` | Server-side lag compensation | P3 | 8 hours |
| `src/network/DeltaCompressor.js` | Delta compression for world updates | P3 | 5 hours |
| `public/src/network/AdaptiveUpdateSender.js` | Adaptive send rates | P2 | 2 hours |

### Total Estimated Time
- **Phase 1** (Critical): ~12 hours (1.5 days)
- **Phase 2** (Smoothness): ~13 hours (1.5 days)
- **Phase 3** (Advanced): ~20 hours (2.5 days)
- **Phase 4** (Polish): ~15 hours (2 days)

**Total**: ~60 hours (7-8 days of focused work)

---

## Trade-offs Summary

### Bandwidth vs Latency

| Approach | Bandwidth Impact | Latency Impact | Recommended? |
|----------|------------------|----------------|--------------|
| Acknowledgments | +10-20% (small overhead) | No change | ✅ Yes |
| Delta compression | -50-80% | No change | ✅ Yes |
| Adaptive send rate | -30-50% | No change | ✅ Yes |
| Interpolation | No change | +100-150ms (render in past) | ✅ Yes (acceptable) |
| Predictive loading | +10-20% | -400ms (perceived) | ✅ Yes |

### Complexity vs Reliability

| System | Complexity | Reliability Gain | Recommended? |
|--------|------------|------------------|--------------|
| Retry logic | Low | High | ✅ Essential |
| Interpolation | Medium | Medium | ✅ High value |
| Lag compensation | High | High | ✅ For 300ms target |
| Delta compression | Medium | Medium | ✅ Bandwidth savings |
| Predictive loading | Medium | Medium | ⚠️ Nice to have |

---

## Risk Analysis

### High Risk Items
1. **Protocol changes** - Could break existing clients
   - **Mitigation**: Version negotiation in handshake, backward compatibility

2. **Interpolation bugs** - Entities could appear in wrong places
   - **Mitigation**: Extensive testing, fallback to no interpolation

3. **Lag compensation unfairness** - "Died behind cover" complaints
   - **Mitigation**: Limit rewind time to 500ms max, clear communication to players

### Medium Risk Items
1. **Chunk retry storms** - Many clients retrying simultaneously
   - **Mitigation**: Jitter in retry timing, server-side rate limiting

2. **Memory leaks** - Entity history accumulation
   - **Mitigation**: Strict limits on history duration, periodic cleanup

### Low Risk Items
1. **Adaptive rate oscillation** - Send rate bouncing up/down
   - **Mitigation**: Hysteresis in rate adjustment

2. **Delta state desync** - Client and server state diverging
   - **Mitigation**: Periodic full syncs every 10 seconds

---

## Success Metrics

### Quantitative Goals
- **Chunk load time**: < 1 second at 200ms ping (currently: up to 1.5s + 400ms = 1.9s)
- **Shot registration rate**: > 99% (currently: ~90-95% with packet loss)
- **Entity jitter**: < 0.5 tiles per frame (currently: 1-2 tiles)
- **Bandwidth per player**: < 300 KB/s (currently: ~500 KB/s)
- **Playable ping range**: Up to 300ms (currently: < 100ms)

### Qualitative Goals
- Player can move continuously without stuttering
- Shots feel responsive and reliable
- Other players move smoothly
- Hit detection feels fair
- No invisible walls

---

## Conclusion

The current architecture is solid but optimized for low-latency scenarios. To support 200-300ms cross-continent play, we need:

1. **Reliable delivery** for critical messages (shooting, chunks)
2. **Movement freedom** through unloaded chunks
3. **Interpolation** for smooth rendering
4. **Lag compensation** for fair hit detection
5. **Adaptive rates** to reduce bandwidth and packet loss

The proposed changes follow industry-standard techniques used in games like:
- **CS:GO** - Lag compensation with position rewinding
- **Overwatch** - Favor the shooter hit detection
- **Rocket League** - Heavy client prediction with server reconciliation
- **Realm of the Mad God** - Chunk streaming with client prediction

**Estimated effort**: 7-8 days of focused development
**Risk level**: Medium (requires careful testing but uses proven techniques)
**Impact**: Transform game from unplayable → smooth at 200-300ms ping
