# ROTMG Networking Stack Documentation

## Overview

The game uses a hybrid networking architecture with multiple transport protocols for optimal performance:

1. **WebSocket (TCP)** - Primary connection, reliable messaging
2. **WebTransport (QUIC/UDP)** - Low-latency game updates via binary protocol
3. **PlayIt.gg Tunnel** - External access to WebTransport (UDP port forwarding)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT                                   │
├─────────────────────────────────────────────────────────────────┤
│  ClientNetworkManager.js                                         │
│  ├── WebSocket Connection (port 4000)                           │
│  │   └── JSON messages (reliable)                               │
│  └── WebTransport Connection (port 4433 via PlayIt)             │
│      └── Binary protocol (low-latency)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER                                   │
├─────────────────────────────────────────────────────────────────┤
│  Server.js                                                       │
│  ├── Express HTTP (port 4000)                                   │
│  ├── WebSocket Server (ws)                                      │
│  │   └── Handles: handshake, chat, abilities, inventory         │
│  └── WebTransport Server (port 4433)                            │
│      └── Handles: position updates, world state, bullets        │
└─────────────────────────────────────────────────────────────────┘
```

## Transport Protocols

### 1. WebSocket (TCP) - Port 4000

**Purpose:** Reliable messaging for critical game events

**Used for:**
- Initial handshake and authentication
- Player login/registration (name, email, class)
- Chat messages
- Ability usage
- Inventory operations
- Portal transitions
- Death/respawn events

**Key Files:**
- `Server.js` - WebSocket server setup (line 159)
- `public/src/network/ClientNetworkManager.js` - Client WebSocket handling

**Message Format:** JSON
```javascript
{
  type: MessageType.XXX,  // Number from protocol-native.js
  data: { ... }           // Payload
}
```

### 2. WebTransport (QUIC/UDP) - Port 4433

**Purpose:** Low-latency updates for real-time game state

**Used for:**
- Player position updates (60Hz)
- Enemy positions and states
- Bullet positions and spawning
- World state synchronization

**Key Files:**
- `src/network/WebTransportServer.js` - Server implementation
- `public/src/network/WebTransportManager.js` - Client implementation
- `common/BinaryProtocol.js` - Binary encoding/decoding

**External Access:**
WebTransport uses UDP which requires special handling for NAT traversal.
We use PlayIt.gg to tunnel UDP traffic:
- PlayIt forwards: `quic.eternalconquests.com:10615` → `localhost:4433`
- Client connects to: `https://quic.eternalconquests.com:10615/game`

### 3. Binary Protocol

**Purpose:** Minimize bandwidth for frequent updates (5-10x smaller than JSON)

**Key Components:**

```javascript
// common/BinaryProtocol.js

// Delta flags for selective encoding
DeltaFlags = {
  POSITION: 1,    // x, y coordinates
  VELOCITY: 2,    // vx, vy
  HEALTH: 4,      // health, maxHealth
  STATE: 8,       // ownerId, worldId, etc.
}

// Encoding functions
encodePlayer(writer, player, flags)
encodeEnemy(writer, enemy, flags)
encodeBullet(writer, bullet, flags)
encodeWorldDelta(players, enemies, bullets, removed, timestamp)

// Decoding functions
decodeWorldDelta(buffer) → { players, enemies, bullets, removed, timestamp }
```

**Binary Message Structure:**
```
[1 byte]  Message type
[4 bytes] Timestamp
[2 bytes] Player count
  [per player: id, flags, position?, velocity?, health?, state?]
[2 bytes] Enemy count
  [per enemy: id, flags, position?, velocity?, health?, state?]
[2 bytes] Bullet count
  [per bullet: id, flags, position?, velocity?, state?]
[2 bytes] Removed entity count
  [per removed: id]
```

## Message Types

Defined in `common/protocol-native.js`:

```javascript
MessageType = {
  // Connection
  HANDSHAKE: 1,
  HANDSHAKE_ACK: 2,
  PING: 3,
  PONG: 4,

  // Player actions
  PLAYER_UPDATE: 10,      // Position update from client
  PLAYER_SHOOT: 11,       // Bullet creation
  PLAYER_ABILITY: 12,     // Ability usage
  PLAYER_DEATH: 13,
  PLAYER_RESPAWN: 14,
  PLAYER_LEAVE: 15,

  // World state
  WORLD_UPDATE: 20,       // Main game state update
  MAP_DATA: 21,
  CHUNK_DATA: 22,

  // Combat
  BULLET_CREATE: 30,
  BULLET_HIT: 31,
  COLLISION: 32,

  // Items
  ITEM_PICKUP: 40,
  ITEM_DROP: 41,
  INVENTORY_UPDATE: 42,

  // Chat
  CHAT_MESSAGE: 50,

  // Portals
  PORTAL_ENTER: 60,
  PORTAL_EXIT: 61,
}
```

## Client Connection Flow

```
1. Client loads game.html
2. ClientNetworkManager.connect() called
   │
   ├─► WebSocket connects to ws://server:4000
   │   └─► Server assigns clientId
   │   └─► Server sends HANDSHAKE_ACK
   │   └─► Server sends MAP_DATA, initial state
   │
   └─► WebTransport connects to https://quic.eternalconquests.com:10615/game
       └─► Upgrades to binary protocol for world updates
```

## Server Update Loop

```javascript
// Server.js - updateGame() runs at 60Hz

1. Update ability cooldowns
2. Update AI/behavior systems
3. Process collisions (bullets vs players, enemies vs players)
4. Update enemy positions
5. broadcastWorldUpdates()
   │
   ├─► For each client:
   │   ├─► Filter visible entities (UPDATE_RADIUS_TILES = 50)
   │   ├─► Check for death, send PLAYER_DEATH if needed
   │   │
   │   ├─► If WebTransport + binary protocol:
   │   │   ├─► encodeWorldDelta() → binary buffer
   │   │   ├─► sendBinary(buffer)
   │   │   └─► send({ localPlayer: health }) // JSON for health
   │   │
   │   └─► Else (WebSocket only):
   │       └─► send(WORLD_UPDATE, jsonPayload)
```

## Key Files Reference

### Server-Side

| File | Purpose |
|------|---------|
| `Server.js` | Main server, WebSocket handling, game loop |
| `src/network/WebTransportServer.js` | WebTransport/QUIC server |
| `src/network/WebRTCServer.js` | WebRTC (currently disabled) |
| `common/BinaryProtocol.js` | Binary encoding/decoding |
| `common/protocol-native.js` | Message types, constants |
| `src/entities/CollisionManager.js` | Bullet/player collision detection |
| `src/entities/BulletManager.js` | Server-side bullet management |
| `src/entities/EnemyManager.js` | Enemy AI and state |

### Client-Side

| File | Purpose |
|------|---------|
| `public/src/network/ClientNetworkManager.js` | Main network handler |
| `public/src/network/WebTransportManager.js` | WebTransport client |
| `public/src/game/ClientBulletManager.js` | Client-side bullet prediction |
| `public/src/game/ClientEnemyManager.js` | Enemy interpolation |
| `public/src/entities/player.js` | Player entity |

## Client-Side Prediction

The client uses prediction for responsive gameplay:

### Bullet Prediction
```javascript
// ClientNetworkManager.js - sendShoot()
// Bullet created locally BEFORE server confirms

// ClientBulletManager.js - updateBullets()
// When server bullet arrives:
// 1. Check if it's own bullet (normalize ID: "1" vs "entity_1")
// 2. If matching local bullet exists, reconcile (update with server data)
// 3. If no match but own bullet, skip (already have local prediction)
// 4. If not own bullet, add new bullet
```

### Position Interpolation
```javascript
// Bullets and enemies use interpolation for smooth movement
this.targetX = serverX;  // Server position
this.targetY = serverY;
// Actual rendered position lerps toward target
this.x += (this.targetX - this.x) * interpolationSpeed * deltaTime;
```

## Health Synchronization

Health is server-authoritative:

```javascript
// Server.js - broadcastWorldUpdates()
// Local player excluded from binary delta to prevent ghost player
// Health sent separately via JSON:
c.webTransportSession.send(MessageType.WORLD_UPDATE, {
  localPlayer: { health, maxHealth, isDead }
});

// ClientNetworkManager.js - WORLD_UPDATE handler
if (data.localPlayer) {
  window.gameState.character.health = data.localPlayer.health;
  window.gameUI.updateHealth(health, maxHealth);
}
```

## Current Issues & Improvements

### Known Issues
1. **Server Stability** - Server occasionally crashes, watchdog.bat provides auto-restart
2. **ID Format Mismatch** - Server sends `entity_X`, client uses numeric IDs (normalized in code)

### Potential Improvements

1. **Delta Compression**
   - Only send changed properties (already using DeltaFlags)
   - Could add entity-level delta tracking (skip unchanged entities)

2. **Interest Management**
   - Currently: Simple radius-based filtering (50 tiles)
   - Could add: Priority-based updates (closer = more frequent)

3. **Lag Compensation**
   - Server has `LagCompensation.js` with position history
   - Currently uses 200ms rewind window
   - Could improve hit detection accuracy

4. **Connection Recovery**
   - Add reconnection logic for dropped connections
   - Maintain session state during brief disconnects

5. **Bandwidth Optimization**
   - Reduce update frequency for distant entities
   - Use variable-length encoding for IDs
   - Batch multiple small messages

6. **Security**
   - Add rate limiting (partially implemented for bullets)
   - Validate all client inputs server-side
   - Add anti-cheat for movement speed

## Configuration

### Network Constants (`common/constants.js`)
```javascript
NETWORK_SETTINGS = {
  UPDATE_RADIUS_TILES: 50,        // Visibility radius
  MAX_ENTITIES_PER_PACKET: 100,   // Prevent huge packets
  TICK_RATE: 60,                  // Server update rate
}

LAG_COMPENSATION = {
  ENABLED: true,
  MAX_REWIND_MS: 200,             // Max time to rewind
  MIN_RTT_MS: 50,                 // Minimum RTT assumption
}

MOVEMENT_VALIDATION = {
  ENABLED: true,
  MAX_SPEED_TILES_PER_SEC: 7.2,   // Speed hack detection
  TELEPORT_THRESHOLD_TILES: 3,    // Teleport detection
}
```

## Adding New Network Features

### Adding a New Message Type

1. Add to `common/protocol-native.js`:
```javascript
MessageType.MY_NEW_MESSAGE = 70;
```

2. Add server handler in `Server.js`:
```javascript
socket.on('message', (data) => {
  if (packet.type === MessageType.MY_NEW_MESSAGE) {
    handleMyNewMessage(clientId, packet.data);
  }
});
```

3. Add client handler in `ClientNetworkManager.js`:
```javascript
this.handlers[MessageType.MY_NEW_MESSAGE] = (data) => {
  // Handle the message
};
```

4. Add send function:
```javascript
// Client
sendMyNewMessage(data) {
  return this.send(MessageType.MY_NEW_MESSAGE, data);
}

// Server
sendToClient(socket, MessageType.MY_NEW_MESSAGE, data);
```

### Adding Binary Protocol for New Entity Type

1. Add encode/decode in `common/BinaryProtocol.js`:
```javascript
export function encodeMyEntity(writer, entity, flags) {
  writer.writeUint32(getEntityId(entity.id));
  writer.writeUint8(flags);
  if (flags & DeltaFlags.POSITION) {
    writer.writeFloat32(entity.x);
    writer.writeFloat32(entity.y);
  }
  // ... more fields
}
```

2. Include in `encodeWorldDelta()` and `decodeWorldDelta()`

3. Update client to handle in binary message processing
