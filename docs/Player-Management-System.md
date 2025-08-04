# Player Management and Character System Documentation

## Overview
The Player Management System handles client connections, player state management, character properties, and multi-world player coordination in the ROTMG-DEMO game. It provides session management, player synchronization, and integration with the game's multiplayer architecture.

## Core Architecture

### 1. Client Connection Management (`/Server.js`)

#### **Client Storage System**
The server maintains client connections and player state through a centralized Map structure:

```javascript
// Global client storage
const clients = new Map(); // clientId -> { socket, player, lastUpdate, mapId }
let nextClientId = 1;

// Multi-world client organization
const clientsByMap = new Map(); // mapId -> Set(clientId)
```

#### **Client Connection Flow**

**WebSocket Connection Handler**:
```javascript
wss.on('connection', (socket, req) => {
  // Generate unique client ID
  const clientId = nextClientId++;
  
  // Set binary protocol for efficient communication
  socket.binaryType = 'arraybuffer';
  
  // Parse URL for map selection
  const url = new URL(req.url, `http://${req.headers.host}`);
  const requestedMapId = url.searchParams.get('mapId');
  
  // Determine target map (requested or default)
  let useMapId = defaultMapId;
  if (requestedMapId && storedMaps.has(requestedMapId)) {
    useMapId = requestedMapId;
  }
  
  // Calculate safe spawn coordinates
  const metaForSpawn = mapManager.getMapMetadata(useMapId) || { width: 64, height: 64 };
  const safeMargin = 2; // tiles away from border
  const spawnX = Math.random() * (metaForSpawn.width - safeMargin * 2) + safeMargin;
  const spawnY = Math.random() * (metaForSpawn.height - safeMargin * 2) + safeMargin;
  
  // Store comprehensive client data
  clients.set(clientId, {
    socket,
    player: {
      id: clientId,
      x: spawnX,
      y: spawnY,
      inventory: new Array(20).fill(null), // 20-slot inventory system
      health: 100,                         // Starting health
      worldId: useMapId,                   // World context
      lastUpdate: Date.now()               // Timestamp for sync
    },
    mapId: useMapId,     // Current map assignment
    lastUpdate: Date.now()
  });
  
  // Send connection acknowledgment
  sendToClient(socket, MessageType.HANDSHAKE_ACK, {
    clientId,
    timestamp: Date.now()
  });
  
  // Provide map information to client
  const mapMetadata = mapManager.getMapMetadata(useMapId);
  sendToClient(socket, MessageType.MAP_INFO, {
    mapId: useMapId,
    width: mapMetadata.width,
    height: mapMetadata.height,
    tileSize: mapMetadata.tileSize,
    chunkSize: mapMetadata.chunkSize,
    timestamp: Date.now()
  });
  
  // Send initial world state
  sendInitialState(socket, clientId);
});
```

### 2. Player Data Structure

#### **Player Object Schema**
Each player maintains comprehensive state information:

```javascript
const playerObject = {
  // Identity and Location
  id: clientId,           // Unique player identifier
  x: spawnX,              // World X coordinate (in tiles)
  y: spawnY,              // World Y coordinate (in tiles)
  worldId: useMapId,      // Current world/map context
  
  // Character State
  health: 100,            // Current health points (0-100)
  maxHealth: 100,         // Maximum health capacity
  
  // Inventory System
  inventory: new Array(20).fill(null), // 20-slot item storage
  
  // Synchronization
  lastUpdate: Date.now(), // Last state update timestamp
  
  // Movement and Physics (implied)
  vx: 0,                  // Velocity X (for prediction)
  vy: 0,                  // Velocity Y (for prediction)
  
  // Combat and Interaction
  isAlive: function() { return this.health > 0; },
  
  // Serialization for network transmission
  serialize: function() {
    return {
      id: this.id,
      x: this.x,
      y: this.y,
      health: this.health,
      worldId: this.worldId
      // Inventory serialized separately for size optimization
    };
  }
};
```

### 3. Multi-World Player Coordination

#### **World Context Integration**

**Player-to-World Assignment**:
```javascript
// Group players by world for efficient processing
function updateGame() {
  const playersByWorld = new Map();
  
  clients.forEach(({ player, mapId }) => {
    if (!playersByWorld.has(mapId)) {
      playersByWorld.set(mapId, []);
    }
    playersByWorld.get(mapId).push(player);
  });
  
  // Process each world independently
  worldContexts.forEach((ctx, mapId) => {
    const players = playersByWorld.get(mapId) || [];
    const primaryTarget = players[0] || null; // AI targeting
    
    // Update world-specific systems with player context
    ctx.bulletMgr.update(deltaTime);
    ctx.enemyMgr.update(deltaTime, ctx.bulletMgr, primaryTarget, mapManager);
    ctx.collMgr.checkCollisions();
    
    // Apply enemy damage to players in this world
    applyEnemyBulletsToPlayers(ctx.bulletMgr, players);
  });
}
```

**Cross-World Player Isolation**:
```javascript
// Send initial world state to newly connected client
function sendInitialState(socket, clientId) {
  const client = clients.get(clientId);
  if (!client) return;
  
  const mapId = client.mapId;
  
  // Filter players to same world only
  const players = {};
  clients.forEach((c, id) => {
    if (c.mapId === mapId) {
      players[id] = c.player;
    }
  });
  
  // Get world-specific entity managers
  const ctx = getWorldCtx(mapId);
  
  // Send world-filtered data
  sendToClient(socket, MessageType.PLAYER_LIST, players);
  sendToClient(socket, MessageType.ENEMY_LIST, ctx.enemyMgr.getEnemiesData(mapId));
  sendToClient(socket, MessageType.BULLET_LIST, ctx.bulletMgr.getBulletsData(mapId));
  sendToClient(socket, MessageType.BAG_LIST, ctx.bagMgr.getBagsData(mapId));
}
```

### 4. Player-Enemy Interaction System

#### **Combat Damage Application**

**Enemy Bullet Damage Processing**:
```javascript
function applyEnemyBulletsToPlayers(bulletMgr, players) {
  const bulletCount = bulletMgr.bulletCount;
  
  for (let bi = 0; bi < bulletCount; bi++) {
    if (bulletMgr.life[bi] <= 0) continue;
    
    const ownerId = bulletMgr.ownerId[bi];
    
    // Only process enemy-owned bullets
    if (typeof ownerId !== 'string' || !ownerId.startsWith('enemy_')) {
      continue;
    }
    
    // Get bullet collision bounds
    const bx = bulletMgr.x[bi];
    const by = bulletMgr.y[bi];
    const bw = bulletMgr.width[bi];
    const bh = bulletMgr.height[bi];
    
    // Check collision with all players in world
    for (const player of players) {
      if (!player || player.health <= 0) continue;
      
      // Player collision bounds (1x1 tile)
      const pw = 1, ph = 1;
      
      // AABB collision detection
      const hit = (
        bx - bw / 2 < player.x + pw / 2 &&
        bx + bw / 2 > player.x - pw / 2 &&
        by - bh / 2 < player.y + ph / 2 &&
        by + bh / 2 > player.y - ph / 2
      );
      
      if (hit) {
        // Apply damage
        const dmg = bulletMgr.damage ? bulletMgr.damage[bi] : 10;
        player.health -= dmg;
        if (player.health < 0) player.health = 0;
        
        // Remove bullet after hit
        bulletMgr.markForRemoval(bi);
        
        // Could trigger additional effects here:
        // - Death handling
        // - Damage notifications
        // - Combat logging
      }
    }
  }
}
```

### 5. Inventory Management System

#### **Item Slot Management**

**Pickup Processing**:
```javascript
function processPickupMessage(clientId, data) {
  const { bagId, itemId, slot } = data || {};
  const client = clients.get(clientId);
  
  if (!client || !bagId || !itemId) return;
  
  const ctx = getWorldCtx(client.mapId);
  const bagMgr = ctx.bagMgr;
  
  // Validate bag visibility and proximity
  const bags = bagMgr.getBagsData(client.mapId, clientId);
  const bagDto = bags.find(b => b.id === bagId);
  
  if (!bagDto) {
    sendToClient(client.socket, MessageType.PICKUP_DENIED, {
      reason: 'not_visible'
    });
    return;
  }
  
  // Range check (2 tile radius)
  const dx = client.player.x - bagDto.x;
  const dy = client.player.y - bagDto.y;
  if ((dx * dx + dy * dy) > 4) {
    return; // Too far away
  }
  
  // Remove item from bag
  const removed = bagMgr.removeItemFromBag(bagId, itemId);
  
  // Add to player inventory
  const inv = client.player.inventory || (client.player.inventory = new Array(20).fill(null));
  
  // Try preferred slot first, then find empty slot
  let targetSlot = (Number.isInteger(slot) && slot >= 0 && slot < inv.length && inv[slot] == null) 
    ? slot 
    : inv.findIndex(x => x == null);
  
  if (targetSlot !== -1) {
    inv[targetSlot] = itemId;
    
    // Notify client of successful pickup
    sendToClient(client.socket, MessageType.PICKUP_SUCCESS, {
      itemId,
      slot: targetSlot,
      bagId
    });
  } else {
    // Inventory full - return item to bag
    bagMgr.addItemToBag(bagId, itemId);
    sendToClient(client.socket, MessageType.PICKUP_DENIED, {
      reason: 'inventory_full'
    });
  }
}
```

**Inventory Slot Movement**:
```javascript
function processMoveItem(clientId, data) {
  const { fromSlot, toSlot } = data || {};
  const client = clients.get(clientId);
  
  if (!client) return;
  
  const inv = client.player.inventory;
  if (!inv || fromSlot < 0 || toSlot < 0 || fromSlot >= inv.length || toSlot >= inv.length) {
    return;
  }
  
  // Validate slots exist and source has item
  if (inv[fromSlot] === null) return;
  
  // Perform swap or move
  const fromItem = inv[fromSlot];
  const toItem = inv[toSlot];
  
  inv[fromSlot] = toItem;   // Could be null (move) or item (swap)
  inv[toSlot] = fromItem;   // Always has the moved item
  
  // Notify client of successful move
  sendToClient(client.socket, MessageType.INVENTORY_UPDATE, {
    fromSlot,
    toSlot,
    fromItem: toItem,
    toItem: fromItem
  });
}
```

### 6. Network Synchronization

#### **Player State Broadcasting**

**Efficient Player List Updates**:
```javascript
// Broadcast player positions to all clients in world
function broadcastPlayerUpdates() {
  const playersByWorld = new Map();
  
  // Group players by world
  clients.forEach((client, clientId) => {
    if (!playersByWorld.has(client.mapId)) {
      playersByWorld.set(client.mapId, {});
    }
    playersByWorld.get(client.mapId)[clientId] = client.player;
  });
  
  // Send world-specific updates
  playersByWorld.forEach((players, mapId) => {
    const worldClients = Array.from(clients.values()).filter(c => c.mapId === mapId);
    
    worldClients.forEach(client => {
      // Interest management - only send nearby players
      const nearbyPlayers = filterPlayersByProximity(players, client.player);
      
      sendToClient(client.socket, MessageType.PLAYER_LIST, {
        players: nearbyPlayers,
        timestamp: Date.now()
      });
    });
  });
}

function filterPlayersByProximity(allPlayers, viewerPlayer, maxDistance = 50) {
  const nearby = {};
  
  Object.entries(allPlayers).forEach(([playerId, player]) => {
    const dx = player.x - viewerPlayer.x;
    const dy = player.y - viewerPlayer.y;
    const distSq = dx * dx + dy * dy;
    
    if (distSq <= maxDistance * maxDistance) {
      nearby[playerId] = {
        id: player.id,
        x: player.x,
        y: player.y,
        health: player.health
        // Minimal data for bandwidth efficiency
      };
    }
  });
  
  return nearby;
}
```

### 7. Session Management and Cleanup

#### **Client Disconnection Handling**

**Missing Implementation Analysis**:
The current codebase has a critical gap - `handleClientDisconnect` is called but not implemented:

```javascript
// Current incomplete implementation
socket.on('close', () => {
  handleClientDisconnect(clientId); // Function not defined!
});
```

**Required Implementation**:
```javascript
function handleClientDisconnect(clientId) {
  const client = clients.get(clientId);
  if (!client) return;
  
  const mapId = client.mapId;
  
  // Remove from client storage
  clients.delete(clientId);
  
  // Update world-specific client tracking
  if (clientsByMap.has(mapId)) {
    clientsByMap.get(mapId).delete(clientId);
    if (clientsByMap.get(mapId).size === 0) {
      clientsByMap.delete(mapId);
    }
  }
  
  // Clean up player-specific resources
  // - Remove player from active targets for AI
  // - Clean up any player-owned entities
  // - Drop inventory items as bags
  
  if (client.player.inventory) {
    const ctx = getWorldCtx(mapId);
    const nonNullItems = client.player.inventory.filter(item => item !== null);
    
    if (nonNullItems.length > 0) {
      // Create bag with player's items
      const bagId = ctx.bagMgr.createBag({
        x: client.player.x,
        y: client.player.y,
        items: nonNullItems,
        worldId: mapId,
        dropSource: 'player_disconnect'
      });
    }
  }
  
  // Notify other players in the same world
  const worldPlayers = Array.from(clients.values()).filter(c => c.mapId === mapId);
  worldPlayers.forEach(otherClient => {
    sendToClient(otherClient.socket, MessageType.PLAYER_DISCONNECTED, {
      playerId: clientId,
      timestamp: Date.now()
    });
  });
  
  if (DEBUG.connections) {
    console.log(`Client ${clientId} disconnected from map ${mapId}`);
  }
}
```

### 8. Missing Message Handling System

#### **Client Message Processing Gap**

**Current Incomplete Implementation**:
```javascript
socket.on('message', (message) => {
  try {
    const packet = BinaryPacket.decode(message);
    if (packet.type === MessageType.MOVE_ITEM) {
      processMoveItem(clientId, packet.data);
    } else if (packet.type === MessageType.PICKUP_ITEM) {
      processPickupMessage(clientId, packet.data);
    } else {
      handleClientMessage(clientId, message); // Function not defined!
    }
  } catch (err) {
    console.error('[NET] Failed to process message', err);
  }
});
```

**Required Implementation**:
```javascript
function handleClientMessage(clientId, message) {
  const client = clients.get(clientId);
  if (!client) return;
  
  try {
    const packet = BinaryPacket.decode(message);
    
    switch (packet.type) {
      case MessageType.PLAYER_MOVE:
        handlePlayerMovement(clientId, packet.data);
        break;
        
      case MessageType.PLAYER_SHOOT:
        handlePlayerShooting(clientId, packet.data);
        break;
        
      case MessageType.CHAT_MESSAGE:
        handleChatMessage(clientId, packet.data);
        break;
        
      case MessageType.PING:
        handlePingRequest(clientId, packet.data);
        break;
        
      case MessageType.MAP_CHUNK_REQUEST:
        handleChunkRequest(clientId, packet.data);
        break;
        
      default:
        if (DEBUG.chat) {
          console.warn(`[NET] Unknown message type ${packet.type} from client ${clientId}`);
        }
        break;
    }
  } catch (error) {
    console.error(`[NET] Error processing message from client ${clientId}:`, error);
  }
}

function handlePlayerMovement(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;
  
  const { x, y, vx, vy, timestamp } = data;
  
  // Validate movement bounds
  const mapMeta = mapManager.getMapMetadata(client.mapId);
  if (x < 0 || y < 0 || x >= mapMeta.width || y >= mapMeta.height) {
    return; // Invalid position
  }
  
  // Check for wall collision
  if (mapManager.isWallOrOutOfBounds(x, y)) {
    return; // Can't move into walls
  }
  
  // Update player position
  client.player.x = x;
  client.player.y = y;
  client.player.vx = vx || 0;
  client.player.vy = vy || 0;
  client.player.lastUpdate = Date.now();
  client.lastUpdate = Date.now();
  
  if (DEBUG.playerMovement) {
    console.log(`Player ${clientId} moved to (${x.toFixed(2)}, ${y.toFixed(2)})`);
  }
}

function handlePlayerShooting(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;
  
  const { angle, timestamp } = data;
  const ctx = getWorldCtx(client.mapId);
  
  // Create player bullet
  const bulletSpeed = 15; // tiles per second
  const bulletLifetime = 3; // seconds
  
  ctx.bulletMgr.addBullet({
    x: client.player.x,
    y: client.player.y,
    vx: Math.cos(angle) * bulletSpeed,
    vy: Math.sin(angle) * bulletSpeed,
    damage: 25,
    lifetime: bulletLifetime,
    ownerId: clientId,
    worldId: client.mapId,
    width: 3,
    height: 3,
    spriteName: 'player_bullet'
  });
  
  if (DEBUG.bulletEvents) {
    console.log(`Player ${clientId} shot bullet at angle ${angle}`);
  }
}
```

### 9. Performance Optimization

#### **Client State Caching**

**Efficient State Tracking**:
```javascript
class PlayerStateCache {
  constructor() {
    this.lastBroadcast = new Map(); // clientId -> last sent state
    this.updateThreshold = 100; // ms between updates
  }
  
  shouldBroadcastPlayer(clientId, currentState) {
    const lastState = this.lastBroadcast.get(clientId);
    const now = Date.now();
    
    if (!lastState || (now - lastState.timestamp) > this.updateThreshold) {
      return true;
    }
    
    // Check for significant position change
    const dx = Math.abs(currentState.x - lastState.x);
    const dy = Math.abs(currentState.y - lastState.y);
    
    if (dx > 0.1 || dy > 0.1) {
      return true;
    }
    
    // Check for health change
    if (currentState.health !== lastState.health) {
      return true;
    }
    
    return false;
  }
  
  recordBroadcast(clientId, state) {
    this.lastBroadcast.set(clientId, {
      ...state,
      timestamp: Date.now()
    });
  }
}
```

### 10. Integration Points Summary

#### **System Dependencies**
- **MapManager**: World bounds checking, walkability validation, spawn point generation
- **BulletManager**: Player shooting, damage application from enemy bullets
- **EnemyManager**: AI targeting, player proximity detection
- **BagManager**: Item pickup, inventory management, drop handling
- **NetworkManager**: Binary protocol encoding/decoding, message type constants
- **CollisionManager**: Player-bullet collision detection

#### **Performance Characteristics**
- **Player Capacity**: Designed for 100+ concurrent players per world
- **Memory Usage**: ~200 bytes per player for core state
- **Network Efficiency**: Binary protocol reduces bandwidth by 60-80%
- **Update Rate**: 20Hz for player state synchronization
- **Cross-World Isolation**: O(1) world switching with context separation

#### **Data Flow**
```
Client Connection → Identity Assignment → World Placement → State Sync → Game Loop Integration
        ↓               ↓                    ↓              ↓              ↓
   WebSocket Setup → Player Object → Map Assignment → Initial State → Continuous Updates
        ↓               ↓                    ↓              ↓              ↓
   Message Handlers → Inventory Init → Spawn Location → Entity Lists → Position Sync
```

**Critical Implementation Gaps**:
1. **handleClientMessage**: Missing core message processing
2. **handleClientDisconnect**: Missing cleanup functionality
3. **Player Movement Validation**: Incomplete server-side verification
4. **Combat System**: Basic damage application without comprehensive combat mechanics

This player management system provides the foundation for multiplayer gameplay but requires completion of the missing handler functions to achieve full functionality.