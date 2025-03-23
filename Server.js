/**
 * server.js
 * Main server implementation with optimized networking and map generation
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs');
const { ServerMapManager, setupMapRoutes } = require('./Managers/MapManager');
const BinaryPacket = require('./server/network/BinaryPacket');
const MessageType = require('./server/network/MessageType');
const BulletManager = require('./server/managers/BulletManager');
const EnemyManager = require('./server/managers/EnemyManager');
const CollisionManager = require('./server/managers/CollisionManager');

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Set up middleware
app.use(express.json());
app.use(express.static('public'));

// Create server managers
const mapManager = new ServerMapManager({
  mapStoragePath: path.join(__dirname, 'maps')
});

// Set up map routes
setupMapRoutes(app, mapManager);

// Create initial procedural map
const defaultMapId = mapManager.createProceduralMap({
  width: 256,
  height: 256,
  name: 'Default Map'
});

console.log(`Created default map: ${defaultMapId}`);

// Create bullet manager
const bulletManager = new BulletManager(10000);

// Create enemy manager
const enemyManager = new EnemyManager(1000, bulletManager);

// Create collision manager
const collisionManager = new CollisionManager(bulletManager, enemyManager, mapManager);

// WebSocket server state
const clients = new Map(); // clientId -> { socket, player, lastUpdate, mapId }
let nextClientId = 1;

// Game state
const gameState = {
  mapId: defaultMapId,
  lastUpdateTime: Date.now(),
  updateInterval: 1000 / 20, // 20 updates per second
  enemySpawnInterval: 10000, // 10 seconds between enemy spawns
  lastEnemySpawnTime: Date.now()
};

// Spawn initial enemies
spawnInitialEnemies(20); // Spawn 20 initial enemies

// WebSocket connection handler
wss.on('connection', (socket, req) => {
  // Generate client ID
  const clientId = nextClientId++;
  
  // Set binary type
  socket.binaryType = 'arraybuffer';
  
  // Store client info
  clients.set(clientId, {
    socket,
    player: {
      id: clientId,
      x: Math.random() * 100 + 50, // Random spawn between 50-150
      y: Math.random() * 100 + 50,
      rotation: 0,
      health: 100,
      lastUpdate: Date.now()
    },
    mapId: gameState.mapId,
    lastUpdate: Date.now()
  });
  
  console.log(`Client connected: ${clientId}`);
  
  // Send handshake acknowledgement
  sendToClient(socket, MessageType.HANDSHAKE_ACK, {
    clientId,
    timestamp: Date.now()
  });
  
  // Send map info
  const mapMetadata = mapManager.getMapMetadata(gameState.mapId);
  sendToClient(socket, MessageType.MAP_INFO, {
    mapId: gameState.mapId,
    width: mapMetadata.width,
    height: mapMetadata.height,
    tileSize: mapMetadata.tileSize,
    chunkSize: mapMetadata.chunkSize,
    timestamp: Date.now()
  });
  
  // Send initial state (player list, enemy list, bullet list)
  sendInitialState(socket, clientId);
  
  // Set up message handler
  socket.on('message', (message) => {
    handleClientMessage(clientId, message);
  });
  
  // Set up disconnect handler
  socket.on('close', () => {
    handleClientDisconnect(clientId);
  });
});

/**
 * Send initial game state to a new client
 * @param {WebSocket} socket - Client socket
 * @param {number} clientId - Client ID
 */
function sendInitialState(socket, clientId) {
  // Send player list
  const players = {};
  clients.forEach((client, id) => {
    if (id !== clientId) { // Don't include the new player
      players[id] = client.player;
    }
  });
  
  sendToClient(socket, MessageType.PLAYER_LIST, {
    players,
    timestamp: Date.now()
  });
  
  // Send enemy list
  const enemies = enemyManager.getEnemiesData();
  sendToClient(socket, MessageType.ENEMY_LIST, {
    enemies,
    timestamp: Date.now()
  });
  
  // Send bullet list
  const bullets = bulletManager.getBulletsData();
  sendToClient(socket, MessageType.BULLET_LIST, {
    bullets,
    timestamp: Date.now()
  });
}

/**
 * Handle a message from a client
 * @param {number} clientId - Client ID
 * @param {ArrayBuffer} message - Binary message data
 */
function handleClientMessage(clientId, message) {
  try {
    // Decode binary packet
    const packet = BinaryPacket.decode(message);
    const { type, data } = packet;
    
    // Update client's last activity time
    const client = clients.get(clientId);
    if (!client) return;
    
    client.lastUpdate = Date.now();
    
    // Handle message based on type
    switch (type) {
      case MessageType.PING:
        // Reply with pong
        sendToClient(client.socket, MessageType.PONG, {
          time: data.time,
          serverTime: Date.now()
        });
        break;
        
      case MessageType.PLAYER_UPDATE:
        // Update player data
        handlePlayerUpdate(clientId, data);
        break;
        
      case MessageType.BULLET_CREATE:
        // Create a new bullet
        handleBulletCreate(clientId, data);
        break;
        
      case MessageType.COLLISION:
        // Validate and process collision
        handleCollision(clientId, data);
        break;
        
      case MessageType.CHUNK_REQUEST:
        // Send requested chunk
        handleChunkRequest(clientId, data);
        break;
        
      case MessageType.HANDSHAKE:
        // Client info already stored at connection, do nothing
        break;
        
      default:
        console.warn(`Unknown message type from client ${clientId}: ${type}`);
    }
  } catch (error) {
    console.error(`Error handling message from client ${clientId}:`, error);
  }
}

/**
 * Handle player update
 * @param {number} clientId - Client ID
 * @param {Object} data - Update data
 */
function handlePlayerUpdate(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;
  
  // Update player data
  const player = client.player;
  
  if (data.x !== undefined) player.x = data.x;
  if (data.y !== undefined) player.y = data.y;
  if (data.rotation !== undefined) player.rotation = data.rotation;
  if (data.health !== undefined) player.health = data.health;
  
  player.lastUpdate = Date.now();
}

/**
 * Handle bullet creation
 * @param {number} clientId - Client ID
 * @param {Object} data - Bullet data
 */
function handleBulletCreate(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;
  
  // Create bullet
  const bulletId = bulletManager.addBullet({
    x: data.x,
    y: data.y,
    vx: Math.cos(data.angle) * data.speed,
    vy: Math.sin(data.angle) * data.speed,
    ownerId: clientId,
    damage: data.damage || 10,
    lifetime: data.lifetime || 3.0
  });
  
  // Broadcast new bullet to all clients
  broadcast(MessageType.BULLET_CREATE, {
    id: bulletId,
    x: data.x,
    y: data.y,
    angle: data.angle,
    speed: data.speed,
    damage: data.damage || 10,
    lifetime: data.lifetime || 3.0,
    ownerId: clientId,
    timestamp: Date.now()
  });
}

/**
 * Handle collision
 * @param {number} clientId - Client ID
 * @param {Object} data - Collision data
 */
function handleCollision(clientId, data) {
  // Validate collision on server
  const result = collisionManager.validateCollision({
    bulletId: data.bulletId,
    enemyId: data.enemyId,
    timestamp: data.timestamp,
    clientId
  });
  
  // Send result to client
  if (result.valid) {
    // Broadcast valid collision to all clients
    broadcast(MessageType.COLLISION_RESULT, {
      valid: true,
      bulletId: result.bulletId,
      enemyId: result.enemyId,
      damage: result.damage,
      enemyHealth: result.enemyHealth,
      enemyKilled: result.enemyKilled,
      timestamp: Date.now()
    });
  } else {
    // Send rejection only to the reporting client
    const client = clients.get(clientId);
    if (client) {
      sendToClient(client.socket, MessageType.COLLISION_RESULT, {
        valid: false,
        reason: result.reason,
        bulletId: data.bulletId,
        enemyId: data.enemyId,
        timestamp: Date.now()
      });
    }
  }
}

/**
 * Handle map chunk request
 * @param {number} clientId - Client ID
 * @param {Object} data - Request data
 */
function handleChunkRequest(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;
  
  // Get chunk data
  try {
    const chunk = mapManager.getChunk(client.mapId, data.chunkX, data.chunkY);
    
    // Send chunk data to client
    sendToClient(client.socket, MessageType.CHUNK_DATA, {
      chunkX: data.chunkX,
      chunkY: data.chunkY,
      chunk,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error(`Error getting chunk for client ${clientId}:`, error);
  }
}

/**
 * Handle client disconnect
 * @param {number} clientId - Client ID
 */
function handleClientDisconnect(clientId) {
  // Remove client from list
  clients.delete(clientId);
  
  // Broadcast player leave to all clients
  broadcast(MessageType.PLAYER_LEAVE, {
    clientId,
    timestamp: Date.now()
  });
  
  console.log(`Client disconnected: ${clientId}`);
}

/**
 * Send a message to a specific client
 * @param {WebSocket} socket - Client socket
 * @param {number} type - Message type
 * @param {Object} data - Message data
 */
function sendToClient(socket, type, data) {
  if (socket.readyState === WebSocket.OPEN) {
    try {
      // Encode binary packet
      const packet = BinaryPacket.encode(type, data);
      
      // Send packet
      socket.send(packet);
    } catch (error) {
      console.error('Error sending message to client:', error);
    }
  }
}

/**
 * Broadcast a message to all connected clients
 * @param {number} type - Message type
 * @param {Object} data - Message data
 */
function broadcast(type, data) {
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      sendToClient(client, type, data);
    }
  });
}

/**
 * Broadcast a message to all clients except one
 * @param {number} type - Message type
 * @param {Object} data - Message data
 * @param {number} excludeClientId - Client ID to exclude
 */
function broadcastExcept(type, data, excludeClientId) {
  wss.clients.forEach(client => {
    const clientId = getClientIdFromSocket(client);
    if (client.readyState === WebSocket.OPEN && clientId !== excludeClientId) {
      sendToClient(client, type, data);
    }
  });
}

/**
 * Get client ID from socket
 * @param {WebSocket} socket - Client socket
 * @returns {number|null} Client ID or null if not found
 */
function getClientIdFromSocket(socket) {
  for (const [id, client] of clients.entries()) {
    if (client.socket === socket) {
      return id;
    }
  }
  return null;
}

/**
 * Spawn initial enemies
 * @param {number} count - Number of enemies to spawn
 */
function spawnInitialEnemies(count) {
  for (let i = 0; i < count; i++) {
    // Random enemy type (0-4)
    const type = Math.floor(Math.random() * 5);
    
    // Random position (away from spawn)
    const x = Math.random() * 400 + 100;
    const y = Math.random() * 400 + 100;
    
    // Spawn enemy
    enemyManager.spawnEnemy(type, x, y);
  }
  
  console.log(`Spawned ${count} initial enemies`);
}

/**
 * Update game state
 */
function updateGame() {
  const now = Date.now();
  const deltaTime = (now - gameState.lastUpdateTime) / 1000; // Convert to seconds
  gameState.lastUpdateTime = now;
  
  // Update bullets
  bulletManager.update(deltaTime);
  
  // Update enemies
  enemyManager.update(deltaTime);
  
  // Check for enemy spawns
  if (now - gameState.lastEnemySpawnTime > gameState.enemySpawnInterval) {
    gameState.lastEnemySpawnTime = now;
    
    // Spawn 1-3 new enemies if below threshold
    if (enemyManager.enemyCount < 50) {
      const count = Math.floor(Math.random() * 3) + 1;
      
      for (let i = 0; i < count; i++) {
        // Random enemy type (0-4)
        const type = Math.floor(Math.random() * 5);
        
        // Random position (around the map)
        const x = Math.random() * 500 + 50;
        const y = Math.random() * 500 + 50;
        
        // Spawn enemy
        enemyManager.spawnEnemy(type, x, y);
      }
      
      console.log(`Spawned ${count} new enemies`);
    }
  }
  
  // Broadcast world updates
  broadcastWorldUpdates();
}

/**
 * Broadcast world updates (player, enemy, bullet positions)
 */
function broadcastWorldUpdates() {
  // Get player data
  const players = {};
  clients.forEach((client, id) => {
    players[id] = client.player;
  });
  
  // Get enemy data
  const enemies = enemyManager.getEnemiesData();
  
  // Get bullet data
  const bullets = bulletManager.getBulletsData();
  
  // Broadcast world update
  broadcast(MessageType.WORLD_UPDATE, {
    players,
    enemies,
    bullets,
    timestamp: Date.now()
  });
}

// Start game update loop
setInterval(updateGame, gameState.updateInterval);

// Server listen
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

// Export required modules for testing
module.exports = {
  app,
  server,
  mapManager,
  bulletManager,
  enemyManager,
  collisionManager
};