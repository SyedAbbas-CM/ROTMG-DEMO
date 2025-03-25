// File: server.js

import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { MapManager } from './Managers/MapManager.js';
import { BinaryPacket, MessageType } from './Managers/NetworkManager.js';
import BulletManager from './Managers/BulletManager.js';
import EnemyManager from './Managers/EnemyManager.js';
import CollisionManager from './Managers/CollisionManager.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server });

// Set up middleware
app.use(express.json());
app.use(express.static('public'));

// Create server managers
const mapManager = new MapManager({
  mapStoragePath: path.join(__dirname, 'maps')
});

// Setup map routes
app.get('/api/maps', (req, res) => {
  // Return list of available maps
  const mapsList = Array.from(mapManager.maps.values()).map(map => ({
    id: map.id,
    name: map.name,
    width: map.width,
    height: map.height,
    procedural: map.procedural
  }));
  
  res.json({ maps: mapsList });
});

app.get('/api/maps/:id', (req, res) => {
  // Return specific map info
  const mapId = req.params.id;
  const mapInfo = mapManager.getMapMetadata(mapId);
  
  if (!mapInfo) {
    return res.status(404).json({ error: 'Map not found' });
  }
  
  res.json(mapInfo);
});

app.get('/api/maps/:id/chunk/:x/:y', (req, res) => {
  // Return chunk data
  const mapId = req.params.id;
  const chunkX = parseInt(req.params.x);
  const chunkY = parseInt(req.params.y);
  
  try {
    const chunk = mapManager.getChunkData(mapId, chunkX, chunkY);
    res.json(chunk || { error: 'Chunk not found' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Create initial procedural map
let defaultMapId;
try {
  defaultMapId = mapManager.createProceduralMap({
    width: 256,
    height: 256,
    name: 'Default Map'
  });
} catch (error) {
  console.error("Error creating procedural map:", error);
  defaultMapId = "default";
}

console.log(`Created default map: ${defaultMapId}`);

// Create bullet manager
const bulletManager = new BulletManager(10000);

// Create enemy manager
const enemyManager = new EnemyManager(1000);

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
  let mapMetadata;
  try {
    mapMetadata = mapManager.getMapMetadata(gameState.mapId);
  } catch (error) {
    console.error("Error getting map metadata:", error);
    mapMetadata = {
      width: 256,
      height: 256,
      tileSize: 12,
      chunkSize: 16
    };
  }
  
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
 * Update game state
 */
function updateGame() {
  const now = Date.now();
  const deltaTime = (now - gameState.lastUpdateTime) / 1000; // Convert to seconds
  gameState.lastUpdateTime = now;
  
  // Create a target object using the first connected player
  // If no players, use a default position
  let target = null;
  for (const [id, client] of clients.entries()) {
    target = client.player;
    break;
  }
  
  // Default target if no players
  if (!target) {
    target = { x: 256, y: 256 };
  }
  
  // Update bullets
  bulletManager.update(deltaTime);
  
  // Update enemies with target
  enemyManager.update(deltaTime, bulletManager, target);
  
  // Check for collisions
  collisionManager.checkCollisions();
  
  // Check for enemy spawns
  if (now - gameState.lastEnemySpawnTime > gameState.enemySpawnInterval) {
    gameState.lastEnemySpawnTime = now;
    
    // Spawn 1-3 new enemies if below threshold
    if (enemyManager.getActiveEnemyCount() < 50) {
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
  
  // Clean up resources
  if (collisionManager.cleanup) collisionManager.cleanup();
  if (enemyManager.cleanup) enemyManager.cleanup();
  if (bulletManager.cleanup) bulletManager.cleanup();
  
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

// Export required modules for testing
export {
  app,
  server,
  mapManager,
  bulletManager,
  enemyManager,
  collisionManager
};

/** Send initial game state to a new client
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
  let bulletId;
  try {
    bulletId = bulletManager.addBullet({
      x: data.x,
      y: data.y,
      vx: Math.cos(data.angle) * data.speed,
      vy: Math.sin(data.angle) * data.speed,
      ownerId: clientId,
      damage: data.damage || 10,
      lifetime: data.lifetime || 3.0
    });
  } catch (error) {
    console.error("Error adding bullet:", error);
    return;
  }
  
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
  let result;
  try {
    result = collisionManager.validateCollision({
      bulletId: data.bulletId,
      enemyId: data.enemyId,
      timestamp: data.timestamp,
      clientId
    });
  } catch (error) {
    console.error("Error validating collision:", error);
    return;
  }
  
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
  
  try {
    const chunk = mapManager.getChunkData(client.mapId, data.chunkX, data.chunkY);
    
    if (!chunk) {
      sendToClient(client.socket, MessageType.CHUNK_NOT_FOUND, {
        chunkX: data.chunkX,
        chunkY: data.chunkY
      });
      return;
    }
    
    sendToClient(client.socket, MessageType.CHUNK_DATA, {
      chunkX: data.chunkX,
      chunkY: data.chunkY,
      chunk
    });
  } catch (error) {
    console.error('Error handling chunk request:', error);
    sendToClient(client.socket, MessageType.ERROR, {
      error: 'Failed to load chunk',
      chunkX: data.chunkX,
      chunkY: data.chunkY
    });
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
  if (socket.readyState === 1) { // WebSocket.OPEN
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
    if (client.readyState === 1) { // WebSocket.OPEN
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
    if (client.readyState === 1 && clientId !== excludeClientId) { // WebSocket.OPEN
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
