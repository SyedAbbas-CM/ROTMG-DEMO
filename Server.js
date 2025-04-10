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
  // Set map storage path for the server
  mapManager.mapStoragePath = './maps';
  
  // Create a procedural map with reduced size
  defaultMapId = mapManager.createProceduralMap({
    width: 64,
    height: 64,
    seed: 123456789,
    name: 'Default Map'
  });
  console.log(`Created default map: ${defaultMapId} - This is the map ID that will be sent to clients`);
  
  // Direct save - guaranteed to work with imported fs
  const enableDirectMapSave = false; // Set to true to enable direct map saving
  if (enableDirectMapSave) {
    try {
      // Get map metadata
      const mapData = mapManager.getMapMetadata(defaultMapId);
      
      // Create a simple tile map
      const simpleTileMap = [];
      for (let y = 0; y < mapData.height; y++) {
        const row = [];
        for (let x = 0; x < mapData.width; x++) {
          // Get tile type, default to -1 if not found
          const tile = mapManager.getTile(x, y);
          row.push(tile ? tile.type : -1);
        }
        simpleTileMap.push(row);
      }
      
      // Save the simple map directly to the root directory with custom formatting
      const directFilePath = './direct_simple_map.json';
      
      // Custom formatting: one row per line, no commas between rows
      const formattedJson = "[\n" + 
        simpleTileMap.map(row => "  " + JSON.stringify(row)).join(",\n") + 
        "\n]";
      
      fs.writeFileSync(directFilePath, formattedJson);
      console.log(`DIRECT SAVE: Simple map saved to ${directFilePath} with custom formatting`);
    } catch (directSaveError) {
      console.error("Error with direct save:", directSaveError);
    }
  }
  
  // Save the map to a file for debugging purposes
  (async () => {
    try {
      // Save detailed version using the MapManager methods in the maps folder
      const filename = `server_map_${defaultMapId}.json`;
      const saveResult = await mapManager.saveMap(defaultMapId, filename);
      if (saveResult) {
        console.log(`SERVER MAP SAVED: Map saved to maps/${filename}`);
      } else {
        console.error(`Failed to save map to file`);
      }
      
      // Save simple tile type array version using MapManager method
      const simpleFilename = `simple_map_${defaultMapId}.json`;
      const simpleResult = await mapManager.saveSimpleMap(defaultMapId, simpleFilename);
      if (simpleResult) {
        console.log(`SIMPLE MAP SAVED: Simple map format saved to maps/${simpleFilename}`);
      } else {
        console.error(`Failed to save simple map format`);
      }
    } catch (saveError) {
      console.error("Error saving map to file:", saveError);
    }
  })();
} catch (error) {
  console.error("Error creating procedural map:", error);
  defaultMapId = "default";
}

// Store maps for persistence
const storedMaps = new Map(); // mapId -> map data
storedMaps.set(defaultMapId, mapManager.getMapMetadata(defaultMapId));

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
  enemySpawnInterval: 30000, // 30 seconds between enemy spawns (was 10000)
  lastEnemySpawnTime: Date.now()
};

// Spawn initial enemies - already set to 2 as requested
spawnInitialEnemies(2);

// WebSocket connection handler
wss.on('connection', (socket, req) => {
  // Generate client ID
  const clientId = nextClientId++;
  
  // Set binary type
  socket.binaryType = 'arraybuffer';
  
  // Parse URL to check for map ID in query parameters
  const url = new URL(req.url, `http://${req.headers.host}`);
  const requestedMapId = url.searchParams.get('mapId');
  
  // Determine which map to use (requested or default)
  let useMapId = defaultMapId;
  if (requestedMapId && storedMaps.has(requestedMapId)) {
    console.log(`Client ${clientId} requested existing map: ${requestedMapId}`);
    useMapId = requestedMapId;
  } else if (requestedMapId) {
    console.log(`Client ${clientId} requested unknown map: ${requestedMapId}, using default`);
  } else {
    console.log(`Client ${clientId} connected without map request, using default map: ${defaultMapId}`);
  }
  
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
    mapId: useMapId,  // Use the appropriate map ID
    lastUpdate: Date.now()
  });
  
  console.log(`Client connected: ${clientId}, assigned to map: ${useMapId}`);
  
  // Send handshake acknowledgement
  sendToClient(socket, MessageType.HANDSHAKE_ACK, {
    clientId,
    timestamp: Date.now()
  });
  
  // Send map info
  let mapMetadata;
  try {
    mapMetadata = mapManager.getMapMetadata(useMapId);
    console.log(`Sending map info to client ${clientId} for map ${useMapId}:`, mapMetadata);
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
    mapId: useMapId,  // Use the appropriate map ID
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
  
  // Periodically log connected clients for debugging
  if (now % 5000 < 50) { // Every 5 seconds
    console.log(`Server has ${clients.size} connected clients:`);
    clients.forEach((client, id) => {
      console.log(`- Client ${id}: pos(${client.player.x.toFixed(0)}, ${client.player.y.toFixed(0)}), hp: ${client.player.health}`);
    });
  }
  
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
  const activeBullets = bulletManager.update(deltaTime);
  if (activeBullets > 0) {
    console.log(`Active bullets: ${activeBullets}`);
  }
  
  // Update enemies with target
  const activeEnemies = enemyManager.update(deltaTime, bulletManager, target);
  if (activeEnemies > 0) {
    //console.log(`Active enemies: ${activeEnemies}, targeting position (${target.x.toFixed(2)}, ${target.y.toFixed(2)})`);
  }
  
  // Check for collisions
  const collisions = collisionManager.checkCollisions();
  if (collisions > 0) {
    console.log(`${collisions} collisions detected by server collision system`);
  }
  
  // Check for enemy spawns
  if (now - gameState.lastEnemySpawnTime > gameState.enemySpawnInterval) {
    gameState.lastEnemySpawnTime = now;
    
    // Spawn only 1 new enemy if below threshold (was 1-3)
    if (enemyManager.getActiveEnemyCount() < 10) { // Reduced enemy cap from 50 to 10
      const count = 1; // Fixed to 1 enemy per spawn instead of random 1-3
      
      for (let i = 0; i < count; i++) {
        // Random enemy type (0-4)
        const type = Math.floor(Math.random() * 5);
        
        // Random position (around the map)
        const x = Math.random() * 500 + 50;
        const y = Math.random() * 500 + 50;
        
        // Spawn enemy
        enemyManager.spawnEnemy(type, x, y);
      }
      
      console.log(`Spawned ${count} new enemy`);
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
  
  // Also periodically broadcast the player list separately to ensure clients have the latest player info
  // Only do this every 2 seconds to reduce network traffic
  if (Date.now() % 2000 < 50) {
    console.log(`Broadcasting player list: ${Object.keys(players).length} players`);
    broadcast(MessageType.PLAYER_LIST, players);
  }
}

// Start game update loop
setInterval(updateGame, gameState.updateInterval);

// Add a separate player status logging interval (every 5 seconds)
setInterval(() => {
  const playerCount = clients.size;
  
  if (playerCount > 0) {
    console.log(`[SERVER] ${playerCount} client${playerCount > 1 ? 's' : ''} connected`);
    
    // Log player positions and details for debugging
    const playerPositions = [];
    clients.forEach((client, id) => {
      const player = client.player;
      playerPositions.push(`  - Player ${id}: (${player.x.toFixed(1)}, ${player.y.toFixed(1)}), health: ${player.health}`);
    });
    
    console.log('Player positions:');
    playerPositions.forEach(p => console.log(p));
    
    // Log what's being sent in PLAYER_LIST messages
    const players = {};
    clients.forEach((client, id) => {
      players[id] = client.player;
    });
    
    console.log(`Player list message would contain ${Object.keys(players).length} players: ${Object.keys(players).join(', ')}`);
  } else {
    console.log('[SERVER] No clients connected');
  }
}, 5000);

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
  
  // FIXED: Send just the players object directly
  sendToClient(socket, MessageType.PLAYER_LIST, players);
  
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
        
      case MessageType.MAP_REQUEST:
        // Handle map request by ID
        handleMapRequest(clientId, data);
        break;
        
      case MessageType.PLAYER_LIST_REQUEST:
        // Handle request for player list
        handlePlayerListRequest(clientId);
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
  
  // Store old position for debug logs
  const oldX = player.x;
  const oldY = player.y;
  const oldRotation = player.rotation;
  
  if (data.x !== undefined) player.x = data.x;
  if (data.y !== undefined) player.y = data.y;
  if (data.rotation !== undefined) player.rotation = data.rotation;
  if (data.health !== undefined) player.health = data.health;
  
  player.lastUpdate = Date.now();
  
  // Log player movement if position actually changed
  if (oldX !== player.x || oldY !== player.y || oldRotation !== player.rotation) {
    console.log(`Player ${clientId} moved: (${oldX.toFixed(2)}, ${oldY.toFixed(2)}) → (${player.x.toFixed(2)}, ${player.y.toFixed(2)}), rotation: ${player.rotation.toFixed(2)}`);
  }
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
    
    console.log(`Player ${clientId} fired bullet ${bulletId} at angle ${data.angle.toFixed(2)}, position (${data.x.toFixed(2)}, ${data.y.toFixed(2)})`);
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
    
    console.log(`Player ${clientId} reported collision: bullet ${data.bulletId} hit enemy ${data.enemyId}, valid: ${result.valid}`);
  } catch (error) {
    console.error("Error validating collision:", error);
    return;
  }
  
  // Send result to client
  if (result.valid) {
    console.log(`Valid collision: bullet ${result.bulletId} hit enemy ${result.enemyId}, enemy health: ${result.enemyHealth}, killed: ${result.enemyKilled}`);
    
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
    console.log(`Invalid collision rejected: ${result.reason}`);
    
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
 * Handle map request by ID
 * @param {number} clientId - Client ID
 * @param {Object} data - Request data
 */
function handleMapRequest(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;
  
  console.log(`Client ${clientId} requesting map: ${data.mapId}`);
  
  // Check if the requested map exists
  if (!data.mapId || !storedMaps.has(data.mapId)) {
    console.log(`Map ${data.mapId} not found, using default`);
    // Keep existing map
    return;
  }
  
  // Update client's map ID
  client.mapId = data.mapId;
  console.log(`Updated client ${clientId} to use map ${data.mapId}`);
  
  // Send map info to client
  const mapMetadata = mapManager.getMapMetadata(data.mapId);
  if (!mapMetadata) {
    console.error(`Failed to get metadata for map ${data.mapId}`);
    return;
  }
  
  sendToClient(client.socket, MessageType.MAP_INFO, {
    mapId: data.mapId,
    width: mapMetadata.width,
    height: mapMetadata.height,
    tileSize: mapMetadata.tileSize,
    chunkSize: mapMetadata.chunkSize,
    timestamp: Date.now()
  });
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
    console.log(`Client ${clientId} requesting chunk (${data.chunkX}, ${data.chunkY}) for map ${client.mapId}`);
    const chunk = mapManager.getChunkData(client.mapId, data.chunkX, data.chunkY);
    
    if (!chunk) {
      console.log(`Chunk (${data.chunkX}, ${data.chunkY}) not found for map ${client.mapId}`);
      sendToClient(client.socket, MessageType.CHUNK_NOT_FOUND, {
        chunkX: data.chunkX,
        chunkY: data.chunkY
      });
      return;
    }
    
    console.log(`Sending chunk (${data.chunkX}, ${data.chunkY}) for map ${client.mapId}`);
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

/**
 * Handle request for player list
 * @param {number} clientId - Client ID
 */
function handlePlayerListRequest(clientId) {
    console.log(`Client ${clientId} requested player list`);
    
    const client = clients.get(clientId);
    if (!client) return;
    
    // Get player data
    const players = {};
    clients.forEach((otherClient, id) => {
        players[id] = otherClient.player;
    });
    
    // Send player list directly to the client
    console.log(`Sending player list to client ${clientId}: ${Object.keys(players).length} players`);
    sendToClient(client.socket, MessageType.PLAYER_LIST, players);
}

// Helper function to create a simple tile map format (2D array of tile types)
function createSimpleTileMap(mapManager, mapId) {
  const mapData = mapManager.getMapMetadata(mapId);
  const width = mapData.width;
  const height = mapData.height;
  const chunkSize = mapData.chunkSize;
  
  // Create a 2D array initialized with -1 (unknown)
  const tileMap = Array(height).fill().map(() => Array(width).fill(-1));
  
  // Populate the map with actual tile types from all chunks
  for (const [key, chunk] of mapManager.chunks.entries()) {
    if (!key.startsWith(`${mapId}_`)) continue;
    
    const chunkKey = key.substring(mapId.length + 1);
    const [chunkX, chunkY] = chunkKey.split(',').map(Number);
    const startX = chunkX * chunkSize;
    const startY = chunkY * chunkSize;
    
    // Fill in the tile types from this chunk
    if (chunk.tiles) {
      for (let y = 0; y < chunk.tiles.length; y++) {
        if (!chunk.tiles[y]) continue;
        
        for (let x = 0; x < chunk.tiles[y].length; x++) {
          const globalX = startX + x;
          const globalY = startY + y;
          
          // Skip if outside map bounds
          if (globalX >= width || globalY >= height) continue;
          
          const tile = chunk.tiles[y][x];
          if (tile) {
            tileMap[globalY][globalX] = tile.type;
          }
        }
      }
    }
  }
  
  return tileMap;
}
