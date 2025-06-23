// File: server.js

import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { MapManager } from './src/MapManager.js';
import { BinaryPacket, MessageType } from './src/NetworkManager.js';
import BulletManager from './src/BulletManager.js';
import EnemyManager from './src/EnemyManager.js';
import CollisionManager from './src/CollisionManager.js';
// Import BehaviorSystem
import BehaviorSystem from './src/BehaviorSystem.js';
import { entityDatabase } from './src/assets/EntityDatabase.js';

// Debug flags to control logging
const DEBUG = {
  // Keep everything silent by default – we'll re-enable specific areas when
  // we actually need them for diagnostics.
  mapCreation: false,
  connections: true,
  enemySpawns: false,
  collisions: false,
  playerPositions: false,
  activeCounts: false,
  // Keep chunkRequests on so we can confirm the request-throttling fix works
  chunkRequests: false,
  chat: false,
  playerMovement: false,
  bulletEvents: false
};

// Expose debug flags globally so helper classes can reference them
globalThis.DEBUG = DEBUG;

// -----------------------------------------------------------------------------
// Feature flags
// -----------------------------------------------------------------------------
// Toggle loading of a hand-crafted (editor-exported) map and automatic portal
// insertion that links the procedural default map to that fixed map.  Set this
// to `true` when you actively want to test the multi-map / portal flow.
// When `false` (the default) the server will create only the procedural map
// exactly as it did originally – giving us the "classic" single-world session
// until we are ready to test portals again.
const ENABLE_FIXED_MAP_LOADING = false;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Create Express app and HTTP server
const app = express();
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server });

// Prevent unhandled 'error' events (e.g. EADDRINUSE) from crashing the process
wss.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    // HTTP server retry logic will handle incrementing the port. Just swallow here.
    console.warn('[WSS] Underlying HTTP server port busy – waiting for retry logic.');
    return;
  }
  console.error('[WSS] Unhandled error:', err);
});

// Set up middleware
app.use(express.json());

// Root route – main menu
app.get('/', (req,res)=>{
  res.sendFile(path.join(__dirname,'public','menu.html'));
});

// ---------------- Asset Browser API ----------------
// These routes provide the Sprite Editor and other tools with lists of accessible images and
// atlas JSON files from the /public/assets directory tree. They were previously only present
// in Server.js (capital S) so running `node server.js` missed them – causing 404 errors in the
// browser. Duplicated here so the lowercase entry-point serves them too.

const imagesDirBase = path.join(__dirname, 'public', 'assets', 'images');
const atlasesDirBase = path.join(__dirname, 'public', 'assets', 'atlases');

// GET /api/assets/images – flat list of image paths relative to public/
app.get('/api/assets/images', (req, res) => {
  const images = [];
  const walk = (dir, base = '') => {
    fs.readdirSync(dir, { withFileTypes: true }).forEach((ent) => {
      const full = path.join(dir, ent.name);
      const rel = path.posix.join(base, ent.name);
      if (ent.isDirectory()) return walk(full, rel);
      if (/\.(png|jpe?g|gif)$/i.test(ent.name)) {
        images.push('assets/images/' + rel);
      }
    });
  };
  try {
    walk(imagesDirBase);
    res.json({ images });
  } catch (err) {
    console.error('[ASSETS] Error generating image list', err);
    res.status(500).json({ error: 'Failed to list images' });
  }
});

// GET /api/assets/images/tree – nested folder tree structure
app.get('/api/assets/images/tree', (req, res) => {
  const buildNode = (dir) => {
    const node = { name: path.basename(dir), type: 'folder', children: [] };
    fs.readdirSync(dir, { withFileTypes: true }).forEach((ent) => {
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        node.children.push(buildNode(full));
      } else if (/\.(png|jpe?g|gif)$/i.test(ent.name)) {
        node.children.push({
          name: ent.name,
          type: 'image',
          path: 'assets/images/' + path.relative(imagesDirBase, full).replace(/\\/g, '/'),
        });
      }
    });
    return node;
  };
  try {
    res.json(buildNode(imagesDirBase));
  } catch (err) {
    console.error('[ASSETS] Error building image tree', err);
    res.status(500).json({ error: 'Failed to build tree' });
  }
});

// GET /api/assets/atlases – list of atlas JSON files
app.get('/api/assets/atlases', (req, res) => {
  try {
    const atlases = fs
      .readdirSync(atlasesDirBase)
      .filter((f) => f.endsWith('.json'))
      .map((f) => '/assets/atlases/' + f);
    res.json({ atlases });
  } catch (err) {
    console.error('[ASSETS] Error listing atlases', err);
    res.status(500).json({ error: 'Failed to list atlases' });
  }
});

// GET /api/assets/atlas/:file – fetch a single atlas JSON by filename
app.get('/api/assets/atlas/:file', (req, res) => {
  const filename = req.params.file;
  // Allow only simple filenames like "chars2.json" – prevents path traversal
  if (!/^[\w-]+\.json$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  const atlasPath = path.join(atlasesDirBase, filename);
  if (!fs.existsSync(atlasPath)) {
    return res.status(404).json({ error: 'Atlas not found' });
  }
  try {
    const data = JSON.parse(fs.readFileSync(atlasPath, 'utf8'));
    res.json(data);
  } catch (err) {
    console.error('[ASSETS] Error reading atlas', err);
    res.status(500).json({ error: 'Failed to read atlas' });
  }
});

// POST /api/assets/atlases/save – persist atlas JSON sent from the editor
app.post('/api/assets/atlases/save', (req, res) => {
  const { filename, data } = req.body || {};
  if (!filename || !data) {
    return res.status(400).json({ error: 'filename and data required' });
  }
  // Simple filename sanitisation – disallow path traversal, require .json extension
  if (!/^[\w-]+\.json$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  const atlasPath = path.join(atlasesDirBase, filename);
  try {
    fs.writeFileSync(atlasPath, JSON.stringify(data, null, 2));
    res.json({ success: true, path: '/assets/atlases/' + filename });
  } catch (err) {
    console.error('[ASSETS] Error saving atlas', err);
    res.status(500).json({ error: 'Failed to save atlas' });
  }
});

// POST /api/assets/images/save – save a base64-encoded PNG image to public/assets/images
app.post('/api/assets/images/save', (req, res) => {
  const { path: relPath, data } = req.body || {};
  if (!relPath || !data || !data.startsWith('data:image/png;base64,')) {
    return res.status(400).json({ error: 'path and base64 data required' });
  }
  // Prevent path traversal, force .png only
  if (relPath.includes('..') || !relPath.toLowerCase().endsWith('.png')) {
    return res.status(400).json({ error: 'Invalid path' });
  }
  const abs = path.join(__dirname, 'public', relPath);
  try {
    const pngBuf = Buffer.from(data.split(',')[1], 'base64');
    fs.writeFileSync(abs, pngBuf);
    res.json({ success: true });
  } catch (err) {
    console.error('[ASSETS] Error saving image', err);
    res.status(500).json({ error: 'Failed to save image' });
  }
});

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

// After map routes definitions
// --- Asset listing routes for editor -----------------
// List PNG images under public/assets/images (recursive)
app.get('/api/assets/images', (req, res) => {
  const imagesDir = path.join(__dirname, 'public', 'assets', 'images');
  const images = [];
  const walk = (dir, base='') => {
    fs.readdirSync(dir, { withFileTypes: true }).forEach(ent => {
      const full = path.join(dir, ent.name);
      const rel = path.join(base, ent.name);
      if (ent.isDirectory()) return walk(full, rel);
      if (/\.(png|jpg|jpeg|gif)$/i.test(ent.name)) images.push("assets/images/" + rel.replace(/\\/g,'/'));
    });
  };
  try {
    walk(imagesDir);
    res.json({ images });
  } catch (err) {
    console.error('Error listing images', err);
    res.status(500).json({ error: 'Failed to list images' });
  }
});

// List atlas JSON files
app.get('/api/assets/atlases', (req, res) => {
  const atlasesDir = path.join(__dirname, 'public', 'assets', 'atlases');
  try {
    const files = fs.readdirSync(atlasesDir).filter(f => f.endsWith('.json'));
    res.json({ atlases: files.map(f => '/assets/atlases/' + f) });
  } catch (err) {
    console.error('Error listing atlases', err);
    res.status(500).json({ error: 'Failed to list atlases' });
  }
});

// Return directory tree of assets/images for hierarchical UI
app.get('/api/assets/images/tree', (req, res) => {
  const imagesDir = path.join(__dirname, 'public', 'assets', 'images');

  function walkDir(dir) {
    const result = { name: path.basename(dir), type: 'folder', children: [] };
    fs.readdirSync(dir, { withFileTypes: true }).forEach(ent => {
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) {
        result.children.push(walkDir(full));
      } else if (/\.(png|jpg|jpeg|gif)$/i.test(ent.name)) {
        result.children.push({
          name: ent.name,
          type: 'image',
          path: 'assets/images/' + path.relative(imagesDir, full).replace(/\\/g, '/')
        });
      }
    });
    return result;
  }

  try {
    const tree = walkDir(imagesDir);
    res.json(tree);
  } catch (err) {
    console.error('Error building image tree', err);
    res.status(500).json({ error: 'Failed to build tree' });
  }
});

// Save an atlas JSON sent from the editor
app.post('/api/assets/atlases/save', (req, res) => {
  const { filename, data } = req.body;
  if (!filename || !data) {
    return res.status(400).json({ error: 'filename and data required' });
  }
  // Sanitize filename (no path traversal)
  if (!/^[a-zA-Z0-9_-]+\.json$/.test(filename)) {
    return res.status(400).json({ error: 'Invalid filename' });
  }
  const atlasPath = path.join(__dirname, 'public', 'assets', 'atlases', filename);
  try {
    fs.writeFileSync(atlasPath, JSON.stringify(data, null, 2));
    res.json({ success: true, path: '/assets/atlases/' + filename });
  } catch (err) {
    console.error('Error saving atlas', err);
    res.status(500).json({ error: 'Failed to save atlas' });
  }
});

// Create initial procedural map
let defaultMapId;
let fixedMapId; // map created from editor file
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
  if (DEBUG.mapCreation) {
    console.log(`Created default map: ${defaultMapId} - This is the map ID that will be sent to clients`);
  }
  
  // (we will move async load after storedMaps declaration below)
  // -------------------------------------------------------------------
} catch (error) {
  console.error("Error creating procedural map:", error);
  defaultMapId = "default";
}

// Store maps for persistence
const storedMaps = new Map(); // mapId -> map data
storedMaps.set(defaultMapId, mapManager.getMapMetadata(defaultMapId));

console.log(`Created default map: ${defaultMapId}`);

// -----------------------------------------------------------
// Load editor-exported map and register portal inside default map
// -----------------------------------------------------------
if (ENABLE_FIXED_MAP_LOADING) {
  (async () => {
    try {
      // Use absolute path so MapManager treats it as a filesystem read rather than a URL.
      const fixedMapPath = path.join(__dirname, 'public', 'maps', 'test.json');
      fixedMapId = await mapManager.loadFixedMap(fixedMapPath);
      storedMaps.set(fixedMapId, mapManager.getMapMetadata(fixedMapId));

      // Spawn enemies defined inside that map right away
      spawnMapEnemies(fixedMapId);

      // Place a portal in centre of procedural map linking to this fixed map
      const defMeta = mapManager.getMapMetadata(defaultMapId);
      if (defMeta) {
        // Ensure objects array exists
        if (!defMeta.objects) defMeta.objects = [];

        const portalObj = {
          id: 'portal_to_' + fixedMapId,
          type: 'portal',
          sprite: 'portal',
          x: 5,
          y: 5,
          destMap: fixedMapId
        };
        defMeta.objects.push(portalObj);
        console.log(`[PORTAL] Spawned portal from ${defaultMapId} to ${fixedMapId} at (${portalObj.x},${portalObj.y})`);
      }
    } catch (err) {
      console.error('[PORTAL] Failed to load fixed map or set up portal', err);
    }
  })();
}

// Create bullet manager
const bulletManager = new BulletManager(10000);
console.log('bulletManager is', typeof bulletManager, bulletManager?.addBullet ? 'OK' : 'NO addBullet'); // PROBE: Check bulletManager validity

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
  updateInterval: 1000 / 30, // 30 updates per second (was 20)
  enemySpawnInterval: 30000, // 30 seconds between enemy spawns (was 10000)
  lastEnemySpawnTime: Date.now()
};

// Spawn initial enemies for the game world
spawnInitialEnemies(2);

// also spawn any enemies defined inside the procedural map metadata (none by default)
spawnMapEnemies(gameState.mapId);

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
    if (DEBUG.connections) {
      console.log(`Client ${clientId} requested existing map: ${requestedMapId}`);
    }
    useMapId = requestedMapId;
  } else if (requestedMapId) {
    if (DEBUG.connections) {
      console.log(`Client ${clientId} requested unknown map: ${requestedMapId}, using default`);
    }
  } else if (DEBUG.connections) {
    console.log(`Client ${clientId} connected without map request, using default map: ${defaultMapId}`);
  }
  
  // Determine safe spawn coordinates within map bounds (avoid edges)
  const metaForSpawn = mapManager.getMapMetadata(useMapId) || { width: 64, height: 64 };
  const safeMargin = 2; // tiles away from the border
  const spawnX = Math.random() * (metaForSpawn.width  - safeMargin * 2) + safeMargin;
  const spawnY = Math.random() * (metaForSpawn.height - safeMargin * 2) + safeMargin;
  
  // Store client info
  clients.set(clientId, {
    socket,
    player: {
      id: clientId,
      x: spawnX,
      y: spawnY,
      rotation: 0,
      health: 100,
      lastUpdate: Date.now()
    },
    mapId: useMapId,  // Use the appropriate map ID
    lastUpdate: Date.now()
  });
  
  if (DEBUG.connections) {
    console.log(`Client connected: ${clientId}, assigned to map: ${useMapId}`);
  }
  
  // Send handshake acknowledgement
  sendToClient(socket, MessageType.HANDSHAKE_ACK, {
    clientId,
    timestamp: Date.now()
  });
  
  // Send map info
  let mapMetadata;
  try {
    mapMetadata = mapManager.getMapMetadata(useMapId);
    if (DEBUG.connections) {
      console.log(`Sending map info to client ${clientId} for map ${useMapId}:`, mapMetadata);
    }
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
  if (DEBUG.playerPositions && now % 30000 < 50) { // Every 30 seconds instead of 5
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
  if (DEBUG.activeCounts && activeBullets > 0 && now % 5000 < 50) { // Only log every 5 seconds
    console.log(`Active bullets: ${activeBullets}`);
  }
  
  // Update enemies with target
  const activeEnemies = enemyManager.update(deltaTime, bulletManager, target, mapManager);
  if (DEBUG.activeCounts && activeEnemies > 0 && now % 5000 < 50) { // Only log every 5 seconds
    console.log(`Active enemies: ${activeEnemies}, targeting position (${target.x.toFixed(2)}, ${target.y.toFixed(2)})`);
  }
  
  // Check for collisions
  const collisions = collisionManager.checkCollisions();
  if (DEBUG.collisions && collisions > 0) {
    console.log(`${collisions} collisions detected by server collision system`);
  }
  
  /* ---------------- PLAYER HIT DETECTION ---------------- */
  // Detect enemy bullets hitting players (simple AABB in tile units)
  const bulletCount = bulletManager.bulletCount;
  for (let bi = 0; bi < bulletCount; bi++) {
    if (bulletManager.life[bi] <= 0) continue; // dead bullet

    const ownerId = bulletManager.ownerId[bi];
    // Only process bullets whose owner is an enemy (starts with "enemy_")
    if (typeof ownerId !== 'string' || !ownerId.startsWith('enemy_')) {
      continue;
    }

    const bx = bulletManager.x[bi];
    const by = bulletManager.y[bi];
    const bw = bulletManager.width[bi];
    const bh = bulletManager.height[bi];

    // Iterate over players
    clients.forEach((client, pid) => {
      const player = client.player;
      if (!player || player.health <= 0) return;

      const pw = 1; // player width in tile units (assume 1×1)
      const ph = 1;

      // Treat positions as centres rather than top-left corners
      const bMinX = bx - bw / 2;
      const bMaxX = bx + bw / 2;
      const bMinY = by - bh / 2;
      const bMaxY = by + bh / 2;

      const pMinX = player.x - pw / 2;
      const pMaxX = player.x + pw / 2;
      const pMinY = player.y - ph / 2;
      const pMaxY = player.y + ph / 2;

      const hit = (
        bMinX < pMaxX &&
        bMaxX > pMinX &&
        bMinY < pMaxY &&
        bMaxY > pMinY
      );

      if (hit) {
        // Apply damage
        const dmg = bulletManager.damage ? bulletManager.damage[bi] : 10;
        player.health -= dmg;
        if (player.health < 0) player.health = 0;

        // Remove bullet
        bulletManager.markForRemoval(bi);

        if (DEBUG.collisions) {
          console.log(`Player ${pid} hit by ${ownerId} bullet (${bulletManager.id[bi]}), dmg ${dmg}, hp ${player.health}`);
        }

        // Optionally broadcast immediate hit packet (future)
      }
    });
  }
  
  // ---------------- PORTAL HANDLING ----------------
  handlePortals();
  
  // Check for enemy spawns
  if (now - gameState.lastEnemySpawnTime > gameState.enemySpawnInterval) {
    gameState.lastEnemySpawnTime = now;
    
    // Spawn only 1 new enemy if below threshold (was 1-3)
    if (enemyManager.getActiveEnemyCount() < 10) { // Reduced enemy cap from 50 to 10
      const count = 1; // Fixed to 1 enemy per spawn instead of random 1-3
      
      // Get a random connected player to spawn near
      const playerClients = Array.from(clients.values());
      if (playerClients.length > 0) {
        const randomPlayer = playerClients[Math.floor(Math.random() * playerClients.length)];
        
        for (let i = 0; i < count; i++) {
          // Red Demon entity ID
          const type = 'red_demon';
          
          // Spawn near the selected player (within 100-200 units)
          const distance = 100 + Math.random() * 100; // Distance from player: 100-200 units
          const angle = Math.random() * Math.PI * 2; // Random angle
          
          let x = randomPlayer.player.x + Math.cos(angle) * distance;
          let y = randomPlayer.player.y + Math.sin(angle) * distance;
          
          // Clamp spawn inside world bounds
          if (mapManager) {
            x = Math.max(1, Math.min(mapManager.width - 1, x));
            y = Math.max(1, Math.min(mapManager.height - 1, y));
          }
          
          // Spawn enemy
          enemyManager.spawnEnemyById(type, x, y);
        }
        
        if (DEBUG.enemySpawns) {
          console.log(`Spawned ${count} new enemy near player at (${randomPlayer.player.x.toFixed(1)}, ${randomPlayer.player.y.toFixed(1)})`);
        }
      }
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
  
  // Get static objects for current map (decor/environment)
  const objects = mapManager.getObjects(gameState.mapId);
  
  // Broadcast world update (include optional debug stats)
  const worldPayload = {
    players,
    enemies,
    bullets,
    objects,
    timestamp: Date.now()
  };

  if (bulletManager.stats) {
    worldPayload.bulletStats = { ...bulletManager.stats };
    // Reset per-tick counters so next frame starts clean
    bulletManager.stats.wallHit = 0;
    bulletManager.stats.entityHit = 0;
    bulletManager.stats.created = 0;
  }

  broadcast(MessageType.WORLD_UPDATE, worldPayload);
  
  // Always send the player list with every update for smoother movement
  // Removed the 2-second throttling
  broadcast(MessageType.PLAYER_LIST, players);
}

// Start game update loop
setInterval(updateGame, gameState.updateInterval);

// Add a separate player status logging interval (every 30 seconds)
setInterval(() => {
  // Skip logging if debug flag is disabled
  if (!DEBUG.playerPositions) return;
  
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
  } else if (DEBUG.connections) {
    console.log('[SERVER] No clients connected');
  }
}, 30000); // Changed from 5000 to 30000 (30 seconds)

// Server listen
const START_PORT = Number(process.env.PORT) || 3000;

function tryListen(port, attemptsLeft = 5) {
  const onError = (err) => {
    if (err.code === 'EADDRINUSE' && attemptsLeft > 0) {
      console.warn(`[SERVER] Port ${port} in use, trying ${port + 1}...`);
      tryListen(port + 1, attemptsLeft - 1);
    } else {
      console.error('[SERVER] Failed to bind port:', err);
      process.exit(1);
    }
  };

  const onListening = () => {
    const actualPort = server.address().port;
    console.log(`[SERVER] Running on port ${actualPort}`);
    // Remove error listener; server is good.
    server.off('error', onError);
    server.off('listening', onListening);
  };

  server.once('error', onError);
  server.once('listening', onListening);
  server.listen(port);
}

tryListen(START_PORT);

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
        
      case MessageType.CHAT_MESSAGE:
        // Handle chat message
        handleChatMessage(clientId, data);
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
  
  // Validate proposed coordinates against the current map so the server
  // never trusts a client that tries to walk through walls.  If the new
  // position is blocked we simply keep the previous coordinate.
  if (data.x !== undefined) {
    const newX = data.x;
    if (!mapManager.isWallOrOutOfBounds(newX, player.y)) {
      player.x = newX;
    }
  }

  if (data.y !== undefined) {
    const newY = data.y;
    if (!mapManager.isWallOrOutOfBounds(player.x, newY)) {
      player.y = newY;
    }
  }
  
  if (data.rotation !== undefined) player.rotation = data.rotation;
  if (data.health !== undefined) player.health = data.health;
  
  player.lastUpdate = Date.now();
  
  // Verbose movement trace – disable by default
  if (globalThis.DEBUG?.playerMovement && (oldX !== player.x || oldY !== player.y || oldRotation !== player.rotation)) {
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
      lifetime: data.lifetime || 5.0,
      spriteName: data.spriteName || null
    });
    
    if (globalThis.DEBUG?.bulletEvents) {
      console.log(`Player ${clientId} fired bullet ${bulletId} at angle ${data.angle.toFixed(2)}, position (${data.x.toFixed(2)}, ${data.y.toFixed(2)})`);
    }
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
    lifetime: data.lifetime || 5.0,
    ownerId: clientId,
    spriteName: data.spriteName || null,
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
    
    if (globalThis.DEBUG?.collisions) {
      console.log(`Player ${clientId} reported collision: bullet ${data.bulletId} hit enemy ${data.enemyId}, valid: ${result.valid}`);
    }
  } catch (error) {
    console.error("Error validating collision:", error);
    return;
  }
  
  // Send result to client
  if (result.valid) {
    if (globalThis.DEBUG?.collisions) {
      console.log(`Valid collision: bullet ${result.bulletId} hit enemy ${result.enemyId}, enemy health: ${result.enemyHealth}, killed: ${result.enemyKilled}`);
    }
    
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
    if (globalThis.DEBUG?.collisions) {
      console.log(`Invalid collision rejected: ${result.reason}`);
    }
    
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
  
  // Spawn enemies defined in that map (if not already spawned). This naive version spawns every time the first client switches, but duplicate spawns are prevented by internal manager cap.
  spawnMapEnemies(data.mapId);
  
  // Send map info to client
  const mapMetadata = mapManager.getMapMetadata(data.mapId);
  if (!mapMetadata) {
    if (DEBUG.chunkRequests) {
      console.log(`Client ${clientId} requested unknown map: ${data.mapId}`);
    }
    return;
  }
  
  if (DEBUG.chunkRequests) {
    console.log(`Sent map info to client ${clientId} for map ${data.mapId}`);
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
      mapId: client.mapId,
      x: data.chunkX,
      y: data.chunkY,
      data: chunk,
      timestamp: Date.now()
    });
    
    if (DEBUG.chunkRequests) {
      console.log(`Sent chunk data to client ${clientId} for map ${client.mapId} at (${data.chunkX}, ${data.chunkY})`);
    }
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
  // Remove client
  clients.delete(clientId);
  
  if (DEBUG.connections) {
    console.log(`Client disconnected: ${clientId}`);
  }
  
  // Broadcast disconnect
  broadcastExcept(MessageType.PLAYER_LEAVE, {
    clientId,
    timestamp: Date.now()
  }, clientId);
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
 * Spawn initial enemies for the game world
 * @param {number} count - Number of enemies to spawn
 */
function spawnInitialEnemies(count) {
  // Use current map dimensions for a valid centre point
  const mapMeta = mapManager.getMapMetadata(gameState.mapId);
  const mapWidth = mapMeta?.width || 64;
  const mapHeight = mapMeta?.height || 64;
  const centerX = mapWidth / 2;
  const centerY = mapHeight / 2;
  const spawnRadius = 10; // smaller radius inside map bounds

  if (DEBUG.enemySpawns) {
    console.log(`Spawning ${count} initial enemies around centre (${centerX},${centerY})`);
  }

  for (let i = 0; i < count; i++) {
    const type = 'red_demon';
    const angle = Math.random() * Math.PI * 2;
    const distance = Math.random() * spawnRadius;
    let x = centerX + Math.cos(angle) * distance;
    let y = centerY + Math.sin(angle) * distance;

    // Clamp spawn inside world bounds
    if (mapManager) {
      x = Math.max(1, Math.min(mapManager.width - 1, x));
      y = Math.max(1, Math.min(mapManager.height - 1, y));
    }

    enemyManager.spawnEnemyById(type, x, y);
  }
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

/**
 * Handle a chat message from a client
 * @param {string} clientId - Client ID
 * @param {Object} data - Message data
 */
function handleChatMessage(clientId, data) {
  // Validate message data
  if (!data || !data.message) {
    console.warn(`Received invalid chat message from client ${clientId}`);
    return;
  }
  
  // Get client info
  const client = clients.get(clientId);
  if (!client) {
    console.warn(`Received chat message from unknown client ${clientId}`);
    return;
  }
  
  // Get player name based on client ID
  let playerName;
  
  // If client has an explicitly set player name, use it
  if (client.player && client.player.name) {
    playerName = client.player.name;
  }
  // Otherwise use a default format
  else {
    playerName = `Player-${clientId}`;
    
    // Store this name in the player object for future use
    if (client.player) {
      client.player.name = playerName;
    }
  }
  
  if (DEBUG.chat) {
    console.log(`Setting chat sender name to: ${playerName} for client ${clientId}`);
  }
  
  // Prepare the message for broadcasting, preserving the original ID if provided
  const chatMessage = {
    id: data.id || Date.now(), // Preserve the ID from the client if it exists
    message: data.message.slice(0, 200), // Limit message length
    sender: playerName,
    channel: data.channel || 'All',
    timestamp: Date.now(),
    clientId: clientId // Always include the sender's client ID
  };
  
  if (DEBUG.chat) {
    console.log(`Chat message from ${playerName} (${clientId}): ${chatMessage.message}`);
  }
  
  // Broadcast to all clients in the same map
  broadcastChat(chatMessage, client.mapId);
}

/**
 * Broadcast a chat message to all clients in the same map
 * @param {Object} chatMessage - Message to broadcast
 * @param {string} mapId - Map ID
 */
function broadcastChat(chatMessage, mapId) {
  const messageClientId = chatMessage.clientId;
  
  // Iterate through all clients
  for (const [clientId, client] of clients.entries()) {
    // Check if client is on the same map
    if (client.mapId === mapId) {
      // Create a copy of the message for each recipient
      const messageForClient = {...chatMessage};
      
      // Flag if this message is being sent to the original sender
      if (clientId === messageClientId) {
        messageForClient.isOwnMessage = true;
      }
      
      // Send chat message to client
      sendToClient(client.socket, MessageType.CHAT_MESSAGE, messageForClient);
    }
  }
  
  if (DEBUG.chat) {
    console.log(`Broadcast chat message to map ${mapId}: ${chatMessage.message}`);
  }
}

// ---------------- Map Editor Endpoints -----------------
const mapsDir = path.join(__dirname, 'public', 'maps');
if (!fs.existsSync(mapsDir)) fs.mkdirSync(mapsDir, { recursive: true });

// List maps
app.get('/api/map-editor/maps', (req, res) => {
  try {
    const files = fs.readdirSync(mapsDir).filter(f=>f.endsWith('.json'));
    res.json({ maps: files });
  } catch(err){
    console.error('Error listing maps', err);
    res.status(500).json({ error:'Failed to list maps' });
  }
});

// Save map JSON
app.post('/api/map-editor/save', (req, res) => {
  const { filename, data } = req.body;
  if(!filename || !data) return res.status(400).json({ error:'filename and data required'});
  if(!/^[a-zA-Z0-9_-]+\.json$/.test(filename)) return res.status(400).json({ error:'Invalid filename'});
  const full = path.join(mapsDir, filename);
  try {
    fs.writeFileSync(full, JSON.stringify(data, null, 2));
    res.json({ success:true, path:`maps/${filename}`});
  }catch(err){
    console.error('Error saving map',err);
    res.status(500).json({error:'Failed to save map'});
  }
});

// ---------- ENTITY DATABASE ROUTES ----------
const entitiesDir = path.join(__dirname, 'public', 'assets', 'entities');
app.get('/api/entities/:group', (req, res) => {
  const group = req.params.group;
  const safe = ['tiles', 'objects', 'enemies'];
  if (!safe.includes(group)) return res.status(400).json({ error: 'Invalid group' });
  const file = path.join(entitiesDir, `${group}.json`);
  if (!fs.existsSync(file)) return res.json([]);
  res.sendFile(file);
});
app.get('/api/entities', (_req, res) => {
  const out = {};
  ['tiles', 'objects', 'enemies'].forEach(g => {
    const file = path.join(entitiesDir, `${g}.json`);
    out[g] = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : [];
  });
  res.json(out);
});

app.post('/api/entities/:group', (req,res)=>{
  const group=req.params.group;
  const safe=['tiles','objects','enemies'];
  if(!safe.includes(group)) return res.status(400).json({error:'Invalid group'});
  const entry=req.body;
  if(!entry||!entry.id) return res.status(400).json({error:'Entry with id required'});
  const file=path.join(entitiesDir,`${group}.json`);
  let arr=[];
  if(fs.existsSync(file)) arr=JSON.parse(fs.readFileSync(file,'utf8'));
  const idx=arr.findIndex(e=>e.id===entry.id);
  if(idx>=0) arr[idx]=entry; else arr.push(entry);
  fs.writeFileSync(file,JSON.stringify(arr,null,2));
  // Reload group in memory
  entityDatabase.loadSync();
  res.json({success:true});
});

// ----- Static files ----- Move this BELOW asset-api so that /api/assets/* is not intercepted by serve-static
app.use(express.static('public'));

app.get('/api/sprites/groups', (req,res)=>{
  try{
    const out={};
    const files=fs.readdirSync(atlasesDirBase).filter(f=>f.endsWith('.json'));
    files.forEach(f=>{
      const data=JSON.parse(fs.readFileSync(path.join(atlasesDirBase,f),'utf8'));
      // groups at top-level
      if(data.groups){
        Object.entries(data.groups).forEach(([g,arr])=>{
          if(!out[g]) out[g]=new Set();
          arr.forEach(n=>out[g].add(n));
        });
      }
      // per-sprite tags
      if(Array.isArray(data.sprites)){
        data.sprites.forEach(s=>{
          const list=Array.isArray(s.tags)?s.tags: (Array.isArray(s.groups)?s.groups: (s.group?[s.group]:null));
          if(list){
            list.forEach(g=>{
              if(!out[g]) out[g]=new Set();
              if(s.name) out[g].add(s.name);
            });
          }
        });
      }
    });
    // convert sets to arrays
    const jsonObj={};
    Object.entries(out).forEach(([g,set])=>{jsonObj[g]=Array.from(set);});
    res.json(jsonObj);
  }catch(err){
    console.error('[sprites/groups] error',err);
    res.status(500).json({error:'Failed to aggregate'});
  }
});

function spawnMapEnemies(mapId){
  const spawns=mapManager.getEnemySpawns(mapId);
  if(!spawns||spawns.length===0) return;
  spawns.forEach(e=>{
    if(e&&e.sprite!==undefined){
      const id=typeof e.sprite==='string'?e.sprite:e.id||e.type;
      enemyManager.spawnEnemyById(id,e.x||0,e.y||0);
    }
  });
  if(DEBUG.enemySpawns){console.log(`[MAP] spawned ${spawns.length} enemies from map ${mapId}`);}
}

/**
 * Check if any players are standing on a portal tile/object and trigger map switch.
 */
function handlePortals(){
  if (!gameState.mapId) return;
  const portals = mapManager.getObjects(gameState.mapId).filter(o => o.type === 'portal' && o.destMap);
  if (portals.length === 0) return;
  portals.forEach(portal => {
    clients.forEach((client, id) => {
      if (client.mapId !== gameState.mapId) return;
      const dx = client.player.x - portal.x;
      const dy = client.player.y - portal.y;
      if (Math.abs(dx) <= 0.5 && Math.abs(dy) <= 0.5) {
        switchEntireWorldToMap(portal.destMap);
      }
    });
  });
}

/**
 * Simple implementation: switch the *whole* session to a new map.
 * Sends MAP_INFO to all clients so they reload chunks.
 */
function switchEntireWorldToMap(destMapId){
  if (!destMapId || gameState.mapId === destMapId) return;
  const meta = mapManager.getMapMetadata(destMapId);
  if (!meta) return;
  console.log(`Switching world to map ${destMapId}`);

  gameState.mapId = destMapId;

  // Reposition players at map centre
  const spawnX = meta.width/2;
  const spawnY = meta.height/2;
  clients.forEach((client,id)=>{
    client.mapId = destMapId;
    client.player.x = spawnX;
    client.player.y = spawnY;
    // Send map info so client begins chunk requests
    sendToClient(client.socket, MessageType.MAP_INFO, {
      mapId: destMapId,
      width: meta.width,
      height: meta.height,
      tileSize: meta.tileSize,
      chunkSize: meta.chunkSize,
      timestamp: Date.now()
    });
  });

  spawnMapEnemies(destMapId);
}
