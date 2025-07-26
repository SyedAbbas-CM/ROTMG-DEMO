// File: server.js

import 'dotenv/config';
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
import BagManager from './src/BagManager.js';
import { ItemManager } from './src/ItemManager.js';
// Import BehaviorSystem
import BehaviorSystem from './src/BehaviorSystem.js';
import { entityDatabase } from './src/assets/EntityDatabase.js';
import { NETWORK_SETTINGS } from './public/src/constants/constants.js';
// ---- Hyper-Boss LLM stack ----
import BossManager from './src/BossManager.js';
import LLMBossController from './src/LLMBossController.js';
import BossSpeechController from './src/BossSpeechController.js';
import './src/telemetry/index.js'; // OpenTelemetry setup
import llmRoutes from './src/routes/llmRoutes.js';
import hotReloadRoutes from './src/routes/hotReloadRoutes.js';
import { logger } from './src/utils/logger.js';

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

// --------------------------------------------------
// Hyper-Boss globals (single boss testbed)
// --------------------------------------------------
let bossManager       = null;
let llmBossController = null;
let bossSpeechCtrl    = null;

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

// Create global server managers
const mapManager  = new MapManager({ mapStoragePath: path.join(__dirname, 'maps') });
const itemManager = new ItemManager();

// Load item definitions from JSON (sync read once at startup)
try {
  const itemsPath = path.join(__dirname, 'src', 'assets', 'items.json');
  const defs = JSON.parse(fs.readFileSync(itemsPath, 'utf8'));
  defs.forEach(def => itemManager.registerItemDefinition(def));
  console.log(`[ItemManager] Loaded ${defs.length} item definitions.`);
} catch (err) {
  console.error('[ItemManager] Failed to load item definitions:', err);
}

// Expose globally so other modules (EnemyManager) can access lazily
globalThis.itemManager = itemManager;

// ---------------------------------------------------------------------------
// Per-world manager containers
// ---------------------------------------------------------------------------
// Each world / map gets its own trio of managers so logic runs fully isolated.
const worldContexts = new Map(); // mapId → { bulletMgr, enemyMgr, collisionMgr, bagMgr }

/**
 * Lazy-create (or fetch) the manager bundle for a given mapId.
 * Always returns the same object for the same world.
 */
function getWorldCtx(mapId) {
  if (!worldContexts.has(mapId)) {
    const bulletMgr = new BulletManager(10000);
    const enemyMgr  = new EnemyManager(1000);
    const collMgr   = new CollisionManager(bulletMgr, enemyMgr, mapManager);
    const bagMgr    = new BagManager(500);

    enemyMgr._bagManager = bagMgr; // inject for drops

    logger.info('worldCtx',`Created managers for world ${mapId}`);
    worldContexts.set(mapId, { bulletMgr, enemyMgr, collMgr, bagMgr });
  }
  return worldContexts.get(mapId);
}

// ---------------------------------------------------------------------------
// Damage players from enemy bullets belonging to the same world context
// ---------------------------------------------------------------------------
function applyEnemyBulletsToPlayers(bulletMgr, players) {
  const bulletCount = bulletMgr.bulletCount;
  for (let bi = 0; bi < bulletCount; bi++) {
    if (bulletMgr.life[bi] <= 0) continue;

    const ownerId = bulletMgr.ownerId[bi];
    // Only bullets owned by enemies hurt players
    if (typeof ownerId !== 'string' || !ownerId.startsWith('enemy_')) continue;

    const bx = bulletMgr.x[bi];
    const by = bulletMgr.y[bi];
    const bw = bulletMgr.width[bi];
    const bh = bulletMgr.height[bi];

    for (const player of players) {
      if (!player || player.health <= 0) continue;

      const pw = 1, ph = 1;
      const hit = (
        bx - bw / 2 < player.x + pw / 2 &&
        bx + bw / 2 > player.x - pw / 2 &&
        by - bh / 2 < player.y + ph / 2 &&
        by + bh / 2 > player.y - ph / 2
      );

      if (hit) {
        const dmg = bulletMgr.damage ? bulletMgr.damage[bi] : 10;
        player.health -= dmg;
        if (player.health < 0) player.health = 0;
        bulletMgr.markForRemoval(bi);
      }
    }
  }
}

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

/* ------------------------------------------------------------------
 * TEMPORARY PORTAL HOOK
 * ------------------------------------------------------------------
 * Until we finish the full multi-world system this helper will spawn a
 * single portal in the centre of the procedural map that links to a
 * handcrafted map file located at public/maps/test.json.  Remove once
 * proper portal placement / map loading pipeline is ready.
 */
(async () => {
  try {
    const fixedMapPath = path.join(__dirname, 'public', 'maps', 'test.json');

    // Load (or retrieve) the handcrafted map so we have its ID.
    let handmadeId;
    // Check if this map has already been loaded previously
    for (const m of storedMaps.values()) {
      if (m && m.sourcePath === fixedMapPath) {
        handmadeId = m.id;
        break;
      }
    }
    if (!handmadeId) {
      handmadeId = await mapManager.loadFixedMap(fixedMapPath);
      storedMaps.set(handmadeId, mapManager.getMapMetadata(handmadeId));
      // Spawn any static enemies that map defines
      spawnMapEnemies(handmadeId);
      console.log(`[TEMP-PORTAL] Loaded handcrafted map ${handmadeId} from ${fixedMapPath}`);

      // --- NEW: Restore procedural map as active so new players spawn there ---
      mapManager.activeMapId = defaultMapId;
      const defMeta = mapManager.getMapMetadata(defaultMapId);
      if (defMeta) {
        mapManager.width  = defMeta.width;
        mapManager.height = defMeta.height;
      }

      // Re-enable procedural generation for other maps
      if (mapManager.enableProceduralGeneration) {
        mapManager.enableProceduralGeneration();
        // Also mark global fixed flag false so procedural chunks generate
        mapManager.isFixedMap = false;
      }
    }

    // Inject a portal into the default procedural map if one is not present
    const defMeta = mapManager.getMapMetadata(defaultMapId);
    if (!defMeta) return;
    if (!defMeta.objects) defMeta.objects = [];

    const alreadyExists = defMeta.objects.some(o => o.type === 'portal' && o.destMap === handmadeId);
    if (!alreadyExists) {
      const portalObj = {
        id: `debug_portal_${handmadeId}`,
        type: 'portal',
        sprite: 'portal',
        x: 5,
        y: 5,
        destMap: handmadeId
      };
      defMeta.objects.push(portalObj);
      console.log(`[TEMP-PORTAL] Spawned portal at (${portalObj.x},${portalObj.y}) linking ${defaultMapId} → ${handmadeId}`);
    }
  } catch (err) {
    console.error('[TEMP-PORTAL] Failed to set up temporary portal', err);
  }
})();

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

// Initialise manager bundle for the procedural default world
const { bulletMgr: bulletManager, enemyMgr: enemyManager, collMgr: collisionManager } = getWorldCtx(defaultMapId);
console.log('[WORLD_CTX] Default world managers ready – bullets:', typeof bulletManager, 'enemies:', typeof enemyManager);

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
// spawnInitialEnemies(); // Function not defined, enemies will spawn via spawnMapEnemies

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
      worldId: useMapId,
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
  const deltaTime = (now - gameState.lastUpdateTime) / 1000;
  gameState.lastUpdateTime = now;

  // Log connected clients occasionally
  if (DEBUG.playerPositions && now % 30000 < 50) {
    console.log(`[SERVER] ${clients.size} connected client(s)`);
  }

  // Group players by world for quick look-ups
  const playersByWorld = new Map();
  clients.forEach(({ player, mapId }) => {
    if (!playersByWorld.has(mapId)) playersByWorld.set(mapId, []);
    playersByWorld.get(mapId).push(player);
  });

  // Iterate over EVERY world context (even empty ones – keeps bullets moving)
  let totalActiveEnemies = 0;
  worldContexts.forEach((ctx, mapId) => {
    const players = playersByWorld.get(mapId) || [];
    const target  = players[0] || null;

    // ---------- Boss logic first so mirroring happens before physics & collisions ----------
    if (bossManager && mapId === gameState.mapId) {
      bossManager.tick(deltaTime, ctx.bulletMgr);
      if (llmBossController) llmBossController.tick(deltaTime, players).catch(()=>{});
      if (bossSpeechCtrl)    bossSpeechCtrl.tick(deltaTime, players).catch(()=>{});
    }

    // ---------- Physics & AI update ----------
    ctx.bulletMgr.update(deltaTime);
    ctx.bagMgr.update(now / 1000);
    totalActiveEnemies += ctx.enemyMgr.update(deltaTime, ctx.bulletMgr, target, mapManager);

    ctx.collMgr.checkCollisions();
    applyEnemyBulletsToPlayers(ctx.bulletMgr, players);

    // Hyper-boss lives in default world only (gameState.mapId)
    if (bossManager && mapId === gameState.mapId) {
      bossManager.tick(deltaTime, ctx.bulletMgr);
      if (llmBossController) llmBossController.tick(deltaTime, players).catch(()=>{});
      if (bossSpeechCtrl)    bossSpeechCtrl.tick(deltaTime, players).catch(()=>{});
    }
  });

  if (DEBUG.activeCounts && totalActiveEnemies > 0 && now % 5000 < 50) {
    console.log(`[SERVER] Active enemies: ${totalActiveEnemies} across ${worldContexts.size} worlds`);
  }

  // ---------------- PORTAL HANDLING ----------------
  handlePortals();
  
  // Broadcast world updates
  broadcastWorldUpdates();
}

/**
 * Broadcast world updates (player, enemy, bullet positions)
 */
function broadcastWorldUpdates() {
  const now = Date.now();
  const UPDATE_RADIUS = NETWORK_SETTINGS.UPDATE_RADIUS_TILES;
  const UPDATE_RADIUS_SQ = UPDATE_RADIUS * UPDATE_RADIUS;

  // Group clients by map so we can send tailored payloads and avoid leaking
  // enemies / players from other worlds.
  const clientsByMap = new Map(); // mapId -> Set(clientId)
  clients.forEach((client, id) => {
    const m = client.mapId || gameState.mapId;
    if (!clientsByMap.has(m)) clientsByMap.set(m, new Set());
    clientsByMap.get(m).add(id);
  });

  // Iterate per mapId and broadcast to only those clients
  clientsByMap.forEach((idSet, mapId) => {
    // Collect players in this map
    const players = {};
    idSet.forEach(cid => { players[cid] = clients.get(cid).player; });

    // Use per-world managers
    const ctx = getWorldCtx(mapId);
    const enemies = ctx.enemyMgr.getEnemiesData(mapId);
    const bullets = ctx.bulletMgr.getBulletsData(mapId);

    // Optionally clamp by map bounds to avoid stray entities outside map
    const meta = mapManager.getMapMetadata(mapId) || { width: 0, height: 0 };
    const clamp = (arr) => arr.filter(o => o.x >= 0 && o.y >= 0 && o.x < meta.width && o.y < meta.height);
    const enemiesClamped = clamp(enemies);
    const bulletsClamped = clamp(bullets);

    // Collect bag data & clamp to map bounds
    const bags = ctx.bagMgr.getBagsData(mapId);
    const bagsClamped = clamp(bags);

    const objects = mapManager.getObjects(mapId);

    // Send tailored update to each client (interest management)
    idSet.forEach(cid => {
      const c = clients.get(cid);
      if (!c) return;

      const px = c.player.x;
      const py = c.player.y;

      const visibleEnemies = enemiesClamped.filter(e => {
        const dx = e.x - px;
        const dy = e.y - py;
        return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
      });

      const visibleBullets = bulletsClamped.filter(b => {
        const dx = b.x - px;
        const dy = b.y - py;
        return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
      });

      const visibleBags = bagsClamped.filter(b => {
        const dx = b.x - px;
        const dy = b.y - py;
        return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
      });

      const payload = {
        players,
        enemies: visibleEnemies.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
        bullets: visibleBullets.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
        bags:   visibleBags,
        objects,
        timestamp: now
      };

      if (ctx.bulletMgr.stats) {
        payload.bulletStats = { ...ctx.bulletMgr.stats };
      }

      sendToClient(c.socket, MessageType.WORLD_UPDATE, payload);
    });

    // Also send player list
    idSet.forEach(cid => {
      const c = clients.get(cid);
      if (c) sendToClient(c.socket, MessageType.PLAYER_LIST, players);
    });

    // Reset bullet stats counters once per frame for this world
    if (ctx.bulletMgr.stats) {
      ctx.bulletMgr.stats.wallHit = 0;
      ctx.bulletMgr.stats.entityHit = 0;
      ctx.bulletMgr.stats.created = 0;
    }
  });
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
  
  // Clean up resources for every world
  worldContexts.forEach((ctx) => {
    if (ctx.collMgr.cleanup) ctx.collMgr.cleanup();
    if (ctx.enemyMgr.cleanup) ctx.enemyMgr.cleanup();
    if (ctx.bulletMgr.cleanup) ctx.bulletMgr.cleanup();
  });
  
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

