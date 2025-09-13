// File: server.js

import 'dotenv/config';
import express from 'express';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { MapManager } from './src/MapManager.js';
import { BinaryPacket, MessageType } from './common/protocol.js';
import BulletManager from './src/BulletManager.js';
import EnemyManager from './src/EnemyManager.js';
import CollisionManager from './src/CollisionManager.js';
import BagManager from './src/BagManager.js';
import { ItemManager } from './src/ItemManager.js';
// Import BehaviorSystem
import BehaviorSystem from './src/BehaviorSystem.js';
import { entityDatabase } from './src/assets/EntityDatabase.js';
import { NETWORK_SETTINGS } from './common/constants.js';
// Import Unit Systems
import SoldierManager from './src/units/SoldierManager.js';
import UnitSystems from './src/units/UnitSystems.js';
import UnitNetworkAdapter from './src/units/UnitNetworkAdaptor.js';
import { CommandSystem } from './src/CommandSystem.js';
// ---- Hyper-Boss LLM stack ----
import { BossManager, LLMBossController, BossSpeechController } from './server/world/llm/index.js';
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

// Create WebSocket server with compression enabled (permessage-deflate)
const wss = new WebSocketServer({
  server,
  perMessageDeflate: {
    zlibDeflateOptions: {
      chunkSize: 1024,
      level: 3,
      memLevel: 7,
    },
    zlibInflateOptions: {
      chunkSize: 10 * 1024,
    },
    // Other options settable:
    clientNoContextTakeover: true,
    serverNoContextTakeover: true,
    serverMaxWindowBits: 10,
  },
});

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
app.use(express.static(path.join(__dirname,'public')));
// Expose common/ for browser consumption so client can import shared protocol/constants
app.use('/common', express.static(path.join(__dirname,'common')));

// Fallback for SPA routing – send index.html for unknown GETs under /public
app.get('/play', (req,res)=>{
  res.sendFile(path.join(__dirname,'public','index.html'));
});

// Direct hit to /index.html (some browsers/extensions bypass static)
app.get('/index.html',(req,res)=>{
  res.sendFile(path.join(__dirname,'public','index.html'));
});
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

// Utility: safely send a binary packet to one client
function sendToClient(socket, type, data = {}) {
  // Skip if socket is closed or undefined
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  try {
    socket.send(BinaryPacket.encode(type, data));
  } catch (err) {
    console.error('[NET] Failed to send packet', type, err);
  }
}

/**
 * Handle client request to pick up an item from a bag
 */
function processPickupMessage(clientId, data){
  const { bagId, itemId, slot } = data || {};
  const client = clients.get(clientId);
  if(!client || !bagId || !itemId) return;
  const ctx = getWorldCtx(client.mapId);
  const bagMgr = ctx.bagMgr;
  // Validate bag visibility
  const bags = bagMgr.getBagsData(client.mapId, clientId);
  const bagDto = bags.find(b=>b.id===bagId);
  if(!bagDto){
    sendToClient(client.socket, MessageType.PICKUP_DENIED, {reason:'not_visible'});
    return;
  }
  // Range check (2 tiles)
  const dx = client.player.x - bagDto.x;
  const dy = client.player.y - bagDto.y;
  if((dx*dx + dy*dy) > 4) return; // too far
  // Attempt to remove from bag
  const emptied = bagMgr.removeItemFromBag(bagId, itemId);
  if(!emptied && !bagDto.items.includes(itemId)){
    // already removed
  }
  // Add to inventory
  const inv = client.player.inventory || (client.player.inventory = new Array(20).fill(null));
  let idx = (Number.isInteger(slot) && slot>=0 && slot<inv.length && inv[slot]==null) ? slot : inv.findIndex(x=>x==null);
  if(idx===-1){
    sendToClient(client.socket, MessageType.PICKUP_DENIED, {reason:'inventory_full'});
    return;
  }
  inv[idx]=itemId;
  // Send updated inventory back
  sendToClient(client.socket, MessageType.INVENTORY_UPDATE, { inventory: inv });
  // Notify all players in world if bag emptied
  if(emptied){
    broadcastToWorld(client.mapId, MessageType.BAG_REMOVE, { bagId });
  }
}

/**
 * Handle client request to reorder inventory slots
 */
function processMoveItem(clientId,data){
  const {fromSlot,toSlot}=data||{};
  const client=clients.get(clientId);
  if(!client) return;
  const inv=client.player.inventory;
  if(!inv) return;
  if(fromSlot<0||fromSlot>=inv.length||toSlot<0||toSlot>=inv.length){
    sendToClient(client.socket,MessageType.MOVE_DENIED,{reason:'bad_slot'});
    return;
  }
  const temp=inv[fromSlot];
  inv[fromSlot]=inv[toSlot];
  inv[toSlot]=temp;
  sendToClient(client.socket,MessageType.INVENTORY_UPDATE,{inventory:inv});
}

// Helper to broadcast to all clients in world
function broadcastToWorld(mapId, type, payload){
  clients.forEach((c)=>{
    if(c.mapId===mapId){
      sendToClient(c.socket, type, payload);
    }
  });
}

// ---------------------------------------------------------------
// Helper: Send the full starting state to a newly-connected client
// ---------------------------------------------------------------
function sendInitialState(socket, clientId) {
  const client = clients.get(clientId);
  if (!client) return;

  const mapId = client.mapId;

  // Players currently in that world
  const players = {};
  clients.forEach((c, id) => {
    if (c.mapId === mapId) players[id] = c.player;
  });

  const ctx = getWorldCtx(mapId);

  // Send separate packets so the client can reuse existing handlers
  // Standardize shape: wrap in { players }
  sendToClient(socket, MessageType.PLAYER_LIST, { players });
  sendToClient(socket, MessageType.ENEMY_LIST,  ctx.enemyMgr.getEnemiesData(mapId));
  sendToClient(socket, MessageType.BULLET_LIST, ctx.bulletMgr.getBulletsData(mapId));
  sendToClient(socket, MessageType.BAG_LIST,    ctx.bagMgr.getBagsData(mapId));
}

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
    
    // Initialize Unit Systems
    const soldierMgr = new SoldierManager(2000); // Support 2000 units per world
    const unitSystems = new UnitSystems(soldierMgr, mapManager);
    const unitNetAdapter = new UnitNetworkAdapter(wss, soldierMgr, unitSystems);

    enemyMgr._bagManager = bagMgr; // inject for drops

    // Note: Initial unit spawning will happen after gameState is initialized

    logger.info('worldCtx',`Created managers for world ${mapId} including unit systems`);
    worldContexts.set(mapId, { bulletMgr, enemyMgr, collMgr, bagMgr, soldierMgr, unitSystems, unitNetAdapter });
  }
  return worldContexts.get(mapId);
}

/**
 * Spawn initial units for demonstration/testing
 */
function spawnInitialUnits(soldierMgr) {
  // Spawn a small army formation
  const formations = [
    { x: 30, y: 30, type: 0, count: 8, team: 'blue' },   // Infantry
    { x: 30, y: 40, type: 1, count: 6, team: 'blue' },   // Heavy Infantry  
    { x: 30, y: 50, type: 4, count: 10, team: 'blue' },  // Archers
    
    { x: 170, y: 30, type: 2, count: 6, team: 'red' },   // Light Cavalry
    { x: 170, y: 40, type: 3, count: 4, team: 'red' },   // Heavy Cavalry
    { x: 170, y: 50, type: 5, count: 8, team: 'red' },   // Crossbowmen
  ];
  
  formations.forEach(formation => {
    for (let i = 0; i < formation.count; i++) {
      const x = formation.x + (i % 4) * 3;
      const y = formation.y + Math.floor(i / 4) * 3;
      const unitId = soldierMgr.spawn(formation.type, x, y, { team: formation.team });
      
      if (unitId) {
        const index = soldierMgr.findIndexById(unitId);
        // Set team for combat identification
        if (!soldierMgr.owner) {
          soldierMgr.owner = new Array(soldierMgr.max);
        }
        soldierMgr.owner[index] = formation.team;
      }
    }
  });
  
  logger.info('units', `Spawned ${formations.reduce((sum, f) => sum + f.count, 0)} initial units`);
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

// Duplicate route definitions removed - already defined above

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

// ---------------------------------------------------------------------------
// Simple portal handler – teleports a player standing on a portal object
function handlePortals() {
  clients.forEach((client) => {
    const { player, mapId } = client;
    if (!player) return;

    const px = Math.round(player.x);
    const py = Math.round(player.y);

    // Gather portals in the current map
    const portals = (mapManager.getObjects(mapId) || []).filter(o => o.type === 'portal' && o.destMap);
    const portal = portals.find(p => p.x === px && p.y === py);
    if (!portal) return; // player not on a portal tile

    const destMapId = portal.destMap;
    const destMeta  = mapManager.getMapMetadata(destMapId);
    if (!destMeta) {
      console.warn('[PORTAL] Destination map metadata missing:', destMapId);
      return;
    }

    // Choose spawn position in destination map
    let destX = 5, destY = 5;
    if (Array.isArray(destMeta.entryPoints) && destMeta.entryPoints.length > 0) {
      destX = destMeta.entryPoints[0].x ?? destX;
      destY = destMeta.entryPoints[0].y ?? destY;
    }

    // Move player server-side
    player.x = destX;
    player.y = destY;
    player.worldId = destMapId;
    client.mapId = destMapId;

    // Ensure managers exist for the new world
    getWorldCtx(destMapId);

    // Notify client so it can switch worlds
    sendToClient(client.socket, MessageType.WORLD_SWITCH, {
      mapId: destMapId,
      x: destX,
      y: destY,
      timestamp: Date.now()
    });

    console.log(`[PORTAL] Teleported player ${player.id} → ${destMapId} (${destX},${destY})`);
  });
}

// Game state
const gameState = {
  mapId: defaultMapId,
  lastUpdateTime: Date.now(),
  updateInterval: 1000 / 30, // 30 updates per second (was 20)
  enemySpawnInterval: 30000, // 30 seconds between enemy spawns (was 10000)
  lastEnemySpawnTime: Date.now()
};

// Unit spawning will be done via in-game commands
let commandSystem = null;

// Initialize command system
function initializeCommandSystem() {
  if (!commandSystem) {
    commandSystem = new CommandSystem({
      clients: clients,
      gameState: gameState,
      getWorldCtx: getWorldCtx
    });
    console.log('[SERVER] Command system initialized');
  }
  return commandSystem;
}

/**
 * Spawn enemies defined in a map's metadata
 * @param {string} mapId - The map ID to spawn enemies for
 */
function spawnMapEnemies(mapId) {
  try {
    const mapMeta = mapManager.getMapMetadata(mapId);
    if (!mapMeta || !mapMeta.enemies || !Array.isArray(mapMeta.enemies)) {
      console.log(`[ENEMIES] No enemies defined for map ${mapId}`);
      return;
    }

    const worldCtx = getWorldCtx(mapId);
    let spawnedCount = 0;

    mapMeta.enemies.forEach(enemyDef => {
      try {
        const { type = 0, x = 10, y = 10, id } = enemyDef;
        
        if (id !== undefined) {
          // Spawn by entity ID from JSON definitions
          worldCtx.enemyMgr.spawnEnemyById(id, x, y, mapId);
        } else {
          // Spawn by type index
          worldCtx.enemyMgr.spawnEnemy(type, x, y, mapId);
        }
        spawnedCount++;
      } catch (err) {
        console.error(`[ENEMIES] Failed to spawn enemy:`, enemyDef, err);
      }
    });

    if (spawnedCount > 0) {
      console.log(`[ENEMIES] Spawned ${spawnedCount} enemies for map ${mapId}`);
    }
  } catch (err) {
    console.error(`[ENEMIES] Error spawning enemies for map ${mapId}:`, err);
  }
}

// Spawn initial enemies for the game world
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
      inventory: new Array(20).fill(null),
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
    timestamp: Date.now(),
    serverTick: gameState.lastUpdateTime
  });
  
  // Send initial state (player list, enemy list, bullet list)
  sendInitialState(socket, clientId);
  
  // Set up message handler
  socket.on('message', (message) => {
    try {
      const packet = BinaryPacket.decode(message);
      if(packet.type === MessageType.MOVE_ITEM){
        processMoveItem(clientId, packet.data);
      } else if(packet.type === MessageType.PICKUP_ITEM){
        processPickupMessage(clientId, packet.data);
      } else if(packet.type === MessageType.PLAYER_TEXT){
        // Initialize command system if not already done
        const cmdSystem = initializeCommandSystem();
        const client = clients.get(clientId);
        if (client && packet.data && packet.data.text) {
          cmdSystem.processMessage(clientId, packet.data.text, client.player);
        }
      } else {
        handleClientMessage(clientId, message);
      }
    } catch(err){
      console.error('[NET] Failed to process message', err);
    }
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
    
    // Update Unit Systems (military units with tactical combat)
    if (ctx.unitSystems) {
      ctx.unitSystems.update(deltaTime);
    }

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
  if (typeof handlePortals === 'function') {
    try { handlePortals(); } catch(err) {
      console.error('[PORTAL] handlePortals error', err);
    }
  }
  
  // Broadcast world updates
  broadcastWorldUpdates();
}

/**
 * Broadcast world updates (player, enemy, bullet positions)
 */
function broadcastWorldUpdates() {
  const now = Date.now();
  const serverTick = gameState.lastUpdateTime;
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
    const playersObj = {};
    idSet.forEach(cid => { playersObj[cid] = clients.get(cid).player; });

    // Use per-world managers
    const ctx = getWorldCtx(mapId);
    const enemies = ctx.enemyMgr.getEnemiesData(mapId);
    const bullets = ctx.bulletMgr.getBulletsData(mapId);
    
    // Get unit data for this world
    const units = ctx.soldierMgr ? ctx.soldierMgr.getSoldiersData() : [];

    // Optionally clamp by map bounds to avoid stray entities outside map
    const meta = mapManager.getMapMetadata(mapId) || { width: 0, height: 0 };
    const clamp = (arr) => arr.filter(o => o.x >= 0 && o.y >= 0 && o.x < meta.width && o.y < meta.height);
    const enemiesClamped = clamp(enemies);
    const bulletsClamped = clamp(bullets);
    const unitsClamped = clamp(units);

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

      const visibleUnits = unitsClamped.filter(u => {
        const dx = u.x - px;
        const dy = u.y - py;
        return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
      });

      const payload = {
        players: playersObj,
        enemies: visibleEnemies.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
        bullets: visibleBullets.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
        units: visibleUnits.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
        bags:   visibleBags,
        objects,
        timestamp: now,
        serverTick
      };

      if (ctx.bulletMgr.stats) {
        payload.bulletStats = { ...ctx.bulletMgr.stats };
      }

      sendToClient(c.socket, MessageType.WORLD_UPDATE, payload);
    });

    // Also send player list in standardized wrapped shape
    idSet.forEach(cid => {
      const c = clients.get(cid);
      if (c) sendToClient(c.socket, MessageType.PLAYER_LIST, { players: playersObj });
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

// ------------------------------------------------------------------
// TEMP stubs to prevent startup crashes
// (Replaced by full implementation above)
// function spawnInitialEnemies (mapId = defaultMapId) {
//   // Placeholder: initial scripted spawn now handled via metadata; nothing to do.
// }

if (typeof sendInitialState !== 'function') {
  function sendInitialState(socket, clientId) {
    // Fallback: send empty world data to satisfy client expectations
    const payloadEmptyArr = [];
    const payloadEmptyObj = {};
    sendToClient(socket, MessageType.PLAYER_LIST, { players: payloadEmptyObj });
    sendToClient(socket, MessageType.ENEMY_LIST,  { enemies: payloadEmptyArr });
    sendToClient(socket, MessageType.BULLET_LIST, { bullets: payloadEmptyArr });
    sendToClient(socket, MessageType.BAG_LIST,    { bags:    payloadEmptyArr });
  }
}

