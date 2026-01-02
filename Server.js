// File: server.js

import 'dotenv/config';
import express from 'express';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import { MapManager } from './src/world/MapManager.js';
import OVERWORLD_CONFIG from './src/world/OverworldConfig.js';
import { SetPieceManager } from './src/world/SetPieceManager.js';
import { BinaryPacket, MessageType, ProtocolStats, UDP_MESSAGES } from './common/protocol-native.js';
import BulletManager from './src/entities/BulletManager.js';
import EnemyManager from './src/entities/EnemyManager.js';
import CollisionManager from './src/entities/CollisionManager.js';
import BagManager from './src/entities/BagManager.js';
import { ItemManager } from './src/entities/ItemManager.js';
// Import BehaviorSystem
import BehaviorSystem from './src/Behaviours/BehaviorSystem.js';
import { entityDatabase } from './src/assets/EntityDatabase.js';
import { NETWORK_SETTINGS } from './common/constants.js';
// Import Unit Systems
import SoldierManager from './src/units/SoldierManager.js';
import UnitSystems from './src/units/UnitSystems.js';
import UnitNetworkAdapter from './src/units/UnitNetworkAdaptor.js';
import { CommandSystem } from './src/CommandSystem.js';
// ---- Hyper-Boss LLM stack ----
import { BossManager, LLMBossController, BossSpeechController } from './server/world/llm/index.js';
import { AIPatternBoss } from './src/boss/AIPatternBoss.js';
import './src/telemetry/index.js'; // OpenTelemetry setup
import llmRoutes from './src/routes/llmRoutes.js';
import hotReloadRoutes from './src/routes/hotReloadRoutes.js';
import mapEditorRoutes from './src/routes/mapEditorRoutes.js';
import enemyEditorRoutes from './src/routes/enemyEditorRoutes.js';
import behaviorDesignerRoutes from './src/routes/behaviorDesignerRoutes.js';
import { logger } from './src/utils/logger.js';
// ---- Network Logger ----
import NetworkLogger from './NetworkLogger.js';
// ---- File Logger ----
import FileLogger from './FileLogger.js';
// ---- Lag Compensation ----
import CircularBuffer from './src/utils/CircularBuffer.js';
import LagCompensation from './src/entities/LagCompensation.js';
import MovementValidator from './src/entities/MovementValidator.js';
import { LAG_COMPENSATION, MOVEMENT_VALIDATION } from './common/constants.js';
// ---- Tile System ----
import { initTileSystem } from './src/assets/initTileSystem.js';
// ---- World Spawn Configuration ----
import { worldSpawns, getWorldSpawns } from './config/world-spawns.js';
// ---- Database for player persistence ----
import { initDatabase, getDatabase } from './src/database/Database.js';
// ---- Player Classes & Abilities ----
import { PlayerClasses, getClassById } from './src/player/PlayerClasses.js';
import AbilitySystem from './src/player/AbilitySystem.js';
// ---- WebRTC for UDP-like transport ----
import { getWebRTCServer } from './src/network/WebRTCServer.js';
// ---- WebTransport for true UDP transport (QUIC) ----
import { getWebTransportServer } from './src/network/WebTransportServer.js';
// ---- Binary Protocol for optimized network updates ----
import {
  BinaryWriter,
  DeltaTracker,
  encodeBullet,
  encodeEnemy,
  encodePlayer,
  encodeWorldDelta,
  DeltaFlags,
  registerSprite,
  getEntityId,
  calculateSavings
} from './common/BinaryProtocol.js';

// Debug flags to control logging
const DEBUG = {
  // Keep everything silent by default – we'll re-enable specific areas when
  // we actually need them for diagnostics.
  mapCreation: false,
  connections: false,  // Disable to reduce connection spam
  enemySpawns: false,
  collisions: false,
  playerPositions: false,
  activeCounts: false,
  chunkRequests: false, // Disable chunk request logging
  chat: false,
  playerMovement: false,
  bulletEvents: false   // Disable bullet event spam
};

// Expose debug flags globally so helper classes can reference them
globalThis.DEBUG = DEBUG;

// --------------------------------------------------
// Hyper-Boss globals (single boss testbed)
// --------------------------------------------------
let bossManager       = null;
let llmBossController = null;
let bossSpeechCtrl    = null;
let aiPatternBoss     = null;

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

// PVP: Allow players to damage each other with bullets (enabled by default)
const PVP_ENABLED = process.env.PVP_ENABLED !== 'false';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// -----------------------------------------------------------------------------
// Helper: Random Spawn Location
// -----------------------------------------------------------------------------
/**
 * Generates a random spawn location within the map bounds
 * @param {Object} mapMetadata - Map metadata with width/height
 * @param {Object} mapManager - MapManager instance for walkability checks
 * @returns {Object} {x, y} coordinates in tile units
 */
function generateRandomSpawnLocation(mapMetadata, mapManager) {
  const { width, height } = mapMetadata;
  const margin = 50; // Stay 50 tiles away from edges
  const maxAttempts = 100;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Generate random position with margin
    const x = margin + Math.random() * (width - 2 * margin);
    const y = margin + Math.random() * (height - 2 * margin);

    // Check if location is walkable
    const tileX = Math.floor(x);
    const tileY = Math.floor(y);
    const tile = mapManager.getTile(tileX, tileY);

    // If tile exists and is walkable, use this spawn
    if (tile && tile.walkable !== false) {
      console.log(`[SPAWN] Generated random spawn at (${x.toFixed(2)}, ${y.toFixed(2)}) after ${attempt + 1} attempts`);
      return { x, y };
    }
  }

  // Fallback to center if no walkable location found
  console.warn(`[SPAWN] Could not find walkable spawn after ${maxAttempts} attempts, using center`);
  return { x: width / 2, y: height / 2 };
}

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

// Artificial latency middleware for testing (controlled by ARTIFICIAL_LATENCY_MS env var)
const ARTIFICIAL_LATENCY_MS = parseInt(process.env.ARTIFICIAL_LATENCY_MS || '0', 10);
if (ARTIFICIAL_LATENCY_MS > 0) {
  console.log(`[SERVER] Adding ${ARTIFICIAL_LATENCY_MS}ms artificial latency to all requests`);
  app.use((req, res, next) => {
    setTimeout(next, ARTIFICIAL_LATENCY_MS);
  });
}

// Initialize Network Logger
const NETWORK_LOGGER_ENABLED = process.env.NETWORK_LOGGER_ENABLED !== 'false';
const NETWORK_LOGGER_VERBOSE = process.env.NETWORK_LOGGER_VERBOSE === 'true';
const NETWORK_LOGGER_INTERVAL = parseInt(process.env.NETWORK_LOGGER_INTERVAL || '30000', 10);

const networkLogger = new NetworkLogger({
  enabled: NETWORK_LOGGER_ENABLED,
  verbose: NETWORK_LOGGER_VERBOSE,
  logInterval: NETWORK_LOGGER_INTERVAL
});

// Initialize File Logger
const FILE_LOGGER_ENABLED = process.env.FILE_LOGGER_ENABLED !== 'false';
const fileLogger = new FileLogger({
  enabled: FILE_LOGGER_ENABLED,
  logsDir: path.join(__dirname, 'logs'),
  maxFileSize: 10 * 1024 * 1024, // 10MB
  maxFiles: 10
});

// Expose fileLogger globally so NetworkLogger can access it
global.fileLogger = fileLogger;

// Initialize Lag Compensation System
const lagCompensation = new LagCompensation({
  enabled: LAG_COMPENSATION.ENABLED,
  maxRewindMs: LAG_COMPENSATION.MAX_REWIND_MS,
  minRTT: LAG_COMPENSATION.MIN_RTT_MS,
  debug: LAG_COMPENSATION.DEBUG,
  fileLogger: fileLogger
});

// Initialize Movement Validator
const movementValidator = new MovementValidator({
  enabled: MOVEMENT_VALIDATION.ENABLED,
  maxSpeedTilesPerSec: MOVEMENT_VALIDATION.MAX_SPEED_TILES_PER_SEC,
  teleportThresholdTiles: MOVEMENT_VALIDATION.TELEPORT_THRESHOLD_TILES,
  logInterval: MOVEMENT_VALIDATION.LOG_INTERVAL_MS,
  fileLogger: fileLogger
});

// Initialize Ability System for player classes
const abilitySystem = new AbilitySystem();

// Initialize Database for player persistence (async)
let gameDatabase = null;

(async () => {
  try {
    gameDatabase = await initDatabase();
    const dbStats = gameDatabase.getStats();
    console.log(`[Database] Ready - ${dbStats.players} players, ${dbStats.characters} characters`);
  } catch (err) {
    console.error('[Database] Failed to initialize:', err.message);
    console.warn('[Database] Running without persistence - player data will not be saved');
  }
})();

// Log server startup
if (fileLogger.enabled) {
  fileLogger.info('SERVER', `Server starting with PID ${process.pid}`);
  fileLogger.info('CONFIG', 'Artificial latency', { latencyMs: ARTIFICIAL_LATENCY_MS });
  fileLogger.info('CONFIG', 'Network logger', { enabled: NETWORK_LOGGER_ENABLED, verbose: NETWORK_LOGGER_VERBOSE });
  fileLogger.info('CONFIG', 'Lag compensation', {
    enabled: LAG_COMPENSATION.ENABLED,
    maxRewindMs: LAG_COMPENSATION.MAX_REWIND_MS,
    minRTT: LAG_COMPENSATION.MIN_RTT_MS,
    debug: LAG_COMPENSATION.DEBUG
  });
  fileLogger.info('CONFIG', 'Movement validation', {
    enabled: MOVEMENT_VALIDATION.ENABLED,
    maxSpeed: MOVEMENT_VALIDATION.MAX_SPEED_TILES_PER_SEC,
    teleportThreshold: MOVEMENT_VALIDATION.TELEPORT_THRESHOLD_TILES
  });
}

// Disable caching for all static files during development
app.use((req, res, next) => {
  res.set('Cache-Control', 'no-store, no-cache, must-revalidate, private');
  res.set('Pragma', 'no-cache');
  res.set('Expires', '0');
  next();
});

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

// Mount editor routes
app.use('/api/map-editor', mapEditorRoutes);
app.use('/api/enemy-editor', enemyEditorRoutes);
app.use('/api/behavior-designer', behaviorDesignerRoutes);

// Create global server managers
const mapManager  = new MapManager({ mapStoragePath: path.join(__dirname, 'maps') });
const itemManager = new ItemManager();
const setPieceManager = new SetPieceManager();

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
// Routes UDP-suitable messages through WebTransport when available
function sendToClient(socket, type, data = {}) {
  // Skip if socket is closed or undefined
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  const clientId = socket.clientId;
  const client = clientId ? clients.get(clientId) : null;

  // Check if this message type should use UDP and client has WebTransport
  const useWebTransport = UDP_MESSAGES.has(type) &&
                          client?.webTransportSession?.isReady;

  const doSend = () => {
    try {
      if (useWebTransport) {
        // Send via WebTransport (UDP-like)
        const success = client.webTransportSession.send(type, data);
        if (success) {
          // Log outgoing message
          if (networkLogger.enabled && clientId) {
            networkLogger.onMessageSent(clientId, type, 0, 'UDP');
          }
          return;
        }
        // Fall through to WebSocket if WebTransport send failed
      }

      // Send via WebSocket (TCP)
      const encoded = BinaryPacket.encode(type, data);
      socket.send(encoded);

      // Log outgoing message
      if (networkLogger.enabled && clientId) {
        networkLogger.onMessageSent(clientId, type, encoded.byteLength || encoded.length || 0);
      }
    } catch (err) {
      console.error('[NET] Failed to send packet', type, err);
      if (networkLogger.enabled && clientId) {
        networkLogger.onMessageError(clientId, err);
      }
    }
  };

  // Apply artificial latency to outgoing messages if configured
  if (ARTIFICIAL_LATENCY_MS > 0) {
    setTimeout(doSend, ARTIFICIAL_LATENCY_MS);
  } else {
    doSend();
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
// Helper: Handle chunk requests from client
// ---------------------------------------------------------------
function handleChunkRequest(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;

  const { chunkX, chunkY } = data;
  const mapId = client.mapId;

  // Chunk request logging disabled to reduce spam

  // Get chunk data from map manager
  const chunkData = mapManager.getChunkData(mapId, chunkX, chunkY);

  if (chunkData) {
    // DEBUG: Log sample tile to verify sprite data
    if (chunkX === 0 && chunkY === 0 && chunkData.tiles && chunkData.tiles[0] && chunkData.tiles[0][0]) {
      const sampleTile = chunkData.tiles[0][0];
      console.log('[SERVER] Sample tile before send:', {
        type: sampleTile.type,
        spriteX: sampleTile.spriteX,
        spriteY: sampleTile.spriteY,
        spriteName: sampleTile.spriteName,
        biome: sampleTile.biome,
        hasProperties: !!sampleTile.properties
      });
    }

    // Send chunk data back to client
    sendToClient(client.socket, MessageType.CHUNK_DATA, {
      x: chunkX,
      y: chunkY,
      data: chunkData
    });
  } else {
    // Chunk not found
    sendToClient(client.socket, MessageType.CHUNK_NOT_FOUND, {
      x: chunkX,
      y: chunkY
    });

    console.warn(`[CHUNK] Chunk (${chunkX}, ${chunkY}) not found for map ${mapId}`);
  }
}

// ---------------------------------------------------------------
// Helper: Handle player shooting with rate limiting
// ---------------------------------------------------------------
function handlePlayerShoot(clientId, bulletData) {
  const client = clients.get(clientId);
  if (!client) return;

  // BLOCK SHOOTING IF PLAYER IS DEAD
  if (client.player && client.player.isDead) {
    return; // Silently block - no spam log
  }

  const now = Date.now();

  // RATE LIMITING: Check bullet timestamps
  let timestamps = playerBulletTimestamps.get(clientId);
  if (!timestamps) {
    timestamps = [];
    playerBulletTimestamps.set(clientId, timestamps);
  }

  // Remove timestamps older than 1 second
  const oneSecondAgo = now - 1000;
  while (timestamps.length > 0 && timestamps[0] < oneSecondAgo) {
    timestamps.shift();
  }

  // Check rate limit: max bullets per second
  if (timestamps.length >= BULLET_RATE_LIMIT.MAX_BULLETS_PER_SECOND) {
    // Rate limited - silently drop the bullet
    return;
  }

  // Check minimum interval between bullets
  if (timestamps.length > 0) {
    const lastBulletTime = timestamps[timestamps.length - 1];
    if (now - lastBulletTime < BULLET_RATE_LIMIT.MIN_BULLET_INTERVAL_MS) {
      // Too fast - silently drop
      return;
    }
  }

  // Record this bullet timestamp
  timestamps.push(now);

  const { x, y, angle, speed, damage } = bulletData;
  const mapId = client.mapId;

  // Get world context
  const ctx = getWorldCtx(mapId);
  if (!ctx || !ctx.bulletMgr) {
    console.error(`[SHOOT] No bullet manager for map ${mapId}`);
    return;
  }

  // Create bullet owned by player
  // Spawn bullet slightly offset from player position to prevent immediate boundary collision
  const BULLET_SPAWN_OFFSET = 0.4; // Spawn 0.4 tiles ahead of player
  // CRITICAL FIX: Trust client-provided position instead of stale server player position
  // Using (x || client.player.x) caused 3-tile position discrepancy due to network latency
  const bulletX = x + Math.cos(angle) * BULLET_SPAWN_OFFSET;
  const bulletY = y + Math.sin(angle) * BULLET_SPAWN_OFFSET;

  const bullet = {
    id: `bullet_${Date.now()}_${clientId}_${Math.random()}`,
    x: bulletX,
    y: bulletY,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    damage: damage || 10,
    ownerId: clientId,
    worldId: mapId,
    width: 0.6,   // TILE UNITS: 60% of a tile (slightly larger for better hit detection)
    height: 0.6,  // TILE UNITS: 60% of a tile (slightly larger for better hit detection)
    lifetime: 1.0 // 1 second = 10 tile range (speed 10 tiles/sec * 1.0s)
  };

  // Add to bullet manager
  const bulletId = ctx.bulletMgr.addBullet(bullet);

  if (DEBUG.bulletEvents) {
    // DIAGNOSTIC: Show BOTH client position and actual bullet spawn position
    console.log(`[SERVER BULLET CREATE] ID: ${bulletId}, ClientPos: (${x.toFixed(4)}, ${y.toFixed(4)}), BulletPos: (${bulletX.toFixed(4)}, ${bulletY.toFixed(4)}), Tile: (${Math.floor(bulletX)}, ${Math.floor(bulletY)}), Owner: ${clientId}, Angle: ${angle.toFixed(2)}, Speed: ${speed.toFixed(2)}`);
  }
}

// ---------------------------------------------------------------
// Helper: Check and resolve player-enemy collisions
// ---------------------------------------------------------------
function checkPlayerEnemyCollision(player, mapId) {
  const ctx = getWorldCtx(mapId);
  if (!ctx || !ctx.enemyMgr) return;

  const enemyMgr = ctx.enemyMgr;
  const playerRadius = 0.5; // Player collision radius

  // Check all enemies
  for (let i = 0; i < enemyMgr.enemyCount; i++) {
    // Skip dead/dying enemies
    if (enemyMgr.health[i] <= 0 || enemyMgr.isDying[i]) continue;

    // Calculate distance
    const dx = player.x - enemyMgr.x[i];
    const dy = player.y - enemyMgr.y[i];
    const dist = Math.sqrt(dx * dx + dy * dy);

    // Calculate minimum separation distance
    const enemyRadius = enemyMgr.width[i] / 2;
    const minDist = playerRadius + enemyRadius;

    // If colliding, push player back
    if (dist < minDist && dist > 0.01) {
      const overlap = minDist - dist;
      const angle = Math.atan2(dy, dx);

      // Push player away from enemy
      player.x += Math.cos(angle) * overlap;
      player.y += Math.sin(angle) * overlap;
    }
  }
}

// ---------------------------------------------------------------
// Helper: Load starting area spawn points
// ---------------------------------------------------------------
let startingAreaCache = null;
async function loadStartingAreaSpawnPoints() {
  if (startingAreaCache) return startingAreaCache;

  try {
    const fs = await import('fs/promises');
    const path = await import('path');
    const __dirname = path.dirname(new URL(import.meta.url).pathname);
    const startingAreaPath = path.join(__dirname, 'public', 'maps', 'StartingArea.json');

    const data = await fs.readFile(startingAreaPath, 'utf-8');
    const mapData = JSON.parse(data);

    // Extract entry points
    if (mapData.entryPoints && Array.isArray(mapData.entryPoints) && mapData.entryPoints.length > 0) {
      startingAreaCache = mapData.entryPoints;
      console.log(`[SERVER] 🏠 Loaded ${startingAreaCache.length} starting area spawn points from StartingArea.json`);
    } else {
      // Fallback to center if no entry points defined
      startingAreaCache = [{ x: 16, y: 16 }];
      console.log('[SERVER] ⚠️ No entry points found in StartingArea.json, using default center spawn');
    }

    return startingAreaCache;
  } catch (error) {
    console.error('[SERVER] ❌ Failed to load StartingArea.json:', error.message);
    // Fallback to default spawn point
    startingAreaCache = [{ x: 16, y: 16 }];
    return startingAreaCache;
  }
}

// ---------------------------------------------------------------
// Helper: Handle player respawn (MMO-style: full delete and recreate)
// ---------------------------------------------------------------
async function handlePlayerRespawn(clientId) {
  const client = clients.get(clientId);
  if (!client) {
    console.warn(`[SERVER] ⚠️ Respawn request from unknown client ${clientId}`);
    return;
  }

  console.log(`[SERVER] 🔄 Player ${clientId} requesting respawn - DELETING old character and CREATING new one`);

  const socket = client.socket;
  const mapId = client.mapId || 'map_1';

  // MMO-STYLE: Completely delete the old player object from the clients Map
  // This simulates a full "logout" - player is removed from the world
  if (client.player) {
    console.log(`[SERVER] 💀 Deleting dead player ${clientId} from backend (x: ${client.player.x?.toFixed(2)}, y: ${client.player.y?.toFixed(2)})`);
  }

  // Remove player from clients Map (like they disconnected)
  clients.delete(clientId);
  console.log(`[SERVER] ✅ Player ${clientId} removed from backend`);

  // Calculate random spawn location
  const metaForSpawn = mapManager.getMapMetadata(mapId) || { width: 2560, height: 2560 };
  const { x: spawnX, y: spawnY } = generateRandomSpawnLocation(metaForSpawn, mapManager);

  console.log(`[SERVER] 🌍 Respawning at random location: (${spawnX.toFixed(2)}, ${spawnY.toFixed(2)}) in world ${mapId}`);

  // Get player's class for correct stats (preserve from before death, or default to warrior)
  const playerClassName = oldClient?.player?.class || 'warrior';
  const playerClass = getClassById(playerClassName);

  // MMO-STYLE: Create a completely NEW player object (like a fresh login)
  const newPlayer = {
    id: clientId,
    x: spawnX,
    y: spawnY,
    inventory: new Array(20).fill(null),
    health: playerClass.stats.health,
    maxHealth: playerClass.stats.maxHealth,
    class: playerClass.id,
    className: playerClass.name,
    worldId: mapId,
    lastUpdate: Date.now(),
    isDead: false,
    positionHistory: new CircularBuffer(10), // Required for lag compensation
    rtt: 50 // Default RTT
  };

  // Initialize position history with spawn position
  newPlayer.positionHistory.add(spawnX, spawnY, Date.now());

  // Re-add to clients Map with the new player object
  clients.set(clientId, {
    socket: socket,
    player: newPlayer,
    mapId: mapId,
    lastUpdate: Date.now()
  });

  console.log(`[SERVER] ✨ Created NEW player ${clientId} at (${spawnX.toFixed(2)}, ${spawnY.toFixed(2)}) with ${playerClass.stats.health} HP`);

  // Send confirmation to client with new character data
  sendToClient(socket, MessageType.PLAYER_RESPAWN, {
    x: spawnX,
    y: spawnY,
    health: playerClass.stats.health,
    maxHealth: playerClass.stats.maxHealth,
    timestamp: Date.now(),
    clientId: clientId
  });

  console.log(`[SERVER] 🎮 Respawn complete - Player ${clientId} is now alive and in the game`);
}

// ---------------------------------------------------------------
// Helper: Handle player position update
// ---------------------------------------------------------------
function handlePlayerUpdate(clientId, positionData) {
  const client = clients.get(clientId);
  if (!client || !client.player) return;

  // BLOCK MOVEMENT IF PLAYER IS DEAD
  if (client.player.isDead) {
    console.log(`[SERVER] 🚫 Blocked movement from dead player ${clientId}`);
    return;
  }

  const { x, y, vx, vy } = positionData;

  // Update server-side player position
  if (x !== undefined && y !== undefined) {
    // Validate movement (soft validation - logs only, never blocks)
    if (movementValidator && movementValidator.enabled) {
      movementValidator.validate(clientId, x, y, Date.now());
    }

    client.player.x = x;
    client.player.y = y;

    // Record position in history for lag compensation
    client.player.positionHistory.add(x, y, Date.now());

    // Check and resolve collision with enemies
    const mapId = client.mapId || gameState.mapId;
    checkPlayerEnemyCollision(client.player, mapId);
  }

  // Update velocity if provided
  if (vx !== undefined && vy !== undefined) {
    client.player.vx = vx;
    client.player.vy = vy;
  }

  // No need to broadcast - this will be handled by the normal game state update cycle
}

// ---------------------------------------------------------------
// Helper: Handle unit spawn requests
// ---------------------------------------------------------------
function handleUnitSpawn(clientId, data) {
  const client = clients.get(clientId);
  if (!client || !client.player) return;

  const { unitType, x, y } = data;
  const mapId = client.mapId || gameState.mapId;
  const ctx = getWorldCtx(mapId);

  if (!ctx || !ctx.soldierMgr) {
    console.warn(`[UNIT SPAWN] No soldier manager for map ${mapId}`);
    return;
  }

  // Spawn unit at specified position (or near player if not specified)
  const spawnX = x !== undefined ? x : client.player.x + 2;
  const spawnY = y !== undefined ? y : client.player.y;

  const unitId = ctx.soldierMgr.spawn(unitType, spawnX, spawnY, {
    team: String(clientId), // Units belong to the player who spawned them
    owner: String(clientId)
  });

  if (unitId) {
    console.log(`[UNIT SPAWN] Player ${clientId} spawned ${unitType} at (${spawnX.toFixed(2)}, ${spawnY.toFixed(2)})`);
  }
}

// ---------------------------------------------------------------
// Helper: Handle unit commands (move, attack, etc.)
// ---------------------------------------------------------------
function handleUnitCommand(clientId, data) {
  const client = clients.get(clientId);
  if (!client || !client.player) return;

  const { unitIds, command, targetX, targetY } = data;
  const mapId = client.mapId || gameState.mapId;
  const ctx = getWorldCtx(mapId);

  if (!ctx || !ctx.soldierMgr) {
    console.warn(`[UNIT COMMAND] No soldier manager for map ${mapId}`);
    return;
  }

  // Apply command to each selected unit
  unitIds.forEach(unitId => {
    const index = ctx.soldierMgr.findIndexById(unitId);
    if (index === -1) return;

    // Verify this unit belongs to the player
    if (ctx.soldierMgr.owner[index] !== String(clientId)) {
      return; // Not their unit
    }

    // Apply command
    if (command === 'move' && targetX !== undefined && targetY !== undefined) {
      ctx.soldierMgr.cmd[index] = 'move';
      ctx.soldierMgr.cmdX[index] = targetX;
      ctx.soldierMgr.cmdY[index] = targetY;
      console.log(`[UNIT COMMAND] Unit ${unitId} move to (${targetX.toFixed(2)}, ${targetY.toFixed(2)})`);
    }
  });
}

// ---------------------------------------------------------------
// Helper: Handle player ability usage
// ---------------------------------------------------------------
function handleUseAbility(clientId, data) {
  const client = clients.get(clientId);
  if (!client || !client.player) return;

  const player = client.player;
  if (player.isDead) return;

  const ability = player.ability;
  if (!ability) {
    sendToClient(client.socket, MessageType.ABILITY_RESULT, { success: false, reason: 'no_ability' });
    return;
  }

  const mapId = client.mapId || gameState.mapId;
  const ctx = getWorldCtx(mapId);

  const result = abilitySystem.executeAbility(
    clientId,
    ability,
    player,
    ctx?.bulletMgr,
    ctx?.enemyMgr,
    mapId
  );

  // Send result back to client
  sendToClient(client.socket, MessageType.ABILITY_RESULT, result);

  // Broadcast ability effect to nearby players if needed
  if (result.success) {
    console.log(`[ABILITY] Player ${clientId} used ${ability.name}: ${JSON.stringify(result)}`);
  }
}

// ---------------------------------------------------------------
// Helper: Handle player chat/text messages
// ---------------------------------------------------------------
function handlePlayerText(clientId, data) {
  const client = clients.get(clientId);
  if (!client || !data || !data.text) return;

  const text = data.text.trim();
  if (!text) return;

  // Check if it's a command (starts with /)
  if (text.startsWith('/')) {
    const cmdSystem = initializeCommandSystem();
    cmdSystem.processMessage(clientId, text, client.player);
  } else {
    // Regular chat message - broadcast to all players in same world
    const chatMessage = {
      sender: client.player.name || `Player ${clientId}`,
      text: text,
      senderId: clientId,
      timestamp: Date.now()
    };

    // Broadcast to all players in same map
    clients.forEach((c, id) => {
      if (c.mapId === client.mapId) {
        sendToClient(c.socket, MessageType.CHAT_MESSAGE, chatMessage);
      }
    });

    if (DEBUG.chat) {
      console.log(`[CHAT] ${chatMessage.sender}: ${text}`);
    }
  }
}

// ---------------------------------------------------------------
// Helper: Handle unhandled client messages
// ---------------------------------------------------------------
function handleClientMessage(clientId, message) {
  if (DEBUG.messages) {
    console.log(`[NET] Unhandled message from client ${clientId}:`, message);
  }
}

// ---------------------------------------------------------------
// Helper: Route incoming packet by type (shared by WebSocket/WebTransport)
// ---------------------------------------------------------------
function routePacket(clientId, type, data) {
  const client = clients.get(clientId);
  if (!client) return;

  if (type === MessageType.PING) {
    if (data && data.timestamp) {
      const rtt = Date.now() - data.timestamp;
      if (client.player) {
        client.player.rtt = client.player.rtt * 0.8 + rtt * 0.2;
      }
    }
    sendToClient(client.socket, MessageType.PONG, data);
  } else if (type === MessageType.MOVE_ITEM) {
    processMoveItem(clientId, data);
  } else if (type === MessageType.PICKUP_ITEM) {
    processPickupMessage(clientId, data);
  } else if (type === MessageType.PLAYER_TEXT) {
    handlePlayerText(clientId, data);
  } else if (type === MessageType.CHUNK_REQUEST) {
    handleChunkRequest(clientId, data);
  } else if (type === MessageType.BULLET_CREATE) {
    handlePlayerShoot(clientId, data);
  } else if (type === MessageType.PLAYER_UPDATE) {
    handlePlayerUpdate(clientId, data);
  } else if (type === MessageType.PLAYER_RESPAWN) {
    handlePlayerRespawn(clientId);
  } else if (type === MessageType.UNIT_SPAWN) {
    handleUnitSpawn(clientId, data);
  } else if (type === MessageType.UNIT_COMMAND) {
    handleUnitCommand(clientId, data);
  } else if (type === MessageType.USE_ABILITY) {
    handleUseAbility(clientId, data);
  } else if (type === MessageType.RTC_OFFER) {
    handleRTCOffer(clientId, data);
  } else if (type === MessageType.RTC_ICE_CANDIDATE) {
    handleRTCIceCandidate(clientId, data);
  } else if (type === MessageType.RTC_READY) {
    console.log(`[Transport] Client ${clientId} UDP transport ready`);
  }
}

// ---------------------------------------------------------------
// Helper: Handle WebRTC SDP offer from client
// ---------------------------------------------------------------
async function handleRTCOffer(clientId, data) {
  const client = clients.get(clientId);
  if (!client) return;

  const webrtcServer = getWebRTCServer();
  if (!webrtcServer.isEnabled()) {
    console.log(`[WebRTC] Server not available, client ${clientId} will use WebSocket only`);
    return;
  }

  // Create a sendToClient function for this client
  const sendFunc = (type, msgData) => {
    sendToClient(client.socket, type, msgData);
  };

  const success = await webrtcServer.handleOffer(clientId, data, sendFunc);
  if (success) {
    console.log(`[WebRTC] Client ${clientId}: Offer handled successfully`);
  }
}

// ---------------------------------------------------------------
// Helper: Handle WebRTC ICE candidate from client
// ---------------------------------------------------------------
async function handleRTCIceCandidate(clientId, data) {
  const webrtcServer = getWebRTCServer();
  if (webrtcServer.isEnabled()) {
    await webrtcServer.handleIceCandidate(clientId, data);
  }
}

// ---------------------------------------------------------------
// Helper: Handle client disconnect
// ---------------------------------------------------------------
function handleClientDisconnect(clientId) {
  const client = clients.get(clientId);
  if (!client) return;

  const mapId = client.mapId;
  const player = client.player;

  // Save player data to database before disconnecting
  if (gameDatabase && player.dbCharacterId) {
    try {
      gameDatabase.updateCharacter(player.dbCharacterId, {
        x: player.x,
        y: player.y,
        world_id: player.worldId,
        health: player.health,
        max_health: player.maxHealth,
        mana: player.mana,
        max_mana: player.maxMana,
        level: player.level,
        experience: player.experience,
        fame: player.fame,
        is_dead: player.isDead ? 1 : 0
      });
      console.log(`[Database] 💾 Saved character ${player.dbCharacterId} for ${player.playerName}`);
    } catch (err) {
      console.error(`[Database] Failed to save character:`, err.message);
    }
  }

  if (DEBUG.connections) {
    console.log(`[SERVER] Client disconnected: ${clientId} from map ${mapId}`);
  }

  // Cleanup WebRTC peer connection
  const webrtcServer = getWebRTCServer();
  if (webrtcServer.isEnabled()) {
    webrtcServer.removeClient(clientId);
  }

  // Remove client from the map
  clients.delete(clientId);

  // Cleanup bullet rate limiting tracking
  playerBulletTimestamps.delete(clientId);

  // Cleanup movement validator tracking
  if (movementValidator) {
    movementValidator.removePlayer(clientId);
  }

  // Notify other clients in the same world about player leaving
  broadcastToWorld(mapId, MessageType.PLAYER_LEAVE, {
    playerId: clientId,
    timestamp: Date.now()
  });
}

// ---------------------------------------------------------------
// Helper: Send the full starting state to a newly-connected client
// ---------------------------------------------------------------
function sendInitialState(socket, clientId) {
  const client = clients.get(clientId);
  if (!client) return;

  const mapId = client.mapId;

  // Players currently in that world (EXCLUDE the client themselves to prevent ghost player)
  const players = {};
  clients.forEach((c, id) => {
    if (c.mapId === mapId && id !== clientId) players[id] = c.player;
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
    const collMgr   = new CollisionManager(bulletMgr, enemyMgr, mapManager, lagCompensation, fileLogger, { pvpEnabled: PVP_ENABLED });
    const bagMgr    = new BagManager(500);

    // Initialize Unit Systems
    const soldierMgr = new SoldierManager(2000); // Support 2000 units per world
    const unitSystems = new UnitSystems(soldierMgr, mapManager);
    const unitNetAdapter = new UnitNetworkAdapter(wss, soldierMgr, unitSystems);

    // Delta tracking for binary protocol optimization
    const deltaTracker = new DeltaTracker();

    enemyMgr._bagManager = bagMgr; // inject for drops

    // Note: Initial unit spawning will happen after gameState is initialized

    logger.info('worldCtx',`Created managers for world ${mapId} including unit systems`);
    worldContexts.set(mapId, { bulletMgr, enemyMgr, collMgr, bagMgr, soldierMgr, unitSystems, unitNetAdapter, deltaTracker });
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

    { x: 100, y: 100, type: 6, count: 1, team: 'boss' }, // Boss Infantry
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

// Initialize tile system before map generation
try {
  console.log('[SERVER] Initializing tile system...');
  // Use top-level await (Node 14.8+)
  await initTileSystem();
  console.log('[SERVER] Tile system initialized successfully');
} catch (error) {
  console.error('[SERVER] Failed to initialize tile system:', error);
}

// Create initial procedural map
let defaultMapId;
let fixedMapId; // map created from editor file
try {
  // Set map storage path for the server
  mapManager.mapStoragePath = './maps';

  // ============================================================================
  // CREATE OVERWORLD - 4x4 REGION GRID (512x512 tiles total)
  // ============================================================================
  // Each region = 128x128 tiles
  // 4 regions × 128 tiles = 512 tiles per dimension
  // Use timestamp-based seed for unique worlds each restart
  // ============================================================================
  const uniqueSeed = Date.now() + Math.random() * 1000;
  console.log(`[SERVER] Generating overworld with seed: ${uniqueSeed}`);
  console.log(`[SERVER] World size: ${OVERWORLD_CONFIG.worldSize}x${OVERWORLD_CONFIG.worldSize} tiles (${OVERWORLD_CONFIG.gridSize}x${OVERWORLD_CONFIG.gridSize} regions)`);

  defaultMapId = mapManager.createProceduralMap({
    width: OVERWORLD_CONFIG.worldSize,   // 512 tiles
    height: OVERWORLD_CONFIG.worldSize,  // 512 tiles
    seed: uniqueSeed,
    name: 'Overworld',
    overworldConfig: OVERWORLD_CONFIG     // Pass overworld config to MapManager
  });
  if (DEBUG.mapCreation) {
    console.log(`Created default map: ${defaultMapId} - This is the map ID that will be sent to clients`);
  }

  // ============================================================================
  // LOAD AND PLACE SET PIECES
  // ============================================================================
  try {
    console.log('[SETPIECE] Loading set pieces...');
    setPieceManager.loadSetPieces(path.join(__dirname, 'public', 'maps'));

    if (setPieceManager.getCount() > 0) {
      console.log(`[SETPIECE] Loaded ${setPieceManager.getCount()} set pieces`);

      // Generate random placement locations (3-5 set pieces)
      const mapMeta = mapManager.getMapMetadata(defaultMapId);
      const placements = setPieceManager.generatePlacements(
        mapMeta.width,
        mapMeta.height,
        3, // Place 3 random set pieces
        200 // Minimum 200 tiles apart
      );

      // Apply each set piece to the map
      for (const placement of placements) {
        setPieceManager.applySetPiece(
          mapManager,
          placement.x,
          placement.y,
          placement.setPieceId
        );
      }

      console.log(`[SETPIECE] Placed ${placements.length} set pieces on the map`);
    }
  } catch (error) {
    console.error('[SETPIECE] Error loading set pieces:', error);
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

// ====== LOAD ENEMY SPAWNS FROM CONFIG ======
// Apply enemy spawns from config/world-spawns.js to overworld
const defaultMeta = mapManager.getMapMetadata(defaultMapId);
if (defaultMeta) {
  const overworldSpawns = getWorldSpawns(defaultMapId);
  defaultMeta.enemies = overworldSpawns;
  console.log(`[SPAWNS] Loaded ${overworldSpawns.length} enemy spawns for overworld from config`);
}

/* ------------------------------------------------------------------
 * MULTI-MAP PORTAL SYSTEM
 * ------------------------------------------------------------------
 * Load all maps from /public/maps/ and create bidirectional portals
 * between the procedural overworld and each dungeon/map.
 */
(async () => {
  try {
    const mapsDir = path.join(__dirname, 'public', 'maps');
    const mapFiles = fs.readdirSync(mapsDir).filter(f => f.endsWith('.json'));
    console.log(`[PORTALS] Found ${mapFiles.length} map files in ${mapsDir}`);

    const loadedMaps = [];
    let portalX = 5;

    // Load each map and create its portal in the overworld
    for (const mapFile of mapFiles) {
      const fixedMapPath = path.join(mapsDir, mapFile);

      // Check if already loaded
      let mapId;
      for (const m of storedMaps.values()) {
        if (m && m.sourcePath === fixedMapPath) {
          mapId = m.id;
          break;
        }
      }

      if (!mapId) {
        mapId = await mapManager.loadFixedMap(fixedMapPath);
        const meta = mapManager.getMapMetadata(mapId);

        // Load enemy spawns from config for this map
        const configSpawns = getWorldSpawns(mapId);
        if (configSpawns && configSpawns.length > 0) {
          meta.enemies = configSpawns;
          console.log(`[SPAWNS] Loaded ${configSpawns.length} enemy spawns for ${mapId} from config`);
        }

        storedMaps.set(mapId, meta);
        spawnMapEnemies(mapId);
        console.log(`[PORTALS] Loaded map ${mapId} from ${mapFile}`);
      }

      loadedMaps.push({ id: mapId, name: mapFile.replace('.json', '') });
    }

    // Restore procedural map as active so new players spawn there
    mapManager.activeMapId = defaultMapId;
    const defMeta = mapManager.getMapMetadata(defaultMapId);
    if (defMeta) {
      mapManager.width  = defMeta.width;
      mapManager.height = defMeta.height;
    }

    // Re-enable procedural generation
    if (mapManager.enableProceduralGeneration) {
      mapManager.enableProceduralGeneration();
      mapManager.isFixedMap = false;
    }

    // Create portals in overworld to each dungeon
    if (!defMeta.objects) defMeta.objects = [];

    loadedMaps.forEach((map) => {
      const alreadyExists = defMeta.objects.some(o => o.type === 'portal' && o.destMap === map.id);
      if (!alreadyExists) {
        const portalObj = {
          id: `portal_to_${map.id}`,
          type: 'portal',
          sprite: 'portal',
          x: portalX,
          y: 5,
          destMap: map.id
        };
        defMeta.objects.push(portalObj);
        console.log(`[PORTALS] Created portal at (${portalX},5) → ${map.name}`);
        portalX += 3; // Space portals 3 tiles apart
      }
    });

    // Create return portals in each dungeon back to overworld
    loadedMaps.forEach(map => {
      const mapMeta = mapManager.getMapMetadata(map.id);
      if (!mapMeta) return;
      if (!mapMeta.objects) mapMeta.objects = [];

      const returnExists = mapMeta.objects.some(o => o.type === 'portal' && o.destMap === defaultMapId);
      if (!returnExists) {
        const returnPortal = {
          id: `portal_return_${map.id}`,
          type: 'portal',
          sprite: 'portal',
          x: 2,
          y: 2,
          destMap: defaultMapId
        };
        mapMeta.objects.push(returnPortal);
        console.log(`[PORTALS] Created return portal in ${map.name} at (2,2) → overworld`);
      }
    });

    console.log(`[PORTALS] Setup complete: ${loadedMaps.length} dungeons accessible from overworld`);
  } catch (err) {
    console.error('[PORTALS] Failed to set up portal system', err);
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

// Rate limiting for bullet spam prevention
const BULLET_RATE_LIMIT = {
  MAX_BULLETS_PER_SECOND: 10,      // Max 10 bullets per second per player
  MIN_BULLET_INTERVAL_MS: 100,      // Minimum 100ms between bullets
  MAX_ACTIVE_BULLETS_PER_PLAYER: 50 // Max 50 active bullets per player
};
const playerBulletTimestamps = new Map(); // clientId -> [timestamps]

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
      // Notify player of the error
      sendToClient(client.socket, MessageType.CHAT_MESSAGE, {
        sender: 'System',
        message: `Portal destination unavailable (${destMapId}). Please try again later.`,
        color: '#FF0000'
      });
      // Move player slightly off portal to prevent getting stuck
      player.x += 1;
      player.y += 1;
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
    // Create a server interface object that CommandSystem expects
    const serverInterface = {
      clients: clients,
      gameState: gameState,
      getWorldCtx: getWorldCtx,
      mapManager: mapManager,
      sendToClient: sendToClient,
      broadcastToWorld: broadcastToWorld
    };
    commandSystem = new CommandSystem(serverInterface);
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
// DISABLED: Only boss unit spawns now
// spawnMapEnemies(gameState.mapId);

// WebSocket connection handler
wss.on('connection', async (socket, req) => {
  // Generate client ID
  const clientId = nextClientId++;

  // Store client ID on socket for network logger
  socket.clientId = clientId;

  // Set binary type
  socket.binaryType = 'arraybuffer';

  // Log connection open
  if (networkLogger.enabled) {
    const remoteAddress = req.socket.remoteAddress || 'unknown';
    networkLogger.onConnectionOpen(clientId, remoteAddress, 'ws');
  }

  // Parse URL to check for map ID, class, and player info in query parameters
  const url = new URL(req.url, `http://${req.headers.host}`);
  const requestedMapId = url.searchParams.get('mapId');
  const requestedClass = url.searchParams.get('class') || 'warrior';
  const playerName = url.searchParams.get('name');
  const playerEmail = url.searchParams.get('email');
  const characterId = url.searchParams.get('characterId');
  const playerClass = getClassById(requestedClass);

  // Database-backed player/character handling
  let dbPlayer = null;
  let dbCharacter = null;

  if (gameDatabase && playerName) {
    // Try to find existing player or create new one
    dbPlayer = gameDatabase.getPlayerByName(playerName);
    if (!dbPlayer) {
      dbPlayer = gameDatabase.createPlayer(playerName, playerEmail);
      console.log(`[SERVER] 📝 New player registered: ${playerName}`);
    } else {
      gameDatabase.updateLastLogin(dbPlayer.id);
      console.log(`[SERVER] 👋 Player logged in: ${playerName}`);
    }

    if (dbPlayer) {
      // Try to load existing character or create new one
      if (characterId) {
        dbCharacter = gameDatabase.getCharacterById(parseInt(characterId));
      }
      if (!dbCharacter) {
        // Get most recently played character, or create new one
        const characters = gameDatabase.getCharactersByPlayerId(dbPlayer.id);
        if (characters.length > 0) {
          dbCharacter = characters[0]; // Most recent
          console.log(`[SERVER] 🎮 Loaded character: ${dbCharacter.name} (${dbCharacter.class})`);
        } else {
          // Create new character with player name
          dbCharacter = gameDatabase.createCharacter(dbPlayer.id, playerName, requestedClass);
          console.log(`[SERVER] ✨ Created new character: ${dbCharacter.name}`);
        }
      }
    }
  }
  
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

  // Determine spawn location - use saved position if available, otherwise random
  let spawnX, spawnY;
  if (dbCharacter && dbCharacter.x && dbCharacter.y && !dbCharacter.is_dead) {
    // Use saved position from database
    spawnX = dbCharacter.x;
    spawnY = dbCharacter.y;
    useMapId = dbCharacter.world_id || useMapId;
    console.log(`[SERVER] 📍 Player ${clientId} resuming at saved position: (${spawnX.toFixed(2)}, ${spawnY.toFixed(2)})`);
  } else {
    // Generate random spawn location
    const metaForSpawn = mapManager.getMapMetadata(useMapId) || { width: 2560, height: 2560 };
    const spawn = generateRandomSpawnLocation(metaForSpawn, mapManager);
    spawnX = spawn.x;
    spawnY = spawn.y;
    console.log(`[SERVER] 🌍 New player ${clientId} spawning at random location: (${spawnX.toFixed(2)}, ${spawnY.toFixed(2)})`);
  }

  // Use character data from database if available, otherwise use class defaults
  const useClass = dbCharacter ? getClassById(dbCharacter.class) : playerClass;

  // Store client info with class-based stats (or loaded from database)
  clients.set(clientId, {
    socket,
    player: {
      id: clientId,
      x: spawnX,
      y: spawnY,
      rotation: 0,
      inventory: new Array(20).fill(null),
      // Stats - use database character if available
      class: dbCharacter?.class || useClass.id,
      className: dbCharacter?.class || useClass.name,
      health: dbCharacter?.health || useClass.stats.health,
      maxHealth: dbCharacter?.max_health || useClass.stats.maxHealth,
      mana: dbCharacter?.mana || useClass.stats.mana,
      maxMana: dbCharacter?.max_mana || useClass.stats.maxMana,
      damage: dbCharacter?.attack || useClass.stats.damage,
      speed: dbCharacter?.speed || useClass.stats.speed,
      defense: dbCharacter?.defense || useClass.stats.defense,
      ability: useClass.ability,
      level: dbCharacter?.level || 1,
      experience: dbCharacter?.experience || 0,
      fame: dbCharacter?.fame || 0,
      // Database references
      dbPlayerId: dbPlayer?.id || null,
      dbCharacterId: dbCharacter?.id || null,
      playerName: playerName || `Player_${clientId}`,
      // World state
      worldId: useMapId,
      lastUpdate: Date.now(),
      isDead: false,
      isStealthed: false,
      isShielded: false,
      positionHistory: new CircularBuffer(10), // 300ms history at 30Hz
      rtt: 50 // Default RTT in milliseconds, updated from PING/PONG
    },
    mapId: useMapId,  // Use the appropriate map ID
    lastUpdate: Date.now()
  });

  // Initialize position history with spawn position
  const client = clients.get(clientId);
  client.player.positionHistory.add(spawnX, spawnY, Date.now());

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
  
  // Set up message handler with artificial latency support
  socket.on('message', (message) => {
    const receiveTime = Date.now(); // Record when message arrived

    const processMessage = () => {
      try {
        // Ignore non-binary messages (e.g., text 'test' messages from debug tools)
        if (typeof message === 'string') {
          if (DEBUG.connections) {
            console.log(`[NET] Ignoring text message from client ${clientId}: ${message}`);
          }
          return;
        }

        // Ensure message is an ArrayBuffer or Buffer
        const buffer = message instanceof ArrayBuffer ? message : message.buffer;

        const packet = BinaryPacket.decode(buffer);

        // Log received message
        if (networkLogger.enabled) {
          const messageSize = buffer.byteLength || buffer.length || 0;
          networkLogger.onMessageReceived(clientId, packet.type, messageSize);
        }

        // Log latency for BULLET_CREATE messages to verify artificial delay
        if (packet.type === MessageType.BULLET_CREATE && ARTIFICIAL_LATENCY_MS > 0) {
          const processTime = Date.now();
          const actualDelay = processTime - receiveTime;
          console.log(`🕐 [LATENCY TEST] Bullet message: Received at ${receiveTime}, Processed at ${processTime}, Delay: ${actualDelay}ms (expected: ${ARTIFICIAL_LATENCY_MS}ms)`);
        }

        if(packet.type === MessageType.PING){
          // Track latency for PING messages
          if (networkLogger.enabled && packet.data && packet.data.timestamp) {
            const rtt = Date.now() - packet.data.timestamp;
            networkLogger.recordLatency(clientId, rtt);

            // Update player RTT for lag compensation
            const client = clients.get(clientId);
            if (client && client.player) {
              // Use moving average: 80% old value + 20% new sample
              client.player.rtt = client.player.rtt * 0.8 + rtt * 0.2;
            }
          }
          // Immediately respond with PONG (echo back the timestamp)
          sendToClient(socket, MessageType.PONG, packet.data);
        } else if(packet.type === MessageType.MOVE_ITEM){
          processMoveItem(clientId, packet.data);
        } else if(packet.type === MessageType.PICKUP_ITEM){
          processPickupMessage(clientId, packet.data);
        } else if(packet.type === MessageType.PLAYER_TEXT){
          handlePlayerText(clientId, packet.data);
        } else if(packet.type === MessageType.CHUNK_REQUEST){
          handleChunkRequest(clientId, packet.data);
        } else if(packet.type === MessageType.BULLET_CREATE){
          handlePlayerShoot(clientId, packet.data);
        } else if(packet.type === MessageType.PLAYER_UPDATE){
          handlePlayerUpdate(clientId, packet.data);
        } else if(packet.type === MessageType.PLAYER_RESPAWN){
          handlePlayerRespawn(clientId);
        } else if(packet.type === MessageType.UNIT_SPAWN){
          handleUnitSpawn(clientId, packet.data);
        } else if(packet.type === MessageType.UNIT_COMMAND){
          handleUnitCommand(clientId, packet.data);
        } else if(packet.type === MessageType.USE_ABILITY){
          handleUseAbility(clientId, packet.data);
        } else if(packet.type === MessageType.RTC_OFFER){
          // WebRTC: Handle SDP offer from client
          handleRTCOffer(clientId, packet.data);
        } else if(packet.type === MessageType.RTC_ICE_CANDIDATE){
          // WebRTC: Handle ICE candidate from client
          handleRTCIceCandidate(clientId, packet.data);
        } else if(packet.type === MessageType.RTC_READY){
          // WebRTC: Client's DataChannel is ready
          console.log(`[WebRTC] Client ${clientId} DataChannel ready`);
        } else {
          handleClientMessage(clientId, message);
        }
      } catch(err){
        console.error('[NET] Failed to process message', err);
        if (networkLogger.enabled) {
          networkLogger.onMessageError(clientId, err);
        }
      }
    };

    // Apply artificial latency to WebSocket messages if configured
    if (ARTIFICIAL_LATENCY_MS > 0) {
      setTimeout(processMessage, ARTIFICIAL_LATENCY_MS);
    } else {
      processMessage();
    }
  });
  
  // Set up disconnect handler
  socket.on('close', (code, reason) => {
    if (networkLogger.enabled) {
      networkLogger.onConnectionClose(clientId, code, reason);
    }
    handleClientDisconnect(clientId);
  });

  // Set up error handler
  socket.on('error', (error) => {
    if (networkLogger.enabled) {
      networkLogger.onConnectionError(clientId, error);
    }
    console.error(`[NET] Socket error for client ${clientId}:`, error);
  });
});

/**
 * Update game state
 */
function updateGame() {
  const now = Date.now();
  const deltaTime = (now - gameState.lastUpdateTime) / 1000;
  gameState.lastUpdateTime = now;

  // Update ability cooldowns and effects
  abilitySystem.update(deltaTime);

  // Check for expired effects on players
  clients.forEach(({ player }, clientId) => {
    if (player.isStealthed && !abilitySystem.hasEffect(clientId, 'stealth')) {
      player.isStealthed = false;
    }
    if (player.isShielded && !abilitySystem.hasEffect(clientId, 'shield')) {
      player.isShielded = false;
    }
  });

  // Log connected clients occasionally
  if (DEBUG.playerPositions && now % 30000 < 50) {
    console.log(`[SERVER] ${clients.size} connected client(s)`);
  }

  // Group players by world for quick look-ups (exclude dead players from AI targeting)
  const playersByWorld = new Map();
  clients.forEach(({ player, mapId }) => {
    if (!playersByWorld.has(mapId)) playersByWorld.set(mapId, []);
    // Only add alive players to the targeting list so AI doesn't attack dead players
    if (!player.isDead) {
      playersByWorld.get(mapId).push(player);
    }
  });

  // Iterate over EVERY world context (even empty ones – keeps bullets moving)
  let totalActiveEnemies = 0;
  worldContexts.forEach((ctx, mapId) => {
    const players = playersByWorld.get(mapId) || [];
    const target  = players[0] || null;

    // ---------- Boss logic first so mirroring happens before physics & collisions ----------
    if (bossManager && mapId === gameState.mapId) {
      bossManager.tick(deltaTime, ctx.bulletMgr);

      // AI Pattern attacks
      if (aiPatternBoss) {
        aiPatternBoss.update(deltaTime);
      }

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

    // Check collisions and process results
    const collisionResults = ctx.collMgr.checkCollisions(deltaTime, players);

    // Process enemy contact collisions (damage + knockback)
    if (collisionResults && collisionResults.enemyContactCollisions) {
      for (const collision of collisionResults.enemyContactCollisions) {
        const player = collision.player;

        // Apply contact damage (damage per second * deltaTime)
        const damageThisFrame = collision.contactDamage * deltaTime;
        player.health = Math.max(0, player.health - damageThisFrame);

        // Apply knockback (instant velocity change)
        if (!player.vx) player.vx = 0;
        if (!player.vy) player.vy = 0;
        player.vx += collision.knockbackX;
        player.vy += collision.knockbackY;

        // Push player away from enemy to prevent overlap (hard collision)
        const pushDistance = 0.5; // Push 0.5 tiles away
        const dx = player.x - collision.enemyX;
        const dy = player.y - collision.enemyY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > 0 && distance < 2) {
          // Push player away if too close
          player.x = collision.enemyX + (dx / distance) * 2;
          player.y = collision.enemyY + (dy / distance) * 2;
        }

        // Check for player death
        if (player.health <= 0 && !player.isDead) {
          player.isDead = true;
          console.log(`[CONTACT DAMAGE] Player ${player.id} killed by enemy contact!`);
          // Character deletion handled centrally in broadcastWorldUpdates when PLAYER_DEATH is sent
        }
      }
    }

    // applyEnemyBulletsToPlayers(ctx.bulletMgr, players); // Now handled in CollisionManager
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
    // Collect players in this map (exclude dead players from rendering)
    const playersObj = {};
    idSet.forEach(cid => {
      const player = clients.get(cid).player;
      // Only include alive players in world updates
      if (!player.isDead) {
        playersObj[cid] = player;
      }
    });

    // Use per-world managers
    const ctx = getWorldCtx(mapId);
    const enemies = ctx.enemyMgr.getEnemiesData(mapId);
    const bullets = ctx.bulletMgr.getBulletsData(mapId);

    // Get unit data for this world
    const units = ctx.soldierMgr ? ctx.soldierMgr.getSoldiersData() : [];

    // Optionally clamp by map bounds to avoid stray entities outside map
    const meta = mapManager.getMapMetadata(mapId) || { width: 0, height: 0 };
    const clamp = (arr) => arr.filter(o => {
      const inBounds = o.x >= 0 && o.y >= 0 && o.x < meta.width && o.y < meta.height;
      // DIAGNOSTIC: Log bullets being clamped out at suspect X coordinates
      if (!inBounds && o.id && o.x >= 8 && o.x <= 11) {
        console.error(`❌ [CLAMP] Bullet ${o.id} at X=${o.x.toFixed(2)} CLAMPED OUT! Map bounds: ${meta.width}x${meta.height}`);
      }
      return inBounds;
    });
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
        const distSq = dx * dx + dy * dy;
        const isVisible = distSq <= UPDATE_RADIUS_SQ;

        // DIAGNOSTIC: Log if bullet is being filtered at suspect X coordinates
        if (!isVisible && b.x >= 8 && b.x <= 11) {
          console.error(`❌ [VIS FILTER] Bullet at X=${b.x.toFixed(2)} filtered! Player at X=${px.toFixed(2)}, Dist=${Math.sqrt(distSq).toFixed(2)}, MaxDist=${UPDATE_RADIUS}`);
        }

        return isVisible;
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

      // CRITICAL: Filter out local player POSITION to prevent ghost/duplicate player bug
      // But we MUST send health updates - server is authoritative for damage!
      const playersForThisClient = { ...playersObj };
      delete playersForThisClient[cid];

      // Include local player health separately (server-authoritative)
      const localPlayerHealth = {
        health: c.player.health,
        maxHealth: c.player.maxHealth || 1000,
        isDead: c.player.isDead || false
      };

      const payload = {
        players: playersForThisClient,
        localPlayer: localPlayerHealth,  // Separate health update for self
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

      // Check for player death and send death message
      if (c.player && c.player.isDead && !c.player.deathMessageSent) {
        c.player.deathMessageSent = true;

        // Delete character from database (permadeath) - only once when death is first detected
        if (gameDatabase && c.player.dbCharacterId) {
          try {
            gameDatabase.deleteCharacter(c.player.dbCharacterId);
            console.log(`[Database] 💀 Character ${c.player.dbCharacterId} deleted (${c.player.playerName} died)`);
            c.player.dbCharacterId = null; // Prevent double deletion
          } catch (err) {
            console.error(`[Database] Failed to delete character:`, err.message);
          }
        }

        sendToClient(c.socket, MessageType.PLAYER_DEATH, {
          playerId: cid,
          deathX: c.player.deathX,
          deathY: c.player.deathY,
          timestamp: c.player.deathTimestamp
        });
        console.log(`[SERVER] 📤 Sent PLAYER_DEATH message to client ${cid}`);
      }

      // Try binary protocol for WebTransport clients (5-10x smaller)
      if (c.webTransportSession?.isReady && c.useBinaryProtocol) {
        try {
          // Combine removed IDs from deltaTracker and bulletManager
          const removedFromDelta = ctx.deltaTracker.getAndClearRemoved();
          const removedBullets = ctx.bulletMgr.getAndClearRemovedBullets ?
            ctx.bulletMgr.getAndClearRemovedBullets() : [];
          const allRemoved = [...removedFromDelta, ...removedBullets];

          const binaryPayload = encodeWorldDelta(
            playersForThisClient,
            visibleEnemies.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
            visibleBullets.slice(0, NETWORK_SETTINGS.MAX_ENTITIES_PER_PACKET),
            allRemoved,
            now
          );
          // Use sendBinary() for raw binary - NOT send() which wraps in JSON
          c.webTransportSession.sendBinary(binaryPayload);

          // Also send local player health via JSON (server-authoritative for damage)
          // Binary protocol doesn't include localPlayer yet, so send separately
          c.webTransportSession.send(MessageType.WORLD_UPDATE, { localPlayer: localPlayerHealth });
          // Track binary bandwidth savings (1% sample rate)
          if (networkLogger && Math.random() < 0.01) {
            const jsonSize = JSON.stringify(payload).length;
            const savings = calculateSavings(jsonSize, binaryPayload.byteLength);
            console.log(`[BINARY-TX] Client ${cid}: ${binaryPayload.byteLength} bytes (${savings.percentage}% smaller than JSON)`);
          }
        } catch (err) {
          console.error(`[BINARY] Encode error for client ${cid}:`, err);
          // Fall back to JSON on binary encoding error
          sendToClient(c.socket, MessageType.WORLD_UPDATE, payload);
        }
      } else {
        // Standard JSON for WebSocket clients
        sendToClient(c.socket, MessageType.WORLD_UPDATE, payload);
      }
    });

    // REMOVED: Redundant PLAYER_LIST - players already included in WORLD_UPDATE
    // Saves ~50-75 KB/s per client

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
const START_PORT = Number(process.env.PORT) || 4000;

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

  const onListening = async () => {
    const actualPort = server.address().port;
    console.log(`[SERVER] Running on port ${actualPort}`);
    console.log(`[SERVER] Protocol: ${ProtocolStats.implementation}`);
    if (ProtocolStats.usingNative) {
      console.log(`[SERVER] ⚡ Performance: 2x faster message encoding/decoding`);
    }

    // Initialize boss AI for the main map
    try {
      const mainMapCtx = getWorldCtx(gameState.mapId);
      bossManager = new BossManager(mainMapCtx?.enemyMgr);

      // Initialize AI Pattern Boss system
      try {
        aiPatternBoss = new AIPatternBoss(bossManager, mainMapCtx?.bulletMgr);
        console.log('[SERVER] AI Pattern Boss system enabled');
      } catch (aiErr) {
        console.warn('[SERVER] AI Pattern Boss failed to initialize:', aiErr.message);
        console.log('[SERVER] Boss will use fallback patterns only');
      }

      // Build LLM controller config from environment variables
      const llmConfig = {
        // Tactical tier (enabled by default unless explicitly disabled)
        tacticalEnabled: process.env.TACTICAL_ENABLED !== 'false',

        // Adaptive frequency (enabled by default)
        adaptiveFrequency: process.env.TACTICAL_ADAPTIVE !== 'false',
        tacticalMinInterval: parseInt(process.env.TACTICAL_MIN_INTERVAL) || 10,
        tacticalMaxInterval: parseInt(process.env.TACTICAL_MAX_INTERVAL) || 30,

        // Strategic tier (opt-in via environment variable)
        strategicEnabled: process.env.STRATEGIC_ENABLED === 'true',
        strategicModel: process.env.STRATEGIC_MODEL,
        strategicInterval: parseInt(process.env.STRATEGIC_INTERVAL) || 300,

        // Tactical model override (optional)
        tacticalModel: process.env.TACTICAL_MODEL
      };

      // Pass bulletMgr, mapMgr, enemyMgr (use nulls if not available - controller handles gracefully)
      if (llmConfig.tacticalEnabled) {
        llmBossController = new LLMBossController(
          bossManager,
          mainMapCtx?.bulletMgr || null,
          mainMapCtx?.mapMgr || mapManager,
          mainMapCtx?.enemyMgr || null,
          llmConfig
        );
        console.log('[SERVER] Tactical LLM Boss AI enabled');
      } else {
        console.log('[SERVER] Tactical LLM Boss AI disabled (TACTICAL_ENABLED=false)');
      }

      if (llmConfig.tacticalEnabled) {
        bossSpeechCtrl = new BossSpeechController(bossManager);
        console.log('[SERVER] Boss Speech controller enabled');
      } else {
        console.log('[SERVER] Boss Speech controller disabled (TACTICAL_ENABLED=false)');
      }

      console.log('[SERVER] Boss AI initialized for main map:', gameState.mapId, {
        tacticalEnabled: llmConfig.tacticalEnabled,
        adaptiveFrequency: llmConfig.adaptiveFrequency,
        strategicEnabled: llmConfig.strategicEnabled
      });

      // Spawn 4 AI Pattern Bosses with different configs
      if (bossManager && mainMapCtx?.enemyMgr && aiPatternBoss) {
        // Boss 1: Bloom Guardian (Easy) - Near spawn
        bossManager.spawnBoss('enemy_8', 30, 30, gameState.mapId);
        aiPatternBoss.setBossConfig(0, 'bloom_guardian');

        // Boss 2: Storm Caller (Medium) - East
        bossManager.spawnBoss('enemy_8', 80, 30, gameState.mapId);
        aiPatternBoss.setBossConfig(1, 'storm_caller');

        // Boss 3: Void Burst (Hard) - South
        bossManager.spawnBoss('enemy_8', 30, 80, gameState.mapId);
        aiPatternBoss.setBossConfig(2, 'void_burst');

        // Boss 4: Serpent King (Expert) - Southeast
        bossManager.spawnBoss('enemy_8', 80, 80, gameState.mapId);
        aiPatternBoss.setBossConfig(3, 'serpent_king');

        console.log('[SERVER] Spawned 4 AI Pattern Bosses:');
        console.log('  - Bloom Guardian (Easy) at (30, 30)');
        console.log('  - Storm Caller (Medium) at (80, 30)');
        console.log('  - Void Burst (Hard) at (30, 80)');
        console.log('  - Serpent King (Expert) at (80, 80)');
      }
    } catch (err) {
      console.error('[SERVER] Failed to initialize boss AI:', err);
    }

    // Initialize WebTransport server for UDP-like transport
    try {
      const webTransportServer = getWebTransportServer();
      if (webTransportServer.enabled) {
        const wtStarted = await webTransportServer.start();
        if (wtStarted) {
          console.log('[SERVER] WebTransport server started on port', webTransportServer.port);

          // Set up message handler to route to game logic
          webTransportServer.onMessage = (wtSessionId, type, data) => {
            // Handle WT_LINK specially - links WebTransport session to WebSocket client
            if (type === MessageType.WT_LINK) {
              const wsClientId = data.clientId;
              const client = clients.get(wsClientId);
              if (client) {
                // Store WebTransport session reference on the client
                const wtSession = webTransportServer.sessions.get(wtSessionId);
                if (wtSession) {
                  client.webTransportSession = wtSession;
                  client.webTransportId = wtSessionId;
                  client.useBinaryProtocol = true; // Enable binary protocol for this client
                  console.log(`[WebTransport] Linked WT session ${wtSessionId} to WS client ${wsClientId} (binary protocol enabled)`);
                  // Send acknowledgement via WebTransport
                  wtSession.send(MessageType.WT_LINK_ACK, { success: true, clientId: wsClientId, binaryProtocol: true });
                }
              } else {
                console.warn(`[WebTransport] WT_LINK failed: client ${wsClientId} not found`);
              }
              return;
            }

            // For other messages, try to find the linked WebSocket client
            // and route through normal game logic
            let wsClientId = null;
            for (const [cid, client] of clients) {
              if (client.webTransportId === wtSessionId) {
                wsClientId = cid;
                break;
              }
            }

            if (wsClientId) {
              routePacket(wsClientId, type, data);
            } else {
              // Fallback: route with WT session ID (won't find client, but logs the attempt)
              routePacket(wtSessionId, type, data);
            }
          };

          webTransportServer.onConnect = (wtSessionId, session) => {
            console.log(`[WebTransport] Session ${wtSessionId} connected, awaiting WT_LINK`);
          };

          webTransportServer.onDisconnect = (wtSessionId) => {
            console.log(`[WebTransport] Session ${wtSessionId} disconnected`);
            // Clean up WebTransport reference from client
            for (const [cid, client] of clients) {
              if (client.webTransportId === wtSessionId) {
                client.webTransportSession = null;
                client.webTransportId = null;
                console.log(`[WebTransport] Unlinked from client ${cid}`);
                break;
              }
            }
          };
        } else {
          console.log('[SERVER] WebTransport server failed to start (check certificates)');
        }
      } else {
        console.log('[SERVER] WebTransport disabled (set WEBTRANSPORT_ENABLED=true to enable)');
      }
    } catch (err) {
      console.error('[SERVER] WebTransport initialization error:', err.message);
    }

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

  // Close all WebSocket connections immediately
  wss.clients.forEach((client) => {
    client.close();
  });

  // Force exit after 1 second if server.close() hasn't completed
  setTimeout(() => {
    console.log('Forcing exit...');
    process.exit(0);
  }, 1000);

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

