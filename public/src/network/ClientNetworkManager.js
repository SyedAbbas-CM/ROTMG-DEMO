// public/src/network/ClientNetworkManager.js

/**
 * Utility for throttling log messages to reduce console spam
 */
const logThrottles = {};
function throttledLog(key, message, data, interval = 1000) {
    const now = Date.now();
    if (!logThrottles[key] || now - logThrottles[key] >= interval) {
        logThrottles[key] = now;
        if (data !== undefined) {
            console.log(message, data);
        } else {
            console.log(message);
        }
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// CONFIG FLAGS
// ----------------------------------------------------------------------------
// Toggle whether the client persists the last map it was connected to in
// localStorage and automatically reconnects to that map on page reload.
// During portal-development we always want to spawn in the default (procedural)
// realm first, so disable persistence by default.
export const ENABLE_MAP_ID_PERSISTENCE = false;

// Simple utility to save map data to a file
window.saveMapData = function() {
    try {
        // First check if we have direct access to the clientMapManager
        if (window.gameState && window.gameState.map) {
            console.log("Saving map data from ClientMapManager...");
            console.log("Map object:", window.gameState.map);
            console.log("Loaded chunks:", window.gameState.map.chunks ? window.gameState.map.chunks.size : 0);
            
            // Get map dimensions
            const map = window.gameState.map;
            const width = map.width || 64;
            const height = map.height || 64;
            const mapId = map.activeMapId || 'unknown';
            
            console.log(`Creating map with dimensions ${width}x${height}`);
            
            // Initialize with 0 (floor) as default
            const tileMap = Array(height).fill().map(() => Array(width).fill(0));
            
            // Process all loaded chunks
            let tilesFound = 0;
            const loadedChunks = new Set();
            
            // Check if chunks are loaded
            if (map.chunks && map.chunks.size > 0) {
                console.log(`Found ${map.chunks.size} loaded chunks in map manager`);
                
                // First method: get data from loaded chunks
                for (const [key, chunk] of map.chunks.entries()) {
                    const [chunkX, chunkY] = key.split(',').map(Number);
                    const startX = chunkX * map.chunkSize;
                    const startY = chunkY * map.chunkSize;
                    
                    loadedChunks.add(key);
                    
                    // Process each tile in the chunk
                    if (chunk && Array.isArray(chunk)) {
                        for (let y = 0; y < chunk.length; y++) {
                            if (!chunk[y]) continue;
                            
                            for (let x = 0; x < chunk[y].length; x++) {
                                if (!chunk[y][x]) continue;
                                
                                const globalX = startX + x;
                                const globalY = startY + y;
                                
                                // Make sure we're within the map bounds
                                if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                                    if (chunk[y][x].type !== undefined) {
                                        tileMap[globalY][globalX] = chunk[y][x].type;
                                        tilesFound++;
                                    }
                                }
                            }
                        }
                    }
                }
                
                console.log(`Found ${tilesFound} tiles with defined types from chunks`);
            } else {
                console.log("No loaded chunks found in map manager. Will try direct tile lookup.");
            }
            
            // Second method: If few tiles found, try direct lookup for entire map
            if (tilesFound < width * height * 0.1) { // Less than 10% filled
                console.log("Few tiles found, using direct getTile() lookup for each position...");
                
                // Force load chunks around player if possible
                if (gameState.character && typeof map.updateVisibleChunks === 'function') {
                    console.log("Forcing chunk update around player position...");
                    map.updateVisibleChunks(gameState.character.x, gameState.character.y);
                    
                    // Give some time for chunks to load
                    console.log("Waiting for chunks to load...");
                    setTimeout(() => {
                        console.log(`After forced update: ${map.chunks ? map.chunks.size : 0} chunks loaded`);
                    }, 500);
                }
                
                // Use direct tile lookup with getTile()
                let directTilesFound = 0;
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        // Get tile directly from the map manager
                        const tile = map.getTile(x, y);
                        if (tile && tile.type !== undefined && tile.type !== 0) {
                            tileMap[y][x] = tile.type;
                            directTilesFound++;
                        }
                    }
                }
                
                console.log(`Found ${directTilesFound} additional tiles with defined types using direct lookup`);
                tilesFound += directTilesFound;
            }
            
            // Third method: If we have access to mapDebug, use that data
            if (tilesFound < width * height * 0.1 && window.mapDebug && window.mapDebug.chunks) {
                console.log("Still few tiles found, trying with mapDebug data...");
                
                let debugTilesFound = 0;
                
                // Process mapDebug chunks
                for (const key in window.mapDebug.chunks) {
                    const [chunkX, chunkY] = key.split(',').map(Number);
                    const chunk = window.mapDebug.chunks[key];
                    const startX = chunkX * map.chunkSize;
                    const startY = chunkY * map.chunkSize;
                    
                    // Process the chunk
                    if (chunk && Array.isArray(chunk)) {
                        for (let y = 0; y < chunk.length; y++) {
                            if (!chunk[y]) continue;
                            
                            for (let x = 0; x < chunk[y].length; x++) {
                                if (!chunk[y][x]) continue;
                                
                                const globalX = startX + x;
                                const globalY = startY + y;
                                
                                // Make sure we're within the map bounds
                                if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                                    if (chunk[y][x].type !== undefined) {
                                        tileMap[globalY][globalX] = chunk[y][x].type;
                                        debugTilesFound++;
                                    }
                                }
                            }
                        }
                    }
                }
                
                console.log(`Found ${debugTilesFound} additional tiles with defined types from mapDebug`);
                tilesFound += debugTilesFound;
            }
            
            // Fourth method: If all else fails, try map's direct saving method
            if (tilesFound < width * height * 0.1 && typeof map.saveMapData === 'function') {
                console.log("Using map's direct saveMapData method as last resort...");
                // This will open its own download
                const mapData = map.saveMapData();
                if (mapData && mapData.tileMap) {
                    console.log("Map's direct saveMapData method returned data");
                    // Replace our tile map with the one from the map manager
                    for (let y = 0; y < Math.min(height, mapData.tileMap.length); y++) {
                        for (let x = 0; x < Math.min(width, mapData.tileMap[y].length); x++) {
                            tileMap[y][x] = mapData.tileMap[y][x];
                        }
                    }
                    tilesFound = "unknown (using direct map data)";
                }
            }
            
            console.log(`Total tiles found: ${tilesFound} out of ${width*height} total tiles`);
            
            // Finally, check if we have all zeros and add some variety if so
            let allZeros = true;
            for (let y = 0; y < height && allZeros; y++) {
                for (let x = 0; x < width && allZeros; x++) {
                    if (tileMap[y][x] !== 0) {
                        allZeros = false;
                        break;
                    }
                }
            }
            
            if (allZeros) {
                console.warn("WARNING: All tiles are 0, adding some variety for testing purposes");
                
                // Add some walls and obstacles in a pattern
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        // Create a border
                        if (x === 0 || y === 0 || x === width - 1 || y === height - 1) {
                            tileMap[y][x] = 1; // Wall
                        } 
                        // Add some obstacles in a pattern
                        else if ((x % 10 === 0 && y % 10 === 0) || (x % 10 === 5 && y % 10 === 5)) {
                            tileMap[y][x] = 2; // Obstacle
                        }
                        // Add some water
                        else if ((x > width/2 - 5 && x < width/2 + 5) && (y > height/2 - 5 && y < height/2 + 5)) {
                            tileMap[y][x] = 3; // Water
                        }
                    }
                }
                
                console.log("Added variety to the all-zero map for testing");
            }
            
            // Format JSON with one row per line for readability like simple_map_map_1.json
            const formattedJson = "[\n" + 
                tileMap.map(row => "  " + JSON.stringify(row)).join(",\n") + 
                "\n]";
            
            // Save the simple map
            const blob = new Blob([formattedJson], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `client_simple_map_${mapId}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log(`Map saved to ${a.download} (${loadedChunks.size} chunks)`);
            return true;
        } else if (window.mapDebug && window.mapDebug.chunks) {
            // Fallback to using mapDebug if available
            console.log("Using mapDebug data as fallback...");
            const mapId = window.mapDebug.mapId || 'unknown';
            const chunks = window.mapDebug.chunks;
            
            // Find the bounds of the map from chunks
            let minChunkX = Infinity, minChunkY = Infinity;
            let maxChunkX = -Infinity, maxChunkY = -Infinity;
            
            for (const key in chunks) {
                const [chunkX, chunkY] = key.split(',').map(Number);
                minChunkX = Math.min(minChunkX, chunkX);
                minChunkY = Math.min(minChunkY, chunkY);
                maxChunkX = Math.max(maxChunkX, chunkX);
                maxChunkY = Math.max(maxChunkY, chunkY);
            }
            
            const chunkSize = 16; // Default chunk size
            const width = (maxChunkX - minChunkX + 1) * chunkSize;
            const height = (maxChunkY - minChunkY + 1) * chunkSize;
            
            console.log(`Creating map with dimensions ${width}x${height} from mapDebug`);
            
            // Initialize with 0 (floor) as default
            const tileMap = Array(height).fill().map(() => Array(width).fill(0));
            
            // Fill the map with known tile types from chunks
            for (const key in chunks) {
                const [chunkX, chunkY] = key.split(',').map(Number);
                const chunk = chunks[key];
                
                if (!chunk || !Array.isArray(chunk)) continue;
                
                const startX = (chunkX - minChunkX) * chunkSize;
                const startY = (chunkY - minChunkY) * chunkSize;
                
                for (let y = 0; y < chunk.length; y++) {
                    if (!chunk[y]) continue;
                    
                    for (let x = 0; x < chunk[y].length; x++) {
                        if (!chunk[y][x]) continue;
                        
                        const globalX = startX + x;
                        const globalY = startY + y;
                        
                        if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                            if (chunk[y][x].type !== undefined) {
                                tileMap[globalY][globalX] = chunk[y][x].type;
                            }
                        }
                    }
                }
            }
            
            // Format JSON with one row per line for readability
            const formattedJson = "[\n" + 
                tileMap.map(row => "  " + JSON.stringify(row)).join(",\n") + 
                "\n]";
            
            // Save the simple map
            const blob = new Blob([formattedJson], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `client_simple_map_${mapId}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log(`Map saved to ${a.download} using mapDebug data`);
            return true;
        } else {
            console.warn("No map data available. Please wait for the map to load or explore more of the map first.");
            return false;
        }
    } catch (error) {
        console.error("Error saving map data:", error);
        return false;
    }
};

// Save a simple 2D array of just tile types
function saveTileTypeArray() {
    try {
        // Check if we have access to the map manager directly
        if (!window.gameState || !window.gameState.map) {
            console.error("No map manager available in gameState");
            return false;
        }
        
        const map = window.gameState.map;
        const mapId = window.mapDebug?.mapId || 'unknown';
        const chunkSize = map.chunkSize || 16;
        const width = map.width || 64;
        const height = map.height || 64;
        
        console.log(`Creating client map with dimensions ${width}x${height} from clientMapManager`);
        
        // Initialize with 0 (floor) as default instead of -1
        const tileMap = Array(height).fill().map(() => Array(width).fill(0));
        
        // Use direct tile lookup for each position in the map
        let tilesFound = 0;
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Get tile directly from the map manager
                const tile = map.getTile(x, y);
                if (tile && tile.type !== undefined) {
                    tileMap[y][x] = tile.type;
                    tilesFound++;
                }
            }
        }
        
        console.log(`Found ${tilesFound} tiles with defined types out of ${width*height} total tiles`);
        
        // If we found very few tiles, try another approach
        if (tilesFound < width * height * 0.1) { // Less than 10% of tiles have values
            console.log("Very few tiles found, trying with chunk data from mapDebug");
            
            // Find the bounds of the map from chunks
            let minChunkX = Infinity, minChunkY = Infinity;
            let maxChunkX = -Infinity, maxChunkY = -Infinity;
            
            // Find the bounds of all chunks
            for (const key in window.mapDebug?.chunks || {}) {
                const [chunkX, chunkY] = key.split(',').map(Number);
                minChunkX = Math.min(minChunkX, chunkX);
                minChunkY = Math.min(minChunkY, chunkY);
                maxChunkX = Math.max(maxChunkX, chunkX);
                maxChunkY = Math.max(maxChunkY, chunkY);
            }
            
            console.log(`Chunk bounds: (${minChunkX},${minChunkY}) to (${maxChunkX},${maxChunkY})`);
            
            // Fill the map with known tile types from chunks
            for (const key in window.mapDebug?.chunks || {}) {
                const [chunkX, chunkY] = key.split(',').map(Number);
                const chunk = window.mapDebug.chunks[key];
                
                // Check if the chunk has tiles
                if (!chunk || !Array.isArray(chunk)) {
                    console.log(`Chunk ${key} has invalid format:`, chunk);
                    continue;
                }
                
                // Process each tile in the chunk
                for (let relY = 0; relY < chunkSize; relY++) {
                    if (!chunk[relY]) continue;
                    
                    for (let relX = 0; relX < chunkSize; relX++) {
                        if (!chunk[relY][relX]) continue;
                        
                        const globalX = chunkX * chunkSize + relX;
                        const globalY = chunkY * chunkSize + relY;
                        
                        // Make sure we're within the map bounds
                        if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                            if (chunk[relY][relX].type !== undefined) {
                                tileMap[globalY][globalX] = chunk[relY][relX].type;
                                tilesFound++;
                            }
                        }
                    }
                }
            }
            
            console.log(`After using chunk data: Found ${tilesFound} tiles with defined types`);
        }
        
        // Check if we have the map's debug print method
        if (tilesFound < width * height * 0.1 && typeof map.printMapDebug === 'function') {
            console.log("Still very few tiles found, trying to extract data from debug visualization");
            
            // Get the debug representation which contains all the tile information
            const debugInfo = map.printMapDebug(width, height, true);
            
            // If the debug info returned tile counts, we have success
            if (debugInfo && debugInfo.loadedChunks > 0) {
                console.log(`Successfully extracted map data from debug visualization: ${debugInfo.loadedChunks} chunks loaded`);
            }
        }
        
        // Format JSON with one row per line for readability
        const formattedJson = "[\n" + 
            tileMap.map(row => "  " + JSON.stringify(row)).join(",\n") + 
            "\n]";
        
        // Save the simple map
        const blob = new Blob([formattedJson], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `client_simple_map_${mapId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`Simple tile map saved to ${a.download}`);
        return true;
    } catch (error) {
        console.error("Error saving simple tile map:", error);
        return false;
    }
}

console.log("Map save function available. Use window.saveMapData() in the console to download map data.");

/**
 * ClientNetworkManager
 * Handles WebSocket communication with the game server using binary packet format
 */
import { BinaryPacket, MessageType, UDP_MESSAGES } from '/common/protocol.js';
import { WebRTCManager } from './WebRTCManager.js';
import { WebTransportManager } from './WebTransportManager.js';
import {
    decodeWorldDelta,
    encodePlayerUpdate,
    encodeBulletCreate,
    encodePing,
    ClientBinaryType
} from './BinaryProtocol.js';
import { Player } from '../entities/player.js';

export class ClientNetworkManager {
    /**
     * Create a client network manager
     * @param {string} serverUrl - WebSocket server URL
     * @param {Object} game - Game reference
     */
    constructor(serverUrl, game) {
        this.serverUrl = serverUrl;
        this.socket = null;
        this.connected = false;
        this.connecting = false;
        this.clientId = null;
        this.messageQueue = [];
        this.lastPingTime = 0;
        this.pingInterval = null;
        this.latency = 0;        // Current latency in ms
        this.avgLatency = 0;     // Rolling average latency
        this.latencyHistory = []; // Last 10 latency samples
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000; // ms
        this.timeOffset = 0; // Client-server time offset

        // WebRTC for UDP-like transport (legacy, may not work through tunnels)
        this.webrtc = null;
        this.useWebRTC = false; // Disable WebRTC by default (use WebTransport instead)

        // WebTransport for true UDP-like transport (works through tunnels)
        this.webtransport = null;
        this.useWebTransport = true; // Enable WebTransport by default
        this.webTransportUrl = null; // Set via config or auto-detect

        // Store game reference for callbacks
        this.game = game || {};
        
        // Try to get playerManager from game or gameState
        if (game && game.playerManager) {
            this.playerManager = game.playerManager;
        } else if (window.gameState && window.gameState.playerManager) {
            this.playerManager = window.gameState.playerManager;
            // Also add to game for reference
            this.game.playerManager = this.playerManager;
        }
        
        // Message handlers - default empty functions
        this.handlers = {};
        Object.values(MessageType).forEach(type => {
            this.handlers[type] = () => {};
        });
        
        console.log("ClientNetworkManager initialized with server URL:", serverUrl);

        // Debug: Expose network status globally
        window.getNetworkStatus = () => {
            const wtStats = this.webtransport?.getStats() || { isReady: false };
            const status = {
                wsConnected: this.connected,
                wtReady: wtStats.isReady,
                binaryCount: this._binaryCount || 0,
                transport: this.webtransport?.isReady ? 'WebTransport' : 'WebSocket',
                wtStats: wtStats
            };

            // Visual summary
            const isBinary = status.wtReady && status.binaryCount > 0;
            console.log('%c╔════════════════════════════════════╗', 'color: #0ff');
            console.log('%c║     [NETWORK] STATUS REPORT        ║', 'color: #0ff; font-weight: bold');
            console.log('%c╠════════════════════════════════════╣', 'color: #0ff');
            console.log(`%c║ WebSocket: ${status.wsConnected ? '✓ CONNECTED' : '✗ DISCONNECTED'}`, status.wsConnected ? 'color: #0f0' : 'color: #f00');
            console.log(`%c║ WebTransport: ${status.wtReady ? '✓ READY' : '✗ NOT READY'}`, status.wtReady ? 'color: #0f0' : 'color: #f90');
            console.log(`%c║ Binary Protocol: ${isBinary ? '✓ ACTIVE (' + status.binaryCount + ' msgs)' : '✗ INACTIVE'}`, isBinary ? 'color: #0f0; font-weight: bold' : 'color: #f00');
            console.log(`%c║ Transport: ${status.transport}`, 'color: #fff');
            console.log('%c╚════════════════════════════════════╝', 'color: #0ff');

            if (!status.wtReady) {
                console.log('%c[NETWORK] WebTransport not connected - binary protocol disabled', 'color: #f90');
                console.log('%c[NETWORK] All messages using JSON over WebSocket', 'color: #f90');
            }

            return status;
        };

        // Also expose quick binary check
        window.isBinaryActive = () => {
            const active = this.webtransport?.isReady && (this._binaryCount || 0) > 0;
            console.log(`%c[BINARY] Protocol ${active ? 'ACTIVE' : 'INACTIVE'} (${this._binaryCount || 0} messages sent)`, active ? 'color: #0f0; font-weight: bold' : 'color: #f00');
            return active;
        };
    }
    
    /**
     * Set up message handlers for different message types
     */
    setupMessageHandlers() {
        // Only set up handlers if the socket exists
        if (!this.socket) {
            console.error('Cannot set up message handlers: Socket not initialized');
            setTimeout(() => {
                // Try again after socket initialization
                if (this.socket) {
                    console.log("Socket now available, setting up message handlers");
                    this.setupMessageHandlers();
                }
            }, 1000);
            return;
        }

        this.handlers[MessageType.HANDSHAKE_ACK] = (data) => {
            this.clientId = data.clientId;
            console.log(`Received client ID: ${this.clientId}`);
            if (this.game.setClientId) {
                this.game.setClientId(this.clientId);
            }
        };
        
        this.handlers[MessageType.MAP_INFO] = (data) => {
            console.log('Received map info:', data);
            // Store map ID only if persistence flag enabled (disabled by default)
            if (ENABLE_MAP_ID_PERSISTENCE && data.mapId) {
                console.log(`Storing map ID in localStorage: ${data.mapId}`);
                localStorage.setItem('currentMapId', data.mapId);
            }
            if (this.game.initMap) {
                this.game.initMap(data);
            }
        };
        
        this.handlers[MessageType.PLAYER_LIST] = (data) => {
            // Server now standardizes to { players }
            const playersData = (data && typeof data.players === 'object') ? data.players : {};
            
            // Validate that we have an object with players
            if (!playersData || typeof playersData !== 'object') {
                console.error('Invalid player list data format:', data);
                return;
            }
            
            const playerCount = Object.keys(playersData).length;
            
            // Throttle these logs to once per second
            throttledLog('player-list', `[NETWORK] Received player list: ${playerCount} players (IDs: ${Object.keys(playersData).join(', ')})`);
            
            if (playerCount > 0 && throttledLog('player-sample', 'Player sample:', null, 5000)) {
                // Log a sample player for data validation (only once every 5 seconds)
                const samplePlayerId = Object.keys(playersData)[0];
                const samplePlayer = playersData[samplePlayerId];
                console.log(`Sample player data for ${samplePlayerId}:`, samplePlayer);
            }
            
            // Process player data to ensure it has all properties needed for animation
            const enhancedPlayersData = {};
            
            // Track position changes to detect movement
            const positionChanges = new Map();
            
            for (const [playerId, playerData] of Object.entries(playersData)) {
                // Skip null/undefined players
                if (!playerData) continue;
                
                // Get previous position data if available
                let previousData = null;
                if (this.playerManager && this.playerManager.players) {
                    previousData = this.playerManager.players.get(playerId);
                }
                
                // Check if position changed from previous update
                let positionChanged = true; // Assume movement by default
                if (previousData) {
                    positionChanged = 
                        previousData.x !== playerData.x || 
                        previousData.y !== playerData.y;
                    
                    // Store whether position changed
                    positionChanges.set(playerId, positionChanged);
                }
                
                // Create enhanced player data with animation properties
                enhancedPlayersData[playerId] = {
                    ...playerData,
                    // Ensure these essential properties exist
                    id: playerId,
                    x: playerData.x || 0,
                    y: playerData.y || 0,
                    rotation: playerData.rotation || 0,
                    health: playerData.health !== undefined ? playerData.health : 100,
                    maxHealth: playerData.maxHealth || 100,
                    // Animation properties for EntityAnimator
                    width: playerData.width || 10,
                    height: playerData.height || 10,
                    // Flag for animation system
                    isMoving: positionChanged
                };
            }
            
            // Call game's setPlayers handler with the enhanced players data
            if (this.game.setPlayers) {
                this.game.setPlayers(enhancedPlayersData);
            } else {
                console.error("PLAYER_LIST handler called but this.game.setPlayers not defined!");
            }
        };
        
        this.handlers[MessageType.ENEMY_LIST] = (data) => {
            throttledLog('enemy-list', `Received enemies list: ${data.enemies ? data.enemies.length : 0} enemies`, null, 2000);
            if (this.game.setEnemies && data.enemies) {
                this.game.setEnemies(data.enemies);
            }
        };
        
        this.handlers[MessageType.BULLET_LIST] = (data) => {
            throttledLog('bullet-list', `Received bullets list: ${data.bullets ? data.bullets.length : 0} bullets`, null, 2000);
            if (this.game.setBullets && data.bullets) {
                this.game.setBullets(data.bullets);
            }
        };

        // Loot bag list (initial or refresh)
        this.handlers[MessageType.BAG_LIST] = (data) => {
            if (this.game.setBags && data.bags) {
                this.game.setBags(data.bags);
            }
        };

        // Loot interaction
        this.handlers[MessageType.PICKUP_ITEM] = (data) => {
            if (this.game.pickupItem) {
                this.game.pickupItem(data.itemId);
            }
        };

        this.handlers[MessageType.INVENTORY_UPDATE] = (data) => {
            if (this.game.setInventory) {
                this.game.setInventory(data.inventory);
            }
        };

        this.handlers[MessageType.WORLD_UPDATE] = (data) => {
            // DIAGNOSTIC: Only log when bullets are present
            if (data.bullets && data.bullets.length > 0) {
                // console.error(`🌍 [WORLD UPDATE] Received ${data.bullets.length} bullets, First: ID=${data.bullets[0].id}, Pos=(${data.bullets[0].x?.toFixed(2)}, ${data.bullets[0].y?.toFixed(2)})`);
            }

            // Only log occasionally to reduce spam
            throttledLog('world-update', `World update received`, null, 3000);

            // ============================================================================
            // MERGE WORLD OBJECTS WITH CHUNK OBJECTS
            // ============================================================================
            // CRITICAL FIX: Don't overwrite window.currentObjects!
            // - ClientMapManager.setChunkData() adds chunk objects (trees, boulders, flowers)
            // - WORLD_UPDATE contains map-level objects (portals, NPCs)
            // - We need to MERGE both arrays, not replace
            // ============================================================================
            if (typeof window !== 'undefined') {
              // Get existing objects (from chunks and previous updates)
              const existingObjects = window.currentObjects || [];
              const worldObjects = Array.isArray(data.objects) ? data.objects : [];

              // Separate chunk objects from map objects
              // Chunk objects have IDs like "obj_X_Y_decor"
              // Map objects have IDs like "portal_to_map_2"
              const chunkObjects = existingObjects.filter(obj => obj.id && obj.id.startsWith('obj_'));

              // CRITICAL FIX: Deduplicate chunk objects by ID to prevent accumulation
              const uniqueChunkObjects = new Map();
              chunkObjects.forEach(obj => {
                if (obj.id) {
                  uniqueChunkObjects.set(obj.id, obj);
                }
              });

              // Convert back to array
              const deduplicatedChunkObjects = Array.from(uniqueChunkObjects.values());

              // Combine: deduplicated chunk objects (from MapManager) + world objects (from server)
              window.currentObjects = [...deduplicatedChunkObjects, ...worldObjects];

              // DEBUG: Log object counts and warn if duplication was found
              // COMMENTED OUT: Lag issues have been fixed, no need for performance logs
              // if (chunkObjects.length !== deduplicatedChunkObjects.length) {
              //   console.warn(`[PERF] Deduplicated ${chunkObjects.length - deduplicatedChunkObjects.length} duplicate chunk objects!`);
              // }

              // if (window.currentObjects.length > 0) {
              //   throttledLog('objects-merged',
              //     `[CLIENT] Merged objects: ${deduplicatedChunkObjects.length} chunk + ${worldObjects.length} world = ${window.currentObjects.length} total`,
              //     null, 5000);
              // }
            }

            // Apply local player health from server (authoritative for damage)
            if (data.localPlayer && window.gameState?.character) {
                const oldHealth = window.gameState.character.health;
                window.gameState.character.health = data.localPlayer.health;
                window.gameState.character.maxHealth = data.localPlayer.maxHealth || 200;  // Match class HP default
                window.gameState.character.isDead = data.localPlayer.isDead || false;

                // DEBUG: Log health changes
                if (oldHealth !== data.localPlayer.health) {
                    console.log(`[HEALTH DEBUG] Server health update: ${oldHealth} -> ${data.localPlayer.health}`);
                }

                // Update UI if health changed
                if (oldHealth !== data.localPlayer.health && window.gameUI?.updateHealth) {
                    window.gameUI.updateHealth(data.localPlayer.health, data.localPlayer.maxHealth || 200);
                }
            }

            if (this.game.updateWorld) {
                // Check if players is nested inside a 'players' property (from server inconsistency)
                const players = data.players?.players || data.players;
                this.game.updateWorld(data.enemies, data.bullets, players, data.objects, data.units || []);
                if (this.game.setBags && data.bags) {
                    this.game.setBags(data.bags);
                }
            }
        };

        // Binary protocol handler for optimized world updates (5-10x smaller packets)
        this.handlers[MessageType.BINARY_WORLD_DELTA] = (data) => {
            try {
                // Data is already an ArrayBuffer from WebTransport
                const delta = decodeWorldDelta(data);

                // Debug: Log incoming binary world updates
                if (!this._binaryRxCount) this._binaryRxCount = 0;
                this._binaryRxCount++;
                if (this._binaryRxCount <= 5 || this._binaryRxCount % 100 === 0) {
                    console.log(`[BINARY] RX WORLD_DELTA #${this._binaryRxCount}: ${delta?.bullets?.length || 0} bullets, ${delta?.enemies?.length || 0} enemies, ${Object.keys(delta?.players || {}).length} players`);
                }

                if (delta && this.game.updateWorld) {
                    // Convert delta format to match existing updateWorld signature
                    this.game.updateWorld(
                        delta.enemies,
                        delta.bullets,
                        delta.players,
                        [], // objects not in binary delta yet
                        []  // units not in binary delta yet
                    );
                    // Handle removed entities
                    if (delta.removed && delta.removed.length > 0) {
                        for (const id of delta.removed) {
                            if (this.game.bulletManager) {
                                this.game.bulletManager.removeBulletById(id);
                            }
                        }
                    }
                }
            } catch (err) {
                console.error('[BINARY] Decode error:', err);
            }
        };

        this.handlers[MessageType.PLAYER_JOIN] = (data) => {
            console.log(`Player joined: ${data.player ? data.player.id : 'unknown'}`);
            if (this.game.addPlayer && data.player) {
                this.game.addPlayer(data.player);
            }
        };
        
        this.handlers[MessageType.PLAYER_LEAVE] = (data) => {
            // FIX: Server sends 'playerId', not 'clientId'
            const playerId = data.playerId || data.clientId;
            console.log(`Player left: ${playerId}`);
            if (this.game.removePlayer) {
                this.game.removePlayer(playerId);
            }
        };
        
        this.handlers[MessageType.BULLET_CREATE] = (data) => {
            if (this.game.addBullet) {
                const bulletData = {
                    id: data.id,
                    x: data.x,
                    y: data.y,
                    vx: Math.cos(data.angle) * data.speed,
                    vy: Math.sin(data.angle) * data.speed,
                    ownerId: data.ownerId,
                    damage: data.damage || 10,
                    lifetime: data.lifetime || 3.0,
                    width: data.width || 5,
                    height: data.height || 5,
                    spriteName: data.spriteName || null,
                    worldId: data.worldId || null
                };
                const playerWorld = window.gameState?.character?.worldId;
                if (playerWorld && bulletData.worldId !== playerWorld) {
                    return; // ignore bullets with other or missing worldId
                }
                this.game.addBullet(bulletData);
            }
        };
        
        this.handlers[MessageType.COLLISION_RESULT] = (data) => {
            if (data.valid && this.game.applyCollision) {
                this.game.applyCollision(data);
            }
        };
        
        this.handlers[MessageType.CHUNK_DATA] = (data) => {
            if (this.game.setChunkData) {
                this.game.setChunkData(data.x, data.y, data.data);
            }
        };
        
        this.handlers[MessageType.CHUNK_NOT_FOUND] = (data) => {
            console.warn(`Chunk not found: (${data.chunkX}, ${data.chunkY})`);
            if (this.game.generateFallbackChunk) {
                this.game.generateFallbackChunk(data.chunkX, data.chunkY);
            }
        };
        
        // Handle pong messages for latency calculation
        this.handlers[MessageType.PONG] = (data) => {
            this.latency = Date.now() - this.lastPingTime;

            // Track rolling average (last 10 samples)
            this.latencyHistory.push(this.latency);
            if (this.latencyHistory.length > 10) {
                this.latencyHistory.shift();
            }
            this.avgLatency = Math.round(
                this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length
            );

            // Update UI ping display
            const pingEl = document.getElementById('pingDisplay');
            if (pingEl) {
                pingEl.textContent = `${this.avgLatency}ms`;
                // Color code: green < 100ms, yellow < 200ms, red >= 200ms
                pingEl.style.color = this.avgLatency < 100 ? '#0f0' :
                                     this.avgLatency < 200 ? '#ff0' : '#f00';
            }

            // Also update connection status element if it exists
            const statusEl = document.getElementById('connectionStatus');
            if (statusEl && this.connected) {
                statusEl.textContent = `Connected (${this.avgLatency}ms)`;
            }

            // Track server tick for reconciliation if present
            if (typeof this.game?.onServerTick === 'function' && data?.serverTick) {
                this.game.onServerTick(data.serverTick);
            }
        };
        
        // Chat message handler - forward to ChatPanel immediately
        this.handlers[MessageType.CHAT_MESSAGE] = (data) => {
            const messageText = data.text || data.message;

            // Directly call ChatPanel if available (most reliable)
            if (window.uiManager?.components?.chatPanel?.receiveChatFromServer) {
                window.uiManager.components.chatPanel.receiveChatFromServer(data);
            }

            // Also try game.uiManager path
            if (this.game?.uiManager?.components?.chatPanel?.receiveChatFromServer) {
                this.game.uiManager.components.chatPanel.receiveChatFromServer(data);
            }

            // Legacy: Add to UI manager if available
            if (this.game?.uiManager?.addChatMessage) {
                const messageType = data.type || 'player';
                this.game.uiManager.addChatMessage(messageText, messageType, data.sender || 'Unknown');
            }
        };

        // WebRTC signaling handlers
        this.handlers[MessageType.RTC_ANSWER] = (data) => {
            console.log('[WebRTC] Received answer from server');
            if (this.webrtc) {
                this.webrtc.handleAnswer(data);
            }
        };

        this.handlers[MessageType.RTC_ICE_CANDIDATE] = (data) => {
            console.log('[WebRTC] Received ICE candidate from server');
            if (this.webrtc) {
                this.webrtc.handleIceCandidate(data);
            }
        };

        // Handle authoritative world switch from server
        this.handlers[MessageType.WORLD_SWITCH] = (data) => {
            console.log(`[NETWORK] WORLD_SWITCH → map ${data.mapId} spawn (${data.spawnX},${data.spawnY})`);

            // ------------------------------------------------------------------
            // Reset local caches and entity lists so the new world starts clean
            // ------------------------------------------------------------------

            // 1) Drop all runtime entities from the CURRENT world so they no
            //    neither render nor collide once we teleport.
            if (this.game?.enemyManager?.cleanup) {
                this.game.enemyManager.cleanup();
            }
            if (this.game?.bulletManager?.cleanup) {
                this.game.bulletManager.cleanup();
            }

            // 2) Clear map chunk cache so no old tiles bleed into the new map.
            if (this.game?.mapManager?.clearChunks) {
                this.game.mapManager.clearChunks();
            }

            // Clear render tile caches (strategic / top-down)
            if (window.clearStrategicCache) window.clearStrategicCache();
            if (window.clearTopDownCache)   window.clearTopDownCache();

            // Flush client-side enemies & bullets so we don't see leftovers
            if (this.game?.setEnemies) this.game.setEnemies([]);
            if (this.game?.setBullets) this.game.setBullets([]);

            // Delegate to higher-level game handler if provided
            if (this.game?.onWorldSwitch) {
                this.game.onWorldSwitch(data);
            } else {
                // Fallback: re-init map and teleport the local character
                if (this.game?.initMap) {
                    this.game.initMap(data);
                }
                if (window.gameState?.character) {
                    window.gameState.character.x = data.spawnX;
                    window.gameState.character.y = data.spawnY;
                    window.gameState.character.worldId = data.mapId;
                }
            }
        };

        // Speech bubbles (taunts, player chat, etc.)
        this.handlers[MessageType.SPEECH] = (data) => {
            // Expected shape: { id, idType, text, ttlMs }
            if (window.speechBubbleManager) {
                window.speechBubbleManager.addBubble({
                    id: data.id,
                    idType: data.idType || 'player',
                    text: data.text || '',
                    ttl: data.ttlMs || 4000
                });
            }
        };

        this.handlers[MessageType.BAG_REMOVE] = (data)=>{
            const id = data?.bagId;
            if(id && this.game && typeof this.game.removeBag==='function'){
                this.game.removeBag(id);
            }
        };

        this.handlers[MessageType.PICKUP_DENIED] = (data)=>{
            const reason = data?.reason || 'denied';
            console.warn('Pickup denied:', reason);
            if(window.showToast) window.showToast(reason==='inventory_full'?'Inventory full':reason);
        };

        this.handlers[MessageType.MOVE_ITEM] = (data) => {
            if (this.game.moveItem) {
                this.game.moveItem(data.fromSlot, data.toSlot);
            }
        };

        this.handlers[MessageType.MOVE_DENIED] = (data)=>{
            if(window.showToast) window.showToast(data?.reason || 'Move denied');
        };

        // Handle player death
        this.handlers[MessageType.PLAYER_DEATH] = (data) => {
            console.log('[CLIENT] Player death received:', data);

            // Mark character as dead to hide sprite
            if (window.gameState && window.gameState.character) {
                window.gameState.character.isDead = true;
            }

            // Trigger death screen fade to black
            if (this.game.handlePlayerDeath) {
                this.game.handlePlayerDeath(data);
            }

            // Show death screen UI
            const deathScreen = document.getElementById('death-screen');
            if (deathScreen) {
                deathScreen.style.display = 'flex';
                // Trigger fade-in animation
                setTimeout(() => {
                    deathScreen.classList.add('visible');
                }, 10);
            }
        };

        // Handle player respawn response from server
        this.handlers[MessageType.PLAYER_RESPAWN] = (data) => {
            console.log('[CLIENT] Player respawn confirmed by server:', data);

            if (!window.gameState) {
                console.error('[CLIENT] Cannot respawn: gameState not found!');
                return;
            }

            // Get character ID from existing character or from client ID
            const characterId = window.gameState.character?.id || this.clientId || data.id || 'player_1';

            // Log old position if character exists
            if (window.gameState.character) {
                console.log(`[CLIENT] Destroying old character at (${window.gameState.character.x?.toFixed(2)}, ${window.gameState.character.y?.toFixed(2)})`);
            }

            // Destroy old character completely
            delete window.gameState.character;

            // Create brand new character object at respawn location using imported Player class
            const newClass = data.class || 'warrior';
            const newSpriteRow = data.spriteRow ?? 0;
            console.log(`[CLIENT] Respawning as class: ${newClass}, spriteRow: ${newSpriteRow}`);

            window.gameState.character = new Player({
                id: characterId,
                x: data.x,
                y: data.y,
                health: data.health || 100,
                maxHealth: data.maxHealth || 100,
                mana: data.mana || 100,
                maxMana: data.maxMana || 100,
                damage: data.damage || 10,
                speed: data.speed || 5,
                defense: data.defense || 0,
                isDead: false,
                class: newClass,
                spriteRow: newSpriteRow
            });

            console.log(`[CLIENT] Created NEW character at (${data.x.toFixed(2)}, ${data.y.toFixed(2)}) as ${newClass}`);

            // Update camera position to follow respawned character
            if (window.gameState.camera) {
                window.gameState.camera.position.x = data.x;
                window.gameState.camera.position.y = data.y;
                console.log(`[CLIENT] Camera repositioned to: (${data.x.toFixed(2)}, ${data.y.toFixed(2)})`);
            }

            // Hide death screen
            const deathScreen = document.getElementById('death-screen');
            if (deathScreen) {
                deathScreen.classList.remove('visible');
                setTimeout(() => {
                    deathScreen.style.display = 'none';
                }, 500);
            }
        };

        // Handle CHARACTER_SELECT - all characters dead, show class picker
        this.handlers[MessageType.CHARACTER_SELECT] = (data) => {
            console.log('[CLIENT] Character selection required:', data);

            // Show class selection overlay
            const classSelectOverlay = document.getElementById('class-select-overlay');
            if (classSelectOverlay) {
                classSelectOverlay.style.display = 'flex';
                setTimeout(() => classSelectOverlay.classList.add('visible'), 50);
            } else {
                // Fallback: redirect to menu for class selection
                console.log('[CLIENT] No class select overlay found, redirecting to menu');
                alert(`Your previous character died. Choose a new class.`);
                window.location.href = 'menu.html';
            }
        };
    }
    
    /**
     * Connect to the WebSocket server
     * @returns {Promise} Resolves when connected
     */
    connect() {
        return new Promise((resolve, reject) => {
            if (this.connected) {
                resolve();
                return;
            }
            
            if (this.connecting) {
                this.messageQueue.push({ resolve, reject });
                return;
            }
            
            this.connecting = true;
            
            try {
                // Get stored map ID for reconnection
                const storedMapId = ENABLE_MAP_ID_PERSISTENCE ? localStorage.getItem('currentMapId') : null;
                let serverUrl = this.serverUrl;
                
                // Include map ID in URL if persistence is enabled and an ID exists
                if (ENABLE_MAP_ID_PERSISTENCE && storedMapId) {
                    console.log(`Found stored map ID: ${storedMapId}`);
                    const separator = this.serverUrl.includes('?') ? '&' : '?';
                    serverUrl = `${this.serverUrl}${separator}mapId=${storedMapId}`;
                    console.log(`Connecting with map ID: ${serverUrl}`);
                }
                
                console.log(`Connecting to server: ${serverUrl}`);
                this.socket = new WebSocket(serverUrl);
                
                // Set binary type for ArrayBuffer data
                this.socket.binaryType = 'arraybuffer';
                
                this.socket.onopen = () => {
                    console.log('Connected to server');
                    this.connected = true;
                    this.connecting = false;
                    this.reconnectAttempts = 0;

                    // With the WebSocket established we can now attach all message handlers
                    this.setupMessageHandlers();

                    // Send handshake
                    this.sendHandshake();

                    // Start ping and send immediate first ping for lag compensation
                    this.startPing();
                    this.sendPing(); // Immediate first ping

                    // Drain message queue
                    this.drainMessageQueue();

                    // Initialize WebRTC for UDP-like transport (after short delay to ensure handshake completes)
                    if (this.useWebRTC) {
                        setTimeout(() => this.initializeWebRTC(), 500);
                    }

                    // Initialize WebTransport for QUIC/UDP transport
                    if (this.useWebTransport) {
                        setTimeout(() => this.initializeWebTransport(), 1000);
                    }

                    resolve();
                };
                
                this.socket.onmessage = (event) => {
                    // RAW MESSAGE DIAGNOSTIC: Only log when bullets are present
                    try {
                        const packet = BinaryPacket.decode(event.data);
                        if (packet.type === MessageType.WORLD_UPDATE && packet.data?.bullets?.length > 0) {
                            const bulletCount = packet.data.bullets.length;
                            // console.error(`📨 [RAW WS] WORLD_UPDATE with ${bulletCount} bullets. First: ID=${packet.data.bullets[0].id}, Pos=(${packet.data.bullets[0].x?.toFixed(2)}, ${packet.data.bullets[0].y?.toFixed(2)})`);
                        }
                    } catch (e) {
                        // Ignore decode errors
                    }

                    this.handleMessage(event.data);
                };
                
                this.socket.onclose = (event) => {
                    console.log(`Disconnected from server: ${event.code} - ${event.reason}`);
                    this.connected = false;
                    this.connecting = false;
                    this.stopPing();
                    this.attemptReconnect();
                    reject(new Error(`Disconnected from server: ${event.code} - ${event.reason}`));
                };
                
                this.socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.connecting = false;
                    reject(error);
                };
            } catch (error) {
                console.error('Failed to connect to server:', error);
                this.connecting = false;
                reject(error);
            }
        });
    }
    
    /**
     * Attempt to reconnect to the server
     */
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Maximum reconnect attempts reached');
            if (this.game.handleDisconnect) {
                this.game.handleDisconnect();
            }
            return;
        }
        
        this.reconnectAttempts++;
        
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        
        setTimeout(() => {
            this.connect().catch((error) => {
                console.error("Reconnect attempt failed:", error.message);
            });
        }, this.reconnectDelay * this.reconnectAttempts);
    }
    
    /**
     * Start the ping interval
     * @private
     */
    startPing() {
        this.stopPing();
        this.pingInterval = setInterval(() => {
            this.sendPing();
        }, 1000); // Every 1 second for lag compensation
    }
    
    /**
     * Stop the ping interval
     * @private
     */
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    /**
     * Send a ping message
     * @private
     */
    sendPing() {
        this.lastPingTime = Date.now();
        this.send(MessageType.PING, { time: this.lastPingTime });
    }
    
    /**
     * Send handshake message
     * @private
     */
    sendHandshake() {
        this.send(MessageType.HANDSHAKE, {
            clientTime: Date.now(),
            screenWidth: window.innerWidth,
            screenHeight: window.innerHeight
        });
    }
    
    /**
     * Send queued messages after connection is established
     */
    drainMessageQueue() {
        console.log(`[NetworkManager] Draining message queue, ${this.messageQueue.length} messages queued`);
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (message.resolve) {
                message.resolve();
            } else {
                console.log(`[NetworkManager] Sending queued message type ${message.type}`);
                this.send(message.type, message.data);
            }
        }
    }
    
    /**
     * Handle an incoming WebSocket message
     * @param {ArrayBuffer} data - Binary message data
     */
    handleMessage(data) {
        try {
            // Decode binary packet
            const packet = BinaryPacket.decode(data);
            const { type, data: messageData } = packet;
            
            // Diagnostic: Always log PLAYER_LIST messages raw format for debugging
            // This will help understand exactly what the server is sending
            if (type === MessageType.PLAYER_LIST) {
                //console.log('=== RAW PLAYER_LIST MESSAGE ===');
                //console.log('Raw message data:', messageData);
                //console.log('Type of data:', typeof messageData);
                //console.log('Keys:', Object.keys(messageData));
                //console.log('Has players property:', messageData.hasOwnProperty('players'));
                //console.log('================================');
                
                // Store the last player list message for diagnostics
                this.lastPlayerListMessage = messageData;
            }
            
            // Update server time offset
            if (messageData.timestamp) {
                this.lastServerTime = messageData.timestamp;
                this.serverTimeOffset = Date.now() - messageData.timestamp;
            }
            
            // Enhanced logging for important message types
            switch (type) {
                case MessageType.WORLD_UPDATE:
                    if (messageData.enemies && messageData.enemies.length > 0) {
                        console.log(`World update: ${messageData.enemies.length} enemies, ${messageData.bullets ? messageData.bullets.length : 0} bullets, ${Object.keys(messageData.players || {}).length} players`);
                    }
                    break;
                case MessageType.BULLET_LIST:
                    console.log(`Received bullets list: ${messageData.bullets ? messageData.bullets.length : 0} bullets`);
                    break;
                case MessageType.ENEMY_LIST:
                    console.log(`Received enemies list: ${messageData.enemies ? messageData.enemies.length : 0} enemies`);
                    if (messageData.enemies && messageData.enemies.length > 0) {
                        console.log(`Enemy sample: ID=${messageData.enemies[0].id}, Pos=(${messageData.enemies[0].x.toFixed(2)}, ${messageData.enemies[0].y.toFixed(2)}), Health=${messageData.enemies[0].health}`);
                    }
                    break;
                case MessageType.COLLISION_RESULT:
                    console.log(`Collision result: valid=${messageData.valid}, targetId=${messageData.targetId}, damage=${messageData.damage}, killed=${messageData.killed}`);
                    break;
            }
            
            // Call the handler for this message type
            if (this.handlers[type]) {
                this.handlers[type](messageData);
            } else {
                console.warn(`No handler for message type: ${type}`);
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    }
    
    /**
     * Send a message to the server
     * @param {number} type - Message type
     * @param {Object} data - Message data
     */
    send(type, data) {
        if (!this.connected) {
            this.messageQueue.push({ type, data });
            console.log(`Queued message type ${type} for later sending`);
            return false;
        }

        try {
            // Try WebTransport for UDP-suitable messages (preferred)
            if (this.webtransport?.isReady && UDP_MESSAGES.has(type)) {
                // Use true binary encoding for PLAYER_UPDATE and BULLET_CREATE
                // These are the highest-frequency messages, binary saves ~85% bandwidth
                let binaryPayload = null;

                if (type === MessageType.PLAYER_UPDATE) {
                    binaryPayload = encodePlayerUpdate(
                        data.x, data.y,
                        data.vx || 0, data.vy || 0,
                        data.angle || data.rotation || 0
                    );
                    // Debug: Log binary usage (first 10, then every 5 sec)
                    if (!this._binaryCount) this._binaryCount = 0;
                    this._binaryCount++;
                    if (this._binaryCount <= 10 || (!this._lastBinaryLog || Date.now() - this._lastBinaryLog > 5000)) {
                        this._lastBinaryLog = Date.now();
                        console.log(`%c[BINARY] PLAYER_UPDATE #${this._binaryCount} via WebTransport: ${binaryPayload.byteLength} bytes (vs ~60 JSON)`, 'color: #0f0; font-weight: bold');
                    }
                } else if (type === MessageType.BULLET_CREATE) {
                    binaryPayload = encodeBulletCreate(
                        data.x, data.y,
                        data.angle,
                        data.speed,
                        data.damage || 10
                    );
                    console.log(`[BINARY] BULLET_CREATE sent via WebTransport: ${binaryPayload.byteLength} bytes (vs ~80 JSON)`);
                } else if (type === MessageType.PING) {
                    binaryPayload = encodePing(data.time || Date.now());
                }

                if (binaryPayload) {
                    // Send raw binary via WebTransport (type is embedded in payload)
                    const sentViaWT = this.webtransport.sendBinary(binaryPayload);
                    if (sentViaWT) {
                        return true;
                    }
                } else {
                    // Fall back to JSON-based send for other message types
                    const sentViaWT = this.webtransport.send(type, data);
                    if (sentViaWT) {
                        return true;
                    }
                }
                // Fall through to WebRTC or WebSocket if WebTransport failed
            }

            // Try WebRTC DataChannel for UDP-suitable messages (fallback)
            if (this.webrtc?.isReady && UDP_MESSAGES.has(type)) {
                const sentViaRTC = this.webrtc.send(type, data);
                if (sentViaRTC) {
                    return true; // Successfully sent via WebRTC
                }
                // Fall through to WebSocket if WebRTC failed
            }

            // Encode binary packet for WebSocket
            const packet = BinaryPacket.encode(type, data);

            // Debug: Log when high-frequency messages fall back to WebSocket
            if ((type === MessageType.PLAYER_UPDATE || type === MessageType.BULLET_CREATE) && !this._wsFallbackLogged) {
                this._wsFallbackLogged = true;
                console.warn('%c[FALLBACK] Using WebSocket (JSON) for game messages - WebTransport not connected', 'color: #f90; font-weight: bold');
            }

            // Send packet via WebSocket (TCP)
            this.socket.send(packet);
            return true;
        } catch (error) {
            console.error('Error sending message:', error);
            return false;
        }
    }
    
    /**
     * Send player update to server
     * @param {Object} playerData - Player state data
     */
    sendPlayerUpdate(playerData) {
        // Block sending updates if player is dead
        if (window.gameState && window.gameState.character && window.gameState.character.isDead) {
            console.log('[CLIENT] 🚫 Blocked sendPlayerUpdate - character is dead');
            return false;
        }

        if (playerData) {
            //console.log(`Sending player update: pos=(${playerData.x ? playerData.x.toFixed(2) : 'undefined'}, ${playerData.y ? playerData.y.toFixed(2) : 'undefined'}), angle=${playerData.angle ? playerData.angle.toFixed(2) : 'undefined'}`);
        }
        return this.send(MessageType.PLAYER_UPDATE, playerData);
    }
    
    /**
     * Send shoot event to server
     * @param {Object} bulletData - Bullet data
     */
    sendShoot(bulletData) {
        // Block shooting if player is dead
        if (window.gameState && window.gameState.character && window.gameState.character.isDead) {
            return false;
        }

        // NOTE: Local bullet is already created by gameManager.firePlayerBullet() before calling this
        // Do NOT create another local bullet here - that causes duplicates

        return this.send(MessageType.BULLET_CREATE, bulletData);
    }
    
    /**
     * Send collision to server
     * @param {Object} collisionData - Collision data
     */
    sendCollision(collisionData) {
        return this.send(MessageType.COLLISION, collisionData);
    }

    /**
     * Request to pick up an item from a loot bag
     * @param {string} bagId
     * @param {number} itemId – item instance ID
     * @param {number} slot – preferred inventory slot (optional)
     */
    sendPickupItem(bagId, itemId, slot=null) {
      this.send(MessageType.PICKUP_ITEM, { bagId, itemId, slot });
    }
    
    /**
     * Request a map chunk from the server
     * @param {number} chunkX - Chunk X coordinate
     * @param {number} chunkY - Chunk Y coordinate
     */
    requestChunk(chunkX, chunkY) {
        return this.send(MessageType.CHUNK_REQUEST, { chunkX, chunkY });
    }
    
    /**
     * Register a message handler
     * @param {number} type - Message type
     * @param {Function} handler - Message handler
     */
    on(type, handler) {
        if (typeof handler === 'function') {
            this.handlers[type] = handler;
        }
    }
    
    /**
     * Get the estimated server time
     * @returns {number} Server time in milliseconds
     */
    getServerTime() {
        return Date.now() - this.serverTimeOffset;
    }
    
    /**
     * Check if connected to server
     * @returns {boolean} True if connected
     */
    isConnected() {
        return this.connected;
    }
    
    /**
     * Get the client ID
     * @returns {string} Client ID
     */
    getClientId() {
        return this.clientId;
    }
    
    /**
     * Disconnect from the server
     */
    disconnect() {
        this.stopPing();
        
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        
        this.connected = false;
        this.connecting = false;
    }
    
    /**
     * Request a specific map by ID
     * @param {string} mapId - Map ID to request
     */
    requestMap(mapId) {
        if (!this.isConnected()) {
            console.warn("Cannot request map: not connected to server");
            return;
        }
        
        console.log(`Requesting map: ${mapId}`);
        this.send(MessageType.MAP_REQUEST, {
            mapId
        });
    }
    
    /**
     * Request the current player list from the server
     * This is useful for diagnostics
     */
    requestPlayerList() {
        console.log("Sending player list request to server");
        return this.send(MessageType.PLAYER_LIST_REQUEST, {
            clientId: this.clientId,
            timestamp: Date.now()
        });
    }
    
    /**
     * Send a chat message to the server
     * @param {Object} chatData - Chat message data
     * @returns {boolean} True if sent successfully
     */
    sendChatMessage(chatData) {
        if (!this.isConnected()) {
            console.warn('Cannot send chat message: Not connected');
            return false;
        }

        try {
            // Make sure socket exists before using it
            if (!this.socket) {
                console.error('Cannot send chat message: Socket not initialized');
                return false;
            }

            // Use MessageType.PLAYER_TEXT - server expects { text: "..." }
            return this.send(MessageType.PLAYER_TEXT, {
                text: chatData.message,
                channel: chatData.channel || 'All',
                timestamp: Date.now()
            });
        } catch (error) {
            console.error('Error sending chat message:', error);
            return false;
        }
    }

    /**
     * Notify server that the player has pressed the portal-interact key.
     */
    sendPortalEnter() {
        return this.send(MessageType.PORTAL_ENTER, { ts: Date.now() });
    }

    /**
     * Move/reorder item within inventory
     */
    sendMoveItem(fromSlot, toSlot){
        this.send(MessageType.MOVE_ITEM,{fromSlot,toSlot});
    }

    /**
     * Request player respawn after death
     * @param {string} newClass - Optional new class to respawn as
     */
    sendRespawn(newClass = null) {
        console.log('[NETWORK] Sending respawn request to server, class:', newClass);
        return this.send(MessageType.PLAYER_RESPAWN, {
            timestamp: Date.now(),
            class: newClass  // Server will use this to change class if provided
        });
    }

    /**
     * Initialize WebRTC for UDP-like transport
     */
    async initializeWebRTC() {
        if (!this.useWebRTC) {
            console.log('[WebRTC] Disabled, using WebSocket only');
            return;
        }

        try {
            console.log('[WebRTC] Initializing...');
            this.webrtc = new WebRTCManager(this);
            const success = await this.webrtc.initialize();

            if (success) {
                console.log('[WebRTC] Initialization started, waiting for connection...');
            } else {
                console.warn('[WebRTC] Initialization failed, using WebSocket only');
                this.webrtc = null;
            }
        } catch (error) {
            console.error('[WebRTC] Error during initialization:', error);
            this.webrtc = null;
        }
    }

    /**
     * Get WebRTC connection stats
     */
    getWebRTCStats() {
        if (this.webrtc) {
            return this.webrtc.getStats();
        }
        return { isReady: false, reason: 'WebRTC not initialized' };
    }

    /**
     * Check if WebRTC DataChannel is ready
     */
    isUDPReady() {
        return this.webtransport?.isReady || this.webrtc?.isReady || false;
    }

    /**
     * Initialize WebTransport for UDP-like transport
     * @param {string} url - Optional WebTransport server URL
     */
    async initializeWebTransport(url = null) {
        if (!this.useWebTransport) {
            console.log('[WebTransport] Disabled, using WebSocket only');
            return;
        }

        // Check browser support
        if (typeof WebTransport === 'undefined') {
            console.warn('[WebTransport] Not supported in this browser, falling back to WebSocket');
            return;
        }

        try {
            console.log('[WebTransport] Initializing...');
            this.webtransport = new WebTransportManager(this);

            // Determine WebTransport URL
            // Priority: 1. Passed URL, 2. Config, 3. Auto-detect from WebSocket URL
            let wtUrl = url || this.webTransportUrl;

            if (!wtUrl) {
                // Auto-detect: replace wss:// with https:// and use PlayIt UDP port
                // e.g., wss://eternalconquests.com -> https://quic.eternalconquests.com:10615/game
                // PlayIt tunnels external port 10615 -> local port 4433
                const wsUrl = new URL(this.serverUrl.replace('ws://', 'http://').replace('wss://', 'https://'));
                wtUrl = `https://quic.${wsUrl.hostname}:10615/game`;
                console.log(`[WebTransport] Auto-detected URL: ${wtUrl}`);
            }

            const success = await this.webtransport.initialize(wtUrl);

            if (success) {
                console.log('[WebTransport] Connected successfully!');
            } else {
                console.warn('[WebTransport] Connection failed, using WebSocket only');
                this.webtransport = null;
            }
        } catch (error) {
            console.error('[WebTransport] Error during initialization:', error);
            this.webtransport = null;
        }
    }

    /**
     * Get WebTransport connection stats
     */
    getWebTransportStats() {
        if (this.webtransport) {
            return this.webtransport.getStats();
        }
        return { isReady: false, reason: 'WebTransport not initialized' };
    }

    /**
     * Get combined UDP transport stats
     */
    getUDPStats() {
        return {
            webtransport: this.getWebTransportStats(),
            webrtc: this.getWebRTCStats(),
            activeTransport: this.webtransport?.isReady ? 'WebTransport' :
                             this.webrtc?.isReady ? 'WebRTC' : 'WebSocket'
        };
    }
}

// Remove in-file duplicate protocol definitions in favor of shared module
