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
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000; // ms
        this.timeOffset = 0; // Client-server time offset
        
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
        
        // Setup message handlers
        this.setupMessageHandlers();
        
        console.log("ClientNetworkManager initialized with server URL:", serverUrl);
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
            // Check if data is directly the players object or has a nested 'players' property
            // This handles both formats from the server
            let playersData = data;
            
            // If the data has a 'players' property and it's an object, use that
            if (data.players && typeof data.players === 'object') {
                playersData = data.players;
                throttledLog('player-list-format', '[NETWORK] Found nested players property in PLAYER_LIST message');
            }
            
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
        
        this.handlers[MessageType.WORLD_UPDATE] = (data) => {
            // Only log occasionally to reduce spam
            throttledLog('world-update', `World update received`, null, 3000);
            
            if (this.game.updateWorld) {
                // Check if players is nested inside a 'players' property (from server inconsistency)
                const players = data.players?.players || data.players;
                this.game.updateWorld(data.enemies, data.bullets, players);
            }
        };
        
        this.handlers[MessageType.PLAYER_JOIN] = (data) => {
            console.log(`Player joined: ${data.player ? data.player.id : 'unknown'}`);
            if (this.game.addPlayer && data.player) {
                this.game.addPlayer(data.player);
            }
        };
        
        this.handlers[MessageType.PLAYER_LEAVE] = (data) => {
            console.log(`Player left: ${data.clientId}`);
            if (this.game.removePlayer) {
                this.game.removePlayer(data.clientId);
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
                    spriteName: data.spriteName || null
                };
                this.game.addBullet(bulletData);
            }
        };
        
        this.handlers[MessageType.COLLISION_RESULT] = (data) => {
            if (data.valid && this.game.applyCollision) {
                this.game.applyCollision(data);
            }
        };
        
        this.handlers[MessageType.CHUNK_DATA] = (data) => {
            console.log(`Received chunk data for (${data.x}, ${data.y})`);
            
            // Create a simple global object to store map data for debugging
            if (!window.mapDebug) {
                window.mapDebug = {
                    mapId: localStorage.getItem('currentMapId') || 'unknown',
                    chunks: {}
                };
                console.log("Map debug object created. Access it via window.mapDebug in the console");
            }
            
            // Store chunk data in the global object
            const chunkKey = `${data.x},${data.y}`;
            window.mapDebug.chunks[chunkKey] = data.data;
            
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
            const latency = Date.now() - this.lastPingTime;
            console.log(`Server ping: ${latency}ms`);
        };
        
        // Add chat message handler only if socket.on is available
        try {
            // Use the CHAT_MESSAGE message type constant (90) instead of 'chat' string
            this.handlers[MessageType.CHAT_MESSAGE] = (data) => {
                console.log('Received chat message:', data);
                
                // Call any registered chat handlers
                if (this.handlers.chat) {
                    this.handlers.chat(data);
                }
                
                // Add message to UI if available
                if (this.game && this.game.uiManager) {
                    // Format message properly for UI manager
                    const messageType = data.type || (data.channel === 'system' ? 'system' : 'player');
                    this.game.uiManager.addChatMessage(
                        data.message,
                        messageType,
                        data.sender || 'Unknown'
                    );
                }
            };
            
            console.log('Chat message handler registered for type:', MessageType.CHAT_MESSAGE);
        } catch (error) {
            console.error('Error setting up chat message handler:', error);
        }

        // Handle authoritative world switch from server
        this.handlers[MessageType.WORLD_SWITCH] = (data) => {
            console.log(`[NETWORK] WORLD_SWITCH → map ${data.mapId} spawn (${data.spawnX},${data.spawnY})`);
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
                }
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
                    
                    // Re-setup message handlers now that socket is initialized
                    this.setupMessageHandlers();
                    
                    // Send handshake
                    this.sendHandshake();
                    
                    // Start ping
                    this.startPing();
                    
                    // Drain message queue
                    this.drainMessageQueue();
                    
                    resolve();
                };
                
                this.socket.onmessage = (event) => {
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
        }, 30000); // Every 30 seconds
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
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            if (message.resolve) {
                message.resolve();
            } else {
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
            // Encode binary packet
            const packet = BinaryPacket.encode(type, data);
            
            // Send packet
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
        console.log(`Sending shoot event: pos=(${bulletData.x.toFixed(2)}, ${bulletData.y.toFixed(2)}), angle=${bulletData.angle.toFixed(2)}, speed=${bulletData.speed}`);
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
            
            // Use MessageType.CHAT_MESSAGE (90) instead of the string 'chat'
            // This ensures correct binary packet encoding
            return this.send(MessageType.CHAT_MESSAGE, {
                message: chatData.message,
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
}

/**
* BinaryPacket - Utility for efficient binary packet encoding/decoding
*/
export class BinaryPacket {
    /**
     * Create a binary packet with a specific message type
     * @param {number} type - Message type ID
     * @param {Object} data - Message data
     * @returns {ArrayBuffer} Binary packet
     */
    static encode(type, data) {
        // Convert data to JSON string for flexibility
        const jsonStr = JSON.stringify(data);
        const jsonBytes = new TextEncoder().encode(jsonStr);
        
        // Create packet: [1 byte type][4 byte length][jsonBytes]
        const packet = new ArrayBuffer(5 + jsonBytes.byteLength);
        const view = new DataView(packet);
        
        // Write type and length
        view.setUint8(0, type);
        view.setUint32(1, jsonBytes.byteLength, true); // Little-endian
        
        // Write JSON data
        new Uint8Array(packet, 5).set(jsonBytes);
        
        return packet;
    }
    
    /**
     * Decode a binary packet
     * @param {ArrayBuffer} packet - Binary packet
     * @returns {Object} Decoded packet {type, data}
     */
    static decode(packet) {
        const view = new DataView(packet);
        
        // Read type and length
        const type = view.getUint8(0);
        const length = view.getUint32(1, true); // Little-endian
        
        // Read JSON data
        const jsonBytes = new Uint8Array(packet, 5, length);
        const jsonStr = new TextDecoder().decode(jsonBytes);
        
        // Parse JSON data
        try {
            const data = JSON.parse(jsonStr);
            return { type, data };
        } catch (error) {
            console.error('Error parsing packet JSON:', error);
            return { type, data: {} };
        }
    }
}

/**
* Message type constants
*/
export const MessageType = {
    // System messages
    HEARTBEAT: 0,
    
    // Connection messages
    HANDSHAKE: 1,
    HANDSHAKE_ACK: 2,
    PING: 3,
    PONG: 4,
    
    // Game state messages
    PLAYER_JOIN: 10,
    PLAYER_LEAVE: 11,
    PLAYER_UPDATE: 12,
    PLAYER_LIST: 13,
    
    // Entity messages
    ENEMY_LIST: 20,
    ENEMY_UPDATE: 21,
    ENEMY_DEATH: 22,
    
    // Bullet messages
    BULLET_CREATE: 30,
    BULLET_LIST: 31,
    BULLET_REMOVE: 32,
    
    // Collision messages
    COLLISION: 40,
    COLLISION_RESULT: 41,
    
    // Map messages
    MAP_INFO: 50,
    CHUNK_REQUEST: 51,
    CHUNK_DATA: 52,
    CHUNK_NOT_FOUND: 53,
    
    // World update
    WORLD_UPDATE: 60,
    
    // Map request
    MAP_REQUEST: 70,
    
    // Player list request
    PLAYER_LIST_REQUEST: 80,
    
    // Chat message
    CHAT_MESSAGE: 90,

    // Portal interaction
    PORTAL_ENTER: 54,      // client -> server
    WORLD_SWITCH: 55       // server -> client
};