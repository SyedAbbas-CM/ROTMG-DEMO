// public/src/map/ClientMapManager.js

import { Tile } from './tile.js';
import { TILE_IDS, CHUNK_SIZE } from '../constants/constants.js';
import { gameState } from '../game/gamestate.js';

/**
 * ClientMapManager - Handles loading, caching, and rendering map data from server
 */
export class ClientMapManager {
    /**
     * Create a new client map manager
     * @param {Object} options - Manager options
     * @param {NetworkManager} options.networkManager - Network manager for map data requests
     */
    constructor(options = {}) {
        this.networkManager = options.networkManager;
        this.activeMapId = null;
        this.mapMetadata = null;
        this.chunks = new Map(); // Chunk cache: "x,y" -> chunk data
        this.tileSize = 12; // Default tile size
        this.chunkSize = 16; // Default chunk size
        this.width = 0;
        this.height = 0;
        this.visibleChunks = []; // Currently visible chunks
        this.pendingChunks = new Set(); // Chunks we're currently requesting
        this.maxCachedChunks = 100; // Maximum chunks to keep in memory
        this.chunkLoadDistance = 2; // How many chunks to load around player
        
        // CRITICAL: Default to false to use server's map data
        this.proceduralEnabled = false;
        
        this.fallbackTileTypes = {
            [TILE_IDS.FLOOR]: 'floor',
            [TILE_IDS.WALL]: 'wall',
            [TILE_IDS.OBSTACLE]: 'obstacle',
            [TILE_IDS.WATER]: 'water',
            [TILE_IDS.MOUNTAIN]: 'mountain'
        };
        
        // LRU (Least Recently Used) tracking for chunk cache
        this.chunkLastAccessed = new Map(); // "x,y" -> timestamp
        
        // Event listeners
        this.eventListeners = {};
        
        console.log("ClientMapManager initialized, procedural generation disabled");
    }
    
    /**
     * Initialize the map with metadata from server
     * @param {Object} data - Map metadata
     */
    initMap(data) {
        this.activeMapId = data.mapId;
        this.mapMetadata = data;
        this.tileSize = data.tileSize || this.tileSize;
        this.chunkSize = data.chunkSize || this.chunkSize;
        this.width = data.width || 0;
        this.height = data.height || 0;
        
        console.log(`Map initialized: ${this.activeMapId} (${this.width}x${this.height})`);
        console.log(`Map properties: tileSize=${this.tileSize}, chunkSize=${this.chunkSize}`);
        
        // Clear existing chunks
        this.chunks.clear();
        this.chunkLastAccessed.clear();
        this.pendingChunks.clear();
        
        // CRITICAL: Always disable procedural generation
        this.proceduralEnabled = false;
        
        // Immediately request chunks around player position
        if (gameState && gameState.character) {
            this.updateVisibleChunks(gameState.character.x, gameState.character.y);
        }
        
        // Dispatch event
        this.dispatchEvent('mapinitialized', { mapId: this.activeMapId });
    }
    
    /**
     * Set chunk data received from server
     * @param {number} chunkX - Chunk X coordinate
     * @param {number} chunkY - Chunk Y coordinate
     * @param {Object} chunkData - Chunk data from server
     */
    setChunkData(chunkX, chunkY, chunkData) {
        const key = `${chunkX},${chunkY}`;
        
        // Track time for performance measurement
        const startTime = performance.now();
        
        // Process chunk data to match our format
        const processedChunk = this.processChunkData(chunkData);
        
        // Store chunk
        this.chunks.set(key, processedChunk);
        this.chunkLastAccessed.set(key, Date.now());
        
        // Remove from pending
        this.pendingChunks.delete(key);
        
        // Calculate processing time
        const processingTime = (performance.now() - startTime).toFixed(2);
        
        // Log more detailed chunk info but keep it infrequent to avoid spam
        if (Math.random() < 0.2) { // Only log 20% of chunks
            console.log(`[MapManager] Received chunk at (${chunkX}, ${chunkY}): ${processedChunk.length} rows x ${processedChunk[0]?.length || 0} cols (processed in ${processingTime}ms)`);
            
            // Count tile types for debugging
            const tileCounts = {};
            if (processedChunk && Array.isArray(processedChunk)) {
                for (const row of processedChunk) {
                    for (const tile of row) {
                        const type = tile?.type || 'unknown';
                        tileCounts[type] = (tileCounts[type] || 0) + 1;
                    }
                }
                // Log tile distribution
                console.log(`[MapManager] Chunk ${key} tile distribution:`, tileCounts);
            }
        }
        
        // Trim cache if needed
        this.trimChunkCache();
        
        // Dispatch event
        this.dispatchEvent('chunkloaded', { chunkX, chunkY });
    }
    
    /**
     * Process chunk data from server into our format
     * @param {Object} chunkData - Chunk data from server
     * @returns {Array} Processed chunk data
     */
    processChunkData(chunkData) {
        // Log the structure of the incoming data to help diagnose issues
        console.log(`Processing chunk data: type=${typeof chunkData}`, 
                    chunkData ? 
                    `keys=${Object.keys(chunkData).join(',')}` : 
                    'chunkData is null/undefined');
                    
        // If the chunk data is already in the right format, return it
        if (Array.isArray(chunkData)) {
            console.log(`Chunk data is already an array with ${chunkData.length} rows`);
            return chunkData;
        }
        
        // Convert from server format to client format
        const processedData = [];
        
        // Different possible formats the server might send
        let tilesArray = null;
        
        // Try to extract tiles array from different possible formats
        if (chunkData && typeof chunkData === 'object') {
            if (chunkData.tiles && Array.isArray(chunkData.tiles)) {
                tilesArray = chunkData.tiles;
                console.log(`Found tiles array in chunkData.tiles with ${tilesArray.length} rows`);
            } else if (chunkData.data && Array.isArray(chunkData.data)) {
                tilesArray = chunkData.data;
                console.log(`Found tiles array in chunkData.data with ${tilesArray.length} rows`);
            } else if (Array.isArray(chunkData.data?.tiles)) {
                tilesArray = chunkData.data.tiles;
                console.log(`Found tiles array in chunkData.data.tiles with ${tilesArray.length} rows`);
            }
        }
        
        // Process tiles array if it exists
        if (tilesArray && Array.isArray(tilesArray)) {
            for (let y = 0; y < tilesArray.length; y++) {
                const row = [];
                for (let x = 0; x < tilesArray[y].length; x++) {
                    const tileData = tilesArray[y][x];
                    let tileType, tileHeight;
                    
                    // Handle different possible formats
                    if (typeof tileData === 'number') {
                        tileType = tileData;
                        tileHeight = 0;
                    } else if (tileData && typeof tileData === 'object') {
                        tileType = tileData.type;
                        tileHeight = tileData.height || 0;
                    } else {
                        tileType = TILE_IDS.FLOOR; // Default
                        tileHeight = 0;
                    }
                    
                    // Create tile instance
                    row.push(new Tile(tileType, tileHeight));
                }
                processedData.push(row);
            }
            
            console.log(`Processed chunk data: ${processedData.length} rows x ${processedData[0]?.length || 0} columns`);
        } else {
            console.warn('No valid tiles array found in chunk data, creating default floor tiles');
            
            // Create default chunk data
            for (let y = 0; y < this.chunkSize; y++) {
                const row = [];
                for (let x = 0; x < this.chunkSize; x++) {
                    row.push(new Tile(TILE_IDS.FLOOR, 0));
                }
                processedData.push(row);
            }
            
            console.log(`Created default chunk data: ${processedData.length} rows x ${processedData[0].length} columns`);
        }
        
        return processedData;
    }
    
    /**
     * Update visible chunks based on player position
     * @param {number} playerX - Player X position in world coordinates
     * @param {number} playerY - Player Y position in world coordinates
     */
    updateVisibleChunks(playerX, playerY) {
        if (!this.networkManager) {
            console.warn("Cannot update visible chunks: network manager not available");
            return;
        }
        
        // Convert player position to tile coordinates first
        const playerTileX = Math.floor(playerX / this.tileSize);
        const playerTileY = Math.floor(playerY / this.tileSize);
        
        // Log map boundaries for debugging
        if (Math.random() < 0.01) { // Only log occasionally
            console.log(`Map boundaries: width=${this.width}, height=${this.height}`);
            console.log(`Player position in tiles: (${playerTileX}, ${playerTileY})`);
        }
        
        // Ensure player stays within map bounds (important!)
        if (this.width > 0 && this.height > 0) {
            if (playerTileX < 0 || playerTileX >= this.width || playerTileY < 0 || playerTileY >= this.height) {
                console.warn(`Player outside map bounds: (${playerTileX}, ${playerTileY}) - Map size: ${this.width}x${this.height}`);
                // Don't update chunks for out-of-bounds player
                return;
            }
        }
        
        // Convert player position to chunk coordinates (integers)
        const centerChunkX = Math.floor(playerTileX / this.chunkSize);
        const centerChunkY = Math.floor(playerTileY / this.chunkSize);
        
        // Get chunks in view distance
        const newVisibleChunks = [];
        const chunksRequested = []; // Track new chunk requests for logging
        
        // Determine valid chunk range based on map size
        const maxChunkX = this.width > 0 ? Math.ceil(this.width / this.chunkSize) - 1 : Infinity;
        const maxChunkY = this.height > 0 ? Math.ceil(this.height / this.chunkSize) - 1 : Infinity;
        
        for (let dy = -this.chunkLoadDistance; dy <= this.chunkLoadDistance; dy++) {
            for (let dx = -this.chunkLoadDistance; dx <= this.chunkLoadDistance; dx++) {
                const chunkX = centerChunkX + dx;
                const chunkY = centerChunkY + dy;
                
                // Skip if out of map bounds
                if (chunkX < 0 || chunkY < 0 || chunkX > maxChunkX || chunkY > maxChunkY) {
                    continue;
                }
                
                // Calculate chunk start in tile coordinates
                const chunkStartX = chunkX * this.chunkSize;
                const chunkStartY = chunkY * this.chunkSize;
                
                // Skip if entire chunk is outside map bounds
                if (chunkStartX >= this.width || chunkStartY >= this.height) {
                    continue;
                }
                
                const key = `${chunkX},${chunkY}`;
                newVisibleChunks.push({ x: chunkX, y: chunkY, key });
                
                // Update last accessed time
                if (this.chunks.has(key)) {
                    this.chunkLastAccessed.set(key, Date.now());
                }
                // Request chunk if not already loaded or pending
                else if (!this.pendingChunks.has(key)) {
                    this.pendingChunks.add(key);
                    try {
                        this.networkManager.requestChunk(chunkX, chunkY);
                        chunksRequested.push(`(${chunkX},${chunkY})`);
                    } catch (error) {
                        console.error(`Error requesting chunk (${chunkX}, ${chunkY}):`, error);
                        this.pendingChunks.delete(key);
                    }
                }
            }
        }
        
        // Log chunk requests in a single message to reduce console spam
        if (chunksRequested.length > 0) {
            console.log(`[MapManager] Requested ${chunksRequested.length} new chunks: ${chunksRequested.join(', ')}`);
        }
        
        // Update visible chunks list
        this.visibleChunks = newVisibleChunks;
    }
    
    /**
     * Update visible chunks without making network requests
     * Use this to prevent flickering in strategic view
     * @param {number} playerX - Player X position
     * @param {number} playerY - Player Y position
     * @param {number} [customChunkDistance] - Optional chunk load distance
     */
    updateVisibleChunksLocally(playerX, playerY, customChunkDistance) {
        // Log local update
        console.log(`[MapManager] Updating visible chunks LOCALLY around (${playerX.toFixed(1)}, ${playerY.toFixed(1)})`);
        
        // Convert player position to chunk coordinates
        const centerChunkX = Math.floor(playerX / (this.tileSize * this.chunkSize));
        const centerChunkY = Math.floor(playerY / (this.tileSize * this.chunkSize));
        
        // Use custom distance if provided, otherwise use default
        const effectiveChunkLoadDistance = customChunkDistance !== undefined ? 
            customChunkDistance : this.chunkLoadDistance;
        
        console.log(`[MapManager] Local update center chunk: (${centerChunkX}, ${centerChunkY}), distance: ${effectiveChunkLoadDistance}`);
        
        // Update visible chunks list without requesting any new chunks
        const newVisibleChunks = [];
        const missingChunks = []; // Track chunks that would be loaded if we were making network requests
        
        // Build list of currently visible chunks
        for (let dy = -effectiveChunkLoadDistance; dy <= effectiveChunkLoadDistance; dy++) {
            for (let dx = -effectiveChunkLoadDistance; dx <= effectiveChunkLoadDistance; dx++) {
                const chunkX = centerChunkX + dx;
                const chunkY = centerChunkY + dy;
                
                // Skip if out of map bounds
                if (this.mapMetadata && this.width > 0 && this.height > 0) {
                    const chunkStartX = chunkX * this.chunkSize;
                    const chunkStartY = chunkY * this.chunkSize;
                    
                    if (chunkStartX < 0 || chunkStartY < 0 || 
                        chunkStartX >= this.width || 
                        chunkStartY >= this.height) {
                        continue;
                    }
                }
                
                const key = `${chunkX},${chunkY}`;
                
                // Add to visible chunks list
                newVisibleChunks.push({ x: chunkX, y: chunkY, key });
                
                // Update last accessed time for existing chunks
                if (this.chunks.has(key)) {
                    this.chunkLastAccessed.set(key, Date.now());
                } else {
                    // Track missing chunks (only done for debugging)
                    missingChunks.push(`(${chunkX},${chunkY})`);
                }
                // No network requests here, unlike updateVisibleChunks
            }
        }
        
        // Log missing chunks for debugging
        if (missingChunks.length > 0) {
            console.log(`[MapManager] ${missingChunks.length} chunks in view distance not loaded: ${missingChunks.join(', ')}`);
        }
        
        // Update the visible chunks list
        this.visibleChunks = newVisibleChunks;
        
        // No trimming of the cache here to avoid any visual jitter
        
        // Update last position for next call
        this._lastPlayerPosition = { x: centerChunkX, y: centerChunkY };
        
        return {
            center: { x: centerChunkX, y: centerChunkY },
            loadedChunks: this.visibleChunks.length - missingChunks.length,
            missingChunks: missingChunks.length
        };
    }
    
    /**
     * Trim the chunk cache to stay under the maximum limit
     */
    trimChunkCache() {
        if (this.chunks.size <= this.maxCachedChunks) {
            return;
        }
        
        // Get chunks sorted by last accessed time (oldest first)
        const sortedChunks = Array.from(this.chunkLastAccessed.entries())
            .sort((a, b) => a[1] - b[1]);
        
        // Calculate how many to remove
        const removeCount = this.chunks.size - this.maxCachedChunks;
        
        // Remove oldest chunks
        for (let i = 0; i < removeCount; i++) {
            const [key] = sortedChunks[i];
            this.chunks.delete(key);
            this.chunkLastAccessed.delete(key);
        }
        
            console.log(`Trimmed ${removeCount} chunks from cache`);
    }
    
    /**
     * Get a specific chunk
     * @param {number} chunkX - Chunk X coordinate
     * @param {number} chunkY - Chunk Y coordinate
     * @returns {Array|null} Chunk data or null if not loaded
     */
    getChunk(chunkX, chunkY) {
        const key = `${chunkX},${chunkY}`;
        
        // Update last accessed time
        if (this.chunks.has(key)) {
            this.chunkLastAccessed.set(key, Date.now());
            return this.chunks.get(key);
        }
        
        // Request chunk if not already pending
        if (this.networkManager && !this.pendingChunks.has(key)) {
            this.pendingChunks.add(key);
            try {
                this.networkManager.requestChunk(chunkX, chunkY);
                console.log(`Requested chunk (${chunkX}, ${chunkY}) on-demand`);
            } catch (error) {
                console.error(`Error requesting chunk (${chunkX}, ${chunkY}):`, error);
                this.pendingChunks.delete(key);
            }
        }
        
        return null;
    }
    
    /**
     * Get a specific tile
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {Tile|null} Tile object or null if not found
     */
    getTile(x, y) {
        // STRICT MAP BOUNDARY CHECK: Only allow coordinates within map bounds
        if (x < 0 || y < 0 || (this.width > 0 && x >= this.width) || (this.height > 0 && y >= this.height)) {
            console.log(`Attempted to get tile outside map bounds: (${x}, ${y}), map size: ${this.width}x${this.height}`);
            return new Tile(TILE_IDS.WALL, 0); // Return wall for out-of-bounds
        }
        
        // Convert to chunk coordinates
        const chunkX = Math.floor(x / this.chunkSize);
        const chunkY = Math.floor(y / this.chunkSize);
        const localX = ((x % this.chunkSize) + this.chunkSize) % this.chunkSize; // Handle negative values
        const localY = ((y % this.chunkSize) + this.chunkSize) % this.chunkSize; // Handle negative values
        
        // Get chunk
        const chunk = this.getChunk(chunkX, chunkY);
        
        // No chunk data available
        if (!chunk) {
            console.log(`No chunk data for (${chunkX}, ${chunkY}), requesting from server`);
            // Request the chunk if network manager is available
            if (this.networkManager) {
                this.networkManager.requestChunk(chunkX, chunkY);
            }
            
            // Return wall tile instead of fallback
            return new Tile(TILE_IDS.WALL, 0);
        }
        
        // Get tile from chunk
        try {
            return chunk[localY][localX];
        } catch (e) {
            console.error(`Error getting tile at (${x}, ${y}) from chunk (${chunkX}, ${chunkY}):`, e);
            return new Tile(TILE_IDS.WALL, 0);
        }
    }
    
    /**
     * Get tiles in a range (for rendering)
     * @param {number} startX - Start X coordinate
     * @param {number} startY - Start Y coordinate
     * @param {number} endX - End X coordinate
     * @param {number} endY - End Y coordinate
     * @returns {Array} Array of tile objects
     */
    getTilesInRange(startX, startY, endX, endY) {
        const tiles = [];
        
        for (let y = startY; y <= endY; y++) {
            for (let x = startX; x <= endX; x++) {
                const tile = this.getTile(x, y);
                if (tile) {
                    tiles.push({ x, y, tile });
                }
            }
        }
        
        return tiles;
    }
    
    /**
     * Check if a tile is a wall, obstacle, or out of map bounds
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {boolean} True if wall or obstacle
     */
    isWallOrObstacle(x, y) {

        return false;
        // Map boundary check
        if (x < 0 || y < 0 || (this.width > 0 && x >= this.width) || (this.height > 0 && y >= this.height)) {
            // Treat outside of map as wall
            return true;
        }
        
        // Get actual tile
        const tile = this.getTile(x, y);
        
        // If no tile found, treat as obstacle
        if (!tile) {
            return true;
        }
        
        // Check tile type
        return tile.type === TILE_IDS.WALL || 
               tile.type === TILE_IDS.OBSTACLE || 
               tile.type === TILE_IDS.MOUNTAIN ||
               tile.type === TILE_IDS.WATER;
    }
    
    /**
     * Generate a fallback tile when chunk not loaded
     * THIS FUNCTION IS DISABLED - We want to strictly respect map boundaries
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {Tile} Fallback tile
     */
    /* DISABLED FALLBACK TILE GENERATION
    generateFallbackTile(x, y) {
        // Use a simple pattern for fallback tiles
        // Make map edges walls, interior floor
        if (x < 0 || y < 0 || (this.width > 0 && x >= this.width) || (this.height > 0 && y >= this.height)) {
            return new Tile(TILE_IDS.WALL, 0);
        }
        
        // Checkerboard pattern
        const isEven = (x + y) % 2 === 0;
        const tileType = isEven ? TILE_IDS.FLOOR : TILE_IDS.FLOOR;
        
        return new Tile(tileType, 0);
    }
    */
    
    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {Function} callback - Event callback
     */
    addEventListener(event, callback) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(callback);
    }
    
    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} callback - Event callback
     */
    removeEventListener(event, callback) {
        if (!this.eventListeners[event]) return;
        const index = this.eventListeners[event].indexOf(callback);
        if (index !== -1) {
            this.eventListeners[event].splice(index, 1);
        }
    }
    
    /**
     * Dispatch an event
     * @param {string} event - Event name
     * @param {Object} data - Event data
     */
    dispatchEvent(event, data) {
        if (!this.eventListeners[event]) return;
        for (const callback of this.eventListeners[event]) {
            callback(data);
        }
    }
    
    /**
     * Print debug information about the current map state
     * @param {boolean} [showFullChunks=false] - Whether to print full chunk data
     */
    debugPrintMapInfo(showFullChunks = false) {
        console.log('=== MAP MANAGER DEBUG INFO ===');
        console.log(`Map ID: ${this.activeMapId || 'None'}`);
        console.log(`Map Size: ${this.width}x${this.height}`);
        console.log(`Tile Size: ${this.tileSize}, Chunk Size: ${this.chunkSize}`);
        console.log(`Procedural Generation: ${this.proceduralEnabled ? 'Enabled' : 'Disabled'}`);
        console.log(`Loaded Chunks: ${this.chunks.size}`);
        console.log(`Pending Chunks: ${this.pendingChunks.size}`);
        console.log(`Visible Chunks: ${this.visibleChunks.length}`);
        
        // Print chunk locations
        const chunkLocations = Array.from(this.chunks.keys()).map(key => {
            const [x, y] = key.split(',').map(Number);
            return `(${x},${y})`;
        });
        console.log(`Chunk Locations: ${chunkLocations.join(', ')}`);
        
        // Print chunk data if requested
        if (showFullChunks) {
            console.log('=== CHUNK DATA ===');
            this.chunks.forEach((chunk, key) => {
                console.log(`Chunk ${key}:`);
                this.printChunkVisually(key, chunk);
            });
        }
        
        console.log('=============================');
    }
    
    /**
     * Print a visual representation of a chunk to the console
     * @param {string} chunkKey - The chunk key (e.g. "0,0")
     * @param {Array} chunk - The chunk data
     */
    printChunkVisually(chunkKey, chunk) {
        if (!chunk || !Array.isArray(chunk)) {
            console.log(`Chunk ${chunkKey} has invalid data format`);
            return;
        }
        
        // Define tile type symbols for visual representation
        const symbols = {
            [TILE_IDS.FLOOR]: '.',
            [TILE_IDS.WALL]: '#',
            [TILE_IDS.OBSTACLE]: 'O',
            [TILE_IDS.WATER]: '~',
            [TILE_IDS.MOUNTAIN]: '^',
            'default': '?'
        };
        
        console.log(`Chunk ${chunkKey} - ${chunk.length}x${chunk[0]?.length || 0}:`);
        
        // Build visual representation
        const visual = [];
        for (let y = 0; y < chunk.length; y++) {
            let row = '';
            for (let x = 0; x < chunk[y].length; x++) {
                const tile = chunk[y][x];
                const tileType = tile?.type || 'default';
                row += symbols[tileType] || symbols['default'];
            }
            visual.push(row);
        }
        
        // Print the visual representation
        visual.forEach(row => console.log(row));
    }
    
    /**
     * Visualize the loaded map in the browser console with color
     * @param {number} centerX - Center tile X coordinate
     * @param {number} centerY - Center tile Y coordinate
     * @param {number} width - Width in tiles to visualize
     * @param {number} height - Height in tiles to visualize
     */
    visualizeMap(centerX = null, centerY = null, width = 40, height = 20) {
        console.log('=== MAP VISUALIZATION ===');
        
        // If no center specified, use player position
        if (centerX === null || centerY === null) {
            if (gameState && gameState.character) {
                centerX = Math.floor(gameState.character.x);
                centerY = Math.floor(gameState.character.y);
            } else {
                centerX = 0;
                centerY = 0;
            }
        }
        
        console.log(`Map centered at (${centerX}, ${centerY}), showing ${width}x${height} tiles`);
        
        // Calculate boundaries
        const startX = Math.floor(centerX - width / 2);
        const startY = Math.floor(centerY - height / 2);
        const endX = startX + width;
        const endY = startY + height;
        
        // Define colors for different tile types
        const colors = {
            [TILE_IDS.FLOOR]: 'color: #8a8a8a', // Gray
            [TILE_IDS.WALL]: 'color: #d43f3f', // Red
            [TILE_IDS.OBSTACLE]: 'color: #d49f3f', // Orange
            [TILE_IDS.WATER]: 'color: #3f8ad4', // Blue
            [TILE_IDS.MOUNTAIN]: 'color: #6f6f6f', // Dark gray
            'current': 'color: #ffffff; background-color: #ff0000', // White on red for player position
            'default': 'color: #ffffff' // White
        };
        
        // Define symbols for tile types
        const symbols = {
            [TILE_IDS.FLOOR]: '·',
            [TILE_IDS.WALL]: '█',
            [TILE_IDS.OBSTACLE]: '▒',
            [TILE_IDS.WATER]: '≈',
            [TILE_IDS.MOUNTAIN]: '▲',
            'current': '⊕',
            'default': '?'
        };
        
        // Track which chunks are loaded or missing
        const loadedChunks = new Set();
        const missingChunks = new Set();
        
        // Build the visualization row by row
        for (let y = startY; y < endY; y++) {
            let row = '%c';
            let formats = [];
            
            for (let x = startX; x < endX; x++) {
                // Check if this is the player position
                const isPlayerPos = (x === centerX && y === centerY);
                
                if (isPlayerPos) {
                    row += symbols['current'];
                    formats.push(colors['current']);
                    continue;
                }
                
                // Get tile type
                const tile = this.getTile(x, y);
                
                // Track chunk status
                const chunkX = Math.floor(x / this.chunkSize);
                const chunkY = Math.floor(y / this.chunkSize);
                const chunkKey = `${chunkX},${chunkY}`;
                
                if (this.chunks.has(chunkKey)) {
                    loadedChunks.add(chunkKey);
                } else {
                    missingChunks.add(chunkKey);
                }
                
                // Add appropriate symbol with color
                if (tile) {
                    const tileType = tile.type;
                    row += '%c' + (symbols[tileType] || symbols['default']);
                    formats.push(colors[tileType] || colors['default']);
                    } else {
                    row += '%c' + '.';
                    formats.push('color: #333333'); // Dark gray for unknown/missing
                }
            }
            
            // Print the row with formats
            console.log(row, ...formats);
        }
        
        // Print chunk information
        console.log('Loaded chunks: ' + Array.from(loadedChunks).join(', '));
        console.log('Missing chunks: ' + Array.from(missingChunks).join(', '));
        console.log('===========================');
        
        // Return summary
        return {
            center: { x: centerX, y: centerY },
            loadedChunks: loadedChunks.size,
            missingChunks: missingChunks.size,
            tilesShown: width * height
        };
    }
    
    /**
     * Save current map data to a file
     * @returns {Object} Map data object
     */
    saveMapData() {
        // Get map dimensions
        const width = this.width || 64;
        const height = this.height || 64;
        
        console.log(`Saving map with dimensions ${width}x${height}`);
        
        // Initialize with 0 (floor) as default
        const tileMap = Array(height).fill().map(() => Array(width).fill(0));
        
        // Keep track of chunks and tiles processed
        const loadedChunks = new Set();
        const tilesFound = 0;
        
        // Process all loaded chunks
        for (const [key, chunk] of this.chunks.entries()) {
            const [chunkX, chunkY] = key.split(',').map(Number);
            const startX = chunkX * this.chunkSize;
            const startY = chunkY * this.chunkSize;
            
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
                            }
                        }
                    }
                }
            }
        }
        
        // Format JSON with one row per line for readability
        const formattedJson = "[\n" + 
            tileMap.map(row => "  " + JSON.stringify(row)).join(",\n") + 
            "\n]";
        
        // Save the map using the browser's download capability
        try {
            const blob = new Blob([formattedJson], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `client_map_direct_${this.activeMapId || 'unknown'}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log(`Map data saved to ${a.download} (${loadedChunks.size} chunks)`);
            
            // Make the save map function available globally for debugging
            window.clientMapData = tileMap;
            console.log("Map data also available at window.clientMapData");
            
            return { 
                tileMap, 
                loadedChunks: loadedChunks.size,
                width,
                height
            };
        } catch (error) {
            console.error("Error saving map data:", error);
            return null;
        }
    }
}