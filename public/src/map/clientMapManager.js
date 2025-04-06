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
        
        // Process chunk data to match our format
        const processedChunk = this.processChunkData(chunkData);
        
        // Store chunk
        this.chunks.set(key, processedChunk);
        this.chunkLastAccessed.set(key, Date.now());
        
        // Remove from pending
        this.pendingChunks.delete(key);
        
        console.log(`Stored chunk (${chunkX}, ${chunkY}) with ${processedChunk.length} rows`);
        
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
        // If the chunk data is already in the right format, return it
        if (Array.isArray(chunkData)) {
            return chunkData;
        }
        
        // Convert from server format to client format
        const processedData = [];
        
        // Process tiles array if it exists
        if (chunkData.tiles && Array.isArray(chunkData.tiles)) {
            for (let y = 0; y < chunkData.tiles.length; y++) {
                const row = [];
                for (let x = 0; x < chunkData.tiles[y].length; x++) {
                    const tileData = chunkData.tiles[y][x];
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
        } else {
            // Create default chunk data
            for (let y = 0; y < this.chunkSize; y++) {
                const row = [];
                for (let x = 0; x < this.chunkSize; x++) {
                    row.push(new Tile(TILE_IDS.FLOOR, 0));
                }
                processedData.push(row);
            }
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
        
        // Convert player position to chunk coordinates (integers)
        const centerChunkX = Math.floor(playerX / (this.tileSize * this.chunkSize));
        const centerChunkY = Math.floor(playerY / (this.tileSize * this.chunkSize));
        
        // Get chunks in view distance
        const newVisibleChunks = [];
        
        for (let dy = -this.chunkLoadDistance; dy <= this.chunkLoadDistance; dy++) {
            for (let dx = -this.chunkLoadDistance; dx <= this.chunkLoadDistance; dx++) {
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
                        console.log(`Requested chunk (${chunkX}, ${chunkY})`);
                    } catch (error) {
                        console.error(`Error requesting chunk (${chunkX}, ${chunkY}):`, error);
                        this.pendingChunks.delete(key);
                    }
                }
            }
        }
        
        this.visibleChunks = newVisibleChunks;
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
     * Get a specific tile by world coordinates
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {Tile|null} Tile object or null if chunk not loaded
     */
    getTile(x, y) {
        // Calculate chunk coordinates
        const chunkX = Math.floor(x / this.chunkSize);
        const chunkY = Math.floor(y / this.chunkSize);
        
        // Get chunk
        const chunk = this.getChunk(chunkX, chunkY);
        if (!chunk) {
            // Return a fallback tile if chunk isn't loaded
            return this.generateFallbackTile(x, y);
        }
        
        // Calculate local coordinates within the chunk
        const localX = ((x % this.chunkSize) + this.chunkSize) % this.chunkSize;
        const localY = ((y % this.chunkSize) + this.chunkSize) % this.chunkSize;
        
        // Make sure we're within the bounds of the chunk data
        if (chunk.length <= localY || !chunk[localY] || chunk[localY].length <= localX) {
            console.warn(`Invalid local coordinates (${localX}, ${localY}) for chunk (${chunkX}, ${chunkY})`);
            return this.generateFallbackTile(x, y);
        }
        
        // Return tile if it exists
        if (chunk[localY] && chunk[localY][localX]) {
            return chunk[localY][localX];
        }
        
        // Return fallback if tile not found
        return this.generateFallbackTile(x, y);
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
     * Check if a position is a wall or obstacle
     * @param {number} x - World X coordinate
     * @param {number} y - World Y coordinate
     * @returns {boolean} True if wall, false if not (or chunk not loaded)
     */
    isWallOrObstacle(x, y) {
        // Convert to tile coordinates
        const tileX = Math.floor(x / this.tileSize);
        const tileY = Math.floor(y / this.tileSize);
        
        // Get tile
        const tile = this.getTile(tileX, tileY);
        
        // If tile not found, assume passable
        if (!tile) {
            return false;
        }
        
        // Check if wall, obstacle, mountain, or water
        return tile.type === TILE_IDS.WALL || 
               tile.type === TILE_IDS.OBSTACLE || 
               tile.type === TILE_IDS.MOUNTAIN ||
               tile.type === TILE_IDS.WATER;
    }
    
    /**
     * Generate a fallback tile when chunk not loaded
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {Tile} Fallback tile
     */
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
}