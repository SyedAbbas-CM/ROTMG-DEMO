/**
 * ClientMapManager.js
 * Client-side map management with efficient chunk loading and caching
 */

/**
 * ClientMapManager - Handles loading, caching, and rendering map data
 */
class clientMapManager {
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
      this.visibleChunks = []; // Currently visible chunks
      this.pendingChunks = new Set(); // Chunks we're currently requesting
      this.maxCachedChunks = 100; // Maximum chunks to keep in memory
      this.chunkLoadDistance = 2; // How many chunks to load around player
      this.fallbackTileTypes = {
        0: 'floor', // Floor
        1: 'wall',  // Wall
        2: 'obstacle', // Obstacle
        3: 'water', // Water
        4: 'mountain' // Mountain
      };
      
      // LRU (Least Recently Used) tracking for chunk cache
      this.chunkLastAccessed = new Map(); // "x,y" -> timestamp
      
      // Register network handlers if network manager provided
      if (this.networkManager) {
        this.registerNetworkHandlers();
      }
    }
    
    /**
     * Register network handlers for map data
     * @private
     */
    registerNetworkHandlers() {
      // Map info handler
      this.networkManager.on(MessageType.MAP_INFO, (data) => {
        this.handleMapInfo(data);
      });
      
      // Chunk data handler
      this.networkManager.on(MessageType.CHUNK_DATA, (data) => {
        this.handleChunkData(data);
      });
    }
    
    /**
     * Handle map info from server
     * @param {Object} data - Map info data
     * @private
     */
    handleMapInfo(data) {
      this.activeMapId = data.mapId;
      this.mapMetadata = data;
      this.tileSize = data.tileSize || this.tileSize;
      this.chunkSize = data.chunkSize || this.chunkSize;
      
      console.log(`Map info received: ${this.activeMapId}`);
      
      // Clear existing chunks
      this.chunks.clear();
      this.chunkLastAccessed.clear();
      this.pendingChunks.clear();
      
      // Request initial chunks around (0,0) as default starting point
      this.updateVisibleChunks(0, 0);
      
      // Dispatch event
      this.dispatchEvent('mapinitialized', { mapId: this.activeMapId });
    }
    
    /**
     * Handle chunk data from server
     * @param {Object} data - Chunk data
     * @private
     */
    handleChunkData(data) {
      const { chunkX, chunkY, chunk } = data;
      
      // Store chunk
      const key = `${chunkX},${chunkY}`;
      this.chunks.set(key, chunk);
      this.chunkLastAccessed.set(key, Date.now());
      
      // Remove from pending
      this.pendingChunks.delete(key);
      
      // Trim cache if needed
      this.trimChunkCache();
      
      // Dispatch event
      this.dispatchEvent('chunkloaded', { chunkX, chunkY });
    }
    
    /**
     * Update visible chunks based on player position
     * @param {number} playerX - Player X position in world coordinates
     * @param {number} playerY - Player Y position in world coordinates
     */
    updateVisibleChunks(playerX, playerY) {
      // Convert player position to chunk coordinates
      const centerChunkX = Math.floor(playerX / this.tileSize / this.chunkSize);
      const centerChunkY = Math.floor(playerY / this.tileSize / this.chunkSize);
      
      // Get chunks in view distance
      const newVisibleChunks = [];
      
      for (let dy = -this.chunkLoadDistance; dy <= this.chunkLoadDistance; dy++) {
        for (let dx = -this.chunkLoadDistance; dx <= this.chunkLoadDistance; dx++) {
          const chunkX = centerChunkX + dx;
          const chunkY = centerChunkY + dy;
          const key = `${chunkX},${chunkY}`;
          
          // Skip if out of map bounds
          if (this.mapMetadata) {
            const chunkStartX = chunkX * this.chunkSize;
            const chunkStartY = chunkY * this.chunkSize;
            
            if (chunkStartX < 0 || chunkStartY < 0 || 
                chunkStartX >= this.mapMetadata.width || 
                chunkStartY >= this.mapMetadata.height) {
              continue;
            }
          }
          
          newVisibleChunks.push({ x: chunkX, y: chunkY, key });
          
          // Update last accessed time
          if (this.chunks.has(key)) {
            this.chunkLastAccessed.set(key, Date.now());
          }
          // Request chunk if not already loaded or pending
          else if (!this.pendingChunks.has(key) && this.networkManager) {
            this.pendingChunks.add(key);
            this.networkManager.requestChunk(chunkX, chunkY);
          }
        }
      }
      
      this.visibleChunks = newVisibleChunks;
    }
    
    /**
     * Trim the chunk cache to stay under the maximum limit
     * @private
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
     * @returns {Object|null} Chunk data or null if not loaded
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
        this.networkManager.requestChunk(chunkX, chunkY);
      }
      
      return null;
    }
    
    /**
     * Get a specific tile by world coordinates
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {Object|null} Tile object or null if chunk not loaded
     */
    getTile(x, y) {
      // Calculate chunk coordinates
      const chunkX = Math.floor(x / this.chunkSize);
      const chunkY = Math.floor(y / this.chunkSize);
      
      // Get chunk
      const chunk = this.getChunk(chunkX, chunkY);
      if (!chunk) {
        return null;
      }
      
      // Calculate local coordinates
      const localX = ((x % this.chunkSize) + this.chunkSize) % this.chunkSize;
      const localY = ((y % this.chunkSize) + this.chunkSize) % this.chunkSize;
      
      // Get tile type
      const tileType = chunk.tiles[localY][localX];
      
      // Return tile object
      return {
        type: tileType,
        x,
        y,
        typeName: this.fallbackTileTypes[tileType] || 'unknown'
      };
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
            tiles.push(tile);
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
      
      // If chunk not loaded, assume passable
      if (!tile) {
        return false;
      }
      
      // Check if wall or obstacle
      return tile.type === 1 || tile.type === 4;
    }
    
    /**
     * Generate a fallback tile when chunk not loaded
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @returns {Object} Fallback tile
     * @private
     */
    generateFallbackTile(x, y) {
      // Simple checkerboard pattern
      const isEven = (x + y) % 2 === 0;
      const tileType = isEven ? 0 : 2;
      
      return {
        type: tileType,
        x,
        y,
        typeName: this.fallbackTileTypes[tileType] || 'unknown',
        isFallback: true
      };
    }
    
    /**
     * Dispatch an event
     * @param {string} eventName - Event name
     * @param {Object} data - Event data
     * @private
     */
    dispatchEvent(eventName, data) {
      const event = new CustomEvent(`map:${eventName}`, { detail: data });
      window.dispatchEvent(event);
    }
  }
  
  // Export for ES modules
  export { ClientMapManager };
  
  // Make available for browser
  if (typeof window !== 'undefined') {
    window.ClientMapManager = ClientMapManager;
  }
  