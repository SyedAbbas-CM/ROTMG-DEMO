// File: /src/managers/MapManager.js

const { PerlinNoise } = require('./world/PerlinNoise');
const { Tile } = require('./world/tile');
const { TILE_IDS, CHUNK_SIZE } = require('./world/constants');

/**
 * MapManager handles the game world, tiles, and chunks.
 */
class MapManager {
  /**
   * Creates a map manager
   */
  constructor() {
    this.chunks = new Map(); // Map of "x,y" -> chunk data
    this.width = 0;         // Width of the world in tiles
    this.height = 0;        // Height of the world in tiles
    this.tileSize = 12;     // Size of each tile in pixels
    this.perlin = new PerlinNoise(Math.random()); // For procedural generation
    this.proceduralEnabled = true;
  }
  
  /**
   * Generate or load a world
   * @param {number} width - Width of the world in tiles
   * @param {number} height - Height of the world in tiles
   */
  generateWorld(width, height) {
    this.width = width;
    this.height = height;
    
    // If we want to pre-generate chunks, we could do it here,
    // but for large worlds it's better to generate on demand
    console.log(`World initialized with size ${width}x${height}`);
  }
  
  /**
   * Get world info for clients
   * @returns {Object} World metadata
   */
  getMapInfo() {
    return {
      width: this.width,
      height: this.height,
      tileSize: this.tileSize,
      chunkSize: CHUNK_SIZE
    };
  }
  
  /**
   * Get data for a specific chunk
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @returns {Object|null} Chunk data or null if not found
   */
  getChunkData(chunkX, chunkY) {
    const key = `${chunkX},${chunkY}`;
    
    // If chunk exists in cache, return it
    if (this.chunks.has(key)) {
      return this.chunks.get(key);
    }
    
    // Otherwise generate it
    if (this.proceduralEnabled) {
      const chunkData = this.generateChunkData(chunkX, chunkY);
      this.chunks.set(key, chunkData);
      return chunkData;
    }
    
    return null;
  }
  
  /**
   * Generate data for a chunk procedurally
   * @param {number} chunkRow - Chunk row (Y coordinate)
   * @param {number} chunkCol - Chunk column (X coordinate)
   * @returns {Object} Generated chunk data
   */
  generateChunkData(chunkRow, chunkCol) {
    const tiles = [];
    
    for (let y = 0; y < CHUNK_SIZE; y++) {
      const row = [];
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const globalX = chunkCol * CHUNK_SIZE + x;
        const globalY = chunkRow * CHUNK_SIZE + y;
        
        // Skip if out of world bounds
        if (globalX >= this.width || globalY >= this.height) {
          row.push(TILE_IDS.WALL); // Use wall for out of bounds
          continue;
        }
        
        // Generate height value using Perlin noise
        const heightValue = this.perlin.get(globalX / 50, globalY / 50);
        
        // Determine tile type based on height
        const tileType = this.determineTileType(heightValue, globalX, globalY);
        
        row.push(tileType);
      }
      tiles.push(row);
    }
    
    return {
      x: chunkCol,
      y: chunkRow,
      tiles: tiles
    };
  }
  
  /**
   * Determine tile type based on height value
   * @param {number} heightValue - Perlin noise height value
   * @param {number} x - Global X coordinate
   * @param {number} y - Global Y coordinate
   * @returns {number} TILE_IDS value
   */
  determineTileType(heightValue, x, y) {
    // Example logic - customize based on your game's needs
    if (heightValue < -0.4) return TILE_IDS.WATER;
    if (heightValue < -0.2) return TILE_IDS.FLOOR;
    if (heightValue < 0.2) return TILE_IDS.FLOOR;
    if (heightValue < 0.5) return TILE_IDS.OBSTACLE;
    return TILE_IDS.MOUNTAIN;
    
    // Add border walls
    if (x === 0 || y === 0 || x === this.width - 1 || y === this.height - 1) {
      return TILE_IDS.WALL;
    }
  }
  
  /**
   * Get a specific tile
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {number|null} Tile type or null if not found
   */
  getTile(x, y) {
    // Convert to chunk coordinates
    const chunkX = Math.floor(x / CHUNK_SIZE);
    const chunkY = Math.floor(y / CHUNK_SIZE);
    const localX = x % CHUNK_SIZE;
    const localY = y % CHUNK_SIZE;
    
    // Get chunk
    const chunk = this.getChunkData(chunkX, chunkY);
    if (!chunk) return null;
    
    // Get tile from chunk
    return chunk.tiles[localY][localX];
  }
  
  /**
   * Check if a position is a wall or out of bounds
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate
   * @returns {boolean} True if wall or out of bounds
   */
  isWallOrOutOfBounds(x, y) {
    // Convert to tile coordinates
    const tileX = Math.floor(x / this.tileSize);
    const tileY = Math.floor(y / this.tileSize);
    
    // Check if out of bounds
    if (tileX < 0 || tileY < 0 || tileX >= this.width || tileY >= this.height) {
      return true;
    }
    
    // Get tile type
    const tileType = this.getTile(tileX, tileY);
    
    // Check if wall or other solid obstacle
    return tileType === TILE_IDS.WALL || 
           tileType === TILE_IDS.MOUNTAIN || 
           tileType === TILE_IDS.WATER;
  }
  
  /**
   * Enable procedural generation
   */
  enableProceduralGeneration() {
    this.proceduralEnabled = true;
  }
  
  /**
   * Disable procedural generation
   */
  disableProceduralGeneration() {
    this.proceduralEnabled = false;
  }
}

module.exports = MapManager;// File: /src/managers/MapManager.js

const { PerlinNoise } = require('./world/PerlinNoise');
const { Tile } = require('./world/tile');
const { TILE_IDS, CHUNK_SIZE } = require('./world/constants');

/**
 * MapManager handles the game world, tiles, and chunks.
 */
class MapManager {
  /**
   * Creates a map manager
   */
  constructor() {
    this.chunks = new Map(); // Map of "x,y" -> chunk data
    this.width = 0;         // Width of the world in tiles
    this.height = 0;        // Height of the world in tiles
    this.tileSize = 12;     // Size of each tile in pixels
    this.perlin = new PerlinNoise(Math.random()); // For procedural generation
    this.proceduralEnabled = true;
  }
  
  /**
   * Generate or load a world
   * @param {number} width - Width of the world in tiles
   * @param {number} height - Height of the world in tiles
   */
  generateWorld(width, height) {
    this.width = width;
    this.height = height;
    
    // If we want to pre-generate chunks, we could do it here,
    // but for large worlds it's better to generate on demand
    console.log(`World initialized with size ${width}x${height}`);
  }
  
  /**
   * Get world info for clients
   * @returns {Object} World metadata
   */
  getMapInfo() {
    return {
      width: this.width,
      height: this.height,
      tileSize: this.tileSize,
      chunkSize: CHUNK_SIZE
    };
  }
  
  /**
   * Get data for a specific chunk
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @returns {Object|null} Chunk data or null if not found
   */
  getChunkData(chunkX, chunkY) {
    const key = `${chunkX},${chunkY}`;
    
    // If chunk exists in cache, return it
    if (this.chunks.has(key)) {
      return this.chunks.get(key);
    }
    
    // Otherwise generate it
    if (this.proceduralEnabled) {
      const chunkData = this.generateChunkData(chunkX, chunkY);
      this.chunks.set(key, chunkData);
      return chunkData;
    }
    
    return null;
  }
  
  /**
   * Generate data for a chunk procedurally
   * @param {number} chunkRow - Chunk row (Y coordinate)
   * @param {number} chunkCol - Chunk column (X coordinate)
   * @returns {Object} Generated chunk data
   */
  generateChunkData(chunkRow, chunkCol) {
    const tiles = [];
    
    for (let y = 0; y < CHUNK_SIZE; y++) {
      const row = [];
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const globalX = chunkCol * CHUNK_SIZE + x;
        const globalY = chunkRow * CHUNK_SIZE + y;
        
        // Skip if out of world bounds
        if (globalX >= this.width || globalY >= this.height) {
          row.push(TILE_IDS.WALL); // Use wall for out of bounds
          continue;
        }
        
        // Generate height value using Perlin noise
        const heightValue = this.perlin.get(globalX / 50, globalY / 50);
        
        // Determine tile type based on height
        const tileType = this.determineTileType(heightValue, globalX, globalY);
        
        row.push(tileType);
      }
      tiles.push(row);
    }
    
    return {
      x: chunkCol,
      y: chunkRow,
      tiles: tiles
    };
  }
  
  /**
   * Determine tile type based on height value
   * @param {number} heightValue - Perlin noise height value
   * @param {number} x - Global X coordinate
   * @param {number} y - Global Y coordinate
   * @returns {number} TILE_IDS value
   */
  determineTileType(heightValue, x, y) {
    // Example logic - customize based on your game's needs
    if (heightValue < -0.4) return TILE_IDS.WATER;
    if (heightValue < -0.2) return TILE_IDS.FLOOR;
    if (heightValue < 0.2) return TILE_IDS.FLOOR;
    if (heightValue < 0.5) return TILE_IDS.OBSTACLE;
    return TILE_IDS.MOUNTAIN;
    
    // Add border walls
    if (x === 0 || y === 0 || x === this.width - 1 || y === this.height - 1) {
      return TILE_IDS.WALL;
    }
  }
  
  /**
   * Get a specific tile
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {number|null} Tile type or null if not found
   */
  getTile(x, y) {
    // Convert to chunk coordinates
    const chunkX = Math.floor(x / CHUNK_SIZE);
    const chunkY = Math.floor(y / CHUNK_SIZE);
    const localX = x % CHUNK_SIZE;
    const localY = y % CHUNK_SIZE;
    
    // Get chunk
    const chunk = this.getChunkData(chunkX, chunkY);
    if (!chunk) return null;
    
    // Get tile from chunk
    return chunk.tiles[localY][localX];
  }
  
  /**
   * Check if a position is a wall or out of bounds
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate
   * @returns {boolean} True if wall or out of bounds
   */
  isWallOrOutOfBounds(x, y) {
    // Convert to tile coordinates
    const tileX = Math.floor(x / this.tileSize);
    const tileY = Math.floor(y / this.tileSize);
    
    // Check if out of bounds
    if (tileX < 0 || tileY < 0 || tileX >= this.width || tileY >= this.height) {
      return true;
    }
    
    // Get tile type
    const tileType = this.getTile(tileX, tileY);
    
    // Check if wall or other solid obstacle
    return tileType === TILE_IDS.WALL || 
           tileType === TILE_IDS.MOUNTAIN || 
           tileType === TILE_IDS.WATER;
  }
  
  /**
   * Enable procedural generation
   */
  enableProceduralGeneration() {
    this.proceduralEnabled = true;
  }
  
  /**
   * Disable procedural generation
   */
  disableProceduralGeneration() {
    this.proceduralEnabled = false;
  }
}

module.exports = MapManager;