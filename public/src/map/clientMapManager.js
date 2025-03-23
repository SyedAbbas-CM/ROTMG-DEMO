/**
 * ClientMapManager.js
 * Handles the game world map on the client side
 */
import { TILE_IDS } from '../constants/constants.js';
import { perlin } from './perlinNoise.js';

export class ClientMapManager {
  /**
   * Creates a client-side map manager
   */
  constructor() {
    this.chunks = new Map(); // Map of "x,y" -> chunk data
    this.width = 0;         // Width of the world in tiles
    this.height = 0;        // Height of the world in tiles
    this.tileSize = 12;     // Size of each tile in pixels
    this.chunkSize = 16;    // Size of each chunk in tiles
    this.viewDistance = 2;  // How many chunks to load around player
  }
  
  /**
   * Initialize map with server info
   * @param {Object} mapInfo - Map metadata from server
   */
  initMap(mapInfo) {
    this.width = mapInfo.width;
    this.height = mapInfo.height;
    this.tileSize = mapInfo.tileSize || this.tileSize;
    this.chunkSize = mapInfo.chunkSize || this.chunkSize;
    
    console.log(`Map initialized: ${this.width}x${this.height}, ${this.chunkSize}x${this.chunkSize} chunks`);
  }
  
  /**
   * Get visible chunks around a position
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate
   * @returns {Array} Array of {x, y} chunk coordinates
   */
  getVisibleChunks(x, y) {
    // Convert world coordinates to tile coordinates
    const tileX = Math.floor(x / this.tileSize);
    const tileY = Math.floor(y / this.tileSize);
    
    // Convert tile coordinates to chunk coordinates
    const centerChunkX = Math.floor(tileX / this.chunkSize);
    const centerChunkY = Math.floor(tileY / this.chunkSize);
    
    const visibleChunks = [];
    
    // Get chunks in view distance
    for (let dy = -this.viewDistance; dy <= this.viewDistance; dy++) {
      for (let dx = -this.viewDistance; dx <= this.viewDistance; dx++) {
        const chunkX = centerChunkX + dx;
        const chunkY = centerChunkY + dy;
        
        // Skip if out of world bounds
        if (chunkX < 0 || chunkY < 0 || 
            chunkX * this.chunkSize >= this.width || 
            chunkY * this.chunkSize >= this.height) {
          continue;
        }
        
        visibleChunks.push({ x: chunkX, y: chunkY });
      }
    }
    
    return visibleChunks;
  }
  
  /**
   * Check if a chunk is loaded
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @returns {boolean} True if chunk is loaded
   */
  hasChunk(chunkX, chunkY) {
    return this.chunks.has(`${chunkX},${chunkY}`);
  }
  
  /**
   * Set chunk data received from server
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @param {Object} data - Chunk data
   */
  setChunkData(chunkX, chunkY, data) {
    this.chunks.set(`${chunkX},${chunkY}`, data);
  }
  
  /**
   * Generate a fallback chunk if server doesn't have it
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   */
  generateFallbackChunk(chunkX, chunkY) {
    const tiles = [];
    
    for (let y = 0; y < this.chunkSize; y++) {
      const row = [];
      for (let x = 0; x < this.chunkSize; x++) {
        const globalX = chunkX * this.chunkSize + x;
        const globalY = chunkY * this.chunkSize + y;
        
        // Simple fallback: Generate a flat floor with some random obstacles
        let tileType = TILE_IDS.FLOOR;
        
        // Use perlin noise to create some variety
        const heightValue = perlin.get(globalX / 20, globalY / 20);
        
        if (heightValue > 0.6) {
          tileType = TILE_IDS.OBSTACLE;
        } else if (heightValue < -0.6) {
          tileType = TILE_IDS.WATER;
        }
        
        // Add border walls
        if (globalX === 0 || globalY === 0 || 
            globalX === this.width - 1 || globalY === this.height - 1) {
          tileType = TILE_IDS.WALL;
        }
        
        row.push(tileType);
      }
      tiles.push(row);
    }
    
    const chunkData = {
      x: chunkX,
      y: chunkY,
      tiles: tiles,
      isFallback: true
    };
    
    this.chunks.set(`${chunkX},${chunkY}`, chunkData);
    console.log(`Generated fallback chunk at (${chunkX}, ${chunkY})`);
    
    return chunkData;
  }
  
  /**
   * Get a specific tile
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {number|null} Tile type or null if not found
   */
  getTile(x, y) {
    // Convert to chunk coordinates
    const chunkX = Math.floor(x / this.chunkSize);
    const chunkY = Math.floor(y / this.chunkSize);
    const localX = x % this.chunkSize;
    const localY = y % this.chunkSize;
    
    // Get chunk
    const key = `${chunkX},${chunkY}`;
    if (!this.chunks.has(key)) return null;
    
    const chunk = this.chunks.get(key);
    
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
    if (tileType === null) return true; // Chunk not loaded yet
    
    // Check if wall or other solid obstacle
    return tileType === TILE_IDS.WALL || 
           tileType === TILE_IDS.MOUNTAIN || 
           tileType === TILE_IDS.WATER;
  }
  
  /**
   * Get tiles in a specific range (for rendering)
   * @param {number} startX - Start X coordinate in tiles
   * @param {number} startY - Start Y coordinate in tiles
   * @param {number} endX - End X coordinate in tiles
   * @param {number} endY - End Y coordinate in tiles
   * @returns {Array} Array of {x, y, type} tile objects
   */
  getTilesInRange(startX, startY, endX, endY) {
    const tiles = [];
    
    for (let y = startY; y <= endY; y++) {
      for (let x = startX; x <= endX; x++) {
        const tileType = this.getTile(x, y);
        
        if (tileType !== null) {
          tiles.push({
            x: x,
            y: y,
            type: tileType
          });
        }
      }
    }
    
    return tiles;
  }
}