// src/map/map.js

import { TILE_IDS, CHUNK_SIZE } from '../constants/constants.js';
import { perlin } from './perlinNoise.js';
import { Tile } from './tile.js';
import { MapLoader } from './mapLoader.js'; // Import MapLoader

class GameMap {
  constructor() {
    this.chunks = new Map();
    this.isFixedMap = false;
    this.proceduralEnabled = true;
  }

  /**
   * Disables procedural generation.
   */
  disableProceduralGeneration() {
    this.proceduralEnabled = false;
  }

  /**
   * Enables procedural generation.
   */
  enableProceduralGeneration() {
    this.proceduralEnabled = true;
    this.isFixedMap = false;
  }

  /**
   * Loads a fixed map from a JSON file.
   * @param {string} url - The URL of the map JSON file.
   * @returns {Promise<void>}
   */
  async loadFixedMap(url) {
    try {
      const mapData = await MapLoader.loadMapFromFile(url);
      this.setMapData(mapData);
      this.isFixedMap = true;
      console.log('Fixed map loaded successfully.');
    } catch (error) {
      console.error('Failed to load fixed map:', error);
    }
  }

  /**
   * Sets the map data from a loaded map.
   * @param {Object} mapData - The map data object.
   */
  setMapData(mapData) {
    this.chunks.clear(); // Clear any existing procedural data

    const { width, height, tiles } = mapData;

    for (let y = 0; y < height; y += CHUNK_SIZE) {
      for (let x = 0; x < width; x += CHUNK_SIZE) {
        const chunkRow = Math.floor(y / CHUNK_SIZE);
        const chunkCol = Math.floor(x / CHUNK_SIZE);
        const chunkData = [];

        for (let dy = 0; dy < CHUNK_SIZE; dy++) {
          const rowData = [];
          for (let dx = 0; dx < CHUNK_SIZE; dx++) {
            const tileX = x + dx;
            const tileY = y + dy;

            if (tileY >= height || tileX >= width) {
              // Out-of-bounds: default to a valid tile (e.g., FLOOR)
              rowData.push(new Tile(TILE_IDS.FLOOR));
              continue;
            }

            const tileType = tiles[tileY][tileX];
            rowData.push(new Tile(tileType));
          }
          chunkData.push(rowData);
        }

        this.chunks.set(`${chunkRow},${chunkCol}`, chunkData);
      }
    }
  }

  /**
   * Procedural generation function
   */
  generateChunkData(chunkRow, chunkCol) {
    if (this.isFixedMap || !this.proceduralEnabled) {
      return [];
    }

    const chunkData = [];
    for (let y = 0; y < CHUNK_SIZE; y++) {
      const rowData = [];
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const globalX = chunkCol * CHUNK_SIZE + x;
        const globalY = chunkRow * CHUNK_SIZE + y;
        const heightValue = perlin.get(globalX / 50, globalY / 50); // Adjust scaling as needed
        const tileType = this.determineTileType(heightValue);
        rowData.push(new Tile(tileType, heightValue));
      }
      chunkData.push(rowData);
    }
    return chunkData;
  }

  /**
   * Function to determine tile type based on height
   */
  determineTileType(heightValue) {
    if (heightValue < -0.2) return TILE_IDS.WATER;
    if (heightValue < 0.2) return TILE_IDS.FLOOR;
    return TILE_IDS.MOUNTAIN;
  }

  /**
   * Function to load a chunk
   */
  loadChunk(chunkRow, chunkCol) {
    const key = `${chunkRow},${chunkCol}`;
    if (this.chunks.has(key)) return this.chunks.get(key);

    if (this.isFixedMap || !this.proceduralEnabled) {
      return null;
    }

    // Generate chunk data procedurally
    const chunkData = this.generateChunkData(chunkRow, chunkCol);
    this.chunks.set(key, chunkData);
    return chunkData;
  }

  /**
   * Function to get tile at world coordinates
   */
  getTile(x, y) {
    const chunkRow = Math.floor(y / CHUNK_SIZE);
    const chunkCol = Math.floor(x / CHUNK_SIZE);
    const chunk = this.loadChunk(chunkRow, chunkCol);
    if (!chunk) return null;

    const localX = x % CHUNK_SIZE;
    const localY = y % CHUNK_SIZE;
    if (localY < 0 || localY >= CHUNK_SIZE || localX < 0 || localX >= CHUNK_SIZE) return null;
    return chunk[localY][localX];
  }

  /**
   * Function to get all tiles in a specific range (for rendering)
   */
  getTilesInRange(xStart, yStart, xEnd, yEnd) {
    const tiles = [];
    for (let y = yStart; y <= yEnd; y++) {
      for (let x = xStart; x <= xEnd; x++) {
        const tile = this.getTile(x, y);
        if (tile) {
          tiles.push({ x, y, tile });
        }
      }
    }
    return tiles;
  }
}

// Export a singleton GameMap instance
export const map = new GameMap();
