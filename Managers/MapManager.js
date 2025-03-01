// server/map/MapManager.js

const PerlinNoise = require('./world/PerlinNoise');
const Tile = require('.world/tile');

// Example tile IDs from your client code
const TILE_IDS = {
  FLOOR: 0,
  WALL: 1,
  WATER: 2,
  MOUNTAIN: 3
};

const CHUNK_SIZE = 32; // or whatever size you prefer

class MapManager {
  constructor() {
    this.chunks = new Map();
    this.isFixedMap = false;
    this.proceduralEnabled = true;
    this.perlin = new PerlinNoise(Math.random());  // or a fixed seed
  }

  // For a fully procedural approach
  generateChunkData(chunkRow, chunkCol) {
    const chunkData = [];
    for (let y = 0; y < CHUNK_SIZE; y++) {
      const rowData = [];
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const globalX = chunkCol * CHUNK_SIZE + x;
        const globalY = chunkRow * CHUNK_SIZE + y;
        const heightValue = this.perlin.get(globalX / 50, globalY / 50);
        const tileType = this.determineTileType(heightValue);
        rowData.push(new Tile(tileType, heightValue));
      }
      chunkData.push(rowData);
    }
    return chunkData;
  }

  determineTileType(heightValue) {
    if (heightValue < -0.2) return TILE_IDS.WATER;
    if (heightValue < 0.2)  return TILE_IDS.FLOOR;
    return TILE_IDS.MOUNTAIN;
  }

  /**
   * Loads or generates chunk at (chunkRow, chunkCol)
   */
  loadChunk(chunkRow, chunkCol) {
    const key = `${chunkRow},${chunkCol}`;
    if (this.chunks.has(key)) {
      return this.chunks.get(key);
    }
    // Not loaded yet, generate it
    if (this.proceduralEnabled) {
      const chunkData = this.generateChunkData(chunkRow, chunkCol);
      this.chunks.set(key, chunkData);
      return chunkData;
    }
    return null;
  }

  /**
   * Get tile at global coordinates (x, y)
   */
  getTile(x, y) {
    const chunkRow = Math.floor(y / CHUNK_SIZE);
    const chunkCol = Math.floor(x / CHUNK_SIZE);
    const chunk = this.loadChunk(chunkRow, chunkCol);
    if (!chunk) return null;
    const localX = x % CHUNK_SIZE;
    const localY = y % CHUNK_SIZE;
    return chunk[localY][localX] || null;
  }

  /**
   * For sending to client, you might want to return simple IDs or a 2D array of tile types.
   */
  getChunkAsIDs(chunkRow, chunkCol) {
    const chunkData = this.loadChunk(chunkRow, chunkCol);
    if (!chunkData) return null;

    // Convert to raw IDs
    const chunkIDs = chunkData.map(row =>
      row.map(tile => tile.type)
    );
    return chunkIDs;
  }

  // Potentially implement saving to disk or a DB:
  saveToFile(filePath) {
    // e.g., convert 'this.chunks' to JSON
    // fs.writeFileSync(...)
  }

  loadFromFile(filePath) {
    // read file, parse JSON, populate this.chunks
  }
}

module.exports = MapManager;
