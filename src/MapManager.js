// File: /src/managers/MapManager.js

import { TILE_IDS, CHUNK_SIZE, TILE_SIZE } from './world/constants.js';
import { tileDatabase } from './assets/TileDatabase.js';
import { PerlinNoise } from './world/PerlinNoise.js';
import { Tile } from './world/tile.js';
import { EnhancedPerlinNoise } from './world/AdvancedPerlinNoise.js';
/**
 * MapManager handles the game world, tiles, and chunks.
 * This is a unified implementation that consolidates functionality
 * from both the original MapManager and GameMap classes.
 */
export class MapManager {
  /**
   * Creates a map manager
   */
  constructor(options = {}) {
    this.chunks = new Map(); // Map of "x,y" -> chunk data
    this.width = 0;          // Width of the world in tiles
    this.height = 0;         // Height of the world in tiles
    this.tileSize = options.tileSize || TILE_SIZE;  // Size of each tile in pixels
    
    // For procedural generation
    this.perlin = new EnhancedPerlinNoise(options.seed || Math.random());
    this.proceduralEnabled = true;
    this.isFixedMap = false;
    
    // Map storage (for server)
    this.mapStoragePath = options.mapStoragePath || '';
    this.maps = new Map(); // For storing multiple maps (id -> mapData)
    this.nextMapId = 1;

    // Track which map is currently active.  This helps internal helper
    // functions (like getTile) that historically passed `null` for mapId.
    this.activeMapId = null;
  }
  
  /**
   * Generate or load a world
   * @param {number} width - Width of the world in tiles
   * @param {number} height - Height of the world in tiles
   * @param {Object} options - Additional options
   * @returns {string} Map ID
   */
  generateWorld(width, height, options = {}) {
    this.width = width;
    this.height = height;
    
    // Clear existing chunks
    this.chunks.clear();
    
    console.log(`World initialized with size ${width}x${height}`);
    
    // Create a map ID
    const mapId = `map_${this.nextMapId++}`;
    
    // Store map metadata
    const meta={
      id: mapId,
      width,
      height,
      tileSize: this.tileSize,
      chunkSize: CHUNK_SIZE,
      name: options.name || 'Untitled Map',
      procedural: this.proceduralEnabled,
      seed: this.perlin.seed,
      objects: [],
      enemySpawns: []
    };
    this.maps.set(mapId, meta);

    // Make this the active map unless caller explicitly opts out
    this.activeMapId = mapId;
    
    return mapId;
  }
  
  /**
   * Get world info for clients
   * @param {string} mapId - Map ID
   * @returns {Object} World metadata
   */
  getMapInfo(mapId) {
    if (mapId && this.maps.has(mapId)) {
      return this.maps.get(mapId);
    }
    
    return {
      width: this.width,
      height: this.height,
      tileSize: this.tileSize,
      chunkSize: CHUNK_SIZE
    };
  }
  
  /**
   * Get data for a specific chunk
   * @param {string} mapId - Map ID
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @returns {Object|null} Chunk data or null if not found
   */
  getChunkData(mapId, chunkX, chunkY) {
    // If no map explicitly supplied, fall back to the currently active one
    if (!mapId) mapId = this.activeMapId;

    // Detailed chunk-send logging is controlled by DEBUG.chunkRequests
    if (globalThis.DEBUG?.chunkRequests) {
      console.log('[SRV] send', mapId, chunkX, chunkY);
    }
    
    const key = `${mapId || 'default'}_${chunkX},${chunkY}`;
    
    // If chunk exists in cache, return it
    if (this.chunks.has(key)) {
      return this.chunks.get(key);
    }
    
    // Otherwise generate or slice it
    if (this.proceduralEnabled && !this.isFixedMap) {
      const chunkData = this.generateChunkData(chunkY, chunkX);
      this.chunks.set(key, chunkData);
      return chunkData;
    }
    
    // If we have a fixed map with a full tileMap, slice on demand
    if (this.isFixedMap) {
      const sliced = this._sliceChunkFromTileMap(mapId, chunkX, chunkY);
      if (sliced) {
        this.chunks.set(key, sliced);
        return sliced;
      }
    }
    
    return null;
  }
  
  /**
   * Generate data for a chunk procedurally using RotMG-style terrain generation
   * @param {number} chunkRow - Chunk row (Y coordinate)
   * @param {number} chunkCol - Chunk column (X coordinate)
   * @returns {Object} Generated chunk data
   */
  generateChunkData(chunkRow, chunkCol) {
    const tiles = [];
    
    // Tuned parameters for richer terrain
    const OCTAVES = 5;          // one extra octave for fine detail
    const PERSISTENCE = 0.55;   // slightly more high-frequency influence
    const BASE_SCALE = 32;      // zoom in => more variation per chunk
    
    for (let y = 0; y < CHUNK_SIZE; y++) {
      const row = [];
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const globalX = chunkCol * CHUNK_SIZE + x;
        const globalY = chunkRow * CHUNK_SIZE + y;
        
        // Skip if out of world bounds
        if (globalX >= this.width || globalY >= this.height) {
          row.push(new Tile(TILE_IDS.WALL)); // Use wall for out of bounds
          continue;
        }
        
        // Generate multiple octaves of noise for more natural terrain
        let height = 0;
        let amplitude = 1.0;
        let frequency = 1.0;
        let maxValue = 0;
        
        // Sum multiple noise octaves
        for (let o = 0; o < OCTAVES; o++) {
          // Scale coordinates based on frequency
          const sampleX = globalX / (BASE_SCALE / frequency);
          const sampleY = globalY / (BASE_SCALE / frequency);
          
          // Add scaled noise value
          height += this.perlin.get(sampleX, sampleY) * amplitude;
          
          // Keep track of max possible value for normalization
          maxValue += amplitude;
          
          // Increase frequency, decrease amplitude for next octave
          amplitude *= PERSISTENCE;
          frequency *= 2;
        }
        
        // Normalize to -1 to 1 range
        height /= maxValue;
        
        // Apply a curve to create more interesting terrain
        // Emphasize extremes (more mountains and water, less flat land)
        height = Math.pow(height, 3);
        
        // For coasts: add a high-frequency noise to create more jagged coastlines
        if (height > -0.4 && height < -0.2) {
          const coastDetail = this.perlin.get(globalX / 10, globalY / 10) * 0.1;
          height += coastDetail;
        }
        
        // Determine tile type based on height and position
        const tileType = this.determineTileType(height, globalX, globalY);

        // Attempt to fetch extended definition from TileDatabase (if loaded)
        const def = tileDatabase?.getByNumeric(tileType) || {};
        
        row.push(new Tile(tileType, height, def));
      }
      tiles.push(row);
    }
    
    // ---------- SIMPLE SMOOTHING PASS ----------
    // Convert isolated single-tile walls into floor and thicken long wall lines
    const neighbours = [
      [1,0],[-1,0],[0,1],[0,-1]
    ];
    const copy = tiles.map(r => r.map(t => t.type));
    for (let y=0;y<tiles.length;y++){
      for (let x=0;x<tiles[y].length;x++){
        const t = copy[y][x];
        if(t!==TILE_IDS.WALL) continue;
        let nWall=0;
        for(const [dx,dy] of neighbours){
          const nx=x+dx, ny=y+dy;
          if(nx>=0&&ny>=0&&ny<copy.length&&nx<copy[0].length){
            if(copy[ny][nx]===TILE_IDS.WALL) nWall++;
          }
        }
        if(nWall===0){
          // isolated pixel wall -> floor
          tiles[y][x].type = TILE_IDS.FLOOR;
        } else if(nWall>=3){
          // core of thick wall: keep, but reinforce neighbours
          for(const [dx,dy] of neighbours){
            const nx=x+dx, ny=y+dy;
            if(nx>=0&&ny>=0&&ny<tiles.length&&nx<tiles[0].length){
              if(tiles[ny][nx].type===TILE_IDS.FLOOR){
                tiles[ny][nx].type=TILE_IDS.WALL;
              }
            }
          }
        }
      }
    }
    
    return {
      x: chunkCol,
      y: chunkRow,
      tiles: tiles
    };
  }
  
  /**
   * Determine tile type based on height value and create biomes like in RotMG
   * @param {number} heightValue - Perlin noise height value
   * @param {number} x - Global X coordinate
   * @param {number} y - Global Y coordinate
   * @returns {number} TILE_IDS value
   */
  determineTileType(heightValue, x, y) {
    // Absolute border walls
    if (x === 0 || y === 0 || x === this.width - 1 || y === this.height - 1) {
      return TILE_IDS.WALL;
    }
    
    // Generate additional noise values for biome variety
    // Use different frequency/scale for variety
    const temperatureNoise = this.perlin.get(x / 100, y / 100);  
    const moistureNoise = this.perlin.get(x / 80 + 500, y / 80 + 500);
    
    // Determine biome based on height, temperature and moisture
    // RotMG-style biome system
    
    // Deep water
    if (heightValue < -0.6) {
      return TILE_IDS.WATER;
    }
    
    // Shallow water/beaches
    if (heightValue < -0.3) {
      // Sometimes place obstacles in shallow water (like reeds or rocks)
      if (moistureNoise > 0.7 && Math.random() < 0.03) {
        return TILE_IDS.OBSTACLE;
      }
      return TILE_IDS.WATER;
    }
    
    // Lowlands (main gameplay areas)
    if (heightValue < 0.2) {
      // Cold biomes with temperature noise
      if (temperatureNoise < -0.5) {
        // Occasionally place obstacles (like ice formations)
        if (Math.random() < 0.02 && moistureNoise > 0.5) {
          return TILE_IDS.OBSTACLE;
        }
        return TILE_IDS.FLOOR; // Could be a special "snow" tile if we add more tile types
      }
      
      // Wet/swampy areas with high moisture
      if (moistureNoise > 0.6) {
        // More obstacles in swampy areas (like bogs)
        if (Math.random() < 0.05) {
          return TILE_IDS.OBSTACLE;
        }
        return TILE_IDS.FLOOR;
      }
      
      // Desert-like areas (low moisture, high temperature)
      if (moistureNoise < -0.3 && temperatureNoise > 0.4) {
        // Few obstacles in deserts
        if (Math.random() < 0.01) {
          return TILE_IDS.OBSTACLE;
        }
        return TILE_IDS.FLOOR;
      }
      
      // Default lowland (grassy fields)
      // Place natural obstacles occasionally
      if (Math.random() < 0.02) {
        return TILE_IDS.OBSTACLE;
      }
      return TILE_IDS.FLOOR;
    }
    
    // Hills and forests (medium elevation)
    if (heightValue < 0.5) {
      // Denser obstacles in hilly/forest areas
      if (Math.random() < 0.1 + (moistureNoise * 0.1)) {
        return TILE_IDS.OBSTACLE;
      }
      return TILE_IDS.FLOOR;
    }
    
    // Mountains (high elevation)
    if (heightValue < 0.7) {
      // Very dense obstacles in mountains
      if (Math.random() < 0.3) {
        return TILE_IDS.OBSTACLE;
      }
      // Some walls in mountains too
      if (Math.random() < 0.15) {
        return TILE_IDS.WALL;
      }
      return TILE_IDS.MOUNTAIN;
    }
    
    // Peaks (highest elevation)
    // Almost impassable
    if (Math.random() < 0.7) {
      return TILE_IDS.WALL;
    }
    return TILE_IDS.MOUNTAIN;
  }
  
  /**
   * Get a specific tile
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {Tile|null} Tile object or null if not found
   */
  getTile(x, y) {
    const mapId=this.activeMapId;
    // Convert to chunk coordinates
    const chunkX = Math.floor(x / CHUNK_SIZE);
    const chunkY = Math.floor(y / CHUNK_SIZE);
    const localX = x % CHUNK_SIZE;
    const localY = y % CHUNK_SIZE;
    
    // Get chunk
    const chunk = this.getChunkData(mapId, chunkX, chunkY);
    if (!chunk) return null;
    
    // Get tile from chunk
    return chunk.tiles[localY][localX];
  }
  
  /**
   * Get the tile type at a specific coordinate
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @returns {number|null} Tile type or null if not found
   */
  getTileType(x, y) {
    const tile = this.getTile(x, y);
    return tile ? tile.type : null;
  }
  
  /**
   * Check if a position is a wall or out of bounds
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate
   * @returns {boolean} True if wall or out of bounds
   */
  isWallOrOutOfBounds(x, y) {
    // Coordinates supplied to this function are already expressed in tile units
    // Do NOT divide by tileSize or we will create mismatches between server and client.
    const tileX = Math.floor(x);
    const tileY = Math.floor(y);
    
    // Debug coordinate conversion occasionally
    if (Math.random() < 0.0001) {
      console.log(`[SERVER] Wall check: World (${x.toFixed(2)}, ${y.toFixed(2)}) -> Tile (${tileX}, ${tileY}) [tileSize=${this.tileSize}]`);
    }
    
    // Check if out of bounds
    if (tileX < 0 || tileY < 0 || tileX >= this.width || tileY >= this.height) {
      return true;
    }
    
    // Prefer property-based walkability if available
    const tile = this.getTile(tileX, tileY);

    if (tile) {
      if (typeof tile.isWalkable === 'function') {
        const blocked = !tile.isWalkable();
        if (blocked && Math.random() < 0.0001) {
          console.log(`[SERVER] Collision (property) at tile (${tileX}, ${tileY}), type: ${tile.type}`);
        }
        return blocked;
      }
      if (tile.properties && tile.properties.walkable !== undefined) {
        return !tile.properties.walkable;
      }
    }

    // Fallback to basic numeric type check
    const tileType = tile ? tile.type : this.getTileType(tileX, tileY);

    const isBlocked = tileType === TILE_IDS.WALL || 
                     tileType === TILE_IDS.OBSTACLE ||
                     tileType === TILE_IDS.MOUNTAIN || 
                     tileType === TILE_IDS.WATER;

    if (isBlocked && Math.random() < 0.0001) {
      console.log(`[SERVER] Collision at tile (${tileX}, ${tileY}), type: ${tileType}`);
    }
    return isBlocked;
  }
  
  /**
   * Load a fixed map from a JSON file
   * @param {string} url - URL or path to map JSON file
   * @returns {Promise<string>} - Promise resolving to map ID
   */
  async loadFixedMap(url) {
    try {
      let mapData;
      
      // If URL looks like http(s) remote, use fetch; otherwise use Node fs.
      if (typeof url === 'string' && /^https?:\/\//i.test(url)) {
        if (typeof fetch !== 'function') throw new Error('fetch not available for remote URL');
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load map: ${response.statusText}`);
        mapData = await response.json();
      }
      // Node.js file read
      else if (typeof require === 'function') {
        // CommonJS environment – straightforward require() access
        const fs = require('fs');
        const path = require('path');
        const resolvedPath = path.isAbsolute(url) ? url : path.join(this.mapStoragePath || process.cwd(), url);
        const data = fs.readFileSync(resolvedPath, 'utf8');
        mapData = JSON.parse(data);
      }
      else {
        // ES-module environment ("type": "module") where require() is not defined.
        // Dynamically import fs/path and read the file synchronously.
        let fs, path;
        try {
          const fsModule = await import('fs');
          const pathModule = await import('path');
          fs = fsModule.default || fsModule;
          path = pathModule.default || pathModule;
        } catch (err) {
          throw new Error('No valid method to load map data (fs unavailable)');
        }

        const resolvedPath = path.isAbsolute(url) ? url : path.join(this.mapStoragePath || process.cwd(), url);
        const data = fs.readFileSync(resolvedPath, 'utf8');
        mapData = JSON.parse(data);
      }
      
      // --- EDITOR COMPATIBILITY -----------------------------------------
      // If mapData comes from our browser editor, it has `ground` (2-D array of
      // sprite names/null) instead of the numeric `tileMap` expected by the
      // engine.  Convert it here: non-null => FLOOR (walkable); null => WALL.
      if (!mapData.tileMap && Array.isArray(mapData.ground)) {
        const tileMap = mapData.ground.map(row =>
          row.map(cell => (cell===null ? TILE_IDS.WALL : TILE_IDS.FLOOR))
        );
        mapData.tileMap = tileMap;
      }

      // Set map data and get ID
      const mapId = this.setMapData(mapData);
      this.isFixedMap = true;
      this.proceduralEnabled = false;
      console.log('Fixed map loaded successfully:', mapId);
      return mapId;
    } catch (error) {
      console.error('Failed to load fixed map:', error);
      throw error;
    }
  }
  
  /**
   * Save map to a file
   * @param {string} mapId - Map ID to save
   * @param {string} filename - Filename to save as
   * @returns {Promise<boolean>} - Promise resolving to success status
   */
  async saveMap(mapId, filename) {
    if (!this.maps.has(mapId)) {
      console.error(`Map ${mapId} not found`);
      return false;
    }
    
    // Check if we're running in Node.js - use dynamic import for ES modules
    let fs, path;
    try {
      // Use dynamic import for ES modules
      const fsModule = await import('fs');
      const pathModule = await import('path');
      fs = fsModule.default || fsModule;
      path = pathModule.default || pathModule;
      console.log("Using ES dynamic import");
    } catch (e) {
      console.error("Cannot access the file system:", e.message);
      return false;
    }
    
    try {
      // Get map metadata
      const mapData = this.maps.get(mapId);
      console.log(`DEBUG: Preparing to save map ${mapId} with dimensions ${mapData.width}x${mapData.height}`);
      
      // Make sure directory exists
      if (!fs.existsSync(this.mapStoragePath)) {
        console.log(`DEBUG: Creating map storage directory: ${this.mapStoragePath}`);
        fs.mkdirSync(this.mapStoragePath, { recursive: true });
      }
      
      // Collect all chunks for this map
      const chunks = {};
      for (const [key, chunk] of this.chunks.entries()) {
        if (key.startsWith(`${mapId}_`)) {
          chunks[key.substring(mapId.length + 1)] = chunk;
        }
      }
      
      console.log(`DEBUG: Found ${Object.keys(chunks).length} chunks to save`);
      
      // Prepare the full map data
      const fullMapData = {
        ...mapData,
        chunks
      };
      
      // Save to file
      const filePath = path.join(this.mapStoragePath, filename);
      console.log(`DEBUG: Attempting to save map to path: ${filePath}`);
      
      // Create maps directory if it doesn't exist
      if (!fs.existsSync(path.dirname(filePath))) {
        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        console.log(`DEBUG: Created directory: ${path.dirname(filePath)}`);
      }
      
      fs.writeFileSync(filePath, JSON.stringify(fullMapData, null, 2));
      
      console.log(`Map saved to ${filePath} successfully!`);
      return true;
    } catch (error) {
      console.error('Failed to save map:', error);
      console.error(`ERROR DETAILS: ${error.message}`);
      console.error(`Stack trace: ${error.stack}`);
      console.error(`Map storage path: ${this.mapStoragePath}`);
      return false;
    }
  }
  
  /**
   * Save map as a simple 2D array of tile types
   * @param {string} mapId - Map ID to save
   * @param {string} filename - Filename to save as
   * @returns {Promise<boolean>} - Promise resolving to success status
   */
  async saveSimpleMap(mapId, filename) {
    if (!this.maps.has(mapId)) {
      console.error(`Map ${mapId} not found`);
      return false;
    }
    
    // Check if we're running in Node.js - use dynamic import for ES modules
    let fs, path;
    try {
      // Use dynamic import for ES modules
      const fsModule = await import('fs');
      const pathModule = await import('path');
      fs = fsModule.default || fsModule;
      path = pathModule.default || pathModule;
      console.log("Using ES dynamic import");
    } catch (e) {
      console.error("Cannot access the file system:", e.message);
      return false;
    }
    
    try {
      // Get map metadata
      const mapData = this.maps.get(mapId);
      const width = mapData.width;
      const height = mapData.height;
      
      console.log(`DEBUG: Preparing to save simple map ${mapId} with dimensions ${width}x${height}`);
      
      // Make sure directory exists
      if (!fs.existsSync(this.mapStoragePath)) {
        console.log(`DEBUG: Creating map storage directory: ${this.mapStoragePath}`);
        fs.mkdirSync(this.mapStoragePath, { recursive: true });
      }
      
      // Create a 2D array initialized with -1 (unknown)
      const tileMap = Array(height).fill().map(() => Array(width).fill(-1));
      
      // FIXED: Use the direct approach to get tile types
      console.log(`Trying direct tile lookup for map ${mapId} - width=${width}, height=${height}`);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          // Get tile directly using getTile method
          const tile = this.getTile(x, y);
          if (tile) {
            tileMap[y][x] = tile.type;
          }
        }
      }
      
      // Check if we still have all -1s, if so try the chunk approach as backup
      let allNegativeOne = true;
      for (let y = 0; y < height && allNegativeOne; y++) {
        for (let x = 0; x < width && allNegativeOne; x++) {
          if (tileMap[y][x] !== -1) {
            allNegativeOne = false;
            break;
          }
        }
      }
      
      // If all still -1, try the chunk approach
      if (allNegativeOne) {
        console.log(`WARNING: Direct tile lookup failed, trying chunk-based approach`);
        let processedChunks = 0;
        
        for (const [key, chunk] of this.chunks.entries()) {
          // Log all keys for debugging
          console.log(`DEBUG: Checking chunk key: ${key} for map prefix ${mapId}_`);
          
          if (!key.startsWith(`${mapId}_`)) continue;
          
          const chunkKey = key.substring(mapId.length + 1);
          const [chunkX, chunkY] = chunkKey.split(',').map(Number);
          const startX = chunkX * this.maps.get(mapId).chunkSize;
          const startY = chunkY * this.maps.get(mapId).chunkSize;
          
          console.log(`DEBUG: Processing chunk ${chunkKey} at position (${startX}, ${startY})`);
          
          // Debug chunk data structure
          console.log(`DEBUG: Chunk structure: ${JSON.stringify(Object.keys(chunk))}`);
          console.log(`DEBUG: Chunk tiles length: ${chunk.tiles ? chunk.tiles.length : 'undefined'}`);
          
          // Fill in the tile types from this chunk
          if (chunk.tiles) {
            for (let y = 0; y < chunk.tiles.length; y++) {
              if (!chunk.tiles[y]) continue;
              
              for (let x = 0; x < chunk.tiles[y].length; x++) {
                const globalX = startX + x;
                const globalY = startY + y;
                
                // Skip if outside map bounds
                if (globalX >= width || globalY >= height) continue;
                
                const tile = chunk.tiles[y][x];
                if (tile) {
                  tileMap[globalY][globalX] = tile.type;
                }
              }
            }
          }
          
          processedChunks++;
        }
        
        console.log(`DEBUG: Processed ${processedChunks} chunks for the simple map`);
      }
      
      // Save to file
      const filePath = path.join(this.mapStoragePath, filename);
      console.log(`DEBUG: Attempting to save simple map to path: ${filePath}`);
      
      // Create maps directory if it doesn't exist
      if (!fs.existsSync(path.dirname(filePath))) {
        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        console.log(`DEBUG: Created directory: ${path.dirname(filePath)}`);
      }
      
      // Format the map with one row per line for readability
      const formattedJson = "[\n" + 
        tileMap.map(row => "  " + JSON.stringify(row)).join(",\n") + 
        "\n]";
      
      fs.writeFileSync(filePath, formattedJson);
      
      console.log(`Simple map saved to ${filePath} successfully!`);
      return true;
    } catch (error) {
      console.error('Failed to save simple map:', error);
      console.error(`ERROR DETAILS: ${error.message}`);
      console.error(`Stack trace: ${error.stack}`);
      console.error(`Map storage path: ${this.mapStoragePath}`);
      return false;
    }
  }
  
  /**
   * Sets the map data from a loaded map
   * @param {Object} mapData - The map data object
   * @returns {string} Map ID
   */
  setMapData(mapData) {
    const mapId = mapData.id || `map_${this.nextMapId++}`;
    
    // Clear existing chunks for this map
    for (const [key] of this.chunks.entries()) {
      if (key.startsWith(`${mapId}_`)) {
        this.chunks.delete(key);
      }
    }
    
    // Store metadata – also keep the raw tileMap (if present) so we can slice chunks on-demand
    const meta={
      id: mapId,
      width: mapData.width,
      height: mapData.height,
      tileSize: mapData.tileSize || this.tileSize,
      chunkSize: mapData.chunkSize || CHUNK_SIZE,
      name: mapData.name || 'Loaded Map',
      procedural: false,
      tileMap: mapData.tileMap || null, // full 2-D array of tile objects / IDs
      objects: mapData.objects || [],   // decorative / interactive objects
      enemySpawns: mapData.enemies || [] // enemy spawn markers
    };
    this.maps.set(mapId, meta);

    // Mark as active map
    this.activeMapId = mapId;
    
    // Update current dimensions
    this.width = mapData.width;
    this.height = mapData.height;
    
    // Load pre-baked chunks if provided
    if (mapData.chunks) {
      for (const [chunkKey, chunkData] of Object.entries(mapData.chunks)) {
        this.chunks.set(`${mapId}_${chunkKey}`, chunkData);
      }
    }
    
    return mapId;
  }
  
  /**
   * Function to get tiles in a range
   * @param {number} xStart - Start X coordinate
   * @param {number} yStart - Start Y coordinate 
   * @param {number} xEnd - End X coordinate
   * @param {number} yEnd - End Y coordinate
   * @returns {Array} Array of tile objects with coordinates
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
  
  /**
   * Enable procedural generation
   */
  enableProceduralGeneration() {
    this.proceduralEnabled = true;
    this.isFixedMap = false;
  }
  
  /**
   * Disable procedural generation
   */
  disableProceduralGeneration() {
    this.proceduralEnabled = false;
  }
  
  /**
   * Create a procedural map (server-side helper)
   * @param {Object} options - Map options
   * @returns {string} Map ID
   */
  createProceduralMap(options = {}) {
    const width = options.width || 256;
    const height = options.height || 256;
    this.enableProceduralGeneration();
    return this.generateWorld(width, height, options);
  }
  
  /**
   * Get map metadata (server-side helper)
   * @param {string} mapId - Map ID
   * @returns {Object} Map metadata
   */
  getMapMetadata(mapId) {
    return this.maps.has(mapId) ? this.maps.get(mapId) : null;
  }
  
  /**
   * Slice a chunk out of a full tileMap for fixed maps that ship only tileMap, not pre-chunked data.
   * @private
   */
  _sliceChunkFromTileMap(mapId, chunkX, chunkY) {
    const meta = this.maps.get(mapId);
    if (!meta || !meta.tileMap) return null;

    const { tileMap, chunkSize } = meta;
    const startX = chunkX * chunkSize;
    const startY = chunkY * chunkSize;

    // Out of bounds?
    if (startX >= meta.width || startY >= meta.height || startX < 0 || startY < 0) {
      return null;
    }

    const tiles = [];
    for (let y = 0; y < chunkSize; y++) {
      const row = [];
      const globalY = startY + y;
      if (globalY >= meta.height) break;
      for (let x = 0; x < chunkSize; x++) {
        const globalX = startX + x;
        if (globalX >= meta.width) break;
        let cell = tileMap[globalY][globalX];
        // If the stored value is a primitive ID, wrap in Tile object; otherwise assume object compatible.
        if (typeof cell === 'number') {
          cell = new Tile(cell, 0);
        }
        row.push(cell);
      }
      tiles.push(row);
    }

    return { x: chunkX, y: chunkY, tiles };
  }
  
  getObjects(mapId){
    const meta=this.getMapMetadata(mapId); return meta&&Array.isArray(meta.objects)?meta.objects:[];
  }
  getEnemySpawns(mapId){
    const meta=this.getMapMetadata(mapId); return meta&&Array.isArray(meta.enemySpawns)?meta.enemySpawns:[];
  }
}

// Export a singleton instance for client use
export const mapManager = new MapManager();

// For CommonJS compatibility
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    MapManager,
    mapManager
  };
}