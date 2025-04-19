// Enhanced PerlinNoise class with additional features
export class EnhancedPerlinNoise {
    constructor(seed = Math.random()) {
      this.seed = seed;
      this.gradients = {};
      this.memory = {};
    }
  
    // Generate a random gradient vector
    randomGradient(ix, iy) {
      const random = 2920 * Math.sin(ix * 21942 + iy * 171324 + this.seed * 8912) *
                    Math.cos(ix * 23157 * iy * 217832 + this.seed * 9758);
      return { x: Math.cos(random), y: Math.sin(random) };
    }
  
    // Dot product of the distance and gradient vectors
    dotGridGradient(ix, iy, x, y) {
      const key = `${ix},${iy}`;
      const gradient = this.gradients[key] || (this.gradients[key] = this.randomGradient(ix, iy));
  
      const dx = x - ix;
      const dy = y - iy;
  
      return dx * gradient.x + dy * gradient.y;
    }
  
    // Improved smoothing function (smoother than linear interpolation)
    fade(t) {
      return t * t * t * (t * (t * 6 - 15) + 10);
    }
  
    // Interpolation function
    lerp(a0, a1, w) {
      return (1 - w) * a0 + w * a1;
    }
  
    // Compute Perlin noise at coordinates x, y
    get(x, y) {
      const memKey = `${x.toFixed(3)},${y.toFixed(3)}`;
      if (this.memory[memKey] !== undefined) {
        return this.memory[memKey];
      }
      
      const x0 = Math.floor(x);
      const x1 = x0 + 1;
      const y0 = Math.floor(y);
      const y1 = y0 + 1;
  
      // Interpolation weights with improved smoothing
      const sx = this.fade(x - x0);
      const sy = this.fade(y - y0);
  
      // Interpolate between grid point gradients
      const n0 = this.dotGridGradient(x0, y0, x, y);
      const n1 = this.dotGridGradient(x1, y0, x, y);
      const ix0 = this.lerp(n0, n1, sx);
  
      const n2 = this.dotGridGradient(x0, y1, x, y);
      const n3 = this.dotGridGradient(x1, y1, x, y);
      const ix1 = this.lerp(n2, n3, sx);
  
      const value = this.lerp(ix0, ix1, sy);
      
      // Cache result for repeated lookups
      this.memory[memKey] = value;
      
      return value;
    }
    
    // Fractal Brownian Motion (fBm) - multiple octaves of noise
    fbm(x, y, octaves = 6, lacunarity = 2.0, persistence = 0.5) {
      let total = 0;
      let frequency = 1;
      let amplitude = 1;
      let maxValue = 0;
      
      for (let i = 0; i < octaves; i++) {
        total += this.get(x * frequency, y * frequency) * amplitude;
        maxValue += amplitude;
        
        // Increase frequency, decrease amplitude for each octave
        amplitude *= persistence;
        frequency *= lacunarity;
      }
      
      return total / maxValue;
    }
    
    // Ridged multifractal noise - creates ridges and valleys
    ridged(x, y, octaves = 6, lacunarity = 2.0, persistence = 0.5) {
      let total = 0;
      let frequency = 1;
      let amplitude = 1;
      let maxValue = 0;
      
      for (let i = 0; i < octaves; i++) {
        // Get noise value
        let noise = this.get(x * frequency, y * frequency);
        
        // Transform to create ridges
        noise = 1 - Math.abs(noise);
        noise = noise * noise; // Square for sharper ridges
        
        total += noise * amplitude;
        maxValue += amplitude;
        
        amplitude *= persistence;
        frequency *= lacunarity;
      }
      
      return total / maxValue;
    }
    
    // Domain warping - distorts the input space for more organic patterns
    warp(x, y, strength = 10.0) {
      // Warp the input coordinates
      const warpX = x + this.get(x * 0.05, y * 0.05) * strength;
      const warpY = y + this.get(x * 0.05 + 100, y * 0.05 + 100) * strength;
      
      // Get noise at warped coordinates
      return this.get(warpX, warpY);
    }
    
    // Clear cache to save memory
    clearCache() {
      this.memory = {};
    }
  }
  
  // Enhanced MapGenerator class using the improved noise generation
  export class EnhancedMapGenerator {
    constructor(options = {}) {
      this.seed = options.seed || Math.random();
      this.perlin = new EnhancedPerlinNoise(this.seed);
      
      // Map dimensions
      this.width = options.width || 256;
      this.height = options.height || 256;
      
      // Optional parameters
      this.waterLevel = options.waterLevel || 0.3;
      this.mountainLevel = options.mountainLevel || 0.7;
      this.forestLevel = options.forestLevel || 0.5;
      
      // Configure tile IDs
      this.TILE_IDS = options.TILE_IDS || {
        FLOOR: 0,
        WALL: 1,
        OBSTACLE: 2,
        WATER: 3,
        MOUNTAIN: 4
      };
    }
    
    // Generate a complete world map
    generateWorld() {
      // Create temperature and moisture maps
      const temperatureMap = this.generateTemperatureMap();
      const moistureMap = this.generateMoistureMap();
      
      // Generate base terrain using multiple noise techniques
      const heightMap = this.generateHeightMap();
      
      // Convert height map to tile types
      const tiles = this.convertToTiles(heightMap, temperatureMap, moistureMap);
      
      // Ensure map edges have walls
      this.addMapBorders(tiles);
      
      // Generate rivers if desired
      // this.generateRivers(tiles, heightMap);
      
      return {
        width: this.width,
        height: this.height,
        tiles: tiles,
        seed: this.seed
      };
    }
    
    // Generate temperature variation across the map (equator to poles)
    generateTemperatureMap() {
      const map = Array(this.height).fill().map(() => Array(this.width).fill(0));
      
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          // Base temperature gradient from south to north (equator in middle)
          const latitudeFactor = 1.0 - Math.abs((y / this.height) - 0.5) * 2;
          
          // Add some noise for local variations
          const noise = this.perlin.get(x * 0.01, y * 0.01) * 0.2;
          
          map[y][x] = latitudeFactor * 0.8 + noise;
        }
      }
      
      return map;
    }
    
    // Generate moisture map (used for biome determination)
    generateMoistureMap() {
      const map = Array(this.height).fill().map(() => Array(this.width).fill(0));
      
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          // Use domain warping for more interesting moisture patterns
          const warpX = x + this.perlin.get(x * 0.02, y * 0.02) * 20;
          const warpY = y + this.perlin.get(x * 0.02 + 40, y * 0.02 + 30) * 20;
          
          map[y][x] = this.perlin.fbm(warpX * 0.01, warpY * 0.01, 4, 2.0, 0.5);
        }
      }
      
      return map;
    }
    
    // Generate base height map using multiple noise techniques
    generateHeightMap() {
      const map = Array(this.height).fill().map(() => Array(this.width).fill(0));
      
      // Parameter for continent vs detailed noise
      const continentScale = 0.002; // Large scale continent shapes
      const detailScale = 0.01;     // Medium scale terrain features
      const microScale = 0.05;      // Small scale details
      
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          // Continent shapes (large scale)
          const continentNoise = this.perlin.fbm(x * continentScale, y * continentScale, 3, 2.0, 0.5);
          
          // Apply some ridged noise for mountain ranges
          const ridgeNoise = this.perlin.ridged(x * detailScale, y * detailScale, 4, 2.0, 0.5);
          
          // Detailed noise
          const detailNoise = this.perlin.fbm(x * microScale, y * microScale, 3, 2.0, 0.5);
          
          // Combine the different scales
          const combinedNoise = continentNoise * 0.6 + ridgeNoise * 0.3 + detailNoise * 0.1;
          
          // Apply curve to emphasize landmasses and oceans
          map[y][x] = this.applyHeightCurve(combinedNoise);
        }
      }
      
      return map;
    }
    
    // Apply non-linear curve to height values to create clearer distinctions
    applyHeightCurve(height) {
      // This makes flatter areas for oceans and land, with steeper transitions
      if (height < this.waterLevel - 0.1) {
        // Deep ocean
        return height * 0.5;
      } else if (height < this.waterLevel + 0.1) {
        // Coastline transition
        return this.waterLevel + (height - this.waterLevel) * 0.8;
      } else if (height > this.mountainLevel) {
        // Mountains get exaggerated
        return this.mountainLevel + (height - this.mountainLevel) * 1.5;
      }
      
      // Normal lands
      return height;
    }
    
    // Convert height map to actual tile types
    convertToTiles(heightMap, temperatureMap, moistureMap) {
      const tiles = Array(this.height).fill().map(() => Array(this.width).fill(0));
      
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          const height = heightMap[y][x];
          const temperature = temperatureMap[y][x];
          const moisture = moistureMap[y][x];
          
          // Use determineTileType to set the tile type
          tiles[y][x] = this.determineTileType(height, temperature, moisture, x, y);
        }
      }
      
      return tiles;
    }
    
    // Determine tile type based on height, temperature, and moisture
    determineTileType(height, temperature, moisture, x, y) {
      // We're explicitly NOT checking for map borders here!
      // That will be handled separately to ensure only the edges get walls
      
      // Determine base tile type from height
      if (height < this.waterLevel) {
        return this.TILE_IDS.WATER;
      } else if (height > this.mountainLevel) {
        return this.TILE_IDS.MOUNTAIN;
      } else {
        // Land tiles - could be floor or obstacle based on biome
        if (moisture > this.forestLevel && temperature > 0.3) {
          // Forest/obstacle in moderate to wet areas
          return this.TILE_IDS.OBSTACLE;
        } else {
          // Basic floor for most land
          return this.TILE_IDS.FLOOR;
        }
      }
    }
    
    // Add walls only at map borders
    addMapBorders(tiles) {
      for (let y = 0; y < this.height; y++) {
        for (let x = 0; x < this.width; x++) {
          // Only put walls at the absolute edges of the map
          if (x === 0 || y === 0 || x === this.width - 1 || y === this.height - 1) {
            tiles[y][x] = this.TILE_IDS.WALL;
          }
        }
      }
    }
    
    // Generate a chunk for the given coordinates
    generateChunkData(chunkRow, chunkCol, chunkSize) {
      const tiles = [];
      
      for (let y = 0; y < chunkSize; y++) {
        const row = [];
        for (let x = 0; x < chunkSize; x++) {
          const globalX = chunkCol * chunkSize + x;
          const globalY = chunkRow * chunkSize + y;
          
          // Check if this position is within the world bounds
          if (globalX >= this.width || globalY >= this.height || globalX < 0 || globalY < 0) {
            // Out of bounds - use floor instead of wall
            row.push({ type: this.TILE_IDS.FLOOR, height: 0 });
            continue;
          }
          
          // Generate the necessary data for this tile
          const heightValue = this.getHeightAt(globalX, globalY);
          const temperature = this.getTemperatureAt(globalX, globalY);
          const moisture = this.getMoistureAt(globalX, globalY);
          
          // Determine tile type (excluding border walls, those are added separately)
          const tileType = this.determineTileType(heightValue, temperature, moisture, globalX, globalY);
          
          // Create a tile object with the type and height value
          row.push({ type: tileType, height: heightValue });
        }
        tiles.push(row);
      }
      
      // Now check and apply border walls ONLY at the edges of the world
      for (let y = 0; y < chunkSize; y++) {
        for (let x = 0; x < chunkSize; x++) {
          const globalX = chunkCol * chunkSize + x;
          const globalY = chunkRow * chunkSize + y;
          
          // Only put walls at the absolute edges of the map
          if (globalX === 0 || globalY === 0 || 
              globalX === this.width - 1 || globalY === this.height - 1) {
            tiles[y][x] = { type: this.TILE_IDS.WALL, height: 0 };
          }
        }
      }
      
      return {
        x: chunkCol,
        y: chunkRow,
        tiles: tiles
      };
    }
    
    // Helper methods to get values at specific coordinates
    getHeightAt(x, y) {
      const continentNoise = this.perlin.fbm(x * 0.002, y * 0.002, 3, 2.0, 0.5);
      const ridgeNoise = this.perlin.ridged(x * 0.01, y * 0.01, 4, 2.0, 0.5);
      const detailNoise = this.perlin.fbm(x * 0.05, y * 0.05, 3, 2.0, 0.5);
      
      const combinedNoise = continentNoise * 0.6 + ridgeNoise * 0.3 + detailNoise * 0.1;
      return this.applyHeightCurve(combinedNoise);
    }
    
    getTemperatureAt(x, y) {
      const latitudeFactor = 1.0 - Math.abs((y / this.height) - 0.5) * 2;
      const noise = this.perlin.get(x * 0.01, y * 0.01) * 0.2;
      return latitudeFactor * 0.8 + noise;
    }
    
    getMoistureAt(x, y) {
      const warpX = x + this.perlin.get(x * 0.02, y * 0.02) * 20;
      const warpY = y + this.perlin.get(x * 0.02 + 40, y * 0.02 + 30) * 20;
      return this.perlin.fbm(warpX * 0.01, warpY * 0.01, 4, 2.0, 0.5);
    }
  }
  
  // Example usage to replace your current MapManager implementation
  
  /**
   * Implementation to integrate with your existing code
   * This shows how to replace the procedural generation parts while
   * keeping your existing MapManager interface
   */
  export function enhanceMapManager(MapManager) {
    // Store the original methods we're going to override
    const originalGenerateChunkData = MapManager.prototype.generateChunkData;
    const originalDetermineTileType = MapManager.prototype.determineTileType;
    
    // Enhanced tile type determination
    MapManager.prototype.determineTileType = function(heightValue, x, y) {
      // ONLY put walls at the absolute edges of the map
      // This is the key fix for your wall problem
      if (x === 0 || y === 0 || x === this.width - 1 || y === this.height - 1) {
        return TILE_IDS.WALL;
      }
      
      // Use enhanced algorithm for everything else
      if (heightValue < -0.4) return TILE_IDS.WATER;
      if (heightValue < 0.3) return TILE_IDS.FLOOR; // Increased this threshold for more floor tiles
      if (heightValue < 0.6) return TILE_IDS.OBSTACLE; // Moved obstacle threshold up
      return TILE_IDS.MOUNTAIN;
    };
    
    // Enhanced chunk generation with better noise
    MapManager.prototype.generateChunkData = function(chunkRow, chunkCol) {
      // Check if we've already created an enhanced perlin instance
      if (!this.enhancedPerlin) {
        this.enhancedPerlin = new EnhancedPerlinNoise(this.perlin.seed);
      }
      
      const tiles = [];
      
      for (let y = 0; y < CHUNK_SIZE; y++) {
        const row = [];
        for (let x = 0; x < CHUNK_SIZE; x++) {
          const globalX = chunkCol * CHUNK_SIZE + x;
          const globalY = chunkRow * CHUNK_SIZE + y;
          
          // Skip if out of world bounds
          if (globalX >= this.width || globalY >= this.height) {
            row.push(new Tile(TILE_IDS.FLOOR)); // Use FLOOR instead of WALL for out of bounds
            continue;
          }
          
          // Use enhanced noise functions for better terrain
          // Combined noise at different scales
          const continentNoise = this.enhancedPerlin.fbm(globalX * 0.002, globalY * 0.002, 3, 2.0, 0.5);
          const ridgeNoise = this.enhancedPerlin.ridged(globalX * 0.01, globalY * 0.01, 4, 2.0, 0.5);
          const detailNoise = this.enhancedPerlin.fbm(globalX * 0.05, globalY * 0.05, 3, 2.0, 0.5);
          
          // Combine noise layers
          const heightValue = continentNoise * 0.6 + ridgeNoise * 0.3 + detailNoise * 0.1;
          
          // Determine tile type with our improved function
          const tileType = this.determineTileType(heightValue, globalX, globalY);
          
          row.push(new Tile(tileType, heightValue));
        }
        tiles.push(row);
      }
      
      return {
        x: chunkCol,
        y: chunkRow,
        tiles: tiles
      };
    };
    
    return MapManager;
  }
  
  // Standalone map generator that you can use if you prefer
  export class ImprovedMapManager {
    constructor(options = {}) {
      this.chunks = new Map(); // Map of "x,y" -> chunk data
      this.width = options.width || 256;
      this.height = options.height || 256;
      this.tileSize = options.tileSize || 12;
      
      // Create enhanced noise generator
      this.seed = options.seed || Math.random();
      this.perlin = new EnhancedPerlinNoise(this.seed);
      
      // Map settings
      this.proceduralEnabled = true;
      this.isFixedMap = false;
      
      console.log(`Improved MapManager initialized with seed: ${this.seed}`);
    }
    
    // Generate a new world
    generateWorld(width, height, options = {}) {
      this.width = width;
      this.height = height;
      
      // Clear existing chunks
      this.chunks.clear();
      
      console.log(`World initialized with size ${width}x${height}`);
      
      // Create a map ID
      const mapId = options.id || `map_${Date.now()}`;
      
      return mapId;
    }
    
    // Get a chunk (generate if needed)
    getChunkData(mapId, chunkX, chunkY) {
      const key = `${mapId || 'default'}_${chunkX},${chunkY}`;
      
      // If chunk exists in cache, return it
      if (this.chunks.has(key)) {
        return this.chunks.get(key);
      }
      
      // Otherwise generate it
      if (this.proceduralEnabled && !this.isFixedMap) {
        const chunkData = this.generateChunkData(chunkX, chunkY);
        this.chunks.set(key, chunkData);
        return chunkData;
      }
      
      return null;
    }
    
    // Generate chunk data with enhanced algorithm
    generateChunkData(chunkRow, chunkCol) {
      const tiles = [];
      
      for (let y = 0; y < CHUNK_SIZE; y++) {
        const row = [];
        for (let x = 0; x < CHUNK_SIZE; x++) {
          const globalX = chunkCol * CHUNK_SIZE + x;
          const globalY = chunkRow * CHUNK_SIZE + y;
          
          // Skip if out of world bounds
          if (globalX >= this.width || globalY >= this.height) {
            row.push(new Tile(TILE_IDS.FLOOR)); // Use FLOOR instead of WALL for out of bounds
            continue;
          }
          
          // Multi-scale noise for better terrain
          const continentNoise = this.perlin.fbm(globalX * 0.002, globalY * 0.002, 3, 2.0, 0.5);
          const ridgeNoise = this.perlin.ridged(globalX * 0.01, globalY * 0.01, 4, 2.0, 0.5);
          const detailNoise = this.perlin.fbm(globalX * 0.05, globalY * 0.05, 3, 2.0, 0.5);
          
          // Combined noise
          const heightValue = continentNoise * 0.6 + ridgeNoise * 0.3 + detailNoise * 0.1;
          
          // Determine tile type
          const tileType = this.determineTileType(heightValue, globalX, globalY);
          
          row.push(new Tile(tileType, heightValue));
        }
        tiles.push(row);
      }
      
      return {
        x: chunkCol,
        y: chunkRow,
        tiles: tiles
      };
    }
    
    // Enhanced tile type determination
    determineTileType(heightValue, x, y) {
      // ONLY put walls at the absolute edges of the map
      if (x === 0 || y === 0 || x === this.width - 1 || y === this.height - 1) {
        return TILE_IDS.WALL;
      }
      
      // Determine tile type based on height - adjusted thresholds
      if (heightValue < -0.4) return TILE_IDS.WATER;
      if (heightValue < 0.3) return TILE_IDS.FLOOR; // More walkable space
      if (heightValue < 0.6) return TILE_IDS.OBSTACLE;
      return TILE_IDS.MOUNTAIN;
    }
    
    // Get a specific tile
    getTile(x, y) {
      // Convert to chunk coordinates
      const chunkX = Math.floor(x / CHUNK_SIZE);
      const chunkY = Math.floor(y / CHUNK_SIZE);
      const localX = x % CHUNK_SIZE;
      const localY = y % CHUNK_SIZE;
      
      // Get chunk
      const chunk = this.getChunkData(null, chunkX, chunkY);
      if (!chunk) return null;
      
      // Get tile from chunk
      return chunk.tiles[localY][localX];
    }
    
    // Check if a position is a wall or obstacle
    isWallOrOutOfBounds(x, y) {
      // Convert to tile coordinates
      const tileX = Math.floor(x / this.tileSize);
      const tileY = Math.floor(y / this.tileSize);
      
      // Check if out of bounds
      if (tileX < 0 || tileY < 0 || tileX >= this.width || tileY >= this.height) {
        return true;
      }
      
      // Get tile type
      const tileType = this.getTileType(tileX, tileY);
      
      // Check if wall or other solid obstacle
      return tileType === TILE_IDS.WALL || 
             tileType === TILE_IDS.MOUNTAIN || 
             tileType === TILE_IDS.WATER;
    }
    
    // Get the tile type at a specific coordinate
    getTileType(x, y) {
      const tile = this.getTile(x, y);
      return tile ? tile.type : null;
    }
  }