/**
 * TileRegistry.js
 * High-performance tile and sprite resolution system
 *
 * This provides O(1) lookups for tiles by name, supports biome-based generation,
 * and creates a bridge between semantic tile names and sprite coordinates.
 */

export class TileRegistry {
  constructor() {
    // Fast lookups: name → tile data
    this.nameToTile = new Map();

    // Reverse lookup: sprite coords → tile name
    this.coordsToName = new Map();

    // Biome definitions with tile pools
    this.biomes = new Map();

    // Tile groups for quick filtering
    this.tilesByCategory = new Map();

    // Loaded atlas metadata
    this.atlases = new Map();

    console.log('[TileRegistry] Initialized');
  }

  /**
   * Load tiles from a sprite atlas JSON
   * @param {string} atlasName - Name of the atlas (e.g., 'lofi_environment')
   * @param {Object} atlasData - The atlas JSON data
   */
  loadAtlas(atlasName, atlasData) {
    if (!atlasData || !atlasData.sprites) {
      console.warn(`[TileRegistry] Invalid atlas data for ${atlasName}`);
      return;
    }

    this.atlases.set(atlasName, {
      meta: atlasData.meta,
      sprites: atlasData.sprites
    });

    let namedCount = 0;
    let autoCount = 0;

    // Register each sprite
    for (const sprite of atlasData.sprites) {
      const tileName = sprite.name;
      const isNamed = !tileName.includes('sprite_'); // User-named tile

      const tileData = {
        name: tileName,
        atlas: atlasName,
        row: sprite.row,
        col: sprite.col,
        width: sprite.width || 8,
        height: sprite.height || 8,
        spriteX: sprite.col * (sprite.width || 8),
        spriteY: sprite.row * (sprite.height || 8),
        group: sprite.group || 'auto',
        isNamed: isNamed
      };

      // Name → tile lookup
      this.nameToTile.set(tileName, tileData);

      // Coords → name lookup (for editor tools)
      const coordKey = `${atlasName}:${sprite.row},${sprite.col}`;
      this.coordsToName.set(coordKey, tileName);

      // Add to appropriate category
      if (isNamed) {
        const category = this._inferCategory(tileName);
        if (!this.tilesByCategory.has(category)) {
          this.tilesByCategory.set(category, []);
        }
        this.tilesByCategory.get(category).push(tileData);
        namedCount++;
      } else {
        autoCount++;
      }
    }

    console.log(`[TileRegistry] Loaded ${atlasName}: ${namedCount} named, ${autoCount} auto-generated tiles`);
  }

  /**
   * Get tile data by name (O(1) lookup)
   * @param {string} name - Tile name (e.g., 'grass', 'water_1')
   * @returns {Object|null} Tile data or null
   */
  getTile(name) {
    return this.nameToTile.get(name) || null;
  }

  /**
   * Get tile by row/col coordinates
   * @param {string} atlasName - Atlas name
   * @param {number} row - Row index
   * @param {number} col - Column index
   * @returns {Object|null} Tile data or null
   */
  getTileByCoords(atlasName, row, col) {
    const coordKey = `${atlasName}:${row},${col}`;
    const tileName = this.coordsToName.get(coordKey);
    return tileName ? this.nameToTile.get(tileName) : null;
  }

  /**
   * Get all tiles in a category
   * @param {string} category - Category name (e.g., 'grass', 'water', 'tree')
   * @returns {Array} Array of tile data objects
   */
  getTilesByCategory(category) {
    return this.tilesByCategory.get(category) || [];
  }

  /**
   * Get a random tile from a list of names
   * @param {Array<string>} tileNames - Array of tile names
   * @returns {Object|null} Random tile data
   */
  getRandomTile(tileNames) {
    if (!tileNames || tileNames.length === 0) return null;
    const randomName = tileNames[Math.floor(Math.random() * tileNames.length)];
    return this.getTile(randomName);
  }

  /**
   * Get a weighted random tile
   * @param {Object} tileWeights - Object mapping tile names to weights
   * @returns {Object|null} Selected tile data
   */
  getWeightedRandomTile(tileWeights) {
    if (!tileWeights) return null;

    const entries = Object.entries(tileWeights);
    if (entries.length === 0) return null;

    const totalWeight = entries.reduce((sum, [_, weight]) => sum + weight, 0);
    let random = Math.random() * totalWeight;

    for (const [tileName, weight] of entries) {
      random -= weight;
      if (random <= 0) {
        return this.getTile(tileName);
      }
    }

    // Fallback
    return this.getTile(entries[0][0]);
  }

  /**
   * Register a biome with its tile palette
   * @param {string} biomeName - Name of the biome
   * @param {Object} biomeConfig - Biome configuration
   */
  registerBiome(biomeName, biomeConfig) {
    const floorTiles = this._resolveTileList(biomeConfig.floorTiles || []);
    const wallTiles = this._resolveTileList(biomeConfig.wallTiles || []);
    const obstacleTiles = this._resolveTileList(biomeConfig.obstacleTiles || []);
    const decorTiles = this._resolveTileList(biomeConfig.decorTiles || []);

    const biome = {
      ...biomeConfig,
      key: biomeName,  // Lowercase registry key for lookups
      // Pre-resolve tile references for fast access
      floorTiles,
      wallTiles,
      obstacleTiles,
      decorTiles
    };

    this.biomes.set(biomeName, biome);
  }

  /**
   * Get biome configuration
   * @param {string} biomeName - Name of the biome
   * @returns {Object|null} Biome config or null
   */
  getBiome(biomeName) {
    return this.biomes.get(biomeName) || null;
  }

  /**
   * Select a biome based on noise values
   * @param {number} height - Height noise value (-1 to 1)
   * @param {number} temperature - Temperature noise value (-1 to 1)
   * @param {number} moisture - Moisture noise value (-1 to 1)
   * @returns {Object|null} Biome config
   */
  selectBiome(height, temperature, moisture) {
    // Priority: height first, then temperature/moisture

    // Deep water (lowered threshold to -0.4 from -0.6 for better generation with gentler height curve)
    if (height < -0.4) {
      return this.getBiome('ocean');
    }

    // Shallow water / coast (lowered threshold to -0.2 from -0.3 for better generation)
    if (height < -0.2) {
      return this.getBiome('coast');
    }

    // Beach / sand
    if (height < -0.1) {
      if (temperature > 0.3) {
        return this.getBiome('beach');
      }
      return this.getBiome('coast');
    }

    // Lowlands (main gameplay area)
    if (height < 0.2) {
      // Cold + wet = swamp
      if (temperature < -0.3 && moisture > 0.3) {
        return this.getBiome('swamp');
      }

      // Cold + dry = tundra
      if (temperature < -0.3) {
        return this.getBiome('tundra');
      }

      // Hot + dry = desert
      if (temperature > 0.4 && moisture < -0.2) {
        return this.getBiome('desert');
      }

      // Warm + somewhat dry = plains
      if (temperature > 0.1 && moisture < 0.2) {
        return this.getBiome('plains');
      }

      // Wet = jungle
      if (moisture > 0.5 && temperature > 0.2) {
        return this.getBiome('jungle');
      }

      // Default = grassland
      return this.getBiome('grassland');
    }

    // Hills / forests
    if (height < 0.5) {
      if (moisture > 0.3) {
        return this.getBiome('forest');
      }
      return this.getBiome('hills');
    }

    // Mountains
    if (height < 0.7) {
      if (temperature < -0.2) {
        return this.getBiome('snow_mountain');
      }
      return this.getBiome('mountain');
    }

    // Peaks / volcanic
    if (temperature > 0.6) {
      return this.getBiome('volcanic');
    }
    return this.getBiome('mountain_peak');
  }

  /**
   * Get a tile for a specific biome and tile type
   * @param {string} biomeName - Biome name
   * @param {string} tileType - Type: 'floor', 'wall', 'obstacle', 'decor'
   * @param {boolean} weighted - Use weighted random selection
   * @returns {Object|null} Tile data
   */
  getBiomeTile(biomeName, tileType = 'floor', weighted = true) {
    const biome = this.getBiome(biomeName);
    if (!biome) {
      return null;
    }

    let tileList;
    let weights;

    switch (tileType) {
      case 'floor':
        tileList = biome.floorTiles;
        weights = biome.floorWeights;
        break;
      case 'wall':
        tileList = biome.wallTiles;
        weights = biome.wallWeights;
        break;
      case 'obstacle':
        tileList = biome.obstacleTiles;
        weights = biome.obstacleWeights;
        break;
      case 'decor':
        tileList = biome.decorTiles;
        weights = biome.decorWeights;
        break;
      default:
        return null;
    }

    if (!tileList || tileList.length === 0) {
      return null;
    }

    // Weighted selection if weights provided
    if (weighted && weights) {
      const tileWeights = {};
      for (const tile of tileList) {
        tileWeights[tile.name] = weights[tile.name] || 1;
      }
      return this.getWeightedRandomTile(tileWeights);
    }

    // Random selection
    return tileList[Math.floor(Math.random() * tileList.length)];
  }

  /**
   * Infer category from tile name
   * @private
   */
  _inferCategory(tileName) {
    const name = tileName.toLowerCase();

    if (name.includes('grass')) return 'grass';
    if (name.includes('water') || name.includes('deep')) return 'water';
    if (name.includes('sand')) return 'sand';
    if (name.includes('tree')) return 'tree';
    if (name.includes('rock') || name.includes('boulder')) return 'rock';
    if (name.includes('flower')) return 'flower';
    if (name.includes('lava')) return 'lava';
    if (name.includes('cobble') || name.includes('stone')) return 'stone';

    return 'misc';
  }

  /**
   * Resolve tile names to tile data objects
   * @private
   */
  _resolveTileList(tileNames) {
    const resolved = [];
    for (const name of tileNames) {
      const tile = this.getTile(name);
      if (tile) {
        resolved.push(tile);
      } else {
        console.warn(`[TileRegistry] Tile not found: ${name}`);
      }
    }
    return resolved;
  }

  /**
   * Debug: Print all named tiles
   */
  listNamedTiles() {
    const named = [];
    for (const [name, tile] of this.nameToTile.entries()) {
      if (tile.isNamed) {
        named.push(`  ${name} → ${tile.atlas} (${tile.row}, ${tile.col})`);
      }
    }
    console.log(`[TileRegistry] Named tiles (${named.length}):\n${named.join('\n')}`);
  }

  /**
   * Debug: Print all biomes
   */
  listBiomes() {
    const biomes = [];
    for (const [name, biome] of this.biomes.entries()) {
      biomes.push(`  ${name} → ${biome.floorTiles.length} floor tiles`);
    }
    console.log(`[TileRegistry] Biomes (${biomes.length}):\n${biomes.join('\n')}`);
  }
}

// Singleton instance
export const tileRegistry = new TileRegistry();
