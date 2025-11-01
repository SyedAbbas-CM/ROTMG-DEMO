/**
 * initTileSystem.js
 * Initialize the tile registry and load biome definitions
 */

import { tileRegistry } from './TileRegistry.js';
import { BIOME_DEFINITIONS } from './BiomeDefinitions.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Initialize the tile system
 * Loads sprite atlases and registers biomes
 */
export async function initTileSystem() {
  console.log('[TileSystem] Initializing...');

  // Load lofi_environment atlas
  const atlasPath = path.join(__dirname, '../../public/assets/atlases/lofi_environment.json');

  try {
    const atlasData = JSON.parse(fs.readFileSync(atlasPath, 'utf8'));
    tileRegistry.loadAtlas('lofi_environment', atlasData);

    console.log(`[TileSystem] Loaded lofi_environment atlas`);
  } catch (error) {
    console.error(`[TileSystem] Failed to load atlas:`, error.message);
    return false;
  }

  // Register all biomes
  for (const [biomeName, biomeConfig] of Object.entries(BIOME_DEFINITIONS)) {
    tileRegistry.registerBiome(biomeName, biomeConfig);
  }

  console.log(`[TileSystem] Registered ${Object.keys(BIOME_DEFINITIONS).length} biomes`);

  // Debug: List named tiles (commented out by default)
  // tileRegistry.listNamedTiles();
  // tileRegistry.listBiomes();

  console.log('[TileSystem] Initialization complete');
  return true;
}

/**
 * Export the tile registry for direct access
 */
export { tileRegistry };

/**
 * Get a tile for map generation
 * @param {number} height - Height noise (-1 to 1)
 * @param {number} temperature - Temperature noise (-1 to 1)
 * @param {number} moisture - Moisture noise (-1 to 1)
 * @param {number} x - World X coordinate
 * @param {number} y - World Y coordinate
 * @returns {Object} Tile selection result
 */
export function selectTileForGeneration(height, temperature, moisture, x, y) {
  // Select biome
  const biome = tileRegistry.selectBiome(height, temperature, moisture);

  if (!biome) {
    console.warn('[TileSystem] No biome selected, using fallback');
    return {
      tile: tileRegistry.getTile('grass'),
      biome: 'grassland',
      type: 'floor'
    };
  }

  // Determine tile type - ALWAYS use floor tiles
  // Objects (trees, rocks, etc.) are generated separately in MapManager
  let tileType = 'floor';
  let tile = null;

  // Get the floor tile for this biome
  tile = tileRegistry.getBiomeTile(biome.key, 'floor', true);
  tileType = 'floor';

  // Fallback to 'grass' if still no tile
  if (!tile) {
    tile = tileRegistry.getTile('grass');
  }

  return {
    tile: tile,
    biome: biome.name,
    type: tileType
  };
}

/**
 * Convert tile data to TILE_ID for legacy compatibility
 * @param {Object} tile - Tile data from registry
 * @param {string} tileType - Type: 'floor', 'obstacle', 'decor'
 * @returns {number} TILE_IDS value
 */
export function tileToLegacyId(tile, tileType) {
  // Import TILE_IDS for conversion
  // For now, map based on type
  if (!tile) return 0;

  // Map tile names to TILE_IDS
  const name = tile.name.toLowerCase();

  // Water tiles
  if (name.includes('water') || name.includes('deep')) {
    return 3; // TILE_IDS.WATER
  }

  // Lava tiles
  if (name.includes('lava')) {
    return 6; // TILE_IDS.LAVA
  }

  // Sand tiles
  if (name.includes('sand')) {
    return 5; // TILE_IDS.SAND
  }

  // Stone/cobble
  if (name.includes('cobble') || name.includes('stone')) {
    return 11; // TILE_IDS.STONE
  }

  // Wall/Mountain obstacles (but NOT trees/rocks - those are objects)
  if (tileType === 'obstacle' && name.includes('wall')) {
    return 2; // TILE_IDS.OBSTACLE
  }

  // Grass tiles (default floor)
  if (name.includes('grass')) {
    return 8; // TILE_IDS.GRASS
  }

  // Default floor
  return 0; // TILE_IDS.FLOOR
}
