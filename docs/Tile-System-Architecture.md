# Tile System Architecture

## Overview

This document explains the high-performance tile and sprite resolution system designed for your ROTMG-inspired game.

## Problem Statement

**Before:**
- Tiles referenced by numeric IDs (`TILE_IDS.FLOOR`, `TILE_IDS.WATER`)
- Hardcoded tile selection in `determineTileType()`
- No way to use your named tiles (`grass`, `tree`, `water_1`)
- Sprite lookup requires manual row/col calculation
- Difficult to create diverse biomes

**After:**
- Tiles referenced by semantic names (`'grass'`, `'tree'`, `'water_1'`)
- O(1) tile lookups via HashMap
- Biome-based procedural generation using named tiles
- Easy sprite resolution (name → atlas coordinates)
- Fast weighted random selection for tile variety

---

## Architecture Components

### 1. TileRegistry (Core System)

**File:** `src/assets/TileRegistry.js`

**Purpose:** Central registry for all tiles with fast lookups.

#### Key Features:

##### A. Fast Tile Lookup (O(1))
```javascript
// Get tile by name
const grassTile = tileRegistry.getTile('grass');
// Returns:
// {
//   name: 'grass',
//   atlas: 'lofi_environment',
//   row: 4,
//   col: 6,
//   spriteX: 48,
//   spriteY: 32,
//   width: 8,
//   height: 8,
//   isNamed: true
// }
```

##### B. Reverse Lookup (Coords → Name)
```javascript
// Get tile by row/col (useful for sprite editor tools)
const tile = tileRegistry.getTileByCoords('lofi_environment', 4, 6);
// Returns the 'grass' tile
```

##### C. Category-Based Queries
```javascript
// Get all grass tiles
const grassTiles = tileRegistry.getTilesByCategory('grass');
// Returns: ['grass', 'grass_yellow', 'grass_dark']

// Get all water tiles
const waterTiles = tileRegistry.getTilesByCategory('water');
// Returns: ['water_1', 'water_2', 'deep_water', 'deep_water_2']
```

##### D. Weighted Random Selection
```javascript
// Select tile with weighted probabilities
const tile = tileRegistry.getWeightedRandomTile({
  'grass': 70,        // 70% chance
  'grass_yellow': 20, // 20% chance
  'grass_dark': 10    // 10% chance
});
```

##### E. Biome Management
```javascript
// Select biome based on noise values
const biome = tileRegistry.selectBiome(height, temperature, moisture);

// Get a tile from a specific biome
const floorTile = tileRegistry.getBiomeTile('grassland', 'floor', true);
const obstacleT ile = tileRegistry.getBiomeTile('grassland', 'obstacle', true);
```

---

### 2. Biome Definitions

**File:** `src/assets/BiomeDefinitions.js`

**Purpose:** Define all biomes using your named tiles.

#### Biome Structure:

```javascript
grassland: {
  name: 'Grassland',
  description: 'Rolling fields of grass with scattered trees',

  // Tile pools (use your named tiles!)
  floorTiles: ['grass', 'grass_yellow', 'grass_dark'],
  wallTiles: [],
  obstacleTiles: ['tree', 'tree_yellow', 'boulder'],
  decorTiles: ['flowers_1', 'flowers_2'],

  // Weighted random selection
  floorWeights: {
    'grass': 70,
    'grass_yellow': 20,
    'grass_dark': 10
  },

  obstacleWeights: {
    'tree': 50,
    'tree_yellow': 30,
    'boulder': 20
  },

  // Generation parameters
  obstacleDensity: 0.03,  // 3% chance per tile
  decorDensity: 0.02,      // 2% chance per tile

  // Noise ranges for biome selection
  heightRange: [-0.1, 0.2],
  temperatureRange: [-0.3, 0.6],
  moistureRange: [-0.2, 0.5]
}
```

#### Included Biomes:

1. **Grassland** - Default, green grass with trees
2. **Forest** - Dense trees, dark grass
3. **Desert** - Sand, dead trees, sparse
4. **Ocean** - Deep water
5. **Coast** - Shallow water
6. **Beach** - Sandy shores
7. **Swamp** - Dark grass, dead trees, water patches
8. **Tundra** - Cold, barren plains
9. **Jungle** - Dense tropical vegetation
10. **Hills** - Rocky terrain
11. **Mountain** - High elevation rocks
12. **Snow Mountain** - Frozen peaks
13. **Mountain Peak** - Highest elevations, nearly impassable
14. **Volcanic** - Lava and burnt landscape

---

### 3. Tile System Initialization

**File:** `src/assets/initTileSystem.js`

**Purpose:** Load atlases and initialize the registry.

#### Initialization Flow:

```javascript
import { initTileSystem, selectTileForGeneration } from './assets/initTileSystem.js';

// 1. Initialize (do this once at server startup)
await initTileSystem();

// 2. Use in map generation
const tileSelection = selectTileForGeneration(
  heightNoise,      // -1 to 1
  temperatureNoise, // -1 to 1
  moistureNoise,    // -1 to 1
  worldX,           // tile X coordinate
  worldY            // tile Y coordinate
);

// Returns:
// {
//   tile: { name: 'grass', atlas: 'lofi_environment', row: 4, col: 6, ... },
//   biome: 'grassland',
//   type: 'floor'  // or 'obstacle', 'decor'
// }
```

---

## Integration with MapManager

### Current MapManager Flow:

```javascript
// MapManager.js:220
const tileType = this.determineTileType(height, globalX, globalY);
```

### Updated MapManager Flow:

```javascript
// MapManager.js (updated)
import { selectTileForGeneration, tileToLegacyId } from '../assets/initTileSystem.js';

generateChunk(chunkRow, chunkCol) {
  for (let localY = 0; localY < CHUNK_SIZE; localY++) {
    for (let localX = 0; localX < CHUNK_SIZE; localX++) {
      const globalX = chunkCol * CHUNK_SIZE + localX;
      const globalY = chunkRow * CHUNK_SIZE + localY;

      // Generate noise values
      const height = this.perlin.get(globalX / scale, globalY / scale);
      const temperature = this.perlin.get(globalX / 100, globalY / 100);
      const moisture = this.perlin.get((globalX + 500) / 80, (globalY + 500) / 80);

      // Get tile from biome system
      const tileSelection = selectTileForGeneration(
        height,
        temperature,
        moisture,
        globalX,
        globalY
      );

      // Convert to legacy TILE_ID for backward compatibility
      const tileType = tileToLegacyId(tileSelection.tile, tileSelection.type);

      // Store tile data (now includes sprite reference!)
      const def = {
        spriteName: tileSelection.tile.name,
        atlas: tileSelection.tile.atlas,
        spriteRow: tileSelection.tile.row,
        spriteCol: tileSelection.tile.col,
        spriteX: tileSelection.tile.spriteX,
        spriteY: tileSelection.tile.spriteY,
        biome: tileSelection.biome
      };

      row.push(new Tile(tileType, height, def));
    }
  }
}
```

---

## Client-Side Rendering

### Current Rendering (Numeric IDs):

```javascript
// render.js - Old way
const tileId = tile.type;  // e.g., 3 for WATER
const spriteX = TILE_SPRITES[tileId].x;  // Hardcoded lookup
const spriteY = TILE_SPRITES[tileId].y;
ctx.drawImage(tileSpriteSheet, spriteX, spriteY, ...);
```

### New Rendering (Named Tiles):

```javascript
// render.js - New way
const tileDef = tile.definition;  // From server

if (tileDef && tileDef.spriteName) {
  // Use sprite coordinates directly from tile data
  spriteManager.drawSprite(
    ctx,
    tileDef.atlas,        // 'lofi_environment'
    tileDef.spriteX,      // Pre-calculated X pixel position
    tileDef.spriteY,      // Pre-calculated Y pixel position
    screenX, screenY,
    TILE_SIZE, TILE_SIZE
  );
} else {
  // Fallback to legacy rendering
  const spriteX = TILE_SPRITES[tile.type].x;
  const spriteY = TILE_SPRITES[tile.type].y;
  ctx.drawImage(tileSpriteSheet, spriteX, spriteY, ...);
}
```

---

## Performance Characteristics

### Lookup Performance:

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `getTile(name)` | O(1) | HashMap lookup |
| `getTileByCoords(atlas, row, col)` | O(1) | HashMap lookup |
| `getTilesByCategory(category)` | O(1) | Pre-grouped arrays |
| `getWeightedRandomTile(weights)` | O(n) | n = number of tile options |
| `selectBiome(h, t, m)` | O(b) | b = number of biomes (~14) |
| `getBiomeTile(biome, type)` | O(1) | Direct array access |

### Memory Usage:

- **TileRegistry**: ~20 KB (256 tiles × 80 bytes each)
- **Biome Definitions**: ~5 KB (14 biomes × ~350 bytes each)
- **Total Overhead**: ~25 KB

This is negligible compared to sprite atlas images (100+ KB each).

---

## Usage Examples

### Example 1: Simple Tile Lookup

```javascript
import { tileRegistry } from './assets/TileRegistry.js';

// Get a specific tile
const grass = tileRegistry.getTile('grass');
console.log(`Grass tile is at row ${grass.row}, col ${grass.col}`);

// Get all variations of a tile type
const waters = tileRegistry.getTilesByCategory('water');
console.log(`Found ${waters.length} water tiles`);
```

### Example 2: Weighted Random Tile Selection

```javascript
// Define tile palette with weights
const forestFloor = {
  'grass_dark': 70,
  'grass': 30
};

// Select random tile
const tile = tileRegistry.getWeightedRandomTile(forestFloor);
console.log(`Selected: ${tile.name}`);
```

### Example 3: Biome-Based Generation

```javascript
// Generate noise values
const height = perlin.get(x / 32, y / 32);
const temperature = perlin.get(x / 100, y / 100);
const moisture = perlin.get(x / 80, y / 80);

// Select biome
const biome = tileRegistry.selectBiome(height, temperature, moisture);
console.log(`Biome: ${biome.name}`);

// Get floor tile for this biome
const floorTile = tileRegistry.getBiomeTile(biome.name, 'floor', true);
console.log(`Floor tile: ${floorTile.name}`);

// Maybe place an obstacle?
if (Math.random() < biome.obstacleDensity) {
  const obstacle = tileRegistry.getBiomeTile(biome.name, 'obstacle', true);
  console.log(`Placing obstacle: ${obstacle.name}`);
}
```

### Example 4: Adding a New Biome

```javascript
// In BiomeDefinitions.js, add:
export const BIOME_DEFINITIONS = {
  // ... existing biomes ...

  my_new_biome: {
    name: 'My New Biome',
    description: 'A custom biome',

    floorTiles: ['grass', 'sand_1'],
    obstacleT iles: ['boulder', 'tree'],
    decorTiles: ['flowers_1'],

    floorWeights: {
      'grass': 50,
      'sand_1': 50
    },

    obstacleDensity: 0.05,
    decorDensity: 0.03,

    heightRange: [0.3, 0.6],
    temperatureRange: [0.0, 0.5],
    moistureRange: [0.2, 0.7]
  }
};
```

---

## Migration Guide

### Step 1: Initialize System

In `Server.js`, add initialization:

```javascript
import { initTileSystem } from './src/assets/initTileSystem.js';

// After other initializations, before server starts
await initTileSystem();
console.log('[SERVER] Tile system initialized');
```

### Step 2: Update MapManager

Replace `determineTileType()` with biome-based selection:

```javascript
import { selectTileForGeneration, tileToLegacyId } from '../assets/initTileSystem.js';

// In generateChunk(), replace:
// const tileType = this.determineTileType(height, globalX, globalY);

// With:
const tileSelection = selectTileForGeneration(height, temp, moisture, globalX, globalY);
const tileType = tileToLegacyId(tileSelection.tile, tileSelection.type);
const def = {
  spriteName: tileSelection.tile.name,
  atlas: tileSelection.tile.atlas,
  spriteX: tileSelection.tile.spriteX,
  spriteY: tileSelection.tile.spriteY
};
```

### Step 3: Update Client Rendering

In render code, check for `tileDef.spriteName` and use sprite coordinates directly.

### Step 4: Test

1. Start server (should see `[TileSystem] Initialization complete`)
2. Connect client and move around
3. You should see varied terrain with your named tiles!

---

## Benefits

1. **Semantic Names**: Use `'grass'` instead of `TILE_IDS.GRASS` or numeric IDs
2. **Fast Lookups**: O(1) HashMap access instead of array searches
3. **Easy Biomes**: Define biomes in JSON-like config, not code
4. **Sprite Resolution**: Automatic sprite coordinate lookup
5. **Extensible**: Add new tiles/biomes without touching MapManager
6. **Weighted Randomness**: Built-in support for tile variety
7. **Type Safety**: Tile objects have consistent structure
8. **Debug Friendly**: Tile names in logs instead of numbers

---

## Future Enhancements

### 1. Tile Transitions

Blend between biomes with transition tiles:

```javascript
function getTransitionTile(biome1, biome2, blendFactor) {
  // Select tile that visually blends the two biomes
  // e.g., grass → sand transition uses 'grass_yellow'
}
```

### 2. Animated Tiles

Support animated water/lava:

```javascript
const waterTile = tileRegistry.getTile('water_1');
waterTile.animated = true;
waterTile.frames = ['water_1', 'water_2'];
waterTile.frameDuration = 500; // ms
```

### 3. Tile Properties

Extend tiles with gameplay properties:

```javascript
const lavaTile = tileRegistry.getTile('lava_1');
lavaTile.damage = 10;  // Damage per second
lavaTile.slows = 0.5;  // Movement speed multiplier
```

### 4. Client-Side Registry

Load TileRegistry on client too for:
- Sprite rendering
- Minimap colors
- Tile tooltips

---

## Conclusion

This tile system provides:
- **Fast** O(1) tile lookups
- **Flexible** biome-based generation
- **Extensible** easy to add new tiles/biomes
- **Performant** minimal memory overhead
- **Clean** semantic tile names throughout codebase

You can now reference tiles by name (`'grass'`, `'tree'`, `'water_1'`) everywhere, and the biome system automatically creates diverse, beautiful procedural worlds using your hand-named sprites!
