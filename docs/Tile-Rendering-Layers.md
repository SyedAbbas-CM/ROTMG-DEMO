# Tile Rendering Layers Issue

## Problem
Currently, all tiles (ground, trees, flowers, obstacles) are rendered on the same layer, causing visual overlap issues. Trees and obstacles appear underneath or mixed with ground tiles.

## Required Solution
Implement proper z-index/layering system with 4 distinct layers:

### Layer 1: Ground/Floor Tiles
- **Types**: grass, sand, water, lava, cobblestone, dirt
- **Biome tiles**: All floor tiles from BiomeDefinitions
- **Z-Index**: 0 (bottom layer)
- **Properties**: Always walkable or unwalkable floor surfaces

### Layer 2: Obstacles
- **Types**: trees, boulders, rocks, walls
- **Biome tiles**: All obstacle tiles from BiomeDefinitions
- **Z-Index**: 1
- **Properties**: Block movement, have collision
- **Examples**: tree, tree_yellow, tree_dead, tree_burnt, boulder, Boulder_yellow, rocks_1, rocks_2, rocks_3

### Layer 3: Decorations/Details
- **Types**: flowers, small plants, ground decorations
- **Biome tiles**: All decor tiles from BiomeDefinitions
- **Z-Index**: 2
- **Properties**: Passable, non-blocking, visual only
- **Examples**: flowers_1, flowers_2

### Layer 4: Entities
- **Types**: players, enemies, projectiles, items
- **Z-Index**: 3 (top layer)
- **Properties**: Dynamic, animated, interactive

## Current Architecture

### Server Side (`src/world/MapManager.js`)
```javascript
// Lines 218-240
const tileSelection = selectTileForGeneration(height, temperature, moisture, globalX, globalY);
// tileSelection contains:
// - tile: tile data with sprite coordinates
// - biome: biome name
// - type: 'floor', 'obstacle', or 'decor'
```

### Tile Type Detection (`src/assets/initTileSystem.js`)
```javascript
// Lines 76-94
// Already determines tile type:
if (obstacleRoll < biome.obstacleDensity) {
  tile = tileRegistry.getBiomeTile(biome.key, 'obstacle', true);
  tileType = 'obstacle';  // ← This is available!
}

if (decorRoll < biome.decorDensity) {
  tile = tileRegistry.getBiomeTile(biome.key, 'decor', true);
  tileType = 'decor';  // ← This is available!
}

// Default floor
tileType = 'floor';  // ← This is available!
```

## Implementation Plan

### 1. Server Side - Already Done! ✅
The server already sends `tileType` ('floor', 'obstacle', 'decor') in tile data. Just need to ensure it's included in network packets.

**File**: `src/world/MapManager.js` line 240
```javascript
tiles.push({
  type: tileType,
  height: height,
  // Include tileType in data
  tileLayerType: tileSelection.type,  // Add this!
  ...tileSelection.tile
});
```

### 2. Client Side - Tile Class
**File**: `public/src/map/tile.js`

Add layer property:
```javascript
constructor(type, height = 0, properties = {}) {
  this.type = type;
  this.height = height;

  // Determine rendering layer based on tile type
  this.layer = this._getLayer(properties.tileLayerType || properties.type);

  // ... rest of constructor
}

_getLayer(tileType) {
  switch(tileType) {
    case 'floor': return 0;
    case 'obstacle': return 1;
    case 'decor': return 2;
    default: return 0;
  }
}
```

### 3. Client Side - Renderer
**File**: `public/src/render/renderTopDown.js`

Render in multiple passes:
```javascript
function renderTiles(ctx, camera, map) {
  // Pass 1: Floor tiles (layer 0)
  for (let tile of visibleTiles) {
    if (tile.layer === 0) {
      drawTile(ctx, tile, x, y);
    }
  }

  // Pass 2: Obstacles (layer 1)
  for (let tile of visibleTiles) {
    if (tile.layer === 1) {
      drawTile(ctx, tile, x, y);
    }
  }

  // Pass 3: Decorations (layer 2)
  for (let tile of visibleTiles) {
    if (tile.layer === 2) {
      drawTile(ctx, tile, x, y);
    }
  }

  // Pass 4: Entities (layer 3) - already handled separately
}
```

## Benefits
- Trees/obstacles render above ground
- Flowers/details render above obstacles
- Players/entities render on top of everything
- Proper visual depth and clarity
- Maintains biome diversity with correct layering

## Files to Modify
1. `src/world/MapManager.js` - Include tileLayerType in tile data
2. `public/src/map/tile.js` - Add layer property
3. `public/src/render/renderTopDown.js` - Implement multi-pass rendering
4. `public/src/render/render.js` - Update main render loop if needed

## Status
- ✅ Biome system working with varied tiles
- ✅ Tile types ('floor', 'obstacle', 'decor') already determined server-side
- ⏸️ Rendering layers not implemented yet
- ⏸️ All tiles currently render on same layer
