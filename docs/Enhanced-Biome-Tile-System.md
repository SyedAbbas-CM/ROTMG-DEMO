# Enhanced Biome & Tile System

## Overview

This document describes the enhanced tile naming, biome generation, and world management system for your ROTMG-inspired game.

## What I've Created

### 1. **Tile Naming Database** (`public/assets/database/tile-names.json`)

Maps sprite atlas coordinates to semantic, human-readable tile names with biome associations.

**Structure:**
```json
{
  "atlases": {
    "lofi_environment": {
      "tiles": {
        "grass": {
          "name": "Grass",
          "sprite": "lofi_environment_sprite_0_0",
          "row": 0,
          "col": 0,
          "biomes": ["grassland", "plains"],
          "walkable": true,
          "category": "floor"
        }
      }
    }
  }
}
```

**Categories:**
- `floor` - Walkable ground tiles
- `wall` - Movement-blocking obstacles
- `water` - Water tiles (usually unwalkable)
- `hazard` - Damaging tiles (lava, spikes, etc.)
- `decoration` - Visual-only elements

### 2. **Biome Definitions** (`public/assets/database/biomes.json`)

Defines 11 biomes with tile associations, environmental parameters, enemies, and dungeons.

**Biomes Included:**
1. **Grassland** - Default starting area
2. **Desert** - Hot, dry wasteland
3. **Ocean** - Deep waters
4. **Beach** - Sandy coastline
5. **Mountain** - Rocky peaks
6. **Tundra** - Cold barren landscape
7. **Ice Plains** - Frozen wasteland
8. **Swamp** - Murky wetlands
9. **Jungle** - Dense tropical forest
10. **Dungeon Stone** - Crafted corridors
11. **Volcanic** - Molten lava landscape

**Parameters:**
- `heightRange` - Elevation thresholds [-1.0 to 1.0]
- `temperatureRange` - Climate values [-1.0 to 1.0]
- `moistureRange` - Humidity levels [-1.0 to 1.0]
- `tiles` - Which tiles to use for this biome
- `tileWeights` - Probability distribution for tile variants
- `enemies` - Enemy types that spawn here
- `dungeons` - Dungeon types that can appear

### 3. **Sprite Labeling Tool** (`public/tools/sprite-labeler.html`)

Interactive web tool for naming and categorizing tiles from sprite atlases.

**Features:**
- Visual atlas grid display
- Click-to-select tiles
- Form-based metadata entry
- Export to JSON format
- Load existing databases
- Preview selected tiles

**How to Use:**
1. Open `http://localhost:3000/tools/sprite-labeler.html`
2. Load your tile atlas image
3. Set tile size (usually 8px for lofi_environment)
4. Click tiles to label them
5. Fill in: name, category, biomes, walkability
6. Click "Save This Tile"
7. Export JSON when done
8. Replace `/public/assets/database/tile-names.json` with your export

## Current System Architecture

### Existing Files

**Frontend:**
- `public/src/assets/SpriteDatabase.js` - Sprite lookup system
- `public/src/assets/EntityDatabase.js` - Entity definitions
- `public/src/assets/TileDatabase.js` - Tile definitions
- `public/assets/atlases/*.json` - Sprite sheet definitions

**Backend:**
- `src/world/MapManager.js` - Map generation & chunk management
- `src/world/PerlinNoise.js` - Noise generation
- `src/world/AdvancedPerlinNoise.js` - Enhanced noise
- `src/routes/portalRoutes.js` - Portal API endpoints
- `Server.js` (line 750-803) - Portal handling logic

### Current Portal System

**How It Works:**
1. Portals are defined as objects in map metadata
2. Server checks player position every tick (line 750)
3. If player stands on portal tile → instant teleport
4. MAP_INFO and WORLD_SWITCH messages sent to client

**Portal Object Format:**
```javascript
{
  id: "portal_unique_id",
  type: "portal",
  sprite: "portal",
  x: 5,
  y: 5,
  destMap: "map_2"
}
```

**API Endpoints** (`/api/portals/*`):
- `GET /api/portals/list?mapId=...` - List portals
- `POST /api/portals/add` - Add portal
- `POST /api/portals/remove` - Remove portal
- `POST /api/portals/link-both` - Create bidirectional portal

## Next Steps - What Needs to Be Implemented

### Phase 1: Biome-Based Tile Generation ⏳

**File to Modify:** `src/world/MapManager.js`

**Current Code** (line 279):
```javascript
determineTileType(heightValue, x, y) {
  // Uses hardcoded TILE_IDS constants
  if (heightValue < -0.6) return TILE_IDS.WATER;
  if (heightValue < -0.3) return TILE_IDS.WATER;
  // ... etc
}
```

**Needed Changes:**
1. Load `biomes.json` and `tile-names.json` at server startup
2. Create `determineBiome(height, temp, moisture)` function
3. Modify `determineTileType()` to:
   - First determine biome
   - Then select tile from that biome's tile palette
   - Use weighted random selection for variants
4. Return tile NAME instead of numeric ID

**Pseudocode:**
```javascript
determineTileType(heightValue, x, y) {
  const temp = this.perlin.get(x / 100, y / 100);
  const moisture = this.perlin.get(x / 80 + 500, y / 80 + 500);

  // Find matching biome
  const biome = this.findBiomeForConditions(heightValue, temp, moisture);

  // Select tile from biome's palette
  const tileName = this.selectTileFromBiome(biome);

  // Look up tile sprite from tile-names database
  return this.getTileDataByName(tileName);
}
```

### Phase 2: Tile Blending & Transitions 🎨

**Goal:** Smooth visual transitions between biomes

**Approach:**
1. For each tile, check neighboring tiles' biomes
2. If on biome boundary, use transition tiles
3. Use `transition_rules` from `biomes.json`

**Algorithm:**
```javascript
function getBlendedTile(x, y, primaryBiome) {
  const neighbors = getNeighborBiomes(x, y);

  if (allSameBiome(neighbors, primaryBiome)) {
    return selectTileFromBiome(primaryBiome);
  }

  // Find dominant neighbor biome
  const secondaryBiome = getMostCommonNeighbor(neighbors);
  const transitionKey = `${primaryBiome}_to_${secondaryBiome}`;

  if (biomeTransitions[transitionKey]) {
    return randomFromArray(biomeTransitions[transitionKey]);
  }

  return selectTileFromBiome(primaryBiome);
}
```

### Phase 3: Dungeon Instance System 🏰

**Current Limitation:** All maps are persistent and shared

**Needed Features:**
1. **Instance Creation:**
   ```javascript
   class DungeonInstanceManager {
     createInstance(dungeonType, party) {
       const instanceId = generateUniqueId();
       const mapId = generateDungeonMap(dungeonType);
       this.instances.set(instanceId, {
         mapId,
         players: new Set(party),
         createdAt: Date.now(),
         expiresAt: Date.now() + (30 * 60 * 1000)
       });
       return { instanceId, mapId };
     }
   }
   ```

2. **Portal Types:**
   - `realm_gate` - Persistent world portals
   - `dungeon_entrance` - Creates instance
   - `boss_room` - Requires key item

3. **Spawn Points:**
   - Define spawn coords in portal object
   - `spawnX`, `spawnY` fields
   - Prevents spawning at portal location

### Phase 4: Enhanced Portal Visuals ✨

**Features:**
- Portal sprite animations (spinning, glowing)
- Entry/exit effects
- Loading transition screens
- Portal discovery notifications

## File Structure Summary

```
ROTMG-DEMO/
├── public/
│   ├── assets/
│   │   └── database/
│   │       ├── tile-names.json        [NEW] ✨ Tile naming database
│   │       ├── biomes.json           [NEW] ✨ Biome definitions
│   │       └── tiles.json            [EXISTING] Legacy tile defs
│   └── tools/
│       └── sprite-labeler.html       [NEW] ✨ Tile labeling tool
├── src/
│   ├── world/
│   │   └── MapManager.js             [TO MODIFY] Add biome logic
│   └── routes/
│       └── portalRoutes.js           [EXISTING] Portal API
└── Server.js                         [TO MODIFY] Add dungeon instances
```

## Quick Start Guide

### 1. Label Your Tiles

```bash
# Open in browser:
http://localhost:3000/tools/sprite-labeler.html

# Steps:
1. Load atlas image (e.g., lofi_environment.png)
2. Set tile size to 8
3. Click each tile
4. Enter: name, category, biomes, walkability
5. Export JSON
6. Save to public/assets/database/tile-names.json
```

### 2. Test Existing System

Current portals work! Try:
```javascript
// In browser console:
fetch('/api/portals/list').then(r => r.json()).then(console.log)
```

### 3. Customize Biomes

Edit `public/assets/database/biomes.json`:
- Adjust height/temp/moisture ranges
- Change tile weights
- Add new biomes
- Define enemy spawns

## Benefits of This System

✅ **Tile Naming:** Human-readable tile IDs instead of numeric constants
✅ **Biome Variety:** 11 distinct biomes with unique tiles and enemies
✅ **Easy Expansion:** Add new tiles/biomes via JSON (no code changes)
✅ **Visual Tool:** Label tiles with sprite-labeler.html
✅ **Existing Integration:** Works with your current SpriteDatabase/EntityDatabase
✅ **Portal System:** Already functional with API endpoints

## What You Need To Do Next

1. ✅ **Use the sprite labeler** to name all your terrain tiles
2. ⏳ **Implement biome generation** in MapManager.js
3. ⏳ **Add tile blending** for smooth biome transitions
4. ⏳ **Create dungeon instances** for instanced gameplay
5. ⏳ **Add portal animations** for better visual feedback

Let me know which part you'd like me to implement first!
