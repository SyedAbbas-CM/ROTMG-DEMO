# Water & Lava Submersion System - Implementation Summary

## Overview

I've created a comprehensive tile effects system (`tileEffects.js`) that handles both movement speed penalties (submersion) and damage from hazardous tiles.

## Current Status

### ✅ Completed:
1. **Created `/public/src/utils/tileEffects.js`** - New comprehensive system
2. **Updated `BiomeDefinitions.js`** - Proper rock/cobblestone distribution
3. **Improved Biome Generation** - Large coherent regions with smooth transitions

### What Already Exists in constants.js:

```javascript
TILE_PROPERTIES = {
  [TILE_IDS.WATER]: {
    isWalkable: true,
    movementCost: 2.0,  // Half speed (1.0 / 2.0 = 0.5x)
    damage: 0
  },
  [TILE_IDS.LAVA]: {
    isWalkable: true,
    movementCost: 3.0,  // Third speed (1.0 / 3.0 = 0.33x)
    damage: 20          // 20 HP every 500ms
  }
}
```

## New Tile Effects System

### Key Features:

**1. Movement Speed Penalties**
- `getTileSpeedMultiplier(character)` - Returns speed multiplier based on current tile
- Water: 0.5x speed
- Lava: 0.33x speed
- Cached for performance

**2. Damage Over Time**
- Lava: 20 damage every 500ms
- Deep water: No damage (just slow)
- Visual feedback (character flashing)
- Console logging for debugging

**3. Combined Updates**
- `updateTileEffects(character)` - Call once per frame
- Updates both speed AND damage
- Efficient caching system

### Integration Points Needed:

**1. Game Loop (game.js:900-902)**
Current code:
```javascript
// Check tile damage (lava, etc.)
if (typeof checkTileDamage === 'function') {
    checkTileDamage();
}
```

Should become:
```javascript
// Update tile effects (speed + damage)
if (typeof updateTileEffects === 'function') {
    updateTileEffects(gameState.character);
}
```

**2. Player Movement Code**
Needs to apply speed multiplier when moving. This is typically in one of:
- `game.js` - `updateCharacter()` function
- `input.js` - Input handling
- `PlayerManager.js` - Player movement logic

Movement should be:
```javascript
const speedMult = getTileSpeedMultiplier(character);
const effectiveSpeed = baseSpeed * speedMult;
character.x += dx * effectiveSpeed * delta;
character.y += dy * effectiveSpeed * delta;
```

**3. Imports**
Add to game.js imports:
```javascript
import { updateTileEffects, getTileSpeedMultiplier } from '../utils/tileEffects.js';
```

Remove old import:
```javascript
// import { checkTileDamage } from '../utils/tileDamage.js'; // OLD - REMOVE
```

## Testing Checklist

Once integrated, test:

1. ✅ Walk on grass/normal tiles - Normal speed
2. ⚠️ Walk into deep water (ocean biome) - Should slow to half speed, NO damage
3. ⚠️ Walk into lava (volcanic biome at high elevations) - Should slow to 1/3 speed AND take damage
4. ⚠️ Console shows damage messages when in lava
5. ⚠️ Character HP decreases in lava
6. ⚠️ Character death triggers if HP reaches 0

## Map System Info

**Current Map**: 512x512 tiles (not infinite)
- Uses procedural chunk-based generation
- Biomes: grassland, plains, forest, desert, ocean, coast, beach, swamp, tundra, jungle, hills, mountain, snow_mountain, mountain_peak, volcanic

**Water Biomes**:
- Ocean: height < -0.4
- Coast: -0.4 < height < -0.2
- Beach: -0.2 < height < -0.1

**Lava Biomes**:
- Volcanic: height >= 0.7 AND temperature > 0.6

## Biome Generation Improvements

**Fixed Issues**:
1. ❌ OLD: Random scattered single-tile biomes
   ✅ NEW: Large 100-200 tile coherent regions

2. ❌ OLD: Trees appearing in lava/water
   ✅ NEW: Incompatible biome check prevents this

3. ❌ OLD: No lava/volcanic biomes appearing
   ✅ NEW: Increased noise multipliers (3.0) to reach extreme values

**Key Parameters** (MapManager.js:245-259):
- BIOME_SCALE = 300 (creates large regions)
- DETAIL_SCALE = 60 (adds variation)
- Multipliers = 3.0 (reaches extreme biome thresholds)
- Layering: 90% biome + 10% detail

## Strategic View Caching

**TODO**: When switching from strategic view to top-down:
- Strategic view tiles/objects should be cached
- Not re-drawn in top-down mode
- Improves performance

This needs investigation in the rendering system (likely in `render.js` or `renderTopDown.js`).

## ✅ INTEGRATION COMPLETE

All integration steps have been completed:

1. ✅ **Integrated `updateTileEffects()` into game loop** (game.js:899-902)
2. ✅ **Applied speed multiplier to player movement** (updateCharacter.js:52-61)
3. ✅ **Updated tile effects system to use sprite names** instead of numeric TILE_IDS
   - System now detects water/lava by checking `tile.spriteName`
   - Supports all water variants: 'deep_water', 'deep_water_2', 'water_1', 'water_2', etc.
   - Supports all lava variants: 'lava_1', 'lava_2', etc.

## Ready for Testing

The system is now ready to test in-game:
- Walk into ocean/coast biomes (deep_water, water tiles) → should slow to 0.5x speed
- Walk into volcanic biomes (lava_1, lava_2 tiles) → should slow to 0.33x speed AND take 20 HP damage every 500ms
- Console will show debug messages when slowdown/damage is applied

## Next Steps

1. Test in-game with water and lava biomes
2. Implement strategic view caching (separate task)
3. Balance damage/speed values if needed based on gameplay testing

## File Locations

- **New System**: `/public/src/utils/tileEffects.js`
- **Old System**: `/public/src/utils/tileDamage.js` (can be removed after migration)
- **Game Loop**: `/public/src/game/game.js:847-967`
- **Constants**: `/public/src/constants/constants.js:43-56`
- **Biome Defs**: `/src/assets/BiomeDefinitions.js`
- **Map Gen**: `/src/world/MapManager.js:243-360`
