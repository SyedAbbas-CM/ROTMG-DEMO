// Managers/world/tile.js

import { TILE_IDS } from './constants.js';

// Server-side tiles can optionally receive a `walkable` flag (e.g. via the
// TileDatabase).  We derive a sensible default from the numeric TILE_IDS if it
// is not explicitly present.

function defaultWalkableForType(type) {
  return !(
    type === TILE_IDS.WALL ||
    type === TILE_IDS.OBSTACLE ||
    type === TILE_IDS.WATER ||
    type === TILE_IDS.MOUNTAIN
  );
}

export class Tile {
    constructor(type, height = 0, properties = {}) {
      this.type = type;             // numeric TILE_IDS value
      this.height = height;         // 3-D height/layer for rendering
      this.properties = properties; // Extended data from TileDatabase

      // Normalise walkable flag
      this.walkable = (properties.walkable !== undefined)
        ? !!properties.walkable
        : defaultWalkableForType(type);

      // Expose sprite metadata as top-level properties for renderer access
      // Use nullish coalescing (??) to preserve 0 values (valid sprite coordinates)
      this.spriteName = properties.spriteName ?? null;
      this.atlas = properties.atlas ?? null;
      this.spriteRow = properties.spriteRow ?? null;
      this.spriteCol = properties.spriteCol ?? null;
      this.spriteX = properties.spriteX ?? null;
      this.spriteY = properties.spriteY ?? null;
      this.biome = properties.biome ?? null;
    }

    /** Convenience wrapper so callers can ask tile.isWalkable() like on client */
    isWalkable() {
      return this.walkable;
    }
  }
  