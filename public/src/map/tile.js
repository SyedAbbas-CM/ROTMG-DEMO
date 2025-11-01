// src/map/tile.js

/**
 * Enhanced Tile class with more flexible properties for rendering and collision
 */
export class Tile {
    /**
     * Create a new tile
     * @param {number} type - The TILE_IDS value of this tile
     * @param {number} height - Height value for 3D rendering (0 = flat)
     * @param {Object} properties - Additional tile properties
     * @param {number} [properties.spriteIndex] - Override the default sprite index for this tile type
     * @param {boolean} [properties.isWalkable] - Override the walkability for this tile type
     * @param {boolean} [properties.isTransparent] - Whether light can pass through this tile
     * @param {Object} [properties.customData] - Any additional custom data for this tile
     * @param {number} [properties.variation] - Variation index for tiles with multiple appearances
     */
    constructor(type, height = 0, properties = {}) {
      this.type = type; // e.g., TILE_IDS.FLOOR, TILE_IDS.WALL, etc.
      this.height = height; // Height value for 3D rendering
      
      // Default properties based on type
      const defaultProps = {
        isWalkable: ![1, 2, 4].includes(type), // All except WALL, OBSTACLE, MOUNTAIN are walkable (WATER is walkable)
        isTransparent: type !== 1, // All except WALL are transparent
        variation: 0, // Default variation
      };
      
      // Merge default properties with provided properties
      this.properties = {...defaultProps, ...properties};

      // Expose sprite metadata as top-level properties for renderer access
      // This matches the server-side Tile constructor behavior
      this.spriteName = properties.spriteName ?? null;
      this.atlas = properties.atlas ?? null;
      this.spriteRow = properties.spriteRow ?? null;
      this.spriteCol = properties.spriteCol ?? null;
      this.spriteX = properties.spriteX ?? null;
      this.spriteY = properties.spriteY ?? null;
      this.biome = properties.biome ?? null;
    }
    
    /**
     * Check if this tile allows walking
     * @returns {boolean} True if walkable
     */
    isWalkable() {
      return !!this.properties.isWalkable;
    }
    
    /**
     * Get the sprite index for rendering, which might be different from the tile type
     * @returns {number} Sprite index to use for rendering
     */
    getSpriteIndex() {
      // If a specific sprite is specified, use it
      if (this.properties.spriteIndex !== undefined) {
        return this.properties.spriteIndex;
      }
      
      // Calculate sprite based on type and variation
      const variation = this.properties.variation || 0;
      return this.type + (variation * 10); // Leaving room for variations
    }
    
    /**
     * Create a copy of this tile with modified properties
     * @param {Object} newProperties - Properties to update
     * @returns {Tile} New tile with updated properties
     */
    withProperties(newProperties) {
      return new Tile(
        this.type,
        this.height,
        {...this.properties, ...newProperties}
      );
    }
    
    /**
     * Check if this tile blocks movement (is wall or obstacle)
     * @returns {boolean} True if tile blocks movement
     */
    isBlockingMovement() {
      return !this.isWalkable();
    }
  }
  