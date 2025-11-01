/**
 * tileSpriteUtils.js
 *
 * Shared utility for resolving tile sprite information across all view modes.
 * Implements a consistent priority system:
 *   Priority 1: Biome system (tile.spriteX, tile.spriteY)
 *   Priority 2: Custom sprite override (tile.properties.sprite or tile.spriteName)
 *   Priority 3: Legacy tile type mapping (TILE_SPRITES[tile.type])
 */

import { TILE_SPRITES } from '../constants/constants.js';

/**
 * Gets sprite information for a tile with consistent priority system.
 *
 * @param {Object} tile - The tile object to resolve sprite info for
 * @returns {Object|null} Sprite info object with structure:
 *   {
 *     type: 'biome' | 'custom' | 'legacy',
 *     spriteX: number,      // Pixel X coordinate (for biome/legacy)
 *     spriteY: number,      // Pixel Y coordinate (for biome/legacy)
 *     sheetName: string,    // Sprite sheet name
 *     spriteName: string,   // Sprite name (for custom sprites)
 *     width: number,        // Sprite width in pixels
 *     height: number        // Sprite height in pixels
 *   }
 */
export function getTileSpriteInfo(tile) {
  if (!tile) return null;

  // Priority 1: Biome system sprite coordinates (pixel coords from lofi_environment)
  if (tile.spriteX !== null && tile.spriteX !== undefined &&
      tile.spriteY !== null && tile.spriteY !== undefined) {
    return {
      type: 'biome',
      spriteX: tile.spriteX,
      spriteY: tile.spriteY,
      sheetName: 'lofi_environment',
      spriteName: null,
      width: 8,   // lofi_environment uses 8x8 sprites
      height: 8
    };
  }

  // Priority 2: Custom sprite override (properties.sprite or spriteName)
  const customSpriteName = tile.properties?.sprite || tile.spriteName;
  if (customSpriteName) {
    return {
      type: 'custom',
      spriteX: null,
      spriteY: null,
      sheetName: null,  // Will be resolved by spriteManager
      spriteName: customSpriteName,
      width: null,      // Will be resolved by spriteManager
      height: null
    };
  }

  // Priority 3: Legacy tile type mapping
  const legacyPos = TILE_SPRITES[tile.type];
  if (legacyPos) {
    return {
      type: 'legacy',
      spriteX: legacyPos.x,
      spriteY: legacyPos.y,
      sheetName: 'tile_sprites',
      spriteName: null,
      width: null,      // Should be resolved from sprite sheet config
      height: null
    };
  }

  // No sprite information found
  return null;
}

/**
 * Checks if a tile has biome system sprite coordinates.
 * Quick check without creating full sprite info object.
 *
 * @param {Object} tile - The tile to check
 * @returns {boolean} True if tile has biome coordinates
 */
export function hasBiomeCoordinates(tile) {
  return tile &&
         tile.spriteX !== null && tile.spriteX !== undefined &&
         tile.spriteY !== null && tile.spriteY !== undefined;
}

/**
 * Checks if a tile has a custom sprite override.
 *
 * @param {Object} tile - The tile to check
 * @returns {boolean} True if tile has custom sprite
 */
export function hasCustomSprite(tile) {
  return tile && (tile.properties?.sprite || tile.spriteName);
}
