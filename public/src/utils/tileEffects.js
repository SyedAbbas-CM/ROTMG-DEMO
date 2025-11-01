/**
 * Tile Effects System
 * Handles both movement speed penalties (submersion) and damage from hazardous tiles
 * Supports water (slows), lava (slows + damages), and other tile types
 */

import { gameState } from '../game/gamestate.js';
import { TILE_PROPERTIES, TILE_IDS } from '../constants/constants.js';

// Track last damage time per entity to prevent spam
const lastDamageTime = new Map();

// Track current speed multipliers per entity
const speedMultipliers = new Map();

/**
 * Get the movement speed multiplier for a character based on current tile
 * @param {Object} character - The character/entity to check
 * @returns {number} Speed multiplier (1.0 = normal, <1.0 = slower, >1.0 = faster)
 */
export function getTileSpeedMultiplier(character) {
  if (!character || !gameState.mapManager) return 1.0;

  const tileX = Math.floor(character.x);
  const tileY = Math.floor(character.y);
  const tile = gameState.mapManager.getTile(tileX, tileY);

  if (!tile) return 1.0;

  // Determine tile type from sprite name since the biome system uses sprite names
  let tileType = null;
  if (tile.spriteName) {
    const spriteName = tile.spriteName.toLowerCase();

    // Check for lava tiles
    if (spriteName.includes('lava')) {
      tileType = TILE_IDS.LAVA;
    }
    // Check for water tiles (deep_water, water_1, water_2, etc.)
    else if (spriteName.includes('water') || spriteName.includes('deep')) {
      tileType = TILE_IDS.WATER;
    }
    // Check for sand tiles
    else if (spriteName.includes('sand')) {
      tileType = TILE_IDS.SAND;
    }
  }

  // Fallback to tile.type if available
  if (tileType === null && tile.type !== undefined) {
    tileType = tile.type;
  }

  // Get tile properties based on type
  const tileProps = tileType !== null ? TILE_PROPERTIES[tileType] : null;
  if (!tileProps) return 1.0;

  // movementCost is the inverse of speed
  // movementCost 2.0 means half speed (multiplier 0.5)
  // movementCost 1.0 means normal speed (multiplier 1.0)
  const speedMult = 1.0 / (tileProps.movementCost || 1.0);

  // Cache the multiplier for this character
  const charId = character.id || 'character';
  speedMultipliers.set(charId, speedMult);

  return speedMult;
}

/**
 * Get cached speed multiplier (faster than recalculating every frame)
 * @param {Object} character - The character/entity
 * @returns {number} Cached speed multiplier or 1.0 if not cached
 */
export function getCachedSpeedMultiplier(character) {
  const charId = character?.id || 'character';
  return speedMultipliers.get(charId) || 1.0;
}

/**
 * Update tile effects for a character - call this every frame or every few frames
 * Applies both speed penalties and damage
 * @param {Object} character - The character/entity to update
 */
export function updateTileEffects(character) {
  // DEBUG: Log function being called
  if (Math.random() < 0.01) { // Log 1% of calls to avoid spam
    console.log('[TILE_EFFECTS] updateTileEffects called', {
      hasCharacter: !!character,
      hasMapManager: !!gameState.mapManager,
      characterPos: character ? `(${character.x?.toFixed(2)}, ${character.y?.toFixed(2)})` : 'N/A'
    });
  }

  if (!character || !gameState.mapManager) {
    if (Math.random() < 0.01) {
      console.warn('[TILE_EFFECTS] Missing character or mapManager', {
        hasCharacter: !!character,
        hasMapManager: !!gameState.mapManager
      });
    }
    return;
  }

  const tileX = Math.floor(character.x);
  const tileY = Math.floor(character.y);
  const tile = gameState.mapManager.getTile(tileX, tileY);

  // DEBUG: Log tile info
  if (Math.random() < 0.01) {
    console.log('[TILE_EFFECTS] Tile info', {
      pos: `(${tileX}, ${tileY})`,
      hasTile: !!tile,
      spriteName: tile?.spriteName,
      tileType: tile?.type
    });
  }

  if (!tile) return;

  // Determine tile type from sprite name
  let tileType = null;
  if (tile.spriteName) {
    const spriteName = tile.spriteName.toLowerCase();
    if (spriteName.includes('lava')) {
      tileType = TILE_IDS.LAVA;
    } else if (spriteName.includes('water') || spriteName.includes('deep')) {
      tileType = TILE_IDS.WATER;
    } else if (spriteName.includes('sand')) {
      tileType = TILE_IDS.SAND;
    }
  }

  // Fallback to tile.type if available
  if (tileType === null && tile.type !== undefined) {
    tileType = tile.type;
  }

  const tileProps = tileType !== null ? TILE_PROPERTIES[tileType] : null;

  // DEBUG: Log tile type detection
  if (tileType !== null && Math.random() < 0.05) { // Log 5% when we detect a special tile
    console.log('[TILE_EFFECTS] Detected special tile', {
      pos: `(${tileX}, ${tileY})`,
      spriteName: tile.spriteName,
      detectedType: tileType,
      TILE_IDS_WATER: TILE_IDS.WATER,
      TILE_IDS_LAVA: TILE_IDS.LAVA,
      tileProps: tileProps
    });
  }

  if (!tileProps) return;

  // Update speed multiplier
  getTileSpeedMultiplier(character);

  // Apply damage if tile is hazardous
  if (tileProps.damage && tileProps.damage > 0) {
    applyTileDamage(character, tile, tileProps, tileType);
  }
}

/**
 * Apply damage from hazardous tiles (lava, etc.)
 * @param {Object} character - The character taking damage
 * @param {Object} tile - The tile causing damage
 * @param {Object} tileProps - Tile properties including damage value
 * @param {number} tileType - TILE_IDS value for the tile type
 */
function applyTileDamage(character, tile, tileProps, tileType) {
  // Damage is applied every 500ms
  const damageInterval = 500;
  const now = Date.now();
  const charId = character.id || 'character';
  const lastDmg = lastDamageTime.get(charId) || 0;

  if (now - lastDmg < damageInterval) return;

  // Apply damage to character
  if (character.health !== undefined) {
    character.health = Math.max(0, character.health - tileProps.damage);
    lastDamageTime.set(charId, now);

    // Determine tile name for feedback
    let tileName = 'hazard';
    if (tileType === TILE_IDS.LAVA) tileName = 'lava';
    else if (tileType === TILE_IDS.WATER) tileName = 'deep water';

    console.log(`🔥 ${tileName} damage! -${tileProps.damage} HP (${character.health} remaining)`);

    // Visual feedback - flash effect
    if (character.isFlashing !== undefined) {
      character.isFlashing = true;
      setTimeout(() => { character.isFlashing = false; }, 200);
    }

    // Check for death
    if (character.health <= 0) {
      console.log(`💀 Character died from ${tileName}!`);

      // Trigger death handler
      handlePlayerDeath(character, tileName);
    }
  }
}

/**
 * Reset all tile effect timers and caches (call when changing maps, respawning, etc.)
 */
export function resetTileEffects() {
  lastDamageTime.clear();
  speedMultipliers.clear();
}

/**
 * Get the current tile type the character is standing on
 * @param {Object} character - The character to check
 * @returns {number|null} TILE_IDS value or null
 */
export function getCurrentTileType(character) {
  if (!character || !gameState.mapManager) return null;

  const tileX = Math.floor(character.x);
  const tileY = Math.floor(character.y);
  const tile = gameState.mapManager.getTile(tileX, tileY);

  if (!tile) return null;

  // Determine tile type from sprite name
  let tileType = null;
  if (tile.spriteName) {
    const spriteName = tile.spriteName.toLowerCase();
    if (spriteName.includes('lava')) {
      tileType = TILE_IDS.LAVA;
    } else if (spriteName.includes('water') || spriteName.includes('deep')) {
      tileType = TILE_IDS.WATER;
    } else if (spriteName.includes('sand')) {
      tileType = TILE_IDS.SAND;
    }
  }

  // Fallback to tile.type if available
  if (tileType === null && tile.type !== undefined) {
    tileType = tile.type;
  }

  return tileType;
}

/**
 * Check if a tile type is a water tile
 * @param {number} tileType - TILE_IDS value
 * @returns {boolean} True if tile is water
 */
export function isWaterTile(tileType) {
  return tileType === TILE_IDS.WATER;
}

/**
 * Check if a tile type is a lava tile
 * @param {number} tileType - TILE_IDS value
 * @returns {boolean} True if tile is lava
 */
export function isLavaTile(tileType) {
  return tileType === TILE_IDS.LAVA;
}

/**
 * Check if a tile type is hazardous (deals damage)
 * @param {number} tileType - TILE_IDS value
 * @returns {boolean} True if tile deals damage
 */
export function isHazardousTile(tileType) {
  const props = TILE_PROPERTIES[tileType];
  return props && props.damage && props.damage > 0;
}

/**
 * Handle player death - spawn grave and return to menu
 * @param {Object} character - The character that died
 * @param {string} causeOfDeath - What killed the player
 */
function handlePlayerDeath(character, causeOfDeath) {
  console.log(`[DEATH] Player died from ${causeOfDeath} at (${character.x.toFixed(2)}, ${character.y.toFixed(2)})`);

  // Spawn grave at death location
  spawnGrave(character.x, character.y);

  // Wait 2 seconds to show death, then return to main menu
  setTimeout(() => {
    console.log('[DEATH] Returning to main menu...');

    // Redirect to main menu/play screen
    if (window.location.pathname.includes('game.html')) {
      window.location.href = '/index.html'; // Or wherever your main menu is
    }
  }, 2000);
}

/**
 * Spawn a grave sprite at the given location
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 */
function spawnGrave(x, y) {
  // Use grave_1 or grave_2 randomly
  const graveSprite = Math.random() < 0.5 ? 'lofi_obj_grave_1' : 'lofi_obj_grave_2';

  console.log(`[DEATH] Spawning ${graveSprite} at (${x.toFixed(2)}, ${y.toFixed(2)})`);

  // Create grave object
  const grave = {
    x: x,
    y: y,
    spriteName: graveSprite,
    type: 'grave',
    isPersistent: true, // Graves should stay on the map
    createdAt: Date.now()
  };

  // Add to game objects if window.currentObjects exists
  if (window.currentObjects && Array.isArray(window.currentObjects)) {
    window.currentObjects.push(grave);
    console.log('[DEATH] Grave added to currentObjects');
  }

  // Also try to add to gameState objects
  if (gameState.objects && Array.isArray(gameState.objects)) {
    gameState.objects.push(grave);
    console.log('[DEATH] Grave added to gameState.objects');
  }
}
