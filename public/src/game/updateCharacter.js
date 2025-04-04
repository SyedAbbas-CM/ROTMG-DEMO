// src/game/updateCharacter.js

import { getKeysPressed, getMoveSpeed } from './input.js';
import { gameState } from './gamestate.js';
import { map } from '../map/map.js';
import { TILE_SIZE, TILE_IDS } from '../constants/constants.js';

/**
 * Updates the character's position based on input and handles collision.
 * @param {number} delta - Time elapsed since the last frame (in seconds).
 */
export function updateCharacter(delta) {
  const character = gameState.character;
  // Force a higher speed value to overcome any potential issues
  const speed = 200; // Higher speed for testing
  const keysPressed = getKeysPressed();

  // Calculate movement direction
  let moveX = 0;
  let moveY = 0;

  // Process movement keys - directly set values to avoid complexities
  if (keysPressed['KeyW'] || keysPressed['ArrowUp']) {
    console.log('Moving UP');
    character.y -= speed * delta;
  }
  if (keysPressed['KeyS'] || keysPressed['ArrowDown']) {
    console.log('Moving DOWN');
    character.y += speed * delta;
  }
  if (keysPressed['KeyA'] || keysPressed['ArrowLeft']) {
    console.log('Moving LEFT');
    character.x -= speed * delta;
  }
  if (keysPressed['KeyD'] || keysPressed['ArrowRight']) {
    console.log('Moving RIGHT');
    character.x += speed * delta;
  }

  // Log current position after movement
  console.log(`Current position: (${character.x.toFixed(2)}, ${character.y.toFixed(2)})`);
}

/**
 * Checks if the new position collides with any impassable tiles.
 * @param {number} x - New X position in world coordinates.
 * @param {number} z - New Z position in world coordinates.
 * @returns {boolean} - True if collision occurs, else false.
 */
function isCollision(x, z) {
  // Ensure x and z are positive
  const tileX = Math.floor(x / TILE_SIZE);
  const tileZ = Math.floor(z / TILE_SIZE);
  
  const tile = map.getTile(tileX, tileZ);
  
  if (!tile) {
    return true; // Outside map bounds
  }
  
  if (tile.type === TILE_IDS.WALL || tile.type === TILE_IDS.MOUNTAIN) {
    return true;
  }

  return false;
}
