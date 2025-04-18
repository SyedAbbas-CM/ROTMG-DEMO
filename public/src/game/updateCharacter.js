// src/game/updateCharacter.js

import { getKeysPressed } from './input.js';
import { gameState } from './gamestate.js';
import { TILE_SIZE, TILE_IDS } from '../constants/constants.js';
import { createLogger, LOG_LEVELS } from '../utils/logger.js';

// Create a logger for this module
const logger = createLogger('movement');

/**
 * Updates the character's position based on input and handles collision.
 * @param {number} delta - Time elapsed since the last frame (in seconds).
 */
export function updateCharacter(delta) {
  const character = gameState.character;
  // Use character's own speed instead of global MOVE_SPEED
  const speed = character.speed || 6.0; // Fallback to 6.0 if character speed isn't defined
  const keysPressed = getKeysPressed();

  // Debug log for speed value occasionally
  logger.occasional(0.01, LOG_LEVELS.DEBUG, `Character speed: ${speed}`);

  // Calculate movement direction
  let moveX = 0;
  let moveY = 0;

  // Process WASD or arrow keys
  if (keysPressed['KeyW'] || keysPressed['ArrowUp']) {
    moveY -= 1;
  }
  if (keysPressed['KeyS'] || keysPressed['ArrowDown']) {
    moveY += 1;
  }
  if (keysPressed['KeyA'] || keysPressed['ArrowLeft']) {
    moveX -= 1;
  }
  if (keysPressed['KeyD'] || keysPressed['ArrowRight']) {
    moveX += 1;
  }

  // Normalize diagonal movement
  if (moveX !== 0 && moveY !== 0) {
    const length = Math.sqrt(moveX * moveX + moveY * moveY);
    moveX /= length;
    moveY /= length;
  }

  // CRITICAL FIX: Force a clean state change when stopping movement
  const isMoving = (moveX !== 0 || moveY !== 0);
  const wasMoving = character.isMoving;
  
  // Update character's movement state
  character.isMoving = isMoving;
  
  // When stopping movement, zero out the movement direction
  if (!isMoving) {
    character.moveDirection = { x: 0, y: 0 };
    
    // CRITICAL FIX: Force the animator to reset to idle directly
    if (wasMoving && character.animator && character.animator.resetToIdle) {
      character.animator.resetToIdle();
    }
  } else {
    // Update move direction when actually moving
    character.moveDirection = { x: moveX, y: moveY };
    
    // CRITICAL FIX: Force animation state to WALK when starting to move
    if (!wasMoving && character.animator && character.animator.states && character.animator.setCurrentState) {
      character.animator.setCurrentState(character.animator.states.WALK);
    }
  }

  // Call the character's update method to handle cooldowns and animation
  if (character.update && typeof character.update === 'function') {
    character.update(delta);
  }

  // Original position before movement
  const originalX = character.x;
  const originalY = character.y;

  // Apply movement with delta time
  if (isMoving) {
    const distance = speed * delta;
    
    // First try moving along X axis
    const newX = character.x + moveX * distance;
    
    if (!isCollision(newX, character.y)) {
      character.x = newX;
    } else {
      // Try with smaller increments to handle edge cases
      const smallStep = Math.sign(moveX) * Math.min(Math.abs(moveX * distance), 0.1);
      const stepX = character.x + smallStep;
      if (!isCollision(stepX, character.y)) {
        character.x = stepX;
      }
    }
    
    // Now try moving along Y axis
    const newY = character.y + moveY * distance;
    if (!isCollision(character.x, newY)) {
      character.y = newY;
    } else {
      // Try with smaller increments
      const smallStep = Math.sign(moveY) * Math.min(Math.abs(moveY * distance), 0.1);
      const stepY = character.y + smallStep;
      if (!isCollision(character.x, stepY)) {
        character.y = stepY;
      }
    }
    
    // If we moved, log the new position occasionally
    if (Math.abs(character.x - originalX) > 0.001 || Math.abs(character.y - originalY) > 0.001) {
      // Only log position every 10 units to avoid spam
      if (Math.floor(character.x) % 10 === 0 && Math.floor(character.y) % 10 === 0) {
        logger.debug(`Position: (${character.x.toFixed(2)}, ${character.y.toFixed(2)})`);
      }
    }
  }
}

/**
 * Checks if the position collides with a wall or is out of bounds
 * @param {number} x - New X position
 * @param {number} y - New Y position
 * @returns {boolean} - True if collision occurs, else false
 */
function isCollision(x, y) {
  // Skip collision if map manager isn't available
  if (!gameState.map) {
    return false;
  }
  
  // Character dimensions (use properties if available, otherwise use defaults)
  const width = gameState.character.width || 20;
  const height = gameState.character.height || 20;
  
  // Check multiple points on the character's body for collisions
  // Center
  if (isPointColliding(x, y)) return true;
  
  // Check corners (adjusted to be slightly inset from the edges)
  const inset = 2; // Inset from edges to avoid getting stuck
  const halfWidth = width / 2 - inset;
  const halfHeight = height / 2 - inset;
  
  // Top-left corner
  if (isPointColliding(x - halfWidth, y - halfHeight)) return true;
  
  // Top-right corner
  if (isPointColliding(x + halfWidth, y - halfHeight)) return true;
  
  // Bottom-left corner
  if (isPointColliding(x - halfWidth, y + halfHeight)) return true;
  
  // Bottom-right corner
  if (isPointColliding(x + halfWidth, y + halfHeight)) return true;
  
  // No collision detected
  return false;
}

/**
 * Checks if a specific point collides with a wall
 * @param {number} x - X position to check
 * @param {number} y - Y position to check
 * @returns {boolean} True if point collides with a wall
 */
function isPointColliding(x, y) {
  // Convert to tile coordinates
  const tileX = Math.floor(x / TILE_SIZE);
  const tileY = Math.floor(y / TILE_SIZE);
  
  try {
    // Check if position is a wall or obstacle
    if (gameState.map.isWallOrObstacle) {
      return gameState.map.isWallOrObstacle(x, y);
    }
    
    // Fallback to checking tile directly
    const tile = gameState.map.getTile(tileX, tileY);
    if (!tile) {
      // No tile found (out of bounds)
      return true;
    }
    
    // Check if it's a wall, obstacle, or mountain
    return (
      tile.type === TILE_IDS.WALL || 
      tile.type === TILE_IDS.OBSTACLE || 
      tile.type === TILE_IDS.MOUNTAIN ||
      tile.type === TILE_IDS.WATER
    );
  } catch (error) {
    logger.error("Error in collision detection:", error);
    // On error, default to no collision
    return false;
  }
}