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
  const speed = getMoveSpeed(); // Units per second
  const keysPressed = getKeysPressed();

  let moveX = 0;
  let moveZ = 0;

  if (gameState.camera.viewType === 'first-person') {
    // Calculate forward and right vectors based on yaw
    const forward = new THREE.Vector3(
      Math.sin(character.rotation.yaw),
      0,
      Math.cos(character.rotation.yaw)
    ).normalize();

    const right = new THREE.Vector3(
      Math.sin(character.rotation.yaw + Math.PI / 2),
      0,
      Math.cos(character.rotation.yaw + Math.PI / 2)
    ).normalize();

    // Movement based on keys
    if (keysPressed['KeyW']) {
      moveX -= forward.x * speed * delta;
      moveZ -= forward.z * speed * delta;
    }
    if (keysPressed['KeyS']) {
      moveX += forward.x * speed * delta;
      moveZ += forward.z * speed * delta;
    }
    if (keysPressed['KeyA']) {
      moveX -= right.x * speed * delta;
      moveZ -= right.z * speed * delta;
    }
    if (keysPressed['KeyD']) {
      moveX += right.x * speed * delta;
      moveZ += right.z * speed * delta;
    }
  } else {
    // Movement logic for top-down and strategic views
    if (keysPressed['KeyW'] || keysPressed['ArrowUp']) {
      moveZ -= speed * delta;
    }
    if (keysPressed['KeyS'] || keysPressed['ArrowDown']) {
      moveZ += speed * delta;
    }
    if (keysPressed['KeyA'] || keysPressed['ArrowLeft']) {
      moveX -= speed * delta;
    }
    if (keysPressed['KeyD'] || keysPressed['ArrowRight']) {
      moveX += speed * delta;
    }
  }

  // Normalize movement to prevent faster diagonal movement
  const length = Math.hypot(moveX, moveZ);
  if (length > 0) {
    moveX /= length;
    moveZ /= length;
    moveX *= speed * delta;
    moveZ *= speed * delta;
  }

  // Calculate new potential position in world coordinates
  const newX = character.x + moveX;
  const newZ = character.y + moveZ;

  // Collision detection
  if (!isCollision(newX, newZ)) {
    character.x = newX;
    character.y = newZ;
    if (gameState.camera.viewType !== 'first-person') {
      console.log(`Character moved to (${character.x.toFixed(2)}, ${character.y.toFixed(2)}).`);
    }
  } else {
    if (gameState.camera.viewType !== 'first-person') {
      console.log(`Collision detected at (${newX.toFixed(2)}, ${newZ.toFixed(2)}). Movement blocked.`);
    }
  }

  // Note: In first-person view, rotation is controlled by the mouse, not by movement
}

/**
 * Checks if the new position collides with any impassable tiles.
 * @param {number} x - New X position in world coordinates.
 * @param {number} z - New Z position in world coordinates.
 * @returns {boolean} - True if collision occurs, else false.
 */
function isCollision(x, z) {
  const tileX = Math.floor(x / TILE_SIZE);
  const tileZ = Math.floor(z / TILE_SIZE);
  const tile = map.getTile(tileX, tileZ);

  if (!tile) return true; // Outside map bounds
  if (tile.type === TILE_IDS.WALL || tile.type === TILE_IDS.MOUNTAIN) {
    return true;
  }

  return false;
}
