// src/game/updateCharacter.js

import { getKeysPressed } from './input.js';
import { gameState } from './gamestate.js';
import { TILE_IDS } from '../constants/constants.js';
import { createLogger, LOG_LEVELS } from '../utils/logger.js';
import * as THREE from 'three'; // Added for THREE.Vector3 and THREE.Quaternion

// Create a logger for this module
const logger = createLogger('movement');

// Toggle flags – tweak in console when needed
export const DEBUG_MOVEMENT = false;           // Set to true for verbose movement logs
window.DEBUG_COLLISION = false;               // Collision overlay (can be toggled by hot-key)

// Initialize collision stats object at module load time
if (!window.COLLISION_STATS) {
  window.COLLISION_STATS = {
    totalCollisions: 0,
    ghostCollisions: 0,
    collisionPositions: [],
    entityCollisions: 0,
    lastEntityCollisions: []
  };
}

/**
 * Updates the character's position based on input and handles collision.
 * @param {number} delta - Time elapsed since the last frame (in seconds).
 */
export function updateCharacter(delta) {
  const character = gameState.character;
  
  // Abort movement and logic when character is dead
  if (!character || (typeof character.health === 'number' && character.health <= 0)) {
    return; // dead or missing character – skip rest of update
  }
  
  // Verbose coordinate sanity-check – only when DEBUG_MOVEMENT is true
  if (DEBUG_MOVEMENT && character && gameState.map) {
    const worldX = character.x;
    const worldY = character.y;
    const tileX = Math.floor(worldX);
    const tileY = Math.floor(worldY);
    console.log(`[DBG] Character world (${worldX.toFixed(2)},${worldY.toFixed(2)}) → tile (${tileX},${tileY})`);
  }
  
  // Use character's own speed instead of global MOVE_SPEED
  const speed = character.speed || 6.0; // units = tiles / second
  const keysPressed = getKeysPressed();

  /* ---------------- Vertical movement (jump) ---------------- */
  if (character.vz === undefined) character.vz = 0; // vertical velocity (tiles/s)
  if (character.z === undefined) character.z = 0;   // current height in tiles

  const GRAVITY = 20;          // tiles per second² (tweak)
  const JUMP_SPEED = 10;       // initial jump velocity (tiles/s)

  if ((keysPressed['Space'] || keysPressed['Spacebar']) && character.z === 0) {
    character.vz = JUMP_SPEED;
  }

  // Integrate vertical motion
  if (character.z > 0 || character.vz > 0) {
    character.z += character.vz * delta;
    character.vz -= GRAVITY * delta;
    if (character.z <= 0) { // landed
      character.z = 0;
      character.vz = 0;
    }
  }

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

  // --- First-person view: convert local WASD into world-space based on yaw ---
  if (gameState?.camera?.viewType === 'first-person' && gameState.camera?.getGroundBasis) {
    const { forward, right } = gameState.camera.getGroundBasis();
    const fwdX = forward.x;
    const fwdY = forward.y;
    const rightX = right.x;
    const rightY = right.y;
    
    // Local intent:   W/S => moveY  (W = -1, S = +1)
    //                 A/D => moveX  (A = -1, D = +1)
    const localForward = -moveY; // because W made it -1 above
    const localRight   = moveX;

    // Compose world vector
    const worldMoveX = fwdX * localForward + rightX * localRight;
    const worldMoveY = fwdY * localForward + rightY * localRight;

    moveX = worldMoveX;
    moveY = worldMoveY;
  }

  // Normalize diagonal movement (after possible rotation)
  if (moveX !== 0 || moveY !== 0) {
    const length = Math.sqrt(moveX * moveX + moveY * moveY);
    if (length > 0) {
      moveX /= length;
      moveY /= length;
    }
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

    // ------------------------------------------------------------------
    // NEW: Sub-step movement to avoid tunnelling through thin obstacles.
    // We break the full movement vector into ≤0.25-tile chunks and test
    // collision after each micro-step.  This guarantees we cannot skip
    // an obstacle even if the character moves several tiles in one frame.
    // ------------------------------------------------------------------

    const MAX_STEP = 0.25;          // tile-units per micro-step
    const steps = Math.max(1, Math.ceil(distance / MAX_STEP));
    const stepX = (moveX * distance) / steps;
    const stepY = (moveY * distance) / steps;

    for (let s = 0; s < steps; s++) {
      // X axis first
      const attemptX = character.x + stepX;
      if (!isCollision(attemptX, character.y)) {
        character.x = attemptX;
      }

      // Y axis
      const attemptY = character.y + stepY;
      if (!isCollision(character.x, attemptY)) {
        character.y = attemptY;
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
  // Check if collision detection is disabled globally
  if (window.PLAYER_COLLISION_ENABLED === false) {
    return false;
  }
  
  // Skip collision if map manager isn't available
  if (!gameState.map) {
    return false;
  }
  
  // -------------------------------------------------------------
  // FAST PATH – centre-tile check
  // -------------------------------------------------------------
  // If the tile we are standing in is already known to be blocking, we can
  // immediately report a collision and avoid the costly multi-point test
  // below.  This guarantees that simple 1×1 wall tiles (e.g. red obstacle
  // blocks) always stop the player.
  if (gameState.map.isWallOrObstacle && gameState.map.isWallOrObstacle(x, y)) {
    return true;
  }
  
  // Character dimensions (tile-units). Default to one tile square.
  const width = gameState.character.width || 0.8;
  const height = gameState.character.height || 0.8;
  
  // FIXED: Reduce collision box size to fix "distant wall" collision issue
  // Instead of checking at 5 points, we'll use a smaller hitbox that's appropriate
  // for the tile size (which is typically 12px)
  
  // Calculate a more appropriate hitbox size based on tileSize
  const tileSize = gameState.map?.tileSize || 12;
  // Use 40% of one tile width for tighter fit (walls are unit tiles)
  const collisionSize = 0.4; // tile units
  const halfSize = collisionSize / 2;
  
  // ADDED: Visualization of collision points when debugging is enabled
  if (window.DEBUG_COLLISION) {
    visualizeCollisionPoints(x, y, halfSize);
  }
  
  // Log the collision size occasionally for debugging
  if (DEBUG_MOVEMENT && Math.random() < 0.001) {
    console.log(`Collision detection using box size: ${collisionSize.toFixed(2)}px (character: ${width}x${height}, tile: ${tileSize}px)`);
  }
  
  // MODIFIED: Initialize a variable to track if collision would have occurred
  let wouldCollide = false;
  
  // Only check center and 4 cardinal points (not corners)
  // Center
  if (isPointColliding(x, y)) wouldCollide = true;
  
  // Cardinal points (closer to center than before)
  // North
  if (!wouldCollide && isPointColliding(x, y - halfSize)) wouldCollide = true;
  // South
  if (!wouldCollide && isPointColliding(x, y + halfSize)) wouldCollide = true;
  // East
  if (!wouldCollide && isPointColliding(x + halfSize, y)) wouldCollide = true;
  // West
  if (!wouldCollide && isPointColliding(x - halfSize, y)) wouldCollide = true;
  
  // MODIFIED: If collision detected, log it but allow movement anyway
  if (wouldCollide) {
    // Track in global stats
    if (!window.COLLISION_STATS) {
      window.COLLISION_STATS = {
        totalCollisions: 0,
        ghostCollisions: 0,
        collisionPositions: []
      };
    }
    
    // Ensure collisionPositions exists
    if (!window.COLLISION_STATS.collisionPositions) {
      window.COLLISION_STATS.collisionPositions = [];
    }
    
    window.COLLISION_STATS.totalCollisions++;
    window.COLLISION_STATS.ghostCollisions++;
    
    // Store this collision for visualization (limited to 10 recent ones)
    const collisionData = {
      x: x,
      y: y,
      time: Date.now(),
      tileX: Math.floor(x / tileSize),
      tileY: Math.floor(y / tileSize)
    };
    
    window.COLLISION_STATS.collisionPositions.unshift(collisionData);
    if (window.COLLISION_STATS.collisionPositions.length > 10) {
      window.COLLISION_STATS.collisionPositions.pop();
    }
    
    // Log this collision occasionally to avoid console spam
    if (DEBUG_MOVEMENT && Math.random() < 0.1) {
      console.log(`GHOST COLLISION: Character passed through wall at (${x.toFixed(2)}, ${y.toFixed(2)}), ` +
                  `tile (${Math.floor(x / tileSize)}, ${Math.floor(y / tileSize)})`);
    }
  }
  
  // Return TRUE if any of the point probes detected a blocking tile.
  return wouldCollide;
}

/**
 * ADDED: Visualize ghost collisions that occurred but didn't block movement
 * This will show recent collision points as red X marks on the screen
 */
function visualizeGhostCollisions() {
  if (!window.DEBUG_COLLISION || !window.COLLISION_STATS) {
    return;
  }
  
  // Ensure collision positions array exists
  if (!window.COLLISION_STATS.collisionPositions) {
    window.COLLISION_STATS.collisionPositions = [];
    return;
  }
  
  // Get debug canvas or create one if it doesn't exist
  let canvas = document.getElementById('debugCollisionCanvas');
  if (!canvas) {
    return; // Canvas should already exist from visualizeCollisionPoints
  }
  
  const ctx = canvas.getContext('2d');
  const camera = gameState.camera;
  if (!camera || !camera.worldToScreen) {
    return;
  }
  
  // Draw recent ghost collisions
  for (const collision of window.COLLISION_STATS.collisionPositions) {
    // Skip if too old (older than 5 seconds)
    if (Date.now() - collision.time > 5000) {
      continue;
    }
    
    // Convert position to screen coordinates
    const pos = camera.worldToScreen(
      collision.x, 
      collision.y, 
      canvas.width, 
      canvas.height
    );
    
    // Draw a red X to mark ghost collision
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
    ctx.lineWidth = 3;
    
    // X mark
    const size = 10;
    ctx.beginPath();
    ctx.moveTo(pos.x - size, pos.y - size);
    ctx.lineTo(pos.x + size, pos.y + size);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(pos.x + size, pos.y - size);
    ctx.lineTo(pos.x - size, pos.y + size);
    ctx.stroke();
    
    // Add a red circle that fades with time
    const age = (Date.now() - collision.time) / 5000; // 0 to 1
    const alpha = 1 - age;
    ctx.fillStyle = `rgba(255, 0, 0, ${alpha * 0.3})`;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 15, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * ADDED: Visualization of collision points when debugging is enabled
 * @param {number} x - Center X position
 * @param {number} y - Center Y position
 * @param {number} halfSize - Half of the collision box size
 */
function visualizeCollisionPoints(x, y, halfSize) {
  // Get debug canvas or create one if it doesn't exist
  let canvas = document.getElementById('debugCollisionCanvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'debugCollisionCanvas';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.pointerEvents = 'none';
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.zIndex = '9999';
    document.body.appendChild(canvas);
  }
  
  const ctx = canvas.getContext('2d');
  
  // Clear previous debug visualization
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Need to convert world coordinates to screen coordinates
  const camera = gameState.camera;
  if (!camera || !camera.worldToScreen) {
    console.error('Camera not available for collision visualization');
    return;
  }
  
  // Draw center point
  const centerPos = camera.worldToScreen(x, y, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
  ctx.beginPath();
  ctx.arc(centerPos.x, centerPos.y, 4, 0, Math.PI * 2);
  ctx.fill();
  
  // Draw cardinal points
  const northPos = camera.worldToScreen(x, y - halfSize, canvas.width, canvas.height);
  const southPos = camera.worldToScreen(x, y + halfSize, canvas.width, canvas.height);
  const eastPos = camera.worldToScreen(x + halfSize, y, canvas.width, canvas.height);
  const westPos = camera.worldToScreen(x - halfSize, y, canvas.width, canvas.height);
  
  // Draw points with different colors
  ctx.fillStyle = 'rgba(0, 255, 0, 0.7)'; // North: green
  ctx.beginPath();
  ctx.arc(northPos.x, northPos.y, 4, 0, Math.PI * 2);
  ctx.fill();
  
  ctx.fillStyle = 'rgba(0, 0, 255, 0.7)'; // South: blue
  ctx.beginPath();
  ctx.arc(southPos.x, southPos.y, 4, 0, Math.PI * 2);
  ctx.fill();
  
  ctx.fillStyle = 'rgba(255, 255, 0, 0.7)'; // East: yellow
  ctx.beginPath();
  ctx.arc(eastPos.x, eastPos.y, 4, 0, Math.PI * 2);
  ctx.fill();
  
  ctx.fillStyle = 'rgba(255, 0, 255, 0.7)'; // West: magenta
  ctx.beginPath();
  ctx.arc(westPos.x, westPos.y, 4, 0, Math.PI * 2);
  ctx.fill();
  
  // Connect the dots to show collision box
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(northPos.x, northPos.y);
  ctx.lineTo(eastPos.x, eastPos.y);
  ctx.lineTo(southPos.x, southPos.y);
  ctx.lineTo(westPos.x, westPos.y);
  ctx.lineTo(northPos.x, northPos.y);
  ctx.stroke();
  
  // Draw tile grid lines for reference if within reasonable range of player
  const tileSize = gameState.map?.tileSize || 12;
  const startTileX = Math.floor((x - 5 * tileSize) / tileSize);
  const startTileY = Math.floor((y - 5 * tileSize) / tileSize);
  const endTileX = Math.floor((x + 5 * tileSize) / tileSize);
  const endTileY = Math.floor((y + 5 * tileSize) / tileSize);
  
  ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
  ctx.lineWidth = 0.5;
  
  // Draw vertical grid lines
  for (let tx = startTileX; tx <= endTileX; tx++) {
    const worldX = tx * tileSize;
    const screenStart = camera.worldToScreen(worldX, startTileY * tileSize, canvas.width, canvas.height);
    const screenEnd = camera.worldToScreen(worldX, endTileY * tileSize, canvas.width, canvas.height);
    
    ctx.beginPath();
    ctx.moveTo(screenStart.x, screenStart.y);
    ctx.lineTo(screenEnd.x, screenEnd.y);
    ctx.stroke();
  }
  
  // Draw horizontal grid lines
  for (let ty = startTileY; ty <= endTileY; ty++) {
    const worldY = ty * tileSize;
    const screenStart = camera.worldToScreen(startTileX * tileSize, worldY, canvas.width, canvas.height);
    const screenEnd = camera.worldToScreen(endTileX * tileSize, worldY, canvas.width, canvas.height);
    
    ctx.beginPath();
    ctx.moveTo(screenStart.x, screenStart.y);
    ctx.lineTo(screenEnd.x, screenEnd.y);
    ctx.stroke();
  }
  
  // ADDED: Draw ghost collisions on top
  visualizeGhostCollisions();
}

/**
 * Checks if a specific point collides with a wall
 * @param {number} x - X position to check
 * @param {number} y - Y position to check
 * @returns {boolean} True if point collides with a wall
 */
function isPointColliding(x, y) {
  // Get tile size for calculations
  const tileSize = gameState.map?.tileSize || 12;
  const tileX = Math.floor(x);
  const tileY = Math.floor(y);

  // ENHANCED: Log coordinate details on every 50th check (approximately)
  const shouldLogDetails = Math.random() < 0.02;
  
  if (shouldLogDetails) {
    console.log(`COLLISION CHECK:
- World Position: (${x.toFixed(4)}, ${y.toFixed(4)})
- Tile Position: (${tileX}, ${tileY})
- TileSize: ${tileSize}
- Tile Percent: (${((x % tileSize) / tileSize).toFixed(4)}, ${((y % tileSize) / tileSize).toFixed(4)})`);
  }

  // Check if position is a wall or obstacle using the map manager's method
  // This is the correct way - let the map manager handle the conversion
  if (gameState.map.isWallOrObstacle) {
    const collides = gameState.map.isWallOrObstacle(x, y);
    
    // Add enhanced logging for collisions
    if (collides || shouldLogDetails) {
      // Log exact conversion details for debugging
      const exactTileX = x;
      const exactTileY = y;
      
      const tileCenterX = (tileX + 0.5);
      const tileCenterY = (tileY + 0.5);
      
      const distanceFromTileCenter = Math.sqrt(
        Math.pow(x - tileCenterX, 2) + 
        Math.pow(y - tileCenterY, 2)
      );
      
      const tileEdgesInfo = {
        left: tileX,
        right: tileX + 1,
        top: tileY,
        bottom: tileY + 1,
        distToWest: x - tileX,
        distToEast: (tileX + 1) - x,
        distToNorth: y - tileY,
        distToSouth: (tileY + 1) - y
      };
      
      // Get minimum distance to any tile edge
      const minDistance = Math.min(
        tileEdgesInfo.distToWest,
        tileEdgesInfo.distToEast,
        tileEdgesInfo.distToNorth,
        tileEdgesInfo.distToSouth
      );
      
      // Find which edge is closest
      let closestEdge = "unknown";
      if (minDistance === tileEdgesInfo.distToWest) closestEdge = "west";
      else if (minDistance === tileEdgesInfo.distToEast) closestEdge = "east";
      else if (minDistance === tileEdgesInfo.distToNorth) closestEdge = "north";
      else if (minDistance === tileEdgesInfo.distToSouth) closestEdge = "south";
      
      const message = collides 
        ? `WALL COLLISION DETECTED` 
        : `No collision`;
      
      console.log(`${message} at world (${x.toFixed(2)}, ${y.toFixed(2)}), tile (${tileX}, ${tileY}):
- Exact tile coords: (${exactTileX.toFixed(4)}, ${exactTileY.toFixed(4)})
- Distance from tile center: ${distanceFromTileCenter.toFixed(2)}
- Closest edge: ${closestEdge} (${minDistance.toFixed(2)} units)
- Tile edges: W:${tileEdgesInfo.left} E:${tileEdgesInfo.right} N:${tileEdgesInfo.top} S:${tileEdgesInfo.bottom}`);
      
      // Get more detailed information about the tile if possible
      if (gameState.map.getTile) {
        const tile = gameState.map.getTile(tileX, tileY);
        if (tile) {
          console.log(`Tile details: 
- Type: ${tile.type}
- Name: ${TILE_IDS[tile.type] || 'Unknown'}
- Properties: ${JSON.stringify(tile.properties || {})}`);
          
          // Check surrounding tiles if a collision was detected
          if (collides) {
            console.log("Checking surrounding tiles...");
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dy === 0) continue; // Skip center tile
                
                const nearTileX = tileX + dx;
                const nearTileY = tileY + dy;
                const nearTile = gameState.map.getTile(nearTileX, nearTileY);
                
                if (nearTile) {
                  const isWall = gameState.map.isWallOrObstacle(
                    nearTileX + 0.5, 
                    nearTileY + 0.5
                  );
                  
                  console.log(`Tile (${nearTileX}, ${nearTileY}): Type ${nearTile.type}, isWall=${isWall}`);
                }
              }
            }
          }
        }
      }
    }
    
    return collides;
  }
  
  // Fallback: Manual tile lookup and collision check
  // Get tile from map
  const tile = gameState.map.getTile(tileX, tileY);
  if (!tile) {
    // No tile found (out of bounds)
    if (shouldLogDetails) {
      console.log(`No tile found at (${tileX}, ${tileY}) - treating as wall (map boundary)`);
    }
    return true;
  }
  
  // Check if it's a wall, obstacle, or mountain
  const collides = (
    tile.type === TILE_IDS.WALL || 
    tile.type === TILE_IDS.OBSTACLE || 
    tile.type === TILE_IDS.MOUNTAIN ||
    tile.type === TILE_IDS.WATER
  );
  
  // Add logging for collisions
  if (collides || shouldLogDetails) {
    console.log(`Tile ${collides ? 'collision' : 'check'} at world (${x.toFixed(2)}, ${y.toFixed(2)}), tile (${tileX}, ${tileY}):
- Tile type: ${tile.type} (${TILE_IDS[tile.type] || 'Unknown'})
- Is blocking: ${collides}`);
  }
  
  return collides;
}

// ADDED: Keyboard shortcut to toggle collision debugging
window.addEventListener('keydown', (event) => {
  // Press CTRL + SHIFT + C to toggle collision debugging
  if (event.ctrlKey && event.shiftKey && event.code === 'KeyC') {
    window.DEBUG_COLLISION = !window.DEBUG_COLLISION;
    console.log(`Collision debugging ${window.DEBUG_COLLISION ? 'enabled' : 'disabled'}`);
    
    // Clean up canvas if debugging is disabled
    if (!window.DEBUG_COLLISION) {
      const canvas = document.getElementById('debugCollisionCanvas');
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }
});

// Create a more informative toggle button for collision visualization
export function addCollisionVisualizationToggle() {
  // Check if the button already exists
  if (document.getElementById('collision-visualization-toggle')) return;
  
  const button = document.createElement('button');
  button.id = 'collision-visualization-toggle';
  button.innerHTML = 'Collision View: <span style="color:green">ON</span>';
  button.style.position = 'fixed';
  button.style.top = '10px';
  button.style.right = '10px';
  button.style.padding = '8px 12px';
  button.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  button.style.color = 'white';
  button.style.fontWeight = 'bold';
  button.style.border = 'none';
  button.style.borderRadius = '5px';
  button.style.zIndex = '9999';
  button.style.cursor = 'pointer';
  
  // Add collision stats display
  const statsDiv = document.createElement('div');
  statsDiv.id = 'collision-stats';
  statsDiv.style.position = 'fixed';
  statsDiv.style.top = '45px';
  statsDiv.style.right = '10px';
  statsDiv.style.padding = '8px';
  statsDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';
  statsDiv.style.color = 'white';
  statsDiv.style.fontFamily = 'monospace';
  statsDiv.style.fontSize = '12px';
  statsDiv.style.borderRadius = '5px';
  statsDiv.style.zIndex = '9998';
  statsDiv.style.display = window.DEBUG_COLLISION ? 'block' : 'none';
  statsDiv.innerHTML = 'Ghost collisions: 0<br>Total collisions: 0';
  
  // Update stats periodically
  setInterval(() => {
    if (window.COLLISION_STATS && window.DEBUG_COLLISION) {
      statsDiv.innerHTML = `Ghost collisions: ${window.COLLISION_STATS.ghostCollisions || 0}<br>Total collisions: ${window.COLLISION_STATS.totalCollisions || 0}`;
    }
  }, 500);
  
  button.addEventListener('click', () => {
    window.DEBUG_COLLISION = !window.DEBUG_COLLISION;
    console.log(`Collision visualization ${window.DEBUG_COLLISION ? 'enabled' : 'disabled'}`);
    button.innerHTML = `Collision View: <span style="color:${window.DEBUG_COLLISION ? 'green' : 'red'}">${window.DEBUG_COLLISION ? 'ON' : 'OFF'}</span>`;
    
    // Show/hide stats
    statsDiv.style.display = window.DEBUG_COLLISION ? 'block' : 'none';
    
    // Clean up canvas if disabled
    if (!window.DEBUG_COLLISION) {
      const canvas = document.getElementById('debugCollisionCanvas');
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  });
  
  document.body.appendChild(button);
  document.body.appendChild(statsDiv);
}

// Add this function to the game loop to visualize ghost collisions
export function updateCollisionVisualization() {
  if (window.DEBUG_COLLISION) {
    // Get player position for visualization
    const character = gameState.character;
    if (character) {
      // Calculate collision box size - FIX: Complete the calculation
      const width = character.width || 20;
      const height = character.height || 20;
      const tileSize = gameState.map?.tileSize || 12;
      const collisionSize = Math.max(width, height) * 0.8; // FIXED: Complete calculation
      const halfSize = collisionSize / 2;
      
      // Visualize at player's position
      visualizeCollisionPoints(character.x, character.y, halfSize);
    }
  }
}