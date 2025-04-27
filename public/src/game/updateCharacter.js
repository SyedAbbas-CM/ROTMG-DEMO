// src/game/updateCharacter.js

import { getKeysPressed } from './input.js';
import { gameState } from './gamestate.js';
import { TILE_SIZE, TILE_IDS } from '../constants/constants.js';
import { createLogger, LOG_LEVELS } from '../utils/logger.js';

// Create a logger for this module
const logger = createLogger('movement');

// CHANGED: Collision debugging enabled by default
window.DEBUG_COLLISION = true;

/**
 * Updates the character's position based on input and handles collision.
 * @param {number} delta - Time elapsed since the last frame (in seconds).
 */
export function updateCharacter(delta) {
  const character = gameState.character;
  
  // ENHANCED: Log position and verify coordinate conversion when character exists
  if (character && gameState.map) {
    const worldX = character.x;
    const worldY = character.y;
    const tileSize = gameState.map.tileSize || 12;
    const tileX = Math.floor(worldX / tileSize);
    const tileY = Math.floor(worldY / tileSize);
    
    // Check for coordinate discrepancy - log if coordinates don't match expected
    if (Math.abs(tileX - Math.round(worldX / tileSize)) > 0.01 || 
        Math.abs(tileY - Math.round(worldY / tileSize)) > 0.01) {
      console.warn(`COORDINATE MISMATCH: World: (${worldX.toFixed(2)}, ${worldY.toFixed(2)}) -> ` +
                   `Tile: (${tileX}, ${tileY}) with tileSize=${tileSize}`);
      console.warn(`Expected tile: (${Math.round(worldX / tileSize)}, ${Math.round(worldY / tileSize)})`);
      
      // Add more detailed conversion information
      const floorX = Math.floor(worldX / tileSize);
      const floorY = Math.floor(worldY / tileSize);
      const ceilX = Math.ceil(worldX / tileSize);
      const ceilY = Math.ceil(worldY / tileSize);
      const roundX = Math.round(worldX / tileSize);
      const roundY = Math.round(worldY / tileSize);
      
      console.warn(`Conversion details:
- World position: (${worldX.toFixed(4)}, ${worldY.toFixed(4)})
- Division result: (${(worldX / tileSize).toFixed(4)}, ${(worldY / tileSize).toFixed(4)})
- Floor: (${floorX}, ${floorY})
- Ceiling: (${ceilX}, ${ceilY})
- Round: (${roundX}, ${roundY})
- Error margin: (${Math.abs(worldX / tileSize - floorX).toFixed(4)}, ${Math.abs(worldY / tileSize - floorY).toFixed(4)})
- % of tile: (${((worldX % tileSize) / tileSize).toFixed(4)}, ${((worldY % tileSize) / tileSize).toFixed(4)})`);
      
      // Check world boundaries for this tile to see if character is on boundary
      const tileWorldX = tileX * tileSize;
      const tileWorldY = tileY * tileSize;
      const nextTileWorldX = (tileX + 1) * tileSize;
      const nextTileWorldY = (tileY + 1) * tileSize;
      
      console.warn(`Tile boundaries:
- Current tile (${tileX}, ${tileY}) world bounds: (${tileWorldX}, ${tileWorldY}) to (${nextTileWorldX}, ${nextTileWorldY})
- Distance from west edge: ${(worldX - tileWorldX).toFixed(4)} (${((worldX - tileWorldX) / tileSize * 100).toFixed(2)}% of tile)
- Distance from north edge: ${(worldY - tileWorldY).toFixed(4)} (${((worldY - tileWorldY) / tileSize * 100).toFixed(2)}% of tile)
- Distance to east edge: ${(nextTileWorldX - worldX).toFixed(4)} (${((nextTileWorldX - worldX) / tileSize * 100).toFixed(2)}% of tile)
- Distance to south edge: ${(nextTileWorldY - worldY).toFixed(4)} (${((nextTileWorldY - worldY) / tileSize * 100).toFixed(2)}% of tile)`);
      
      // Try alternative conversion methods to diagnose
      console.warn(`Experiment: Direct tile center calculation: (${tileX + 0.5}, ${tileY + 0.5}) -> World: (${(tileX + 0.5) * tileSize}, ${(tileY + 0.5) * tileSize})`);
    }
    
    // Periodically log character position with enhanced details
    if (Math.random() < 0.05) {
      console.log(`CHARACTER POSITION DETAILS:
- World: (${worldX.toFixed(2)}, ${worldY.toFixed(2)})
- Tile: (${tileX}, ${tileY}) with tileSize=${tileSize}
- Center of tile: (${(tileX + 0.5) * tileSize}, ${(tileY + 0.5) * tileSize})
- Tile percentage: (${((worldX % tileSize) / tileSize).toFixed(2)}, ${((worldY % tileSize) / tileSize).toFixed(2)})`);
      
      // Log map boundaries and player position in tiles
      if (gameState.map.width && gameState.map.height) {
        console.log(`Map boundaries: width=${gameState.map.width}, height=${gameState.map.height}`);
        console.log(`Player position in tiles: (${tileX}, ${tileY})`);
      }
    }
  }
  
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
  // Check if collision detection is disabled globally
  if (window.PLAYER_COLLISION_ENABLED === false) {
    return false;
  }
  
  // Skip collision if map manager isn't available
  if (!gameState.map) {
    return false;
  }
  
  // Character dimensions (use properties if available, otherwise use defaults)
  const width = gameState.character.width || 20;
  const height = gameState.character.height || 20;
  
  // FIXED: Reduce collision box size to fix "distant wall" collision issue
  // Instead of checking at 5 points, we'll use a smaller hitbox that's appropriate
  // for the tile size (which is typically 12px)
  
  // Calculate a more appropriate hitbox size based on tileSize
  const tileSize = gameState.map.tileSize || 12;
  // Use 60% of character size or 80% of tile size, whichever is smaller
  const collisionSize = Math.min(width * 0.6, tileSize * 0.8);
  const halfSize = collisionSize / 2;
  
  // ADDED: Visualization of collision points when debugging is enabled
  if (window.DEBUG_COLLISION) {
    visualizeCollisionPoints(x, y, halfSize);
  }
  
  // Log the collision size occasionally for debugging
  if (Math.random() < 0.001) {
    console.log(`Collision detection using box size: ${collisionSize.toFixed(2)}px (character: ${width}x${height}, tile: ${tileSize}px)`);
  }
  
  // Only check center and 4 cardinal points (not corners)
  // Center
  if (isPointColliding(x, y)) return true;
  
  // Cardinal points (closer to center than before)
  // North
  if (isPointColliding(x, y - halfSize)) return true;
  // South
  if (isPointColliding(x, y + halfSize)) return true;
  // East
  if (isPointColliding(x + halfSize, y)) return true;
  // West
  if (isPointColliding(x - halfSize, y)) return true;
  
  // No collision detected
  return false;
}

/**
 * ADDED: Visualize collision points for debugging
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
  const tileSize = gameState.map.tileSize || 12;
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
}

/**
 * Checks if a specific point collides with a wall
 * @param {number} x - X position to check
 * @param {number} y - Y position to check
 * @returns {boolean} True if point collides with a wall
 */
function isPointColliding(x, y) {
  // Important: x and y are in world coordinates, NOT tile coordinates
  try {
    // Get tile size for calculations
    const tileSize = gameState.map.tileSize || 12;
    const tileX = Math.floor(x / tileSize);
    const tileY = Math.floor(y / tileSize);

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
        const exactTileX = x / tileSize;
        const exactTileY = y / tileSize;
        
        const tileCenterX = (tileX + 0.5) * tileSize;
        const tileCenterY = (tileY + 0.5) * tileSize;
        
        const distanceFromTileCenter = Math.sqrt(
          Math.pow(x - tileCenterX, 2) + 
          Math.pow(y - tileCenterY, 2)
        );
        
        const tileEdgesInfo = {
          left: tileX * tileSize,
          right: (tileX + 1) * tileSize,
          top: tileY * tileSize,
          bottom: (tileY + 1) * tileSize,
          distToWest: x - (tileX * tileSize),
          distToEast: (tileX + 1) * tileSize - x,
          distToNorth: y - (tileY * tileSize),
          distToSouth: (tileY + 1) * tileSize - y
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
                      nearTileX * tileSize + tileSize/2, 
                      nearTileY * tileSize + tileSize/2
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
  } catch (error) {
    logger.error("Error in collision detection:", error);
    // On error, default to no collision
    return false;
  }
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

// Create a toggle button for collision visualization
function addCollisionVisualizationToggle() {
  // Check if the button already exists
  if (document.getElementById('collision-visualization-toggle')) return;
  
  const button = document.createElement('button');
  button.id = 'collision-visualization-toggle';
  button.innerText = 'Toggle Collision View';
  button.style.position = 'fixed';
  button.style.top = '10px';
  button.style.right = '10px';
  button.style.padding = '5px';
  button.style.backgroundColor = window.DEBUG_COLLISION ? 'green' : 'red';
  button.style.color = 'white';
  button.style.fontWeight = 'bold';
  button.style.border = 'none';
  button.style.borderRadius = '5px';
  button.style.zIndex = '9999';
  
  button.addEventListener('click', () => {
    window.DEBUG_COLLISION = !window.DEBUG_COLLISION;
    console.log(`Collision visualization ${window.DEBUG_COLLISION ? 'enabled' : 'disabled'}`);
    button.style.backgroundColor = window.DEBUG_COLLISION ? 'green' : 'red';
    
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
}

// Add this call to immediately create the button when the file is loaded
setTimeout(addCollisionVisualizationToggle, 1000);