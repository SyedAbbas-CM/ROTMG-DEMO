// src/render/render.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';
// Import view renderers - comment these out if they cause circular references
// They should be available on the window object anyway
// import { renderTopDownView } from './renderTopDown.js';
// import { renderStrategicView } from './renderStrategic.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Rendering parameters
const scaleFactor = SCALE;

// Add a debug flag for rendering
const DEBUG_ENEMY_SPRITES = false; // Set to false to disable sprite debugging visuals

/**
 * Render the character
 */
export function renderCharacter() {
  const character = gameState.character;
  if (!character) {
    console.error("Cannot render character: character not defined in gameState");
    return;
  }

  // Debug: Log the current view type to verify it's being detected correctly
  //console.log(`[renderCharacter] Current view type: ${gameState.camera?.viewType}`);

  // Determine scale factor based on view type
  let isStrategicView = gameState.camera?.viewType === 'strategic';
  let viewScaleFactor = isStrategicView ? 0.5 : 1.0; // FIXED back to 0.5 for strategic view (was 0.25)
  let effectiveScale = SCALE * viewScaleFactor; 

  // Debug: Log the scale factor being used
  //console.log(`[renderCharacter] Using scale factor: ${viewScaleFactor}, effectiveScale: ${effectiveScale}`);

  // Use the player's draw method if it exists
  if (character.draw && typeof character.draw === 'function') {
    // Directly set a flag on the character to ensure proper scaling
    character._isStrategicView = isStrategicView;
    character._viewScaleFactor = viewScaleFactor;
    
    // We'll still use the default draw method, but with view scaling info attached
    character.draw(ctx, gameState.camera.position);
    return;
  }

  // Fallback to old rendering method if draw method doesn't exist
  const charSheetObj = spriteManager.getSpriteSheet('character_sprites');
  if (!charSheetObj) {
    console.warn("Character sprite sheet not loaded");
    
    // Draw a placeholder rectangle if sprite sheet not available
    ctx.fillStyle = 'red';
    
    // Get screen dimensions
    const screenWidth = canvas2D.width;
    const screenHeight = canvas2D.height;
    
    // Calculate screen position (center of screen)
    const screenX = screenWidth / 2;
    const screenY = screenHeight / 2;
    
    // Draw rectangle with view-dependent scaling
    ctx.fillRect(
      screenX - (character.width * effectiveScale) / 2,
      screenY - (character.height * effectiveScale) / 2,
      character.width * effectiveScale,
      character.height * effectiveScale
    );
    return;
  }

  // Older sprite-based rendering
  const characterSpriteSheet = charSheetObj.image;
  
  // Get screen dimensions
  const screenWidth = canvas2D.width;
  const screenHeight = canvas2D.height;
  
  // Calculate screen position (center of screen)
  const screenX = screenWidth / 2;
  const screenY = screenHeight / 2;
  
  // Save the canvas state
  ctx.save();
  
  // If character has a rotation, rotate around character center
  if (typeof character.rotation === 'number') {
    ctx.translate(screenX, screenY);
    ctx.rotate(character.rotation);
    
    // Get sprite coordinates
    const spriteX = character.spriteX !== undefined ? character.spriteX : 0;
    const spriteY = character.spriteY !== undefined ? character.spriteY : 0;
    
    // Draw character centered at (0,0) after translation with view-dependent scaling
    ctx.drawImage(
      characterSpriteSheet,
      spriteX, spriteY,
      TILE_SIZE, TILE_SIZE,
      -character.width * effectiveScale / 2, -character.height * effectiveScale / 2,
      character.width * effectiveScale, character.height * effectiveScale
    );
  } else {
    // Get sprite coordinates
    const spriteX = character.spriteX !== undefined ? character.spriteX : 0;
    const spriteY = character.spriteY !== undefined ? character.spriteY : 0;
    
    // Draw character without rotation with view-dependent scaling
    ctx.drawImage(
      characterSpriteSheet,
      spriteX, spriteY,
      TILE_SIZE, TILE_SIZE,
      screenX - character.width * effectiveScale / 2,
      screenY - character.height * effectiveScale / 2,
      character.width * effectiveScale,
      character.height * effectiveScale
    );
  }
  
  // Restore canvas state
  ctx.restore();
}

/**
 * Render all enemies
 */
export function renderEnemies() {
  // Get enemies from enemyManager
  if (!gameState.enemyManager) {
    console.warn("Cannot render enemies: enemyManager not available");
    return;
  }
  
  // Get rendered enemies from the manager
  const enemies = gameState.enemyManager.getEnemiesForRender ? 
                  gameState.enemyManager.getEnemiesForRender() : [];
  
  // If no enemies, nothing to render
  if (!enemies || !Array.isArray(enemies) || enemies.length === 0) {
    return;
  }
  
  // Get enemy sprite sheet
  const enemySheetObj = spriteManager.getSpriteSheet('enemy_sprites');
  if (!enemySheetObj) {
    console.warn("Enemy sprite sheet not loaded");
    return;
  }
  const enemySpriteSheet = enemySheetObj.image;
  
  // Log sheet info once to help with debugging
  if (!window.enemySpriteDebugLogged) {
    window.enemySpriteDebugLogged = true;
    console.log(`Enemy sprite sheet loaded: ${enemySpriteSheet.width}x${enemySpriteSheet.height}`, 
      enemySheetObj.config);
  }
  
  // Get view scaling factor - FIXING back to 0.5 for strategic view
  const viewType = gameState.camera?.viewType || 'top-down';
  const viewScaleFactor = viewType === 'strategic' ? 0.5 : 1.0;
  
  // Get screen dimensions
  const screenWidth = canvas2D.width;
  const screenHeight = canvas2D.height;
  
  // Use camera's worldToScreen method if available for consistent coordinate transformation
  const useCamera = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
  
  // Render each enemy
  enemies.forEach(enemy => {
    try {
      // Scale dimensions based on view type
      const width = enemy.width * SCALE * viewScaleFactor;
      const height = enemy.height * SCALE * viewScaleFactor;
      
      // FIXED: Use camera's worldToScreen method for consistent coordinate transformation
      let screenX, screenY;
      
      if (useCamera) {
        // Use camera's consistent transformation method
        const screenPos = gameState.camera.worldToScreen(
          enemy.x, 
          enemy.y, 
          screenWidth, 
          screenHeight, 
          TILE_SIZE
        );
        screenX = screenPos.x;
        screenY = screenPos.y;
      } else {
        // Fallback to direct calculation
        screenX = (enemy.x - gameState.camera.position.x) * TILE_SIZE * viewScaleFactor + screenWidth / 2;
        screenY = (enemy.y - gameState.camera.position.y) * TILE_SIZE * viewScaleFactor + screenHeight / 2;
      }
      
      // Skip if off screen (with buffer)
      const buffer = width;
      if (screenX < -buffer || screenX > screenWidth + buffer || 
          screenY < -buffer || screenY > screenHeight + buffer) {
        return;
      }

      // Save context for rotation
      ctx.save();
      
      // Apply enemy flashing effect if it's been hit
      if (enemy.isFlashing) {
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = 'red';
        ctx.fillRect(screenX - width/2, screenY - height/2, width, height);
        ctx.globalAlpha = 1.0;
      }
      
      // If enemy has a rotation, use it
      if (typeof enemy.rotation === 'number') {
        ctx.translate(screenX, screenY);
        ctx.rotate(enemy.rotation);
        ctx.drawImage(
          enemySpriteSheet,
          enemy.spriteX || 0, enemy.spriteY || 0, 
          8, 8, // Use 8x8 sprite size for source
          -width/2, -height/2, width, height
        );
      } else {
        // Draw without rotation - center at the calculated screen position
        ctx.drawImage(
          enemySpriteSheet,
          enemy.spriteX || 0, enemy.spriteY || 0, 
          8, 8, // Use 8x8 sprite size for source
          screenX - width/2, screenY - height/2, width, height
        );
      }
      
      // Debug visualization of sprite source rectangle
      if (DEBUG_ENEMY_SPRITES) {
        // Draw a small version of the sprite sheet in corner for reference
        const sheetScale = 2;
        const sheetX = 10;
        const sheetY = 10;
        ctx.drawImage(
          enemySpriteSheet,
          sheetX, sheetY,
          enemySpriteSheet.width * sheetScale,
          enemySpriteSheet.height * sheetScale
        );
        
        // Highlight the source rectangle used for this enemy
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(
          sheetX + (enemy.spriteX || 0) * sheetScale,
          sheetY + (enemy.spriteY || 0) * sheetScale,
          8 * sheetScale, 8 * sheetScale
        );
        
        // Show enemy type and coordinates as text
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(
          `Enemy ${enemy.id} - Type: ${gameState.enemyManager.enemyTypes ? 
            gameState.enemyManager.enemyTypes[enemy.type]?.name : enemy.type}`,
          screenX - width/2, screenY + height/2 + 15
        );
        ctx.fillText(
          `Sprite: (${enemy.spriteX},${enemy.spriteY})`,
          screenX - width/2, screenY + height/2 + 30
        );
      }
      
      // Draw health bar
      if (enemy.health !== undefined && enemy.maxHealth !== undefined) {
        const healthPercent = enemy.health / enemy.maxHealth;
        const barWidth = width;
        const barHeight = 4;
        
        // Draw background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(
          screenX - barWidth/2,
          screenY - height/2 - barHeight - 2,
          barWidth,
          barHeight
        );
        
        // Draw health - color based on percentage
        ctx.fillStyle = healthPercent > 0.6 ? 'green' : healthPercent > 0.3 ? 'yellow' : 'red';
        ctx.fillRect(
          screenX - barWidth/2,
          screenY - height/2 - barHeight - 2,
          barWidth * healthPercent,
          barHeight
        );
      }
      
      // Restore context
      ctx.restore();
    } catch (error) {
      console.error("Error rendering enemy:", error);
    }
  });
}

/**
 * Render all bullets
 */
export function renderBullets() {
  const bm = gameState.bulletManager;
  if (!bm) {
    // no bullet manager attached
    return;
  }

  // If nothing active, skip
  if (bm.bulletCount === 0) return;

  const spriteManager = window.spriteManager;
  const viewType = gameState.camera?.viewType || 'top-down';
  const viewScale = viewType === 'strategic' ? 0.5 : 1.0;

  const W = canvas2D.width;
  const H = canvas2D.height;
  const useCam = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
  
  // Get bullet scale factor if available
  const bulletScale = bm.bulletScale || 1.0;

  for (let i = 0; i < bm.bulletCount; i++) {
    // world → screen
    let sx, sy;
    if (useCam) {
      ({ x: sx, y: sy } = gameState.camera.worldToScreen(
        bm.x[i], bm.y[i], W, H, TILE_SIZE
      ));
    } else {
      sx = (bm.x[i] - gameState.camera.position.x) * TILE_SIZE * viewScale + W/2;
      sy = (bm.y[i] - gameState.camera.position.y) * TILE_SIZE * viewScale + H/2;
    }

    // cull
    if (sx < -100 || sx > W+100 || sy < -100 || sy > H+100) continue;

    // Apply bullet scale to width and height
    const w = (bm.width[i] || 8) * viewScale * bulletScale;
    const h = (bm.height[i] || 8) * viewScale * bulletScale;

    // try sprite draw
    if (spriteManager && bm.sprite && bm.sprite[i]) {
      try {
        // Use the bullet's sprite information
        const sprite = bm.sprite[i];
        spriteManager.drawSprite(
          ctx,
          sprite.spriteSheet || 'bullet_sprites',
          sprite.spriteX || 0, sprite.spriteY || 0,
          sx - w/2, sy - h/2,
          w, h
        );
        continue;
      } catch (e) {
        console.error("Error rendering bullet sprite:", e);
        // fall through to circle
      }
    }

    // fallback glow circle
    const isLocal = bm.ownerId[i] === gameState.character?.id;
    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, w);
    if (isLocal) {
      grad.addColorStop(0, 'rgb(255,255,120)');
      grad.addColorStop(0.7, 'rgb(255,160,0)');
      grad.addColorStop(1, 'rgba(255,100,0,0)');
    } else {
      grad.addColorStop(0, 'rgb(255,100,255)');
      grad.addColorStop(0.7, 'rgb(255,0,100)');
      grad.addColorStop(1, 'rgba(200,0,0,0)');
    }

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(sx, sy, w, 0, Math.PI*2);
    ctx.fill();

    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(sx, sy, w*0.3, 0, Math.PI*2);
    ctx.fill();
  }
}

/**
 * Complete render function for game state
 */
export function renderGame() {
  // Make sure canvas is properly sized
  if (canvas2D.width !== window.innerWidth || canvas2D.height !== window.innerHeight) {
    canvas2D.width = window.innerWidth;
    canvas2D.height = window.innerHeight;
  }
  
  // Clear the canvas - first with clearRect for best performance
  ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);
  
  // Then fill with a solid background color to ensure ALL old pixels are gone
  // This is especially important in strategic view to prevent ghost artifacts
  if (gameState.camera.viewType === 'strategic') {
    // Use fully opaque black background in strategic view to prevent ghosting
    ctx.fillStyle = 'rgb(0, 0, 0)';  
  } else {
    // Use standard black in other views
    ctx.fillStyle = 'rgb(0, 0, 0)';
  }
  ctx.fillRect(0, 0, canvas2D.width, canvas2D.height);
  
  // Draw level (different based on view type)
  // Note: We don't import these functions directly to avoid circular references
  // Instead, we get them from the global scope
  const viewType = gameState.camera.viewType;
  
  try {
    // Log the availability of render functions (only occasionally to avoid console spam)
    if (Math.random() < 0.01) {
      //console.log(`Render functions available: topdown=${typeof window.renderTopDownView === 'function'}, strategic=${typeof window.renderStrategicView === 'function'}`);
    }
    
    if (viewType === 'top-down') {
      if (typeof window.renderTopDownView === 'function') {
        window.renderTopDownView();
      } else {
        console.error("Top-down view render function not available");
      }
    } else if (viewType === 'strategic') {
      if (typeof window.renderStrategicView === 'function') {
        window.renderStrategicView();
      } else {
        console.error("Strategic view render function not available - make sure renderStrategic.js is loaded and window.renderStrategicView is set");
      }
    } else {
      console.error(`Unknown view type ${viewType}`);
    }
  } catch (error) {
    console.error("Error rendering game view:", error);
  }
  
  // Draw entities
  renderBullets();
  renderEnemies();
  renderPlayers();
  renderCharacter();
  
  // Draw UI elements
  renderUI();
  
  // Run coordinate system test if debug mode is enabled
  if (gameState.camera && gameState.camera.debugMode) {
    try {
      // Import directly to avoid circular dependencies
      const { testCoordinateSystem } = window.coordinateUtils;
      if (typeof testCoordinateSystem === 'function') {
        testCoordinateSystem(ctx, gameState.camera);
      }
    } catch (error) {
      console.error("Error testing coordinate system:", error);
    }
  }
}

/**
 * Render UI elements
 */
function renderUI() {
  // Draw player health bar
  if (gameState.character && gameState.character.health !== undefined) {
    const health = gameState.character.health;
    const maxHealth = gameState.character.maxHealth || 100;
    const healthPercent = health / maxHealth;
    
    const barWidth = 200;
    const barHeight = 20;
    const barX = 20;
    const barY = canvas2D.height - barHeight - 20;
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(barX, barY, barWidth, barHeight);
    
    // Health
    ctx.fillStyle = healthPercent > 0.6 ? 'green' : healthPercent > 0.3 ? 'yellow' : 'red';
    ctx.fillRect(barX, barY, barWidth * healthPercent, barHeight);
    
    // Border
    ctx.strokeStyle = 'white';
    ctx.strokeRect(barX, barY, barWidth, barHeight);
    
    // Text
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.fillText(`Health: ${Math.round(health)}/${maxHealth}`, barX + 10, barY + 15);
  }
  
  // Draw network status
  if (gameState.networkManager) {
    const status = gameState.networkManager.isConnected() ? 'Connected' : 'Disconnected';
    ctx.fillStyle = gameState.networkManager.isConnected() ? 'green' : 'red';
    ctx.font = '12px Arial';
    ctx.fillText(`Server: ${status}`, canvas2D.width - 120, 20);
    
    // Add player counter to UI
    const playerCount = gameState.playerManager?.players.size || 0;
    const totalPlayers = playerCount + 1; // Add 1 for local player
    ctx.fillStyle = 'white';
    ctx.fillText(`Players: ${totalPlayers} (${playerCount} others)`, canvas2D.width - 120, 40);
    
    // Add collision statistics if available
    if (window.collisionStats) {
      ctx.fillStyle = 'white';
      ctx.fillText(`Collisions: ${window.collisionStats.validated}/${window.collisionStats.reported} (${window.collisionStats.getValidationRate()})`, canvas2D.width - 120, 60);
    }

    // Show behavior mode if enabled
    if (window.ALLOW_CLIENT_ENEMY_BEHAVIOR !== undefined) {
      ctx.fillStyle = window.ALLOW_CLIENT_ENEMY_BEHAVIOR ? 'yellow' : 'white';
      ctx.fillText(`Enemy Behaviors: ${window.ALLOW_CLIENT_ENEMY_BEHAVIOR ? 'ON' : 'OFF'}`, canvas2D.width - 120, 80);
    }
  }
}

/**
 * Render all players (except the local player)
 */
export function renderPlayers() {
  // Skip if no player manager
  if (!gameState.playerManager) {
    console.error("Cannot render players: playerManager not available in gameState");
    return;
  }
  
  // Log minimally, only when really needed
  if (Math.random() < 0.01) { // Just 1% of frames
    //console.log(`[renderPlayers] Rendering ${gameState.playerManager.players.size} players. Local ID: ${gameState.playerManager.localPlayerId}`);
  }
  
  try {
    // Delegate rendering to the player manager's render method
    gameState.playerManager.render(ctx, gameState.camera.position);
  } catch (error) {
    console.error("Error rendering players:", error);
  }
}

/**
 * Render all items on the ground
 */
export function renderItems() {
  // Get items from itemManager
  if (!gameState.itemManager) {
    return;
  }
  
  // Get items for rendering
  const items = gameState.itemManager.getGroundItemsForRender ? 
                gameState.itemManager.getGroundItemsForRender() : [];
  
  // If no items, nothing to render
  if (!items || !Array.isArray(items) || items.length === 0) {
    return;
  }
  
  // Get item sprite sheet
  const itemSheetObj = spriteManager.getSpriteSheet('item_sprites');
  if (!itemSheetObj) {
    console.warn("Item sprite sheet not loaded");
    return;
  }
  const itemSpriteSheet = itemSheetObj.image;
  
  // Get view scaling factor
  const viewType = gameState.camera?.viewType || 'top-down';
  const viewScaleFactor = viewType === 'strategic' ? 0.5 : 1.0;
  
  // Get screen dimensions
  const screenWidth = canvas2D.width;
  const screenHeight = canvas2D.height;
  
  // Use camera's worldToScreen method if available for consistent coordinate transformation
  const useCamera = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
  
  // Render each item
  items.forEach(item => {
    try {
      // Scale dimensions based on view type
      const width = item.width * SCALE * viewScaleFactor;
      const height = item.height * SCALE * viewScaleFactor;
      
      // FIXED: Use camera's worldToScreen method for consistent coordinate transformation
      let screenX, screenY;
      
      if (useCamera) {
        // Use camera's consistent transformation method
        const screenPos = gameState.camera.worldToScreen(
          item.x, 
          item.y, 
          screenWidth, 
          screenHeight, 
          TILE_SIZE
        );
        screenX = screenPos.x;
        screenY = screenPos.y;
      } else {
        // Fallback to direct calculation
        screenX = (item.x - gameState.camera.position.x) * TILE_SIZE * viewScaleFactor + screenWidth / 2;
        screenY = (item.y - gameState.camera.position.y) * TILE_SIZE * viewScaleFactor + screenHeight / 2;
      }
      
      // Skip if off screen (with buffer)
      const buffer = width;
      if (screenX < -buffer || screenX > screenWidth + buffer || 
          screenY < -buffer || screenY > screenHeight + buffer) {
        return;
      }
      
      // Draw the item
      ctx.drawImage(
        itemSpriteSheet,
        item.spriteX || 0, item.spriteY || 0, 
        item.width || 16, item.height || 16,
        screenX - width/2, screenY - height/2, width, height
      );
      
      // Draw item name for close items
      if (gameState.player) {
        const distanceToPlayer = Math.sqrt(
          Math.pow(item.x - gameState.player.x, 2) + 
          Math.pow(item.y - gameState.player.y, 2)
        );
        
        // Only show names for items close to the player
        if (distanceToPlayer < 2.5) {
          ctx.font = '12px Arial';
          ctx.fillStyle = 'white';
          ctx.textAlign = 'center';
          ctx.fillText(item.name || 'Unknown Item', screenX, screenY + height/2 + 15);
        }
      }
    } catch (error) {
      console.error("Error rendering item:", error);
    }
  });
}