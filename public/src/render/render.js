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

/**
 * Render the character
 */
export function renderCharacter() {
  const character = gameState.character;
  if (!character) {
    console.error("Cannot render character: character not defined in gameState");
    return;
  }

  // Use the player's draw method if it exists
  if (character.draw && typeof character.draw === 'function') {
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
    
    // Draw rectangle
    ctx.fillRect(
      screenX - (character.width * SCALE) / 2,
      screenY - (character.height * SCALE) / 2,
      character.width * SCALE,
      character.height * SCALE
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
    
    // Draw character centered at (0,0) after translation
    ctx.drawImage(
      characterSpriteSheet,
      spriteX, spriteY,
      TILE_SIZE, TILE_SIZE,
      -character.width * SCALE / 2, -character.height * SCALE / 2,
      character.width * SCALE, character.height * SCALE
    );
  } else {
    // Get sprite coordinates
    const spriteX = character.spriteX !== undefined ? character.spriteX : 0;
    const spriteY = character.spriteY !== undefined ? character.spriteY : 0;
    
    // Draw character without rotation
    ctx.drawImage(
      characterSpriteSheet,
      spriteX, spriteY,
      TILE_SIZE, TILE_SIZE,
      screenX - character.width * SCALE / 2,
      screenY - character.height * SCALE / 2,
      character.width * SCALE,
      character.height * SCALE
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

  // Render each enemy
  enemies.forEach(enemy => {
    try {
      const width = enemy.width * scaleFactor;
      const height = enemy.height * scaleFactor;

      // Calculate screen position
      const x = (enemy.x - gameState.camera.position.x) * TILE_SIZE * scaleFactor + canvas2D.width / 2 - width / 2;
      const y = (enemy.y - gameState.camera.position.y) * TILE_SIZE * scaleFactor + canvas2D.height / 2 - height / 2;

      // Save context for rotation
      ctx.save();
      
      // Apply enemy flashing effect if it's been hit
      if (enemy.isFlashing) {
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = 'red';
        ctx.fillRect(x, y, width, height);
        ctx.globalAlpha = 1.0;
      }
      
      // If enemy has a rotation, use it
      if (typeof enemy.rotation === 'number') {
        ctx.translate(x + width/2, y + height/2);
        ctx.rotate(enemy.rotation);
        ctx.drawImage(
          enemySpriteSheet,
          enemy.spriteX || 0, enemy.spriteY || 0, 
          enemy.width || 24, enemy.height || 24,
          -width/2, -height/2, width, height
        );
      } else {
        // Draw without rotation
        ctx.drawImage(
          enemySpriteSheet,
          enemy.spriteX || 0, enemy.spriteY || 0, 
          enemy.width || 24, enemy.height || 24,
          x, y, width, height
        );
      }
      
      // Draw health bar
      if (enemy.health !== undefined && enemy.maxHealth !== undefined && enemy.health > 0) {
        const healthPercent = enemy.health / enemy.maxHealth;
        const barWidth = width;
        const barHeight = 4;
        const barY = y - barHeight - 2;
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(x, barY, barWidth, barHeight);
        
        // Health
        ctx.fillStyle = healthPercent > 0.6 ? 'green' : healthPercent > 0.3 ? 'yellow' : 'red';
        ctx.fillRect(x, barY, barWidth * healthPercent, barHeight);
      }
      
      // Restore context
      ctx.restore();
    } catch (error) {
      console.error("Error rendering enemy:", error, enemy);
    }
  });
}

/**
 * Render all bullets with sprite support
 * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
 */
export function renderBullets() {
  // Get canvas context if not provided
  const ctx = document.getElementById('gameCanvas').getContext('2d');
  if (!ctx) return;
  
  // If bulletManager doesn't exist, nothing to render
  if (!gameState.bulletManager) return;
  
  // Get screen center
  const screenCenterX = ctx.canvas.width / 2;
  const screenCenterY = ctx.canvas.height / 2;
  
  // Minimal logging only when needed
  if (Math.random() < 0.01) { // Just 1% of frames
    //console.log(`[renderBullets] Bullets count: ${gameState.bulletManager.bulletCount || 0}`);
  }
  
  // Call the bullet manager's render method with camera position
  if (typeof gameState.bulletManager.render === 'function') {
    gameState.bulletManager.render(ctx, gameState.camera.position);
  } else {
    // Fallback rendering if bullet manager lacks render method
    const bullets = gameState.bulletManager.getBulletsForRender ? 
                  gameState.bulletManager.getBulletsForRender() : [];
                  
    if (!bullets || bullets.length === 0) return;
    
    // Get sprite manager
    const spriteManager = window.spriteManager || null;
    
    // Draw each bullet
    bullets.forEach(bullet => {
      // Calculate screen coordinates - ensure correct coordinate scaling with TILE_SIZE
      const screenX = (bullet.x - gameState.camera.position.x) * TILE_SIZE + screenCenterX;
      const screenY = (bullet.y - gameState.camera.position.y) * TILE_SIZE + screenCenterY;
      
      // Verify bullet is on screen before drawing to avoid rendering off-screen
      const bulletHalfSize = (bullet.width || 8) / 2;
      if (screenX + bulletHalfSize < 0 || screenX - bulletHalfSize > canvas2D.width || 
          screenY + bulletHalfSize < 0 || screenY - bulletHalfSize > canvas2D.height) {
        return; // Skip rendering off-screen bullets
      }
      
      // Draw bullet
      if (spriteManager && bullet.spriteSheet) {
        // Render with sprite
        spriteManager.drawSprite(
          ctx,
          bullet.spriteSheet,
          bullet.spriteX || 8 * 10, // Default X position in sprite sheet (col * width)
          bullet.spriteY || 8 * 11, // Default Y position in sprite sheet (row * height)
          screenX - (bullet.width || 8) / 2,
          screenY - (bullet.height || 8) / 2,
          bullet.width || 8,
          bullet.height || 8
        );
      } else {
        // Fallback to simple but visible bullet
        ctx.fillStyle = bullet.ownerId === gameState.playerManager?.localPlayerId ? 'yellow' : 'red';
        ctx.beginPath();
        ctx.arc(screenX, screenY, (bullet.width || 8) / 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Add highlight outline for better visibility
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
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
  
  // Clear the canvas - first with clearRect
  ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);
  
  // Then fill with a dark background color to ensure old pixels are gone
  ctx.fillStyle = 'rgba(0, 0, 0, 1)';  // Solid black background
  ctx.fillRect(0, 0, canvas2D.width, canvas2D.height);
  
  // Draw level (different based on view type)
  // Note: We don't import these functions directly to avoid circular references
  // Instead, we get them from the global scope
  const viewType = gameState.camera.viewType;
  
  try {
    // Log the availability of render functions (only occasionally to avoid console spam)
    if (Math.random() < 0.01) {
      console.log(`Render functions available: topdown=${typeof window.renderTopDownView === 'function'}, strategic=${typeof window.renderStrategicView === 'function'}`);
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