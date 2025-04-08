// src/render/render.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Rendering parameters
const scaleFactor = SCALE;

/**
 * Render the character
 */
export function renderCharacter() {
  console.log("renderCharacter called");
  
  const character = gameState.character;
  if (!character) {
    console.error("Cannot render character: character not defined in gameState");
    return;
  }

  console.log(`Character at position: (${character.x.toFixed(1)}, ${character.y.toFixed(1)})`);

  const charSheetObj = spriteManager.getSpriteSheet('character_sprites');
  if (!charSheetObj) {
    console.warn("Character sprite sheet not loaded - attempting fallback rendering");
    // Fallback rendering with a simple shape
    const width = character.width * scaleFactor;
    const height = character.height * scaleFactor;
    const x = canvas2D.width / 2 - width / 2;
    const y = canvas2D.height / 2 - height / 2;
    
    // Draw a visible character with a bright color
    ctx.save();
    ctx.translate(x + width/2, y + height/2);
    
    // Apply rotation if character has it
    if (typeof character.rotation === 'object' && character.rotation.yaw !== undefined) {
      ctx.rotate(character.rotation.yaw);
    } else if (typeof character.rotation === 'number') {
      ctx.rotate(character.rotation);
    }
    
    // Draw a brightly colored rectangle for visibility
    ctx.fillStyle = 'lime';
    ctx.fillRect(-width/2, -height/2, width, height);
    
    // Add border for better visibility
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.strokeRect(-width/2, -height/2, width, height);
    
    // Add direction indicator
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -height);
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.restore();
    return;
  }
  
  const characterSpriteSheet = charSheetObj.image;
  const width = character.width * scaleFactor;
  const height = character.height * scaleFactor;
  
  // Calculate screen position in top-down view (pixels)
  const x = canvas2D.width / 2 - width / 2;
  const y = canvas2D.height / 2 - height / 2;

  // Save current transform
  ctx.save();
  
  // Draw character at center of screen
  ctx.translate(x + width/2, y + height/2);
  
  // Apply rotation if character has it
  if (typeof character.rotation === 'object' && character.rotation.yaw !== undefined) {
    ctx.rotate(character.rotation.yaw);
  } else if (typeof character.rotation === 'number') {
    ctx.rotate(character.rotation);
  }
  
  // Log sprite details
  console.log(`Drawing character sprite from (${character.spriteX}, ${character.spriteY}) at screen pos (${x}, ${y})`);
  
  // Draw character image
  ctx.drawImage(
    characterSpriteSheet,
    character.spriteX, character.spriteY, TILE_SIZE, TILE_SIZE, // Source rectangle
    -width/2, -height/2, width, height // Destination rectangle (centered)
  );
  
  // Add outline for better visibility
  ctx.strokeStyle = 'yellow';
  ctx.lineWidth = 1;
  ctx.strokeRect(-width/2, -height/2, width, height);
  
  // Restore transform
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
    console.log(`[renderBullets] Bullets count: ${gameState.bulletManager.bulletCount || 0}`);
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
  // Debug log to see if this function is being called
  console.log(`renderGame called - view type: ${gameState.camera?.viewType}`);
  
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
    if (viewType === 'top-down' && typeof window.renderTopDownView === 'function') {
      window.renderTopDownView();
    } else if (viewType === 'strategic' && typeof window.renderStrategicView === 'function') {
      window.renderStrategicView();
    } else {
      console.error(`Cannot render view type ${viewType}: render function not available`);
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
    console.log(`[renderPlayers] Rendering ${gameState.playerManager.players.size} players. Local ID: ${gameState.playerManager.localPlayerId}`);
  }
  
  try {
    // Delegate rendering to the player manager's render method
    gameState.playerManager.render(ctx, gameState.camera.position);
  } catch (error) {
    console.error("Error rendering players:", error);
  }
}