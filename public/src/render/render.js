// src/render/render.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';
import { unitRenderer } from '../entities/ClientUnitRenderer.js';
// Import view renderers - comment these out if they cause circular references
// They should be available on the window object anyway
// import { renderTopDownView} from './renderTopDown.js';
// import { renderStrategicView } from './renderStrategic.js';

// CRITICAL DEBUG: This log proves render.js is being loaded
console.log('====== render.js FILE LOADED! TIMESTAMP:', new Date().toISOString(), '======');
console.log('[render.js] Module loaded and executing');

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');
ctx.imageSmoothingEnabled = false; // pixel-perfect rendering

// Rendering parameters
const scaleFactor = SCALE;

// Global scaling factor for the strategic (zoomed-out) view
const STRATEGIC_VIEW_SCALE = 0.25;  // was 0.5 – much smaller to fit more of the map

// Add a debug flag for rendering
const DEBUG_ENEMY_SPRITES = false; // Set to false to disable sprite debugging visuals

/**
 * Render the character
 */
export function renderCharacter() {
  console.log('[DEBUG renderCharacter] Called');
  const character = gameState.character;
  console.log('[DEBUG renderCharacter] character =', character);
  console.log('[DEBUG renderCharacter] character.draw =', character?.draw);
  console.log('[DEBUG renderCharacter] typeof character.draw =', typeof character?.draw);
  if (!character) {
    console.error("Cannot render character: character not defined in gameState");
    return;
  }

  // Don't render character if dead
  if (character.isDead) {
    console.log('[DEBUG renderCharacter] Character is dead, skipping render');
    return;
  }

  // Debug: Log the current view type to verify it's being detected correctly
  //console.log(`[renderCharacter] Current view type: ${gameState.camera?.viewType}`);

  // Determine scale factor based on view type
  const isStrategicView = gameState.camera?.viewType === 'strategic';
  const viewScaleFactor = isStrategicView ? STRATEGIC_VIEW_SCALE : 1.0;
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
      screenX - (character.width * character.renderScale * effectiveScale) / 2,
      screenY - (character.height * character.renderScale * effectiveScale) / 2,
      character.width * character.renderScale * effectiveScale,
      character.height * character.renderScale * effectiveScale
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
  
  // EFFICIENT: Cache water status - only check when tile changes
  const tileX = Math.floor(character.x);
  const tileY = Math.floor(character.y);

  if (!character._lastTileX || character._lastTileX !== tileX || character._lastTileY !== tileY) {
    character._lastTileX = tileX;
    character._lastTileY = tileY;
    const tile = gameState.mapManager?.getTile(tileX, tileY);

    // AGGRESSIVE DEBUG: Always log tile data when position changes
    if (Math.random() < 0.1) { // 10% of position changes
      console.log('[WATER DEBUG] Tile check at', tileX, tileY);
      console.log('[WATER DEBUG] tile =', tile);
      console.log('[WATER DEBUG] tile.spriteName =', tile?.spriteName);
      console.log('[WATER DEBUG] tile.type =', tile?.type);
      console.log('[WATER DEBUG] tile.biome =', tile?.biome);
      console.log('[WATER DEBUG] All tile properties:', tile ? Object.keys(tile) : 'tile is null');
    }

    // Check sprite name for water/lava - tile.spriteName is at top level, NOT tile.def.spriteName
    let isWater = false;
    if (tile && tile.spriteName) {
      const spriteName = tile.spriteName.toLowerCase();
      isWater = spriteName.includes('water') || spriteName.includes('deep') || spriteName.includes('lava');
      if (isWater) {
        console.log(`[WATER DETECTED] Found water by spriteName: "${tile.spriteName}" at (${tileX}, ${tileY})`);
      }
    }
    // Fallback: check by type ID (TILE_IDS.WATER = 3)
    else if (tile && tile.type === 3) {
      isWater = true;
      console.log(`[WATER DETECTED] Found water by type ID=3 at (${tileX}, ${tileY})`);
    }

    character._isOnWater = isWater;
  }

  const isOnWater = character._isOnWater || false;

  // Water submersion: Show only top 50% of sprite when on water
  const submersionRatio = isOnWater ? 0.5 : 1.0;
  const spriteSourceHeight = TILE_SIZE * submersionRatio;
  const renderHeight = character.height * character.renderScale * effectiveScale * submersionRatio;
  // Calculate vertical offset to shift sprite downward when submerged
  const submersionYOffset = isOnWater ? (character.height * character.renderScale * effectiveScale * 0.25) : 0;

  // Save the canvas state
  ctx.save();

  // Get sprite coordinates
  const spriteX = character.spriteX !== undefined ? character.spriteX : 0;
  const spriteY = character.spriteY !== undefined ? character.spriteY : 0;

  const charWidth = character.width * character.renderScale * effectiveScale;
  const charX = screenX - charWidth / 2;
  const charY = screenY - renderHeight / 2 + submersionYOffset;

  // Helper function to draw sprite with black outline using shadow
  const drawWithOutline = (x, y, width, height) => {
    // Draw black outline using shadow (drawn 4 times for thick outline)
    ctx.shadowColor = 'black';
    ctx.shadowBlur = 0;

    // Draw shadow in 8 directions for outline effect
    const outlineSize = 1.5;
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 4) {
      ctx.shadowOffsetX = Math.cos(angle) * outlineSize;
      ctx.shadowOffsetY = Math.sin(angle) * outlineSize;
      ctx.drawImage(
        characterSpriteSheet,
        spriteX, spriteY,
        TILE_SIZE, spriteSourceHeight,
        x, y,
        width, height
      );
    }

    // Reset shadow and draw main sprite
    ctx.shadowColor = 'transparent';
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.drawImage(
      characterSpriteSheet,
      spriteX, spriteY,
      TILE_SIZE, spriteSourceHeight,
      x, y,
      width, height
    );
  };

  // If character has a rotation, rotate around character center
  if (typeof character.rotation === 'number') {
    ctx.translate(screenX, screenY);
    ctx.rotate(character.rotation);
    drawWithOutline(-charWidth / 2, -renderHeight / 2 + submersionYOffset, charWidth, renderHeight);
  } else {
    drawWithOutline(charX, charY, charWidth, renderHeight);
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
  const enemySpriteSheet = enemySheetObj ? enemySheetObj.image : null;
  if (!enemySheetObj) {
    console.warn("Enemy sprite sheet not loaded – will use SpriteDatabase aliases or fallback rectangles");
  }
  
  // Log sheet info once to help with debugging
  if (enemySheetObj && !window.enemySpriteDebugLogged) {
    window.enemySpriteDebugLogged = true;
    console.log(`Enemy sprite sheet loaded: ${enemySpriteSheet.width}x${enemySpriteSheet.height}`, enemySheetObj.config);
  }
  
  // Get view scaling factor - FIXING back to 0.5 for strategic view
  const viewType = gameState.camera?.viewType || 'top-down';
  const viewScaleFactor = viewType === 'strategic' ? STRATEGIC_VIEW_SCALE : 1.0;
  
  // Get screen dimensions
  const screenWidth = canvas2D.width;
  const screenHeight = canvas2D.height;
  
  // Use camera's worldToScreen method if available for consistent coordinate transformation
  const useCamera = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
  
  // Render each enemy
  enemies.forEach(enemy => {
    try {
      // DEFENSIVE: Skip enemies with NaN/Infinity positions to prevent render crash
      if (!isFinite(enemy.x) || !isFinite(enemy.y)) {
        console.error(`[RENDER] Skipping enemy with invalid position: id=${enemy.id}, x=${enemy.x}, y=${enemy.y}`);
        return;
      }

      // Use tile size scaling instead of collision width to control sprite size
      const baseScale = enemy.renderScale || 2; // tiles wide/high
      const width = TILE_SIZE * baseScale * viewScaleFactor;
      const height = TILE_SIZE * baseScale * viewScaleFactor;

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

      // DEFENSIVE: Also check screen coordinates for NaN
      if (!isFinite(screenX) || !isFinite(screenY)) {
        return;
      }

      // Skip if off screen (with buffer)
      const buffer = width;
      if (screenX < -buffer || screenX > screenWidth + buffer ||
          screenY < -buffer || screenY > screenHeight + buffer) {
        return;
      }

      // ===================================================================
      // SPRITE SUBMERSION EFFECT - Calculate before rendering paths
      // ===================================================================
      // Check if enemy is standing on water/lava for submersion effect
      const enemyTileX = Math.floor(enemy.x);
      const enemyTileY = Math.floor(enemy.y);
      const enemyTile = gameState.mapManager?.getTile(enemyTileX, enemyTileY);

      // Check if water/lava tile by sprite name or type ID
      let enemyOnWater = false;
      if (enemyTile && enemyTile.spriteName) {
        const spriteName = enemyTile.spriteName.toLowerCase();
        enemyOnWater = spriteName.includes('water') || spriteName.includes('deep') || spriteName.includes('lava');
      } else if (enemyTile && (enemyTile.type === 3 || enemyTile.type === 6)) { // TILE_IDS.WATER or TILE_IDS.LAVA
        enemyOnWater = true;
      }

      // Water submersion: Show only top 50% of sprite when on water/lava
      const enemySubmersionRatio = enemyOnWater ? 0.5 : 1.0;
      const enemySpriteSourceHeight = 8 * enemySubmersionRatio;
      const enemyRenderHeight = height * enemySubmersionRatio;
      // Calculate vertical offset to shift sprite downward when submerged
      const enemySubmersionYOffset = enemyOnWater ? (height * 0.25) : 0;
      // ===================================================================

      const spriteDB = window.spriteDatabase;

      // DEBUG: Log boss sprite info (throttled)
      if (enemy.renderScale >= 6 || enemy.id?.includes('boss')) {
        if (!renderEnemies._bossDebugCount) renderEnemies._bossDebugCount = 0;
        renderEnemies._bossDebugCount++;
        if (renderEnemies._bossDebugCount <= 3 || renderEnemies._bossDebugCount % 100 === 0) {
          const hasSprite = spriteDB?.hasSprite?.(enemy.spriteName);
          console.log(`[BOSS SPRITE] id=${enemy.id}, spriteName=${enemy.spriteName}, hasSprite=${hasSprite}, renderScale=${enemy.renderScale}, spriteDB=${!!spriteDB}`);
        }
      }

      // If we have spriteName and sprite database, draw using it with submersion
      if (enemy.spriteName && spriteDB && spriteDB.hasSprite(enemy.spriteName)) {
        // For spriteDB rendering, we need to use canvas clipping for submersion
        if (enemyOnWater) {
          ctx.save();
          // Create clipping region for top 50% of sprite
          ctx.beginPath();
          ctx.rect(
            screenX - width/2,
            screenY - height/2 + enemySubmersionYOffset,
            width,
            enemyRenderHeight
          );
          ctx.clip();
        }

        spriteDB.drawSprite(
          ctx,
          enemy.spriteName,
          screenX - width/2,
          screenY - height/2,
          width,
          height
        );

        if (enemyOnWater) {
          ctx.restore();
        }

        // Draw health bar if needed (adjusted for submersion)
        if (enemy.health !== undefined && enemy.maxHealth !== undefined) {
          const hpPct = enemy.health / enemy.maxHealth;
          const bw = width;
          const bh = 4;
          const healthBarY = screenY - enemyRenderHeight/2 + enemySubmersionYOffset - bh - 2;
          ctx.fillStyle = 'rgba(0,0,0,0.5)';
          ctx.fillRect(screenX - bw/2, healthBarY, bw, bh);
          ctx.fillStyle = hpPct>0.6?'green':hpPct>0.3?'yellow':'red';
          ctx.fillRect(screenX - bw/2, healthBarY, bw*hpPct, bh);
        }
        return; // skip legacy drawing
      }

      // Save context for rotation
      ctx.save();

      // Apply enemy flashing effect if it's been hit
      if (enemy.isFlashing) {
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = 'red';
        ctx.fillRect(screenX - width/2, screenY - height/2, width, enemyRenderHeight);
        ctx.globalAlpha = 1.0;
      }

      if (enemySpriteSheet) {
        // If enemy has a rotation, use it
        if (typeof enemy.rotation === 'number') {
          ctx.translate(screenX, screenY);
          ctx.rotate(enemy.rotation);
          ctx.drawImage(
            enemySpriteSheet,
            enemy.spriteX || 0, enemy.spriteY || 0,
            8, enemySpriteSourceHeight,  // Clipped source height if on water
            -width/2, -enemyRenderHeight/2 + enemySubmersionYOffset, width, enemyRenderHeight  // Shift downward when submerged
          );
        } else {
          // Draw without rotation - centered
          ctx.drawImage(
            enemySpriteSheet,
            enemy.spriteX || 0, enemy.spriteY || 0,
            8, enemySpriteSourceHeight,  // Clipped source height if on water
            screenX - width/2, screenY - enemyRenderHeight/2 + enemySubmersionYOffset, width, enemyRenderHeight  // Shift downward when submerged
          );
        }
      } else {
        // Fallback: draw magenta rectangle so invisible enemies are obvious
        ctx.fillStyle = 'magenta';
        ctx.fillRect(screenX - width/2, screenY - height/2, width, height);
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
/**
 * Render military units
 */
export function renderUnits() {
  // Get units from unitManager
  if (!gameState.unitManager && !window.unitManager) {
    return; // No unit manager available
  }
  
  const unitManager = gameState.unitManager || window.unitManager;
  if (!unitManager) {
    return;
  }
  
  // Get units for rendering
  let units = [];
  try {
    // Use different methods based on what's available
    if (unitManager.getUnitsForRender) {
      units = unitManager.getUnitsForRender();
    } else if (unitManager.count > 0) {
      // Generate units array from SoA data
      units = [];
      for (let i = 0; i < unitManager.count; i++) {
        units.push({
          id: unitManager.id[i],
          type: unitManager.typeIdx ? unitManager.typeIdx[i] : unitManager.type[i],
          typeName: `Unit${unitManager.typeIdx ? unitManager.typeIdx[i] : unitManager.type[i]}`,
          displayName: `Unit ${unitManager.typeIdx ? unitManager.typeIdx[i] : unitManager.type[i]}`,
          category: 'military',
          x: unitManager.x[i],
          y: unitManager.y[i],
          health: unitManager.hp ? unitManager.hp[i] : unitManager.health[i],
          maxHealth: 100, // Default
          morale: unitManager.morale ? unitManager.morale[i] : 50,
          state: unitManager.state ? unitManager.state[i] : 0,
          team: unitManager.owner ? unitManager.owner[i] : 'neutral',
          sprite: {
            sheet: 'Mixed_Units',
            name: `Mixed_Units_0_${Math.min(5, unitManager.typeIdx ? unitManager.typeIdx[i] : unitManager.type[i])}`,
            scale: 0.5
          }
        });
      }
    }
  } catch (error) {
    console.warn('[renderUnits] Error getting units for render:', error);
    return;
  }
  
  // If no units, nothing to render
  if (!units || !Array.isArray(units) || units.length === 0) {
    return;
  }
  
  // Get view type for proper scaling
  const viewType = gameState.camera?.viewType || 'top-down';

  // Set up selection callback
  unitRenderer.isUnitSelected = (unitId) => {
    return gameState.selectedUnits && gameState.selectedUnits.includes(unitId);
  };

  // Render units using the unit renderer
  try {
    unitRenderer.renderUnits(ctx, units, gameState.camera.position, viewType);
    
    // Debug info occasionally
    if (Math.random() < 0.001) { // Very occasionally
      console.log(`[renderUnits] Rendered ${units.length} units in ${viewType} view`);
    }
  } catch (error) {
    console.error('[renderUnits] Error rendering units:', error);
  }
}

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
  const viewScale = viewType === 'strategic' ? STRATEGIC_VIEW_SCALE : 1.0;

  const W = canvas2D.width;
  const H = canvas2D.height;
  const useCam = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
  
  // Get bullet scale factor if available
  const bulletScale = bm.bulletScale || 1.0;

  // DIAGNOSTIC: Log every bullet position being rendered (first 3 bullets only to avoid spam)
  if (bm.bulletCount > 0 && Math.random() < 0.05) { // 5% sample rate
    console.log(`[BULLET RENDER] Rendering ${bm.bulletCount} bullets. First bullet at world pos: (${bm.x[0].toFixed(2)}, ${bm.y[0].toFixed(2)}), ID: ${bm.id[0]}`);
    if (gameState.camera) {
      console.log(`[BULLET RENDER] Camera at: (${gameState.camera.position.x.toFixed(2)}, ${gameState.camera.position.y.toFixed(2)})`);
    }
  }

  for (let i = 0; i < bm.bulletCount; i++) {
    // DEFENSIVE: Skip bullets with NaN/Infinity positions to prevent render crash
    const bulletX = bm.x[i];
    const bulletY = bm.y[i];
    if (!isFinite(bulletX) || !isFinite(bulletY)) {
      // Log once per bullet ID to identify the source
      if (!renderBullets._nanWarned) renderBullets._nanWarned = new Set();
      const bulletId = bm.id[i];
      if (!renderBullets._nanWarned.has(bulletId)) {
        renderBullets._nanWarned.add(bulletId);
        console.error(`[RENDER] Skipping bullet with invalid position: id=${bulletId}, x=${bulletX}, y=${bulletY}, vx=${bm.vx[i]}, vy=${bm.vy[i]}, targetX=${bm.targetX[i]}, targetY=${bm.targetY[i]}`);
      }
      continue;
    }

    // world → screen
    let sx, sy;
    if (useCam) {
      ({ x: sx, y: sy } = gameState.camera.worldToScreen(
        bulletX, bulletY, W, H, TILE_SIZE
      ));
    } else {
      sx = (bulletX - gameState.camera.position.x) * TILE_SIZE * viewScale + W/2;
      sy = (bulletY - gameState.camera.position.y) * TILE_SIZE * viewScale + H/2;
    }

    // DEFENSIVE: Also check screen coordinates
    if (!isFinite(sx) || !isFinite(sy)) {
      continue;
    }

    // VISUAL SIZE: Make bullets appear 50% larger than their collision size
    // Collision is 0.6 tiles, but visual is 0.6 * 1.5 = 0.9 tiles for better visibility
    const BULLET_VISUAL_SCALE = viewType === 'strategic' ? 0.5 : 1.5; // Smaller in strategic view, larger in normal view
    const minPx = 3;
    // CRITICAL: Default to 0.6 tiles, NOT 8! (8 tiles = ~144px = WAY too large)
    // Use sensible bounds: bullets should be 0.3-2.0 tiles, anything outside is invalid
    const baseW = (bm.width[i] > 0.1 && bm.width[i] < 5) ? bm.width[i] : 0.6;
    const baseH = (bm.height[i] > 0.1 && bm.height[i] < 5) ? bm.height[i] : 0.6;
    const drawW = Math.max(baseW * TILE_SIZE * BULLET_VISUAL_SCALE, minPx); // Scale up visual size
    const drawH = Math.max(baseH * TILE_SIZE * BULLET_VISUAL_SCALE, minPx); // Scale up visual size

    // cull
    if (sx < -100 || sx > W+100 || sy < -100 || sy > H+100) continue;

    // Apply bullet scale to width and height
    const w = drawW;
    const h = drawH;

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
    // FINAL SAFETY: Validate all gradient parameters before creating
    if (!isFinite(drawW) || drawW <= 0) {
      console.error(`[RENDER] Invalid drawW for bullet: id=${bm.id[i]}, drawW=${drawW}, width=${bm.width[i]}`);
      continue;
    }
    // Normalize IDs for comparison - server sends "entity_X" but character.id may be numeric
    const localId = String(gameState.character?.id || '').replace(/^entity_/, '');
    const bulletOwner = String(bm.ownerId[i] || '').replace(/^entity_/, '');
    const isLocal = localId && (localId === bulletOwner);
    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, drawW);
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
    ctx.arc(sx, sy, drawW, 0, Math.PI*2);
    ctx.fill();

    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(sx, sy, drawW*0.3, 0, Math.PI*2);
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
  if (gameState.camera && gameState.camera.viewType === 'strategic') {
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
  const viewType = gameState.camera ? gameState.camera.viewType : 'top-down';

  try {
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
    } else if (viewType === 'first-person') {
      // FIRST-PERSON VIEW: Update and render 3D scene
      if (typeof window.updateFirstPerson === 'function' && window.renderer && window.scene && window.camera) {
        // Update the 3D scene based on character position and game state
        window.updateFirstPerson(window.camera);

        // Render the Three.js scene
        window.renderer.render(window.scene, window.camera);
      } else {
        console.error("First-person view not available - missing updateFirstPerson, renderer, scene, or camera");
      }
    } else {
      console.error(`Unknown view type ${viewType}`);
    }
  } catch (error) {
    console.error("Error rendering game view:", error);
  }
  
  // Draw entities - wrap each in try-catch to prevent one bad entity from crashing everything
  try {
    renderBullets();
  } catch (error) {
    console.error("[RENDER] renderBullets crashed:", error);
  }

  try {
    renderEnemies();
  } catch (error) {
    console.error("[RENDER] renderEnemies crashed:", error);
  }

  try {
    renderUnits();
  } catch (error) {
    console.error("[RENDER] renderUnits crashed:", error);
  }

  try {
    renderPlayers();
  } catch (error) {
    console.error("[RENDER] renderPlayers crashed:", error);
  }

  try {
    renderCharacter();
  } catch (error) {
    console.error("[RENDER] renderCharacter crashed:", error);
  }

  // Draw UI elements
  try {
    renderUI();
  } catch (error) {
    console.error("[RENDER] renderUI crashed:", error);
  }
  
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
  // Draw RTS command mode indicator
  if (typeof window.isInRTSMode === 'function' && window.isInRTSMode()) {
    ctx.save();
    ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
    ctx.strokeStyle = 'rgba(0, 200, 0, 1)';
    ctx.lineWidth = 2;

    // Draw indicator in top-right corner
    const x = canvas2D.width - 150;
    const y = 20;

    ctx.fillRect(x, y, 130, 30);
    ctx.strokeRect(x, y, 130, 30);

    ctx.fillStyle = '#000000';
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('RTS MODE', x + 65, y + 15);

    ctx.restore();
  }

  // REMOVED: Placeholder health bar (now using Character Panel UI component)
  // Draw player health bar
  /*
  if (gameState.character && gameState.character.health !== undefined) {
    if (gameState.character.health <= 0) {
      // Big centred message
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(0,0,canvas2D.width,canvas2D.height);
      ctx.fillStyle='red';
      ctx.font='72px serif';
      ctx.textAlign='center';
      ctx.fillText('YOU DIED', canvas2D.width/2, canvas2D.height/2);
      return; // skip rest of UI when dead
    }
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
  */
  
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
  const viewScaleFactor = viewType === 'strategic' ? STRATEGIC_VIEW_SCALE : 1.0;
  
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