// src/render/renderStrategic.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

// Debug flags
const DEBUG_RENDERING = false;

// ANTI-FLICKERING: Add a tile cache to prevent constantly requesting the same chunks
const strategicTileCache = new Map();

// Track when we last updated chunks to limit request frequency
let lastChunkUpdateTime = 0;
const CHUNK_UPDATE_INTERVAL = 1000;

export function renderStrategicView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  if (!mapManager) {
    console.warn("Cannot render map: map manager not available");
    return;
  }
  ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);
  // Calculate current time for throttling
  const now = performance.now();

  // For strategic view, we want to see more of the map
  // so we'll show more tiles based on the smaller scale factor
    // ② Re‑read current zoom every frame
  const scaleFactor = camera.getViewScaleFactor();
  const tilesInViewX = Math.ceil(canvas2D.width  / (TILE_SIZE * scaleFactor)) + 2;
  const tilesInViewY = Math.ceil(canvas2D.height / (TILE_SIZE * scaleFactor)) + 2;

  const startX = Math.floor(camera.position.x - tilesInViewX / 2);
  const startY = Math.floor(camera.position.y - tilesInViewY / 2);
  const endX = startX + tilesInViewX;
  const endY = startY + tilesInViewY;
  
  // Only update visible chunks periodically to reduce network/CPU load
  if (now - lastChunkUpdateTime > CHUNK_UPDATE_INTERVAL) {
    if (mapManager.updateVisibleChunks) {
      mapManager.updateVisibleChunks(camera.position.x, camera.position.y);
    }
    lastChunkUpdateTime = now;
  }
  
  const tileSheetObj = spriteManager.getSpriteSheet("tile_sprites");
  if (!tileSheetObj) { console.warn('tile_sprites not loaded'); return; }
  const tileSpriteSheet = tileSheetObj.image;

  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      // ANTI-FLICKERING: Check cache first before requesting tile
      const tileKey = `${x},${y}`;
      let tile = strategicTileCache.get(tileKey);
      
      if (!tile) {
        // Get the tile from map manager if not in cache
        tile = mapManager.getTile ? mapManager.getTile(x, y) : null;
        
        // Store in cache if valid
        if (tile) {
          strategicTileCache.set(tileKey, tile);
        }
      }
      
      if (tile) {
        const spritePos = TILE_SPRITES[tile.type];
        
        // Convert tile grid position to world position
        // In this game, tile coordinates are the same as world coordinates
        const worldX = x;
        const worldY = y;
        
        // FIX: Use correct TILE_SIZE parameter (not multiplied by scaleFactor)
        const screenPos = camera.worldToScreen(
          worldX + 0.5, // shift to tile center
          worldY + 0.5, 
          canvas2D.width, 
          canvas2D.height, 
          TILE_SIZE  // Use base TILE_SIZE, let worldToScreen apply scaling
        );
        
        // Draw tile using the consistent screen position
        const sCfg = tileSheetObj.config;
        const spriteW = sCfg.defaultSpriteWidth  || TILE_SIZE;
        const spriteH = sCfg.defaultSpriteHeight || TILE_SIZE;
        ctx.drawImage(
          tileSpriteSheet,
          spritePos.x, spritePos.y, spriteW, spriteH,
          screenPos.x - (spriteW * scaleFactor / 2),
          screenPos.y - (spriteH * scaleFactor / 2),
          spriteW * scaleFactor,
          spriteH * scaleFactor
        );

        // Height shading (lighter to darker)
        if (tile.height && tile.height > 0) {
          const alpha = Math.min(tile.height / 15, 1) * 0.25;
          ctx.fillStyle = `rgba(0,0,0,${alpha.toFixed(3)})`;
          ctx.fillRect(
            screenPos.x - (spriteW * scaleFactor / 2),
            screenPos.y - (spriteH * scaleFactor / 2),
            spriteW * scaleFactor,
            spriteH * scaleFactor
          );
        }
        
        // Add debug visualization to help with alignment
        if (DEBUG_RENDERING) {
          // Draw a grid outline in red (less visible in strategic view)
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
          ctx.lineWidth = 1;
          ctx.strokeRect(
            screenPos.x - (spriteW * scaleFactor / 2),
            screenPos.y - (spriteH * scaleFactor / 2),
            spriteW * scaleFactor,
            spriteH * scaleFactor
          );
          
          // Draw tile coordinates for reference (only every several tiles to avoid clutter)
          if ((x % 10 === 0 && y % 10 === 0) || (x === 0 && y === 0)) {
            ctx.fillStyle = 'white';
            ctx.font = '6px Arial';
            ctx.fillText(
              `(${x},${y})`, 
              screenPos.x - (spriteW * scaleFactor / 2) + 1,
              screenPos.y - (spriteH * scaleFactor / 2) + 6
            );
          }
        }
      }
    }
  }

  // ANTI-FLICKERING: Periodically clean up cache to prevent memory leaks
  // Clean up every ~30 seconds
  if (now % 30000 < 16) {
    // Keep this light - just clear old tiles far outside the view
    const cacheCleanupDistance = Math.max(tilesInViewX, tilesInViewY) * 2;
    
    for (const [key, _] of strategicTileCache) {
      const [tileX, tileY] = key.split(',').map(Number);
      const dx = Math.abs(tileX - startX - tilesInViewX/2);
      const dy = Math.abs(tileY - startY - tilesInViewY/2);
      
      if (dx > cacheCleanupDistance || dy > cacheCleanupDistance) {
        strategicTileCache.delete(key);
      }
    }
  }
}

// Make sure this function is defined in window scope
window.renderStrategicView = renderStrategicView;

// Log when the file loads to verify it's being included
console.log("Strategic view render system initialized. Function available:", typeof window.renderStrategicView === 'function');
