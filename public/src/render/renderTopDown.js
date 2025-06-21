// src/render/renderTopDown.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

const camera = gameState.camera;
const scaleFactor = camera.getViewScaleFactor();

// Debug flags
const DEBUG_RENDERING = false;

// ANTI-FLICKERING: Add a tile cache to prevent constantly requesting the same chunks
// Top-down view has fewer tiles, so we need a smaller cache
const topDownTileCache = new Map();

// Track when we last updated chunks to limit request frequency
let lastChunkUpdateTime = 0;
const CHUNK_UPDATE_INTERVAL = 2000; // Only update chunks every 2 seconds for top-down (less frequently needed)

export function renderTopDownView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  if (!mapManager) {
    console.warn("Cannot render map: map manager not available");
    return;
  }

  // Calculate current time for throttling
  const now = performance.now();

  // Determine visible tiles based on camera position
  const tilesInViewX = Math.ceil(canvas2D.width / (TILE_SIZE * scaleFactor));
  const tilesInViewY = Math.ceil(canvas2D.height / (TILE_SIZE * scaleFactor));

  const startX = Math.floor(camera.position.x - tilesInViewX / 2);
  const startY = Math.floor(camera.position.y - tilesInViewY / 2);
  const endX = startX + tilesInViewX;
  const endY = startY + tilesInViewY;
  
  // ANTI-FLICKERING: Only update visible chunks periodically, not every frame
  // Top-down view needs much less frequent updates since it shows fewer tiles
  if (now - lastChunkUpdateTime > CHUNK_UPDATE_INTERVAL) {
    // If mapManager has updateVisibleChunks method, call it only periodically
    if (mapManager.updateVisibleChunks) {
      mapManager.updateVisibleChunks(camera.position.x, camera.position.y);
    }
    
    lastChunkUpdateTime = now;
  }
  
  const tileSheetObj = spriteManager.getSpriteSheet("tile_sprites");
  if (!tileSheetObj) return;
  const tileSpriteSheet = tileSheetObj.image;

  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      // ANTI-FLICKERING: Check cache first before requesting tile
      const tileKey = `${x},${y}`;
      let tile = topDownTileCache.get(tileKey);
      
      if (!tile) {
        // Get the tile from map manager if not in cache
        tile = mapManager.getTile ? mapManager.getTile(x, y) : null;
        
        // Store in cache if valid
        if (tile) {
          topDownTileCache.set(tileKey, tile);
        }
      }
      
      if (tile) {
        const spritePos = TILE_SPRITES[tile.type];
        
        // Convert tile grid position to world position
        // In this game, tile coordinates are the same as world coordinates
        const worldX = x;
        const worldY = y;
        
        // FIX: Use correct TILE_SIZE parameter (not multiplied by scaleFactor)
        // This was causing the double scaling issue
        const screenPos = camera.worldToScreen(
          worldX + 0.5,  // shift to tile center for proper alignment
          worldY + 0.5, 
          canvas2D.width, 
          canvas2D.height, 
          TILE_SIZE  // Use base TILE_SIZE, let worldToScreen apply scaling
        );
        
        const sCfg = tileSheetObj.config;
        const spriteW = sCfg.defaultSpriteWidth  || TILE_SIZE;
        const spriteH = sCfg.defaultSpriteHeight || TILE_SIZE;
        ctx.drawImage(
          tileSpriteSheet,
          spritePos.x, spritePos.y, spriteW, spriteH, // Source rectangle
          screenPos.x - (spriteW * scaleFactor / 2),
          screenPos.y - (spriteH * scaleFactor / 2),
          spriteW * scaleFactor,
          spriteH * scaleFactor
        );

        // ---- Height shading -------------------------------------------------
        if (tile.height && tile.height > 0) {
          const alpha = Math.min(tile.height / 15, 1) * 0.35; // up to 35% darken
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
          // Draw a grid outline in red
          ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
          ctx.lineWidth = 1;
          ctx.strokeRect(
            screenPos.x - (spriteW * scaleFactor / 2),
            screenPos.y - (spriteH * scaleFactor / 2),
            spriteW * scaleFactor,
            spriteH * scaleFactor
          );
          
          // Draw tile coordinates for reference (only every few tiles to avoid clutter)
          if ((x % 5 === 0 && y % 5 === 0) || (x === 0 && y === 0)) {
            ctx.fillStyle = 'white';
            ctx.font = '8px Arial';
            ctx.fillText(
              `(${x},${y})`, 
              screenPos.x - (spriteW * scaleFactor / 2) + 2,
              screenPos.y - (spriteH * scaleFactor / 2) + 8
            );
          }
        }
      }
    }
  }

  // ANTI-FLICKERING: Periodically clean up cache to prevent memory leaks
  // Clean up less frequently for top-down view
  if (now % 60000 < 16) { // Every minute
    const cacheCleanupDistance = Math.max(tilesInViewX, tilesInViewY) * 2;
    
    for (const [key, _] of topDownTileCache) {
      const [tileX, tileY] = key.split(',').map(Number);
      const dx = Math.abs(tileX - startX - tilesInViewX/2);
      const dy = Math.abs(tileY - startY - tilesInViewY/2);
      
      if (dx > cacheCleanupDistance || dy > cacheCleanupDistance) {
        topDownTileCache.delete(key);
      }
    }
  }
}

// Export to window object to avoid circular references
window.renderTopDownView = renderTopDownView;

// Log the export to ensure it's registered globally
console.log("TopDown view render function registered:", window.renderTopDownView ? "Success" : "Failed");
