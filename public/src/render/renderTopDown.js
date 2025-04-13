// src/render/renderTopDown.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES, SCALE } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

// Use the same SCALE constant as entities for consistent sizing
const scaleFactor = SCALE; // Standard scale factor (5)

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

  // Get the view scale factor from camera
  const viewScaleFactor = camera.getViewScaleFactor();
  
  // Calculate the effective tile size with scaling applied
  const effectiveTileSize = TILE_SIZE * scaleFactor * viewScaleFactor;

  // Calculate visible tiles based on screen dimensions and tile size
  // Add a small margin to prevent pop-in at screen edges
  const tilesInViewX = Math.ceil(canvas2D.width / effectiveTileSize) + 2;
  const tilesInViewY = Math.ceil(canvas2D.height / effectiveTileSize) + 2;

  // Calculate the starting tile coordinates
  const startX = Math.floor(camera.position.x - tilesInViewX / 2);
  const startY = Math.floor(camera.position.y - tilesInViewY / 2);
  const endX = startX + tilesInViewX;
  const endY = startY + tilesInViewY;
  
  // ANTI-FLICKERING: Only update visible chunks periodically, not every frame
  // Top-down view needs much less frequent updates since it shows fewer tiles
  if (now - lastChunkUpdateTime > CHUNK_UPDATE_INTERVAL) {
    // If mapManager has updateVisibleChunks method, call it only periodically
    if (mapManager.updateVisibleChunks) {
      // Request a slightly larger area than visible to prevent edge loading
      mapManager.updateVisibleChunks(camera.position.x, camera.position.y, 5);
    }
    
    lastChunkUpdateTime = now;
  }
  
  const tileSheetObj = spriteManager.getSpriteSheet("tile_sprites");
  if (!tileSheetObj) return;
  const tileSpriteSheet = tileSheetObj.image;

  // Clear the entire canvas before rendering tiles
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas2D.width, canvas2D.height);

  // Debug helper to visualize the view boundaries
  let tilesRendered = 0;

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
        tilesRendered++;
        const spritePos = TILE_SPRITES[tile.type];
        
        // Convert world to screen coordinates using the same formula as entities
        // Apply a small offset (0.5, 0.5) to center tiles on grid positions
        const worldX = x + 0.5; // Center in tile
        const worldY = y + 0.5; // Center in tile
        
        // Use standard world-to-screen transformation
        const screenX = (worldX - camera.position.x) * TILE_SIZE * viewScaleFactor + canvas2D.width / 2;
        const screenY = (worldY - camera.position.y) * TILE_SIZE * viewScaleFactor + canvas2D.height / 2;
        
        // Draw the tile centered at the screen position
        ctx.drawImage(
          tileSpriteSheet,
          spritePos.x, spritePos.y, TILE_SIZE, TILE_SIZE, // Source rectangle
          screenX - (effectiveTileSize / 2), // Center horizontally 
          screenY - (effectiveTileSize / 2), // Center vertically
          effectiveTileSize,
          effectiveTileSize
        );
        
        // DEBUGGING: Draw tile coordinates for troubleshooting
        if (gameState.debug && (x % 10 === 0 && y % 10 === 0)) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
          ctx.font = '10px Arial';
          ctx.fillText(`(${x},${y})`, screenX, screenY);
        }
      }
    }
  }

  // DEBUGGING: Show how many tiles we're rendering
  if (gameState.debug) {
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.fillText(`Rendering ${tilesRendered} tiles, view: ${tilesInViewX}x${tilesInViewY}`, 10, 20);
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
