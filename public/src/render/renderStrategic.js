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

// Rendering parameters for strategic view (zoomed out)
const scaleFactor = 0.5; // Smaller scale for strategic view

// ANTI-FLICKERING: Add a tile cache to prevent constantly requesting the same chunks
const strategicTileCache = new Map();

// Track when we last updated chunks to limit request frequency
let lastChunkUpdateTime = 0;
const CHUNK_UPDATE_INTERVAL = 1000; // Only update chunks every 1 second

export function renderStrategicView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  if (!mapManager) {
    console.warn("Cannot render map: map manager not available");
    return;
  }

  // Calculate current time for throttling
  const now = performance.now();

  // Determine visible tiles based on camera position
  // For strategic view, we show more tiles due to the smaller scale
  const tilesInViewX = Math.ceil(canvas2D.width / (TILE_SIZE * scaleFactor));
  const tilesInViewY = Math.ceil(canvas2D.height / (TILE_SIZE * scaleFactor));

  // Add buffer tiles to avoid visual gaps at edges
  const bufferTiles = 4;
  const startX = Math.floor(camera.position.x / TILE_SIZE - tilesInViewX / 2) - bufferTiles;
  const startY = Math.floor(camera.position.y / TILE_SIZE - tilesInViewY / 2) - bufferTiles;
  const endX = startX + tilesInViewX + bufferTiles * 2;
  const endY = startY + tilesInViewY + bufferTiles * 2;
  
  // ANTI-FLICKERING: Only update visible chunks periodically, not every frame
  if (now - lastChunkUpdateTime > CHUNK_UPDATE_INTERVAL) {
    // If mapManager has updateVisibleChunks method, call it only periodically
    if (mapManager.updateVisibleChunks) {
      // Temporarily disable network requests if possible
      const originalNetworkManager = mapManager.networkManager;
      const wasAutoUpdating = mapManager.autoUpdateChunks;
      
      try {
        // Disable automatic chunk updates during rendering to prevent flickering
        mapManager.autoUpdateChunks = false;
        
        // Update visible chunks without network requests if possible
        if (mapManager.updateVisibleChunksLocally) {
          mapManager.updateVisibleChunksLocally(camera.position.x, camera.position.y);
        } else {
          // If no local update method, use regular update but disable network temporarily
          mapManager.networkManager = null; // Temporarily disable network requests
          mapManager.updateVisibleChunks(camera.position.x, camera.position.y);
          mapManager.networkManager = originalNetworkManager; // Restore network manager
        }
      } catch (err) {
        console.error("Error updating chunks:", err);
        // Restore values in case of error
        mapManager.autoUpdateChunks = wasAutoUpdating;
        mapManager.networkManager = originalNetworkManager;
      }
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
        // Draw tile with strategic scaling
        ctx.drawImage(
          tileSpriteSheet,
          spritePos.x, spritePos.y, TILE_SIZE, TILE_SIZE, // Source rectangle
          (x * TILE_SIZE - camera.position.x) * scaleFactor + canvas2D.width / 2,
          (y * TILE_SIZE - camera.position.y) * scaleFactor + canvas2D.height / 2,
          TILE_SIZE * scaleFactor,
          TILE_SIZE * scaleFactor
        );
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
