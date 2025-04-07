// src/render/renderStrategic.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';
import { renderCharacter, renderEnemies, renderBullets, renderPlayers } from './render.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Rendering parameters
const scaleFactor = 0.5; // Reduce tile size for strategic view

// Track last update time for throttling
let lastRenderFrameTime = 0;

// ANTI-FLICKERING: Tile rendering cache - stores the last rendered state of each tile
const tileRenderCache = {
  // Format: "x,y" -> { spriteX, spriteY, lastRendered }
  tiles: new Map(),
  lastCleanup: 0,
  maxSize: 10000, // Maximum tiles to cache
  cleanupInterval: 30000, // Cleanup less frequently (every 30 seconds)
  
  // Add a tile to the cache
  addTile: function(x, y, spriteX, spriteY) {
    const key = `${x},${y}`;
    this.tiles.set(key, { 
      spriteX, 
      spriteY, 
      lastRendered: Date.now() 
    });
    
    // Cleanup if needed
    const now = Date.now();
    if (now - this.lastCleanup > this.cleanupInterval) {
      this.cleanup();
      this.lastCleanup = now;
    }
  },
  
  // Get a cached tile
  getTile: function(x, y) {
    const key = `${x},${y}`;
    return this.tiles.get(key);
  },
  
  // Cleanup the cache (remove oldest tiles)
  cleanup: function() {
    if (this.tiles.size <= this.maxSize) return;
    
    // Sort tiles by last rendered time
    const sortedTiles = Array.from(this.tiles.entries())
      .sort((a, b) => a[1].lastRendered - b[1].lastRendered);
    
    // Remove oldest tiles
    const removeCount = Math.floor((this.tiles.size - this.maxSize) / 2); // Only remove half the excess
    for (let i = 0; i < removeCount; i++) {
      this.tiles.delete(sortedTiles[i][0]);
    }
  }
};

// Debug variables to track rendering consistency
const DEBUG = {
  enabled: false, // Disable verbose debugging by default
  lastRender: 0,
  frameCount: 0,
  lastRenderTime: 0,
  renderTimes: [],
  chunkCountHistory: [],
  isFirstRender: true,
  visibleTileCount: 0
};

export function renderStrategicView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  // Skip rendering if no map manager
  if (!mapManager) {
    return; // Silently return if no map manager
  }

  // Clear the canvas
  ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);
  
  // ANTI-FLICKERING: Throttle chunk updates
  const now = performance.now();
  const frameTime = now - lastRenderFrameTime;
  
  // Calculate how many tiles we need to render to fill the screen
  const tilesInViewX = Math.ceil(canvas2D.width / (TILE_SIZE * scaleFactor)) + 2;
  const tilesInViewY = Math.ceil(canvas2D.height / (TILE_SIZE * scaleFactor)) + 2;
  
  // Add buffer to prevent black edges and empty areas
  const bufferTiles = 4;

  // Calculate the starting tile coordinates
  const cameraX = Math.floor(camera.position.x);
  const cameraY = Math.floor(camera.position.y);
  const startX = Math.floor(cameraX / TILE_SIZE - tilesInViewX / 2) - bufferTiles;
  const startY = Math.floor(cameraY / TILE_SIZE - tilesInViewY / 2) - bufferTiles;
  const endX = startX + tilesInViewX + bufferTiles * 2;
  const endY = startY + tilesInViewY + bufferTiles * 2;
  
  // Get sprite sheet
  const tileSheetObj = spriteManager.getSpriteSheet("tile_sprites");
  if (!tileSheetObj) return;
  const tileSpriteSheet = tileSheetObj.image;
  
  // Track how many tiles we actually render
  let renderedTileCount = 0;
  let cachedTileCount = 0;
  
  // Track which chunks are being used in this frame
  const usedChunks = new Set();
  
  // ANTI-FLICKERING: Set to track which cached tiles we've rendered
  const renderedCachedTiles = new Set();
  
  // First pass: Render actual tiles from the map
  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      // Calculate chunk coordinates for this tile
      const chunkX = Math.floor(x / mapManager.chunkSize);
      const chunkY = Math.floor(y / mapManager.chunkSize);
      const chunkKey = `${chunkX},${chunkY}`;
      
      // Track used chunk
      usedChunks.add(chunkKey);
      
      // Calculate screen position first to see if it's visible
      const screenX = (x * TILE_SIZE - camera.position.x) * scaleFactor + canvas2D.width / 2;
      const screenY = (y * TILE_SIZE - camera.position.y) * scaleFactor + canvas2D.height / 2;
      
      // Skip if completely out of view (with buffer)
      const tileScreenSize = TILE_SIZE * scaleFactor;
      if (screenX < -tileScreenSize || screenX > canvas2D.width + tileScreenSize ||
          screenY < -tileScreenSize || screenY > canvas2D.height + tileScreenSize) {
        continue;  
      }
      
      const tile = mapManager.getTile ? mapManager.getTile(x, y) : null;
      
      if (tile) {
        const spritePos = TILE_SPRITES[tile.type];
        if (!spritePos) continue; // Skip if missing sprite data
        
        // Draw tile
        ctx.drawImage(
          tileSpriteSheet,
          spritePos.x, spritePos.y, TILE_SIZE, TILE_SIZE, // Source rectangle
          screenX, screenY, tileScreenSize, tileScreenSize // Destination rectangle
        );
        
        // ANTI-FLICKERING: Add to cache
        tileRenderCache.addTile(x, y, spritePos.x, spritePos.y);
        
        // Mark as rendered
        renderedCachedTiles.add(`${x},${y}`);
        
        renderedTileCount++;
      } 
      else {
        // ANTI-FLICKERING: Check the tile cache for previously rendered tiles
        const cachedTile = tileRenderCache.getTile(x, y);
        
        if (cachedTile) {
          // Draw from cache
          ctx.drawImage(
            tileSpriteSheet,
            cachedTile.spriteX, cachedTile.spriteY, TILE_SIZE, TILE_SIZE,
            screenX, screenY, tileScreenSize, tileScreenSize
          );
          
          // Mark as rendered
          renderedCachedTiles.add(`${x},${y}`);
          
          cachedTileCount++;
        }
      }
    }
  }
  
  // Update visible chunks in map manager occasionally
  // Only do this on a timer to prevent flickering
  if (frameTime > 1000) { // Only update chunks every 1 second
    if (mapManager.updateVisibleChunks) {
      // Use slightly larger distance for strategic view to reduce edge flickering
      const strategicChunkDistance = 3; 
      mapManager.updateVisibleChunks(camera.position.x, camera.position.y, strategicChunkDistance);
    }
    lastRenderFrameTime = now;
  }

  // Draw character and entities
  renderCharacter();
  renderEnemies();
  renderBullets();
  renderPlayers();
}

