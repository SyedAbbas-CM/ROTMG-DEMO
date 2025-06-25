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

// Provide a helper to wipe this cache when the client switches maps to avoid
// showing stale tiles from the previous world.
export function clearTopDownCache() {
  topDownTileCache.clear();
  console.log('[TopDownView] Tile cache cleared');
}

// Expose via global window for ease of access without circular imports
window.clearTopDownCache = clearTopDownCache;

// Lazy loader for atlas json – ensures any sheet referenced by map tile gets
// loaded on demand without blocking the main thread.
function ensureSheetLoaded(sheetName){
  if (spriteManager.getSpriteSheet(sheetName)) return;
  fetch(`/assets/atlases/${sheetName}.json`).then(r=>r.json()).then(cfg=>{
    cfg.name ||= sheetName;
    // Provide fallback path for image if missing and meta.image exists
    if(!cfg.path && cfg.meta && cfg.meta.image){
      cfg.path = cfg.meta.image.startsWith('/')? cfg.meta.image : ('/' + cfg.meta.image);
    }
    return spriteManager.loadSpriteSheet(cfg);
  }).catch(err=>{
    if(!ensureSheetLoaded.loggedMissing){ensureSheetLoaded.loggedMissing=new Set();}
    if(!ensureSheetLoaded.loggedMissing.has(sheetName)){
      console.warn(`[TopDown] Failed to auto-load sheet ${sheetName}:`,err);
      ensureSheetLoaded.loggedMissing.add(sheetName);
    }
  });
}

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
  
  // NOTE: sprite sheets are fetched per-tile inside the render loop so
  // that mixed sheets can be rendered in the same view.

  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      // ANTI-FLICKERING: Check cache first before requesting tile
      const tileKey = `${mapManager.activeMapId || 'map'}:${x},${y}`;
      let tile = topDownTileCache.get(tileKey);
      
      if (!tile) {
        // Get the tile from map manager if not in cache
        tile = mapManager.getTile ? mapManager.getTile(x, y) : null;
        
        // Store in cache if valid
        if (tile) {
          topDownTileCache.set(tileKey, tile);
        }
      }
      
      if (!tile) continue;
      
      // Determine sprite – per-tile override takes priority
      let spritePos;
      let spriteSheetName = 'tile_sprites';
      if (tile.properties && tile.properties.sprite) {
        const rawName = tile.properties.sprite;
        // Try to derive sheet name quickly to preload
        const parts = rawName.split('_sprite_');
        if(parts.length>1){ ensureSheetLoaded(parts[0]); }
        const spriteObj = spriteManager.fetchSprite(rawName);
        if (spriteObj) {
          spriteSheetName = spriteObj.sheetName;
          spritePos = { x: spriteObj.x, y: spriteObj.y };
        }
      }
      if (!spritePos && tile.spriteName) {
        const rawName=tile.spriteName;
        const parts=rawName.split('_sprite_');
        if(parts.length>1){ ensureSheetLoaded(parts[0]); }
        const spriteObj = spriteManager.fetchSprite(rawName);
        if (spriteObj) {
          spriteSheetName = spriteObj.sheetName;
          spritePos = { x: spriteObj.x, y: spriteObj.y };
        }
      }
      if (!spritePos) {
        spritePos = TILE_SPRITES[tile.type];
      }
      
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
        mapManager.tileSize || TILE_SIZE
      );
      
      ensureSheetLoaded(spriteSheetName);
      const sheetObj = spriteManager.getSpriteSheet(spriteSheetName);
      if(!sheetObj) continue; // wait until loaded next frame
      const sCfg = sheetObj.config;
      const spriteW = sCfg.defaultSpriteWidth  || TILE_SIZE;
      const spriteH = sCfg.defaultSpriteHeight || TILE_SIZE;
      ctx.drawImage(
        sheetObj.image,
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

  // ANTI-FLICKERING: Periodically clean up cache to prevent memory leaks
  // Clean up less frequently for top-down view
  if (now % 60000 < 16) { // Every minute
    const cacheCleanupDistance = Math.max(tilesInViewX, tilesInViewY) * 2;
    
    for (const [key, _] of topDownTileCache) {
      const [tileX, tileY] = key.split(':').slice(1).map(Number);
      const dx = Math.abs(tileX - startX - tilesInViewX/2);
      const dy = Math.abs(tileY - startY - tilesInViewY/2);
      
      if (dx > cacheCleanupDistance || dy > cacheCleanupDistance) {
        topDownTileCache.delete(key);
      }
    }
  }

  /* ---------------------------------------------------------
   * TEMPORARY PORTAL RENDERING
   * ---------------------------------------------------------
   * Draw a placeholder sprite for the portal at world tile (5,5)
   * until we have a full object-rendering pipeline.
   */
  try {
    const portalWorldX = 5;
    const portalWorldY = 5;

    const portalScreen = camera.worldToScreen(
      portalWorldX + 0.5,
      portalWorldY + 0.5,
      canvas2D.width,
      canvas2D.height,
      mapManager.tileSize || TILE_SIZE
    );

    // Pick a sprite from the spriteManager if available; otherwise fallback to magenta circle
    let drewSprite = false;
    const portalSheetObj = spriteManager.getSpriteSheet('enemy_sprites');
    if (portalSheetObj) {
      const img = portalSheetObj.image;
      // Use sprite at (0,0) for now
      const sCfg = portalSheetObj.config;
      const sw = sCfg.defaultSpriteWidth || TILE_SIZE;
      const sh = sCfg.defaultSpriteHeight || TILE_SIZE;
      const scale = scaleFactor * 1.5; // slightly larger so it pops
      ctx.drawImage(
        img,
        0,
        0,
        sw,
        sh,
        portalScreen.x - (sw * scale / 2),
        portalScreen.y - (sh * scale / 2),
        sw * scale,
        sh * scale
      );
      drewSprite = true;
    }
    if (!drewSprite) {
      ctx.fillStyle = 'magenta';
      const size = TILE_SIZE * scaleFactor;
      ctx.beginPath();
      ctx.arc(portalScreen.x, portalScreen.y, size / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  } catch (err) {
    console.error('[PortalRender] Failed:', err);
  }
  // ---------------------------------------------------------
}

// Export to window object to avoid circular references
window.renderTopDownView = renderTopDownView;

// Log the export to ensure it's registered globally
console.log("TopDown view render function registered:", window.renderTopDownView ? "Success" : "Failed");
