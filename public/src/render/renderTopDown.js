// src/render/renderTopDown.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');
ctx.imageSmoothingEnabled = false; // keep retro pixel art crisp

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

const camera = gameState.camera;
let scaleFactor = 1; // will be updated every frame in renderTopDownView

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

// ----------------------------------------------------------------------------
// Safe auto-loader for sprite sheets.
//  • Guarantees we issue at most ONE network request per sheet.
//  • Subsequent callers await the same promise instead of spamming fetch().
//  • Failed sheets are remembered so we don't re-hit the server every frame.
// ----------------------------------------------------------------------------
const _sheetPromises = new Map();

function ensureSheetLoaded(sheetName) {
  if (!sheetName) return Promise.resolve();

  // Already available – done.
  if (spriteManager.getSpriteSheet(sheetName)) return Promise.resolve();

  // Already loading (or failed once) – return stored promise to dedupe.
  if (_sheetPromises.has(sheetName)) return _sheetPromises.get(sheetName);

  // Start the network request.
  const p = fetch(`/assets/atlases/${encodeURIComponent(sheetName)}.json`)
    .then(res => {
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return res.json();
    })
    .then(cfg => {
      cfg.name ||= sheetName;
      if (!cfg.path && cfg.meta && cfg.meta.image) {
        cfg.path = cfg.meta.image.startsWith('/') ? cfg.meta.image : '/' + cfg.meta.image;
      }
      return spriteManager.loadSpriteSheet(cfg);
    })
    .catch(err => {
      console.warn(`[TopDown] Failed to load sheet '${sheetName}':`, err.message);
      // On failure keep the rejected promise stored so later calls don't retry
      throw err;
    });

  _sheetPromises.set(sheetName, p);
  return p;
}

// ---------------------------------------------------------------------------
// Helper to fetch sprite regardless of naming convention.
// Supports both "tiles2_sprite_6_3" (editor) and "tiles2_6_3" (atlas) names.
// ---------------------------------------------------------------------------
function getSpriteFlexible(name, tryLoadSheet=true){
  if(!name) return null;
  // First try the exact name
  let obj = spriteManager.fetchSprite(name);
  if(obj) return obj;

  // Attempt to normalise: replace "_sprite_" with "_"
  if(name.includes('_sprite_')){
    const alt = name.replace('_sprite_','_');
    obj = spriteManager.fetchSprite(alt);
    if(obj) return obj;
  }

  // Attempt to strip trailing rotation or variant (rare)
  const stripped = name.replace(/_sprite_/,'_').replace(/_rot\d+$/,'');
  obj = spriteManager.fetchSprite(stripped);
  if(obj) return obj;

  // As last resort attempt to load the sheet and retry once (for first-frame)
  if(tryLoadSheet){
    const parts = name.split('_sprite_')[0].split('_');
    const sheet = parts[0];
    ensureSheetLoaded(sheet);
    return getSpriteFlexible(name,false);
  }

  return null;
}

export function renderTopDownView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  if (!mapManager) {
    console.warn("Cannot render map: map manager not available");
    return;
  }

  // Recompute scale factor each frame to reflect zoom changes & ensure
  // we have a valid camera before using it.
  if (camera && typeof camera.getViewScaleFactor === 'function') {
    scaleFactor = camera.getViewScaleFactor();
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
        const parts = rawName.split('_sprite_');
        if(parts.length>1){ ensureSheetLoaded(parts[0]); }
        const spriteObj = getSpriteFlexible(rawName);
        if (spriteObj) {
          spriteSheetName = spriteObj.sheetName;
          spritePos = { x: spriteObj.x, y: spriteObj.y };
        }
      }
      if (!spritePos && tile.spriteName) {
        const rawName=tile.spriteName;
        const parts=rawName.split('_sprite_');
        if(parts.length>1){ ensureSheetLoaded(parts[0]); }
        const spriteObj = getSpriteFlexible(rawName);
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

      // spriteMeta may be defined when we used fetchSprite; otherwise null.
      let spriteW, spriteH;
      if (typeof spriteObj !== 'undefined' && spriteObj) {
        spriteW = spriteObj.width || sCfg.defaultSpriteWidth  || TILE_SIZE;
        spriteH = spriteObj.height|| sCfg.defaultSpriteHeight || TILE_SIZE;
      } else {
        spriteW = sCfg.defaultSpriteWidth  || TILE_SIZE;
        spriteH = sCfg.defaultSpriteHeight || TILE_SIZE;
      }

      // FINAL NORMALISATION: Always draw at mapManager.tileSize so 8×8 / 10×10
      // frames get up-scaled to the full logical tile and no transparent rim
      // reveals the black canvas.
      const drawW = mapManager.tileSize || TILE_SIZE;
      const drawH = mapManager.tileSize || TILE_SIZE;

      ctx.drawImage(
        sheetObj.image,
        spritePos.x, spritePos.y, spriteW, spriteH, // Source rectangle
        screenPos.x - (drawW * scaleFactor / 2),
        screenPos.y - (drawH * scaleFactor / 2),
        drawW * scaleFactor,
        drawH * scaleFactor
      );

      /*
      // ---- Height shading -------------------------------------------------
      // Disabled for now – keeps the tiles crisp.  Re-enable when we add
      // proper sprite silhouettes or volumetric shadows.
      if (tile.height && tile.height > 0) {
        const alpha = Math.min(tile.height / 15, 1) * 0.35; // up to 35% darken
        ctx.fillStyle = `rgba(0,0,0,${alpha.toFixed(3)})`;
        ctx.fillRect(
          screenPos.x - (drawW * scaleFactor / 2),
          screenPos.y - (drawH * scaleFactor / 2),
          drawW * scaleFactor,
          drawH * scaleFactor
        );
      }
      */
      
      // Add debug visualization to help with alignment
      if (DEBUG_RENDERING) {
        // Draw a grid outline in red
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(
          screenPos.x - (drawW * scaleFactor / 2),
          screenPos.y - (drawH * scaleFactor / 2),
          drawW * scaleFactor,
          drawH * scaleFactor
        );
        
        // Draw tile coordinates for reference (only every few tiles to avoid clutter)
        if ((x % 5 === 0 && y % 5 === 0) || (x === 0 && y === 0)) {
          ctx.fillStyle = 'white';
          ctx.font = '8px Arial';
          ctx.fillText(
            `(${x},${y})`, 
            screenPos.x - (drawW * scaleFactor / 2) + 2,
            screenPos.y - (drawH * scaleFactor / 2) + 8
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
   * OBJECT RENDERING – Portals
   * --------------------------------------------------------- */
  try {
    const objects = window.currentObjects || [];
    if (Array.isArray(objects)) {
      objects.forEach(obj => {
        // Render portals, decorative objects *and* billboards so every
        // non-wall layer appears in the mini-map / top-down view.
        if (obj.type !== 'portal' && obj.type !== 'decor' && obj.type !== 'billboard') return;

        const objScreen = camera.worldToScreen(
          obj.x + 0.5,
          obj.y + 0.5,
          canvas2D.width,
          canvas2D.height,
          mapManager.tileSize || TILE_SIZE
        );

        const spriteName = obj.sprite || 'tiles2_sprite_6_3';
        let spriteObj = null;
        if (spriteName) {
          const parts = spriteName.split('_sprite_');
          if (parts.length > 1) ensureSheetLoaded(parts[0]);
          spriteObj = getSpriteFlexible(spriteName);
        }

        if (spriteObj && spriteManager.getSpriteSheet(spriteObj.sheetName)) {
          const sheet = spriteManager.getSpriteSheet(spriteObj.sheetName);
          const sCfg = sheet.config;
          const sw = spriteObj.width || sCfg.defaultSpriteWidth || TILE_SIZE;
          const sh = spriteObj.height || sCfg.defaultSpriteHeight || TILE_SIZE;
          // Draw every object at exactly one tile so wide sprites (32×32) don't overflow.
          const drawW = mapManager.tileSize || TILE_SIZE;
          const drawH = mapManager.tileSize || TILE_SIZE;
          const scale = scaleFactor * 1.0; // keep slight breathing space; tweak if needed

          ctx.drawImage(
            sheet.image,
            spriteObj.x,
            spriteObj.y,
            sw,
            sh,
            objScreen.x - (drawW * scale / 2),
            objScreen.y - (drawH * scale / 2),
            drawW * scale,
            drawH * scale
          );
        } else {
          // Fallback: simple cyan circle
          ctx.fillStyle = '#00FFFF';
          const size = TILE_SIZE * scaleFactor;
          ctx.beginPath();
          ctx.arc(objScreen.x, objScreen.y, size / 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });
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
