// src/render/renderTopDown.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES, TILE_IDS } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';
import { getUIManager } from '../ui/UIManager.js';
import { getCurrentTileType } from '../utils/tileEffects.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');
ctx.imageSmoothingEnabled = false; // keep retro pixel art crisp

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

const camera = gameState.camera;
let scaleFactor = 1; // will be updated every frame in renderTopDownView

// ============================================================================
// PERFORMANCE: Debug flags (set to false for production)
// ============================================================================
const DEBUG_RENDERING = false;
const DISABLE_ALL_RENDER_LOGS = true; // Set to true to disable all console.log in render loop
const ENABLE_PERFORMANCE_STATS = true; // Set to true to show FPS and draw call stats - RE-ENABLED to monitor cache sizes

// Performance tracking
let frameCount = 0;
let lastFPSUpdate = performance.now();
let currentFPS = 0;
let drawCallCount = 0;
let tilesRendered = 0;

// MEMORY PROFILING: Track transparency cache usage
let transparencyCalls = 0;
let transparencyCacheHits = 0;
let transparencyCacheMisses = 0;
let lastMemoryLog = performance.now();

// ANTI-FLICKERING: Add a tile cache to prevent constantly requesting the same chunks
// WITH LRU CACHE LIMIT TO PREVENT MEMORY LEAK
const MAX_TILE_CACHE_SIZE = 300; // Limit to prevent unbounded growth
const topDownTileCache = new Map();
const _tileCacheAccessOrder = []; // Track insertion order for LRU eviction

// Track when we last updated chunks to limit request frequency
let lastChunkUpdateTime = 0;
const CHUNK_UPDATE_INTERVAL = 2000; // Only update chunks every 2 seconds for top-down (less frequently needed)

// Provide a helper to wipe this cache when the client switches maps to avoid
// showing stale tiles from the previous world.
export function clearTopDownCache() {
  topDownTileCache.clear();
  _tileCacheAccessOrder.length = 0; // Clear access order tracking
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

  // DEBUG: Log sheet loading attempt
  if (!DISABLE_ALL_RENDER_LOGS) console.log(`[TopDown] Loading sprite sheet: ${sheetName}`);

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
      if (!DISABLE_ALL_RENDER_LOGS) console.log(`[TopDown] Loaded sprite sheet config for ${sheetName}:`, {
        sprites: cfg.sprites?.length || 0,
        defaultWidth: cfg.defaultSpriteWidth,
        defaultHeight: cfg.defaultSpriteHeight
      });
      return spriteManager.loadSpriteSheet(cfg);
    })
    .then(result => {
      if (!DISABLE_ALL_RENDER_LOGS) {
        console.log(`[TopDown] Successfully loaded sprite sheet: ${sheetName}`);
        // DEBUG: List a few sprites from this sheet
        const allSprites = spriteManager.getAllSprites();
        const sheetSprites = Object.keys(allSprites).filter(k => k.startsWith(sheetName));
        console.log(`[TopDown] ${sheetName} has ${sheetSprites.length} registered sprites. Examples:`, sheetSprites.slice(0, 10));
      }
      return result;
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
// Simple memo for getSpriteFlexible results across frames
const _spriteFlexCache = new Map();

function getSpriteFlexible(name, tryLoadSheet=true){
  if(_spriteFlexCache.has(name)) return _spriteFlexCache.get(name);
  if (!name) return null;
  const parts = name.split('_sprite_');
  if (parts.length > 1 && tryLoadSheet) ensureSheetLoaded(parts[0]);
  const sprite = spriteManager.getSprite(name);
  if(sprite) _spriteFlexCache.set(name, sprite);
  return sprite;
}

let _renderDebugOnce = false;

// ----------------------------------------------------------------------------
// Color-Key Transparency Processing
// Converts black pixels (RGB 0,0,0) to transparent for sprites with black backgrounds
// WITH LRU CACHE LIMIT TO PREVENT MEMORY LEAK
// ----------------------------------------------------------------------------
const MAX_TRANSPARENT_CACHE_SIZE = 100; // Limit cache to prevent memory leak
const _transparentSpriteCache = new Map();
const _cacheAccessOrder = []; // Track insertion order for LRU eviction

function drawSpriteWithBlackTransparency(ctx, image, sx, sy, sw, sh, dx, dy, dw, dh) {
  // MEMORY PROFILING: Count every call
  transparencyCalls++;

  // Create cache key based on sprite coordinates
  const cacheKey = `${sx}_${sy}_${sw}_${sh}`;

  let processedCanvas = _transparentSpriteCache.get(cacheKey);

  // MEMORY PROFILING: Track cache hit/miss
  if (processedCanvas) {
    transparencyCacheHits++;
  } else {
    transparencyCacheMisses++;
  }

  if (!processedCanvas) {
    try {
      // Create temporary canvas for processing
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = sw;
      tempCanvas.height = sh;
      const tempCtx = tempCanvas.getContext('2d');

      // Draw sprite to temp canvas
      tempCtx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);

      // Get pixel data
      const imageData = tempCtx.getImageData(0, 0, sw, sh);
      const pixels = imageData.data;

      // Process pixels: make pure black (0,0,0) transparent
      // Allow slight tolerance for near-black pixels from compression
      for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];

        // If pixel is black or very dark, make it transparent
        if (r < 5 && g < 5 && b < 5) {
          pixels[i + 3] = 0; // Set alpha to 0 (transparent)
        }
      }

      // Put processed pixels back
      tempCtx.putImageData(imageData, 0, 0);

      // MEMORY LEAK FIX: Implement LRU eviction
      // Cache the processed canvas
      _transparentSpriteCache.set(cacheKey, tempCanvas);
      _cacheAccessOrder.push(cacheKey);
      processedCanvas = tempCanvas;

      // Evict oldest entry if cache is too large
      if (_transparentSpriteCache.size > MAX_TRANSPARENT_CACHE_SIZE) {
        const oldestKey = _cacheAccessOrder.shift();
        _transparentSpriteCache.delete(oldestKey);
        if (!DISABLE_ALL_RENDER_LOGS) {
          console.log('[PERF] Evicted sprite from transparency cache, size:', _transparentSpriteCache.size);
        }
      }

      if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.01) {
        console.log('[RENDER] Cached new transparent sprite:', cacheKey, 'Cache size:', _transparentSpriteCache.size);
      }
    } catch (error) {
      console.error('[RENDER] Error processing sprite transparency:', error);
      // Fallback: just draw the sprite directly without processing
      ctx.drawImage(image, sx, sy, sw, sh, dx, dy, dw, dh);
      return;
    }
  }

  // Draw the processed sprite to the main canvas
  ctx.drawImage(processedCanvas, 0, 0, sw, sh, dx, dy, dw, dh);
}

// ----------------------------------------------------------------------------
// Object Position Indexing for O(1) lookups during tile rendering
// ----------------------------------------------------------------------------
function buildObjectPositionMap(objects) {
  const map = new Map();
  if (!Array.isArray(objects)) return map;

  objects.forEach(obj => {
    // Only index decorative objects and billboards that should be layered with tiles
    if (obj.type === 'decor' || obj.type === 'billboard') {
      const key = `${Math.floor(obj.x)},${Math.floor(obj.y)}`;
      map.set(key, obj);
    }
  });

  return map;
}

export function renderTopDownView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  if (!mapManager) {
    console.warn("Cannot render map: map manager not available");
    return;
  }

  // ============================================================================
  // PERFORMANCE: Reset counters for this frame
  // ============================================================================
  drawCallCount = 0;
  tilesRendered = 0;
  const frameStartTime = performance.now();

  // Recompute scale factor each frame to reflect zoom changes & ensure
  // we have a valid camera before using it.
  if (camera && typeof camera.getViewScaleFactor === 'function') {
    scaleFactor = camera.getViewScaleFactor();
  }

  // Calculate current time for throttling
  const now = performance.now();

  // ============================================================================
  // PERFORMANCE: Calculate visible tiles based on viewport size
  // ============================================================================
  // PERFORMANCE FIX: Limit tiles to reduce draw calls from 400-500 to ~100-150
  // With 512x512 world, we need render limits to maintain 60fps
  const MAX_TILES_X = 50; // Reasonable viewport for top-down
  const MAX_TILES_Y = 40;

  const tilesInViewX = Math.min(Math.ceil(canvas2D.width / (TILE_SIZE * scaleFactor)), MAX_TILES_X);
  const tilesInViewY = Math.min(Math.ceil(canvas2D.height / (TILE_SIZE * scaleFactor)), MAX_TILES_Y);

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

  // Build object position map for O(1) lookups during tile rendering
  const objects = window.currentObjects || [];
  const objectMap = buildObjectPositionMap(objects);

  // DEBUG: Log object status once per 60 frames (~1 second)
  if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.016) {
    console.log(`[RENDER] Objects: ${objects.length} total, ${objectMap.size} in map for rendering`);
    if (objects.length > 0 && objectMap.size === 0) {
      console.log('[RENDER] WARNING: Objects exist but map is empty!', objects.slice(0, 3));
    }
  }

  // TILE-BY-TILE LAYERED RENDERING
  // For each tile position, we render:
  //   Layer 0: Ground tile (always visible)
  //   Layer 1: Environmental object (tree, rock, billboard) if present
  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      // ANTI-FLICKERING: Check cache first before requesting tile
      const tileKey = `${mapManager.activeMapId || 'map'}:${x},${y}`;
      let tile = topDownTileCache.get(tileKey);
      
      if (!tile) {
        // Get the tile from map manager if not in cache
        tile = mapManager.getTile ? mapManager.getTile(x, y) : null;

        // Store in cache if valid WITH LRU EVICTION
        if (tile) {
          topDownTileCache.set(tileKey, tile);
          _tileCacheAccessOrder.push(tileKey);

          // LRU EVICTION: Remove oldest entries if cache exceeds limit
          if (topDownTileCache.size > MAX_TILE_CACHE_SIZE) {
            const oldestKey = _tileCacheAccessOrder.shift();
            topDownTileCache.delete(oldestKey);

            if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.01) {
              console.log('[PERF] Evicted tile from cache, size:', topDownTileCache.size);
            }
          }
        }
      }
      
      if (!tile) {
        // If tile isn't loaded yet, skip drawing to avoid defaulting to base floor
        // and poll for more chunks next interval.
        continue;
      }
      
      // Determine sprite – NEW: use biome system sprite coordinates if available
      let spritePos;
      let spriteSheetName = 'lofi_environment'; // DEFAULT to lofi_environment for biome tiles
      let resolvedSprite = null;

      // Priority 1: Use sprite coordinates from TileRegistry (biome system)
      // These are PIXEL coordinates, not row/col
      if (tile.spriteX !== null && tile.spriteY !== null && tile.spriteX !== undefined && tile.spriteY !== undefined) {
        spritePos = { x: tile.spriteX, y: tile.spriteY };
        ensureSheetLoaded('lofi_environment');
        // Create a fake resolved sprite for size info (8x8 tiles)
        resolvedSprite = { width: 8, height: 8, sheetName: 'lofi_environment' };

        // DEBUG: Log first few tiles to verify sprite coordinates
        if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.001) {
          console.log('[TILE] Using biome sprite:', {
            x, y,
            spriteX: tile.spriteX,
            spriteY: tile.spriteY,
            spriteName: tile.spriteName,
            biome: tile.biome,
            type: tile.type
          });
        }
      }
      // Priority 2: per-tile sprite override (for custom/editor tiles)
      else if (tile.properties && tile.properties.sprite) {
        const rawName = tile.properties.sprite;
        const parts = rawName.split('_sprite_');
        if (parts.length > 1) {
          spriteSheetName = parts[0];
          ensureSheetLoaded(parts[0]);
        }
        resolvedSprite = spriteManager.fetchSprite(rawName);
        if (resolvedSprite) {
          spriteSheetName = resolvedSprite.sheetName;
          spritePos = { x: resolvedSprite.x, y: resolvedSprite.y };
        }
      }
      // Priority 3: spriteName lookup (fallback for old system)
      else if (tile.spriteName) {
        const rawName = tile.spriteName;
        const parts = rawName.split('_sprite_');
        if (parts.length > 1) {
          spriteSheetName = parts[0];
          ensureSheetLoaded(parts[0]);
        }
        resolvedSprite = spriteManager.fetchSprite(rawName);
        if (resolvedSprite) {
          spriteSheetName = resolvedSprite.sheetName;
          spritePos = { x: resolvedSprite.x, y: resolvedSprite.y };
        }
      }
      // Priority 4: fallback to legacy TILE_SPRITES mapping (old tile_sprites sheet)
      if (!spritePos) {
        spriteSheetName = 'tile_sprites';
        spritePos = tile.type !== undefined ? TILE_SPRITES[tile.type] : null;
        if (!spritePos) continue;
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

      // Use resolved sprite dimensions when available; fall back to sheet defaults
      const spriteW = (resolvedSprite && resolvedSprite.width)  || sCfg.defaultSpriteWidth  || TILE_SIZE;
      const spriteH = (resolvedSprite && resolvedSprite.height) || sCfg.defaultSpriteHeight || TILE_SIZE;

      // FINAL NORMALISATION: Always draw at mapManager.tileSize so 8×8 / 10×10
      // frames get up-scaled to the full logical tile and no transparent rim
      // reveals the black canvas.
      const drawW = mapManager.tileSize || TILE_SIZE;
      const drawH = mapManager.tileSize || TILE_SIZE;

      // LAYER 0: Draw ground tile sprite
      ctx.drawImage(
        sheetObj.image,
        spritePos.x, spritePos.y, spriteW, spriteH, // Source rectangle
        screenPos.x - (drawW * scaleFactor / 2),
        screenPos.y - (drawH * scaleFactor / 2),
        drawW * scaleFactor,
        drawH * scaleFactor
      );
      drawCallCount++;
      tilesRendered++;

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

      // ============================================================================
      // LAYER 1: Draw environmental object (tree, rock, billboard) at this position
      // NOTE: Objects are drawn AFTER ground tiles, so if the sprite has transparency
      // or proper alpha channel, the ground will show through. If sprites have black
      // backgrounds in the sheet, we may need to use globalCompositeOperation.
      // ============================================================================
      const objectKey = `${x},${y}`;
      const objectAtTile = objectMap.get(objectKey);

      if (objectAtTile) {
        // DEBUG: Log when we find an object (very rarely to avoid spam)
        if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.001) {
          console.log(`[RENDER] Found object at (${x},${y}):`, objectAtTile);
        }

        let spriteName = objectAtTile.sprite || 'tiles2_sprite_6_3';
        let spriteObj = null;

        // DEBUG: Always log sprite resolution attempts
        const debugSprite = !DISABLE_ALL_RENDER_LOGS && Math.random() < 0.01;
        if (debugSprite) {
          console.log(`[RENDER] Attempting to resolve sprite "${spriteName}" for object at (${x},${y})`);
        }

        if (spriteName) {
          // Ensure lofi_environment sheet is loaded for procedural objects
          ensureSheetLoaded('lofi_environment');

          // ============================================================================
          // CRITICAL BUG FIX: Sprite Loading Race Condition
          // ============================================================================
          // PROBLEM: spriteManager.getSprite() has a fallback that returns the FIRST
          // sprite from the FIRST loaded sheet when a sprite isn't found. This caused
          // objects to render with the wrong sprite (character sprites instead of trees).
          //
          // ROOT CAUSE: ensureSheetLoaded() is async, but rendering is sync. If we try
          // to get sprites before lofi_environment finishes loading, getSprite("tree")
          // returns the fallback (first sprite from character_sprites sheet).
          //
          // SOLUTION: Check if lofi_environment sheet is loaded BEFORE calling getSprite().
          // If not loaded yet, skip the object (it will render on next frame once loaded).
          //
          // HOW TO AVOID THIS ERROR:
          // 1. Always check getSpriteSheet(sheetName) returns non-null before getSprite()
          // 2. Never rely on getSprite() fallback behavior for production rendering
          // 3. Validate returned sprite.sheetName matches expected sheet
          // 4. For async sheet loading, either await the promise OR check if loaded
          // ============================================================================
          const lofiSheet = spriteManager.getSpriteSheet('lofi_environment');

          if (!lofiSheet) {
            // Sheet not loaded yet, skip this object for this frame
            // It will appear on the next render cycle once the sheet finishes loading
            if (debugSprite) console.log(`[RENDER] lofi_environment sheet not loaded yet, skipping object`);
            continue;
          }

          // ============================================================================
          // Sprite Name Resolution
          // ============================================================================
          // Sprites can be stored with different naming conventions:
          // 1. Simple names: "tree", "boulder", "rocks_1" (from lofi_environment.json)
          // 2. Prefixed names: "lofi_environment_tree" (how SpriteManager stores them)
          // 3. Legacy editor format: "lofi_environment_sprite_4_9" (grid coordinates)
          //
          // The SpriteManager stores ALL sprites with the key format: "{sheetName}_{spriteName}"
          // So "tree" from lofi_environment.json is stored as "lofi_environment_tree"
          // ============================================================================
          const parts = spriteName.split('_sprite_');
          if (parts.length > 1) {
            // Already has sheet prefix (e.g., "lofi_environment_sprite_4_9")
            ensureSheetLoaded(parts[0]);
            spriteObj = getSpriteFlexible(spriteName);
            if (debugSprite) console.log(`[RENDER] Tried getSpriteFlexible("${spriteName}"):`, spriteObj ? 'FOUND' : 'NOT FOUND');
          } else {
            // Simple name (e.g., "tree") - need to add sheet prefix
            // Try with lofi_environment prefix FIRST (this is the correct format)
            spriteObj = spriteManager.getSprite(`lofi_environment_${spriteName}`);
            if (debugSprite) {
              console.log(`[RENDER] Tried getSprite("lofi_environment_${spriteName}"):`, spriteObj ? {sheet: spriteObj.sheetName, name: spriteObj.name} : 'NOT FOUND');
            }

            // Validate the sprite is actually from lofi_environment (not fallback)
            if (!spriteObj || spriteObj.sheetName !== 'lofi_environment') {
              // Try without prefix as fallback
              const unprefixedSprite = spriteManager.getSprite(spriteName);
              if (debugSprite) console.log(`[RENDER] Tried getSprite("${spriteName}"):`, unprefixedSprite ? {sheet: unprefixedSprite.sheetName, name: unprefixedSprite.name} : 'NOT FOUND');
              if (unprefixedSprite && unprefixedSprite.sheetName === 'lofi_environment') {
                spriteObj = unprefixedSprite;
              }
            }

            // Last resort: flexible lookup (handles various naming patterns)
            if (!spriteObj || spriteObj.sheetName !== 'lofi_environment') {
              const flexSprite = getSpriteFlexible(spriteName);
              if (debugSprite) console.log(`[RENDER] Tried getSpriteFlexible("${spriteName}"):`, flexSprite ? {sheet: flexSprite.sheetName, name: flexSprite.name} : 'NOT FOUND');
              if (flexSprite && flexSprite.sheetName === 'lofi_environment') {
                spriteObj = flexSprite;
              }
            }
          }
        }

        if (debugSprite) {
          console.log(`[RENDER] Final sprite resolution for "${spriteName}":`, spriteObj ? {name: spriteObj.name, sheet: spriteObj.sheetName} : 'NULL');
        }

        if (spriteObj && spriteManager.getSpriteSheet(spriteObj.sheetName)) {
          const sheet = spriteManager.getSpriteSheet(spriteObj.sheetName);
          const sCfg = sheet.config;
          const sw = spriteObj.width || sCfg.defaultSpriteWidth || TILE_SIZE;
          const sh = spriteObj.height || sCfg.defaultSpriteHeight || TILE_SIZE;
          const objDrawW = mapManager.tileSize || TILE_SIZE;
          const objDrawH = mapManager.tileSize || TILE_SIZE;

          // DEBUG: Log rendering (very rarely)
          if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.0005) {
            console.log(`[RENDER] Drawing object sprite:`, {
              sprite: spriteName,
              sheet: spriteObj.sheetName,
              pos: {x, y},
              screenPos,
              spriteObj: {x: spriteObj.x, y: spriteObj.y, w: sw, h: sh},
              drawDims: {w: objDrawW * scaleFactor, h: objDrawH * scaleFactor},
              hasImage: !!sheet.image
            });
          }

          // IMPORTANT: lofi_environment sprites have BLACK backgrounds, not transparency
          // TEMPORARY: Bypass transparency processing to test if objects render at all
          ctx.drawImage(
            sheet.image,
            spriteObj.x,
            spriteObj.y,
            sw,
            sh,
            screenPos.x - (objDrawW * scaleFactor / 2),
            screenPos.y - (objDrawH * scaleFactor / 2),
            objDrawW * scaleFactor,
            objDrawH * scaleFactor
          );
          drawCallCount++;

          // Use color-key transparency processing to remove black pixels
          /*drawSpriteWithBlackTransparency(
            ctx,
            sheet.image,
            spriteObj.x,
            spriteObj.y,
            sw,
            sh,
            screenPos.x - (objDrawW * scaleFactor / 2),
            screenPos.y - (objDrawH * scaleFactor / 2),
            objDrawW * scaleFactor,
            objDrawH * scaleFactor
          );*/
        } else if (objectAtTile) {
          // DEBUG: Log when sprite resolution fails
          if (!DISABLE_ALL_RENDER_LOGS && Math.random() < 0.001) {
            console.log(`[RENDER] Failed to resolve sprite "${spriteName}" for object at (${x},${y})`);
          }
        }
      }
      // ============================================================================

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
   * OBJECT RENDERING – Portals ONLY
   * NOTE: Decorative objects and billboards are now rendered
   * in the tile loop above to ensure proper layering with ground tiles.
   * Only portals are rendered here since they need special handling.
   * --------------------------------------------------------- */
  try {
    if (Array.isArray(objects)) {
      objects.forEach(obj => {
        // Only render portals here - decor and billboards are handled in tile loop
        if (obj.type !== 'portal') return;

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
          const drawW = mapManager.tileSize || TILE_SIZE;
          const drawH = mapManager.tileSize || TILE_SIZE;
          const scale = scaleFactor * 1.0;

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
          drawCallCount++;
        } else {
          // Fallback: simple cyan circle for portals
          ctx.fillStyle = '#00FFFF';
          const size = TILE_SIZE * scaleFactor;
          ctx.beginPath();
          ctx.arc(objScreen.x, objScreen.y, size / 2, 0, Math.PI * 2);
          ctx.fill();
          drawCallCount++;
        }
      });
    }
  } catch (err) {
    console.error('[PortalRender] Failed:', err);
  }

  /* ---------------------------------------------------------
   * BAG RENDERING – Loot bags dropped by enemies
   * --------------------------------------------------------- */
  try {
    const bags = gameState.bags;
    if(!bags || bags.length===0) { /* nothing to draw */ }
    else {
    bags.forEach(b => {
      const bagScreen = camera.worldToScreen(
        b.x + 0.5,
        b.y + 0.5,
        canvas2D.width,
        canvas2D.height,
        mapManager.tileSize || TILE_SIZE
      );

      // Quick cull – if bag is completely off-canvas, skip
      if(bagScreen.x < -TILE_SIZE || bagScreen.x > canvas2D.width + TILE_SIZE ||
         bagScreen.y < -TILE_SIZE || bagScreen.y > canvas2D.height + TILE_SIZE){
        return;
      }

      // Attempt to load bag sprite once; default sheet "objects" with alias 'lootbag_white'
      let spriteKey = 'items_sprite_lootbag_white';
      switch(b.bagType){
        case 1: spriteKey='items_sprite_lootbag_brown'; break;
        case 2: spriteKey='items_sprite_lootbag_purple'; break;
        case 3: spriteKey='items_sprite_lootbag_orange'; break;
        case 4: spriteKey='items_sprite_lootbag_cyan'; break;
        case 5: spriteKey='items_sprite_lootbag_blue'; break;
        case 6: spriteKey='items_sprite_lootbag_red'; break;
      }
      let spriteObj = getSpriteFlexible(spriteKey);
      if (!spriteObj) {
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(bagScreen.x, bagScreen.y, TILE_SIZE * scaleFactor * 0.4, 0, Math.PI*2);
        ctx.fill();
        drawCallCount++;
      } else {
        const sheet = spriteManager.getSpriteSheet(spriteObj.sheetName);
        if (sheet && sheet.image) {
          const draw = TILE_SIZE * scaleFactor;
          ctx.drawImage(sheet.image,
            spriteObj.x, spriteObj.y,
            spriteObj.width, spriteObj.height,
            bagScreen.x - draw/2,
            bagScreen.y - draw/2,
            draw, draw);
          drawCallCount++;
        }
      }
    });
    }
  } catch(err){
    console.error('[BagRender] Failed:', err);
  }
  // ---------------------------------------------------------

  // ============================================================================
  // PERFORMANCE: Calculate and display performance stats
  // ============================================================================
  if (ENABLE_PERFORMANCE_STATS) {
    const frameTime = performance.now() - frameStartTime;

    // Update FPS counter every second
    frameCount++;
    const timeSinceLastUpdate = performance.now() - lastFPSUpdate;
    if (timeSinceLastUpdate >= 1000) {
      currentFPS = Math.round((frameCount * 1000) / timeSinceLastUpdate);
      frameCount = 0;
      lastFPSUpdate = performance.now();
    }

    // Draw performance overlay in top-left corner
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(10, 10, 450, 250); // Larger for more stats

    // FPS and timing stats (green)
    ctx.fillStyle = '#00FF00';
    ctx.font = '14px monospace';
    ctx.fillText(`FPS: ${currentFPS}`, 20, 30);
    ctx.fillText(`Frame Time: ${frameTime.toFixed(2)}ms`, 20, 50);
    ctx.fillText(`Draw Calls: ${drawCallCount}`, 20, 70);
    ctx.fillText(`Tiles Rendered: ${tilesRendered}`, 20, 90);

    // Player position (white)
    ctx.fillStyle = '#FFFFFF';
    const char = gameState?.character;
    if (char) {
      ctx.fillText(`X: ${char.x?.toFixed(2) || 0}`, 20, 110);
      ctx.fillText(`Y: ${char.y?.toFixed(2) || 0}`, 20, 130);
      ctx.fillText(`Z: ${char.z?.toFixed(2) || 0}`, 20, 150);
    }

    // MEMORY LEAK MONITORING: Display cache sizes (yellow)
    ctx.fillStyle = '#FFFF00';
    ctx.fillText(`Sprite Cache: ${_transparentSpriteCache.size}/${MAX_TRANSPARENT_CACHE_SIZE}`, 20, 170);
    ctx.fillText(`Tile Cache: ${topDownTileCache.size}/${MAX_TILE_CACHE_SIZE}`, 20, 190);

    // Object tracking (cyan)
    ctx.fillStyle = '#00FFFF';
    const objectCount = (window.currentObjects && Array.isArray(window.currentObjects)) ? window.currentObjects.length : 0;
    ctx.fillText(`Objects: ${objectCount}`, 20, 210);

    // Memory usage (orange)
    ctx.fillStyle = '#FFA500';
    if (performance.memory) {
      const memoryMB = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
      const memoryLimitMB = (performance.memory.jsHeapSizeLimit / 1024 / 1024).toFixed(0);
      ctx.fillText(`Memory: ${memoryMB}/${memoryLimitMB} MB`, 20, 230);
    } else {
      ctx.fillText(`Memory: N/A (use Chrome/Edge)`, 20, 230);
    }

    // Transparency cache stats (magenta)
    ctx.fillStyle = '#FF00FF';
    const hitRate = transparencyCalls > 0 ? ((transparencyCacheHits / transparencyCalls) * 100).toFixed(1) : 0;
    ctx.fillText(`Transparency Calls: ${transparencyCalls}`, 240, 30);
    ctx.fillText(`Cache Hits: ${transparencyCacheHits} (${hitRate}%)`, 240, 50);
    ctx.fillText(`Cache Misses: ${transparencyCacheMisses}`, 240, 70);

    ctx.restore();
  }

  // ============================================================================
  // PERIODIC CACHE CLEANUP - Prevent unbounded growth
  // ============================================================================
  // Clear transparency cache every 60 seconds if getting large
  if (now - lastChunkUpdateTime > 60000 && _transparentSpriteCache.size > 50) {
    _transparentSpriteCache.clear();
    _cacheAccessOrder.length = 0; // Clear access order array
    console.log('[PERF] Periodic cleanup: Cleared transparency cache');
  }

  // ============================================================================
  // PERIODIC MEMORY PROFILING - Log every second to track accumulation
  // ============================================================================
  if (now - lastMemoryLog >= 1000) {
    lastMemoryLog = now;

    const objectCount = (window.currentObjects && Array.isArray(window.currentObjects)) ? window.currentObjects.length : 0;
    const memoryMB = performance.memory ? (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2) : 'N/A';
    const hitRate = transparencyCalls > 0 ? ((transparencyCacheHits / transparencyCalls) * 100).toFixed(1) : 0;

    console.log(`[PERF] Frame ${frameCount} | FPS: ${currentFPS} | Memory: ${memoryMB} MB | ` +
                `Objects: ${objectCount} | Sprite Cache: ${_transparentSpriteCache.size}/${MAX_TRANSPARENT_CACHE_SIZE} | ` +
                `Tile Cache: ${topDownTileCache.size} | Transparency Calls: ${transparencyCalls} | Hit Rate: ${hitRate}%`);

    // Check for potential memory leaks
    if (_transparentSpriteCache.size >= MAX_TRANSPARENT_CACHE_SIZE) {
      // console.warn('[PERF] WARNING: Sprite cache at maximum size - LRU eviction active');
    }

    if (topDownTileCache.size >= MAX_TILE_CACHE_SIZE) {
      // console.warn('[PERF] WARNING: Tile cache at maximum size - LRU eviction active');
    }

    if (objectCount > 5000) {
      // console.warn(`[PERF] WARNING: High object count (${objectCount}) - potential accumulation`);
    }

    // ============================================================================
    // SUBMERSION VISUAL EFFECTS
    // ============================================================================
    // SUBMERSION EFFECTS - Removed (will implement sprite-level clipping instead)
    // ============================================================================
    // TODO: Implement sprite clipping when standing in water/lava

    // Check for transparency cache issues
    if (transparencyCacheMisses > 100 && transparencyCalls > 200) {
      const missRate = (transparencyCacheMisses / transparencyCalls * 100).toFixed(1);
      if (missRate > 10) {
        // console.warn(`[PERF] WARNING: High cache miss rate (${missRate}%) - cache may not be effective`);
      }
    }
  }
}

// Export to window object to avoid circular references
window.renderTopDownView = renderTopDownView;

canvas2D.addEventListener('click', (e)=>{
  const ui = getUIManager();
  if(!ui || !gameState.bags) return;
  const rect = canvas2D.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;
  // Find nearest bag within 24px screen distance
  const camera = gameState.camera;
  const tileSize = TILE_SIZE;
  let targetBag=null, distSq=Infinity;
  gameState.bags.forEach(b=>{
    const screen = camera.worldToScreen(b.x+0.5,b.y+0.5,canvas2D.width,canvas2D.height,tileSize);
    const dx = screen.x - clickX; const dy = screen.y - clickY;
    const d2 = dx*dx + dy*dy;
    if(d2 < 24*24 && d2 < distSq){ targetBag=b; distSq=d2; }
  });
  if(targetBag){
    const lw = ui.components['lootWindow'];
    if(lw){ lw.openForBag(targetBag, e.clientX, e.clientY); }
  }
});
