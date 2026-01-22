import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_IDS, SCALE, TILE_SPRITES } from '../constants/constants.js';
import { map } from '../map/map.js';
import { spriteManager } from '../assets/spriteManager.js';
import * as THREE from 'three';
import { spriteDatabase } from '../assets/SpriteDatabase.js';

// IMMEDIATE LOG: Proves this file version is loaded
console.log('!!!!!!! renderFirstPerson.js LOADED - DEBUG VERSION ' + Date.now() + ' !!!!!!!!');

// Caches for THREE.Texture per spriteName so we don't create multiple copies
const textureCache = new Map();

// Global reference to the main tile-atlas texture so createTileMaterial can
// fall back to it when we call it with `texture === null` (happens after a
// world-switch when per-sprite meshes are created lazily).
let tileSheetTexture = null; // set in addFirstPersonElements()

// ---------------------------------------------------------------------------
// Shared helpers (duplicated from renderTopDown.js) – sheet auto-loader and
// sprite-name normaliser so the first-person renderer understands the
// "tiles2_sprite_6_3" convention produced by the map editor.
// ---------------------------------------------------------------------------

const _sheetPromisesFP = new Map();
function ensureSheetLoadedFP(sheetName){
  if(!sheetName) return Promise.resolve();
  if(spriteManager.getSpriteSheet(sheetName)) return Promise.resolve();
  if(_sheetPromisesFP.has(sheetName)) return _sheetPromisesFP.get(sheetName);
  const p = fetch(`/assets/atlases/${encodeURIComponent(sheetName)}.json`).then(r=>r.json()).then(cfg=>{
    cfg.name ||= sheetName;
    if(!cfg.path && cfg.meta && cfg.meta.image){
      cfg.path = cfg.meta.image.startsWith('/')? cfg.meta.image : ('/' + cfg.meta.image);
    }
    return spriteManager.loadSpriteSheet(cfg);
  }).catch(err=>{console.warn('[FirstPerson] Failed to load sheet', sheetName, err); throw err;});
  _sheetPromisesFP.set(sheetName,p);
  return p;
}

function getSpriteFlexibleFP(name, tryLoadSheet=true){
  if(!name) return null;
  let obj = spriteManager.fetchSprite(name);
  if(obj) return obj;
  if(name.includes('_sprite_')){
    const alt = name.replace('_sprite_','_');
    obj = spriteManager.fetchSprite(alt);
    if(obj) return obj;
  }
  const stripped = name.replace(/_sprite_/,'_').replace(/_rot\d+$/,'');
  obj = spriteManager.fetchSprite(stripped);
  if(obj) return obj;
  if(tryLoadSheet){
    const sheet = name.split('_sprite_')[0].split('_')[0];
    return ensureSheetLoadedFP(sheet).then(()=>getSpriteFlexibleFP(name,false));
  }
  return null;
}

function getSpriteTexture(spriteName) {
  if (!spriteName) return null;
  if (textureCache.has(spriteName)) return textureCache.get(spriteName);

  // Try multiple name variations to find the sprite
  const namesToTry = [
    spriteName,
    `chars_${spriteName}`,       // Enemy sprites often in chars atlas
    `chars2_${spriteName}`,      // Or chars2
    `lofi_obj_${spriteName}`,    // Objects
    spriteName.replace(/_/g, '') // Try without underscores
  ];

  for (const name of namesToTry) {
    // Try via spriteManager (supports flexible names)
    const spr = getSpriteFlexibleFP(name, false);
    if (spr) {
      const sheetObj = spriteManager.getSpriteSheet(spr.sheetName);
      if (sheetObj?.image) {
        const canvas = document.createElement('canvas');
        canvas.width = spr.width;
        canvas.height = spr.height;
        const ctx2d = canvas.getContext('2d');
        ctx2d.drawImage(sheetObj.image, spr.x, spr.y, spr.width, spr.height, 0, 0, spr.width, spr.height);
        const tex = new THREE.CanvasTexture(canvas);
        tex.magFilter = THREE.NearestFilter;
        tex.minFilter = THREE.NearestFilter;
        textureCache.set(spriteName, tex);
        return tex;
      }
    }

    // Fallback – use legacy spriteDatabase (lofi_* etc.)
    if (spriteDatabase?.hasSprite?.(name)) {
      const frame = spriteDatabase.getSprite(name);
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext('2d');
      spriteDatabase.drawSprite(ctx, name, 0, 0, frame.width, frame.height);
      const texture = new THREE.CanvasTexture(canvas);
      texture.magFilter = THREE.NearestFilter;
      texture.minFilter = THREE.NearestFilter;
      textureCache.set(spriteName, texture);
      return texture;
    }
  }

  // Log missing sprites occasionally for debugging
  if (Math.random() < 0.01) {
    console.warn(`[FirstPerson] Could not find texture for sprite: ${spriteName}`);
  }

  return null;
}

// Groups that hold dynamic sprites
let enemySpriteGroup;
let bulletSpriteGroup;
let billboardSpriteGroup; // holds UI billboards (names, hp bars)
const billboardSpriteMap = new Map(); // id -> Sprite mapping for billboards
const enemySpriteMap = new Map(); // enemyId -> sprite
const bulletSpritePool = []; // reusable pool
let bulletPoolIndex = 0;

const VIEW_RADIUS = 64; // How many tiles to render around player in first-person view
const SCALING_3D = 12.8; // Scale factor for 3D objects
const MAX_INSTANCES = 4096; // Maximum number of instances for instanced meshes
const CAMERA_HEIGHT = 1.5; // Default eye height in 3D world
const UPDATE_DISTANCE = 0.5; // Distance player must move before updating visible tiles

// InstancedMeshes for different tile types
let floorInstancedMesh, wallInstancedMesh, obstacleInstancedMesh, waterInstancedMesh, mountainInstancedMesh, rampInstancedMesh;

// Geometry templates accessible to dynamic mesh factory
let floorGeometryTemplate = null;
let wallGeometryTemplate  = null;
let floorGeometryWaterTemplate = null; // same as floor

// Bullet rendering constants and variables
const BULLET_MAX_INSTANCES = 2048;
const BULLET_SCALE_3D = SCALING_3D * 0.25; // larger bullet spheres
let bulletInstancedMesh;

// Fallback colors for different tile types
const FALLBACK_COLORS = {
  [TILE_IDS.FLOOR]: 0x4a7c4e,      // Grass green
  [TILE_IDS.WALL]: 0x505050,       // Dark Gray
  [TILE_IDS.OBSTACLE]: 0x3d6b3d,   // Tree green
  [TILE_IDS.WATER]: 0x3366aa,      // Water blue
  [TILE_IDS.MOUNTAIN]: 0x6b5344,   // Mountain brown
};

// Track if lofi_environment sheet is loaded and ready
let lofiEnvironmentReady = false;
let lofiEnvironmentLoadPromise = null;

// Track tiles that need re-rendering when sheets load
let pendingTileRefresh = false;

// Track last update position
let lastUpdateX = 0;
let lastUpdateY = 0;

// Debug flags
const DEBUG_RENDERING = false;
const DEBUG_FREQUENCY = 0.01; // How often to print debug info

// Tile cache to prevent unnecessary renderings
const fpsTileCache = new Map();

// ---------------------------------------------------------------------------
// Per-sprite material caches – one Map per tile type so we reuse materials and
// avoid leaking dozens of identical THREE.Material instances when switching
// between worlds.
// ---------------------------------------------------------------------------
const floorMaterialCache     = new Map();
const wallMaterialCache      = new Map();
const obstacleMaterialCache  = new Map();
const waterMaterialCache     = new Map();
const mountainMaterialCache  = new Map();

// ---------------------------------------------------------------------------
// InstancedMesh caches
// ---------------------------------------------------------------------------
//   meshesByType.floor ↦ Map(spriteKey → InstancedMesh)
//   same for wall / obstacle / water / mountain
// These are populated lazily by getInstancedMesh().  On a world switch the
// caller should clear all maps and dispose the meshes via
// forceDisposeFirstPersonView().  This infrastructure is added now; the full
// migration of updateVisibleTiles() will follow in the next patch.
const meshesByType = {
  floor:     new Map(),
  wall:      new Map(),
  obstacle:  new Map(),
  water:     new Map(),
  mountain:  new Map()
};

function getInstancedMesh(tileType, spriteKey) {
  let cache;
  let baseGeometry;
  switch (tileType) {
    case TILE_IDS.FLOOR:    cache = meshesByType.floor;    baseGeometry = floorGeometryTemplate; break;
    case TILE_IDS.WALL:     cache = meshesByType.wall;     baseGeometry = wallGeometryTemplate;  break;
    case TILE_IDS.OBSTACLE: cache = meshesByType.obstacle; baseGeometry = wallGeometryTemplate;  break;
    case TILE_IDS.WATER:    cache = meshesByType.water;    baseGeometry = floorGeometryTemplate; break;
    case TILE_IDS.MOUNTAIN: cache = meshesByType.mountain; baseGeometry = wallGeometryTemplate;  break;
    default: return null;
  }
  if (!spriteKey) spriteKey = '__DEFAULT__';
  if (cache.has(spriteKey)) return cache.get(spriteKey);

  // We need the material – reuse (or create) one via createTileMaterial.
  // NOTE: createTileMaterial expects a spriteOverride for per-sprite art.  For
  // the default key we pass null so it uses TILE_SPRITES mapping.
  const mat = createTileMaterial(null, tileType, spriteKey === '__DEFAULT__' ? null : spriteKey);
  const mesh = new THREE.InstancedMesh(baseGeometry.clone(), mat, MAX_INSTANCES);
  mesh.frustumCulled = false;
  mesh.name = `${TILE_IDS[tileType] || tileType}_${spriteKey}`;
  // We'll add the mesh to the scene from addFirstPersonElements() once we have
  // access to the THREE.Scene reference.  For now we just cache it.
  cache.set(spriteKey, mesh);
  if (sceneGlobalRef) sceneGlobalRef.add(mesh); // sceneGlobalRef set in addFirstPersonElements
  return mesh;
}

/**
 * Gets or creates an InstancedMesh for biome tiles (using spriteX/spriteY coordinates).
 * Similar to getInstancedMesh but uses createBiomeTileMaterial instead of createTileMaterial.
 *
 * @param {number} tileType - The tile type ID
 * @param {number} spriteX - Pixel X coordinate in lofi_environment
 * @param {number} spriteY - Pixel Y coordinate in lofi_environment
 * @returns {THREE.InstancedMesh} The instanced mesh for this biome sprite
 */
function getInstancedMeshForBiome(tileType, spriteX, spriteY) {
  let cache;
  let baseGeometry;
  switch (tileType) {
    case TILE_IDS.FLOOR:    cache = meshesByType.floor;    baseGeometry = floorGeometryTemplate; break;
    case TILE_IDS.WALL:     cache = meshesByType.wall;     baseGeometry = wallGeometryTemplate;  break;
    case TILE_IDS.OBSTACLE: cache = meshesByType.obstacle; baseGeometry = wallGeometryTemplate;  break;
    case TILE_IDS.WATER:    cache = meshesByType.water;    baseGeometry = floorGeometryTemplate; break;
    case TILE_IDS.MOUNTAIN: cache = meshesByType.mountain; baseGeometry = wallGeometryTemplate;  break;
    default: return null;
  }

  // Create unique key from biome coordinates
  const spriteKey = `biome_${spriteX}_${spriteY}`;

  if (cache.has(spriteKey)) return cache.get(spriteKey);

  // Create material using biome sprite coordinates
  // Note: createBiomeTileMaterial now always returns a material (colored fallback if sheet not ready)
  const mat = createBiomeTileMaterial(spriteX, spriteY, tileType);

  // Safety check - should never happen now but keep for robustness
  if (!mat) {
    console.error(`[FirstPerson] Unexpected null material for biome (${spriteX},${spriteY})`);
    // Create emergency fallback
    const emergencyMat = new THREE.MeshStandardMaterial({
      color: FALLBACK_COLORS[tileType] || 0x808080,
      side: THREE.DoubleSide
    });
    const mesh = new THREE.InstancedMesh(baseGeometry.clone(), emergencyMat, MAX_INSTANCES);
    mesh.frustumCulled = false;
    mesh.name = `${TILE_IDS[tileType] || tileType}_emergency_${spriteX}_${spriteY}`;
    cache.set(spriteKey, mesh);
    if (sceneGlobalRef) sceneGlobalRef.add(mesh);
    return mesh;
  }

  const mesh = new THREE.InstancedMesh(baseGeometry.clone(), mat, MAX_INSTANCES);
  mesh.frustumCulled = false;
  mesh.name = `${TILE_IDS[tileType] || tileType}_${spriteKey}`;

  cache.set(spriteKey, mesh);
  if (sceneGlobalRef) sceneGlobalRef.add(mesh);

  return mesh;
}

// We capture the scene reference so getInstancedMesh() can insert meshes later
let sceneGlobalRef = null;

/**
 * FIX #4: Clears all cached biome meshes and disposes their resources.
 * This should be called after sprite sheets load to force recreation of materials.
 */
function clearBiomeMeshCache() {
  let disposedCount = 0;

  Object.values(meshesByType).forEach(cache => {
    cache.forEach((mesh, key) => {
      // Only clear biome meshes (those with "biome_" prefix)
      if (key.startsWith('biome_')) {
        // Remove from scene
        if (sceneGlobalRef && mesh.parent) {
          sceneGlobalRef.remove(mesh);
        }

        // Dispose resources
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) {
          if (mesh.material.map) mesh.material.map.dispose();
          mesh.material.dispose();
        }

        // Remove from cache
        cache.delete(key);
        disposedCount++;
      }
    });
  });

  if (disposedCount > 0) {
    console.log(`[FirstPerson] Cleared ${disposedCount} cached biome meshes after sheet load`);
  }
}

/**
 * Initializes and adds first-person elements to the scene.
 * @param {THREE.Scene} scene - The Three.js scene to add elements to.
 * @param {Function} callback - Function to call once elements are added.
 */
export async function addFirstPersonElements(scene, callback) {
  // Keep global reference so dynamically created InstancedMeshes can attach
  sceneGlobalRef = scene;
  console.log('[FirstPerson] Adding first-person elements to the scene.');

  // CRITICAL: Pre-load all required sprite sheets for first-person view
  // Load in parallel for faster startup
  const sheetsToLoad = [
    'lofi_environment',  // Biome tiles
    'chars',             // Enemy sprites
    'chars2',            // More enemy sprites
    'lofi_obj',          // Objects (trees, rocks)
    'lofi_obj_packB'     // More objects
  ];

  console.log('[FirstPerson] Pre-loading sprite sheets:', sheetsToLoad);

  const loadPromises = sheetsToLoad.map(sheet =>
    ensureSheetLoadedFP(sheet).catch(err => {
      console.warn(`[FirstPerson] Failed to load ${sheet}:`, err.message);
      return null;
    })
  );

  await Promise.all(loadPromises);

  // Check which sheets loaded successfully
  sheetsToLoad.forEach(sheet => {
    const loaded = !!spriteManager.getSpriteSheet(sheet);
    console.log(`[FirstPerson] ${loaded ? '✅' : '❌'} ${sheet}: ${loaded ? 'loaded' : 'failed'}`);
  });

  const sheetObj = spriteManager.getSpriteSheet('lofi_environment');
  if (sheetObj?.image) {
    console.log('[FirstPerson] lofi_environment details:', {
      imageWidth: sheetObj.image.width,
      imageHeight: sheetObj.image.height
    });
  }

  // Mark sheet as ready
  lofiEnvironmentReady = true;

  // Clear any biome meshes that were created with fallback materials
  // before the sheet finished loading (handles race condition)
  if (pendingTileRefresh) {
    console.log('[FirstPerson] Clearing cached biome meshes to use real textures');
    clearBiomeMeshCache();
    pendingTileRefresh = false;
    // Force re-render of visible tiles
    lastUpdateX = -999;
    lastUpdateY = -999;
  }

  // Create a THREE.Texture from the loaded Image
  const tileSheetObj = spriteManager.getSpriteSheet('tile_sprites');
  if (!tileSheetObj) {
    console.warn('[FirstPerson] Tile sprites not loaded, using fallback materials');
    useFallbackMaterials(scene);
    if (callback) callback();
    return;
  }
  
  const tileTexture = new THREE.Texture(tileSheetObj.image);
  tileTexture.needsUpdate = true; // Update the texture

  // Store so later calls to createTileMaterial(null, …) can reuse it.
  tileSheetTexture = tileTexture;

  // Create materials for each tile type
  const floorMaterial = createTileMaterial(tileTexture, TILE_IDS.FLOOR);
  const wallMaterial = createTileMaterial(tileTexture, TILE_IDS.WALL);
  const obstacleMaterial = createTileMaterial(tileTexture, TILE_IDS.OBSTACLE);
  const waterMaterial = createTileMaterial(tileTexture, TILE_IDS.WATER);
  const mountainMaterial = createTileMaterial(tileTexture, TILE_IDS.MOUNTAIN);

  // Define geometry for floor and walls
  const floorGeometry = new THREE.PlaneGeometry(SCALING_3D, SCALING_3D);
  floorGeometry.rotateX(-Math.PI / 2); // Rotate the plane to face upwards

  floorGeometryTemplate = floorGeometry;  // make available globally

  const wallGeometry = new THREE.BoxGeometry(SCALING_3D, SCALING_3D * 3, SCALING_3D);

  wallGeometryTemplate = wallGeometry;

  console.log(`[FirstPerson] Creating InstancedMeshes with maxInstances: ${MAX_INSTANCES}`);

  try {
    // Initialize InstancedMeshes for each tile type
    floorInstancedMesh = new THREE.InstancedMesh(floorGeometry, floorMaterial, MAX_INSTANCES);
    floorInstancedMesh.receiveShadow = true;
    floorInstancedMesh.name = 'floorInstancedMesh';
    floorInstancedMesh.frustumCulled = false; // avoid whole‐world pop-out
    scene.add(floorInstancedMesh);
    console.log('[FirstPerson] Added floorInstancedMesh to the scene');

    wallInstancedMesh = new THREE.InstancedMesh(wallGeometry, wallMaterial, MAX_INSTANCES);
    wallInstancedMesh.castShadow = true;
    wallInstancedMesh.receiveShadow = true;
    wallInstancedMesh.name = 'wallInstancedMesh';
    wallInstancedMesh.frustumCulled = false;
    scene.add(wallInstancedMesh);
    console.log('[FirstPerson] Added wallInstancedMesh to the scene');

    obstacleInstancedMesh = new THREE.InstancedMesh(wallGeometry, obstacleMaterial, MAX_INSTANCES);
    obstacleInstancedMesh.castShadow = true;
    obstacleInstancedMesh.receiveShadow = true;
    obstacleInstancedMesh.name = 'obstacleInstancedMesh';
    obstacleInstancedMesh.frustumCulled = false;
    scene.add(obstacleInstancedMesh);
    console.log('[FirstPerson] Added obstacleInstancedMesh to the scene');

    waterInstancedMesh = new THREE.InstancedMesh(floorGeometry, waterMaterial, MAX_INSTANCES);
    waterInstancedMesh.receiveShadow = true;
    waterInstancedMesh.name = 'waterInstancedMesh';
    waterInstancedMesh.frustumCulled = false;
    scene.add(waterInstancedMesh);
    console.log('[FirstPerson] Added waterInstancedMesh to the scene');

    mountainInstancedMesh = new THREE.InstancedMesh(wallGeometry, mountainMaterial, MAX_INSTANCES);
    mountainInstancedMesh.castShadow = true;
    mountainInstancedMesh.receiveShadow = true;
    mountainInstancedMesh.name = 'mountainInstancedMesh';
    mountainInstancedMesh.frustumCulled = false;
    scene.add(mountainInstancedMesh);
    console.log('[FirstPerson] Added mountainInstancedMesh to the scene');

    /* ---------- RAMP INSTANCED MESH ---------- */
    const rampGeometry = new THREE.PlaneGeometry(SCALING_3D, SCALING_3D);
    rampGeometry.rotateX(-Math.PI / 4); // 45° slope
    const rampMaterial = floorMaterial.clone();
    rampInstancedMesh = new THREE.InstancedMesh(rampGeometry, rampMaterial, MAX_INSTANCES);
    rampInstancedMesh.receiveShadow = true;
    rampInstancedMesh.name = 'rampInstancedMesh';
    rampInstancedMesh.frustumCulled = false;
    scene.add(rampInstancedMesh);
    console.log('[FirstPerson] Added rampInstancedMesh to the scene');

    /* ---------- BULLET INSTANCED MESH ---------- */
    const bulletGeometry = new THREE.SphereGeometry(BULLET_SCALE_3D, 8, 8);
    const bulletMaterial = new THREE.MeshBasicMaterial({ color: 0xffe066 });
    bulletInstancedMesh = new THREE.InstancedMesh(bulletGeometry, bulletMaterial, BULLET_MAX_INSTANCES);
    bulletInstancedMesh.name = 'bulletInstancedMesh';
    scene.add(bulletInstancedMesh);
    console.log('[FirstPerson] Added bulletInstancedMesh to the scene');
    // Disable legacy spheres – we'll render sprites instead
    bulletInstancedMesh.visible = false;

    // Sprite groups for enemies and bullets
    enemySpriteGroup = new THREE.Group();
    enemySpriteGroup.name = 'enemySprites';
    scene.add(enemySpriteGroup);

    bulletSpriteGroup = new THREE.Group();
    bulletSpriteGroup.name = 'bulletSprites';
    scene.add(bulletSpriteGroup);

    billboardSpriteGroup = new THREE.Group();
    billboardSpriteGroup.name = 'billboardSprites';
    scene.add(billboardSpriteGroup);
    console.log('[FirstPerson] Added billboardSpriteGroup to the scene');

    // Set initial player position for tracking updates
    if (gameState && gameState.character) {
      lastUpdateX = gameState.character.x;
      lastUpdateY = gameState.character.y;
    }

    // Initial render of tiles around the character
    updateVisibleTiles();
    console.log('[FirstPerson] Initial tiles rendered around the character.');
  } catch (error) {
    console.error('[FirstPerson] Error creating InstancedMeshes:', error);
    useFallbackMaterials(scene);
  }

  // Call the callback function to signal that elements are added
  if (callback) callback();
}

/**
 * Creates a material for a specific tile type from a tile sprite sheet.
 * @param {THREE.Texture} texture - The loaded texture for the sprite sheet.
 * @param {number} tileType - The TILE_IDS value for which to create the material.
 * @param {string} spriteOverride - Optional sprite name to override the default sprite.
 * @returns {THREE.MeshStandardMaterial} - The created material.
 */
function createTileMaterial(texture, tileType, spriteOverride = null) {
  if (!texture) texture = tileSheetTexture; // fallback to global atlas

  if (spriteOverride) {
    // Try via spriteManager first (supports flexible names)
    const spr = getSpriteFlexibleFP(spriteOverride,false);
    if(spr){
      ensureSheetLoadedFP(spr.sheetName);
      const sheetObj = spriteManager.getSpriteSheet(spr.sheetName);
      if(sheetObj){
        const canvas = document.createElement('canvas');
        canvas.width = spr.width; canvas.height = spr.height;
        const ctx2d = canvas.getContext('2d');
        ctx2d.drawImage(sheetObj.image, spr.x, spr.y, spr.width, spr.height, 0,0, spr.width, spr.height);
        const tex = new THREE.CanvasTexture(canvas);
        tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.NearestFilter;
        return new THREE.MeshStandardMaterial({ map: tex, transparent:true, side:THREE.DoubleSide });
      }
    }

    // Legacy fallback via spriteDatabase
    if (spriteDatabase?.hasSprite?.(spriteOverride)) {
      const tex = getSpriteTexture(spriteOverride);
      if (tex) {
        return new THREE.MeshStandardMaterial({ map: tex, transparent:true, side:THREE.DoubleSide });
      }
    }
  }

  const spritePos = TILE_SPRITES[tileType];
  // Determine sprite size dynamically from the loaded sheet (falls back to TILE_SIZE)
  const sheetCfg = spriteManager.getSpriteSheet('tile_sprites')?.config;
  const spriteW = sheetCfg?.defaultSpriteWidth  || TILE_SIZE;
  const spriteH = sheetCfg?.defaultSpriteHeight || TILE_SIZE;

  if (!spritePos) {
    console.warn(`[FirstPerson] No sprite position defined for tile type ${tileType}. Using fallback color.`);
    // Use a fallback color material
    return new THREE.MeshStandardMaterial({ 
      color: FALLBACK_COLORS[tileType] || 0xffffff, 
      side: THREE.DoubleSide 
    });
  }

  try {
    // Draw the sprite onto a canvas to create an isolated texture (simpler than UV fiddling)
    const baseImage = texture.image;
    const canvas = document.createElement('canvas');
    canvas.width = spriteW;
    canvas.height = spriteH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(baseImage, spritePos.x, spritePos.y, spriteW, spriteH, 0, 0, spriteW, spriteH);

    const tileCanvasTex = new THREE.CanvasTexture(canvas);
    tileCanvasTex.magFilter = THREE.NearestFilter;
    tileCanvasTex.minFilter = THREE.NearestFilter;

    const material = new THREE.MeshStandardMaterial({
      map: tileCanvasTex,
      transparent: true,
      side: THREE.DoubleSide,
      // Keep emissive dark so texture colors show correctly
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0
    });

    return material;
  } catch (error) {
    console.error(`[FirstPerson] Error creating material for tileType ${tileType}:`, error);
    // Fallback to solid color
    return new THREE.MeshStandardMaterial({ 
      color: FALLBACK_COLORS[tileType] || 0xffffff, 
      side: THREE.DoubleSide 
    });
  }
}

/**
 * Creates a Three.js material from biome system sprite coordinates (pixel coords in lofi_environment).
 * This is used for tiles that have spriteX/spriteY properties from the biome system.
 *
 * @param {number} spriteX - Pixel X coordinate in lofi_environment.png
 * @param {number} spriteY - Pixel Y coordinate in lofi_environment.png
 * @param {number} tileType - The tile type ID (for fallback color)
 * @returns {THREE.Material} The created material
 */
function createBiomeTileMaterial(spriteX, spriteY, tileType) {
  // Check if sheet loaded SYNCHRONOUSLY (don't await - sheet should be pre-loaded at init)
  const sheetObj = spriteManager.getSpriteSheet('lofi_environment');

  // DEBUG: Log sheet status on first few calls
  if (!createBiomeTileMaterial._debugCount) createBiomeTileMaterial._debugCount = 0;
  if (createBiomeTileMaterial._debugCount < 5) {
    createBiomeTileMaterial._debugCount++;
    console.log(`[FP-DEBUG] createBiomeTileMaterial called:`, {
      spriteX, spriteY, tileType,
      sheetExists: !!sheetObj,
      hasImage: !!sheetObj?.image,
      imageComplete: sheetObj?.image?.complete,
      imageWidth: sheetObj?.image?.width,
      imageHeight: sheetObj?.image?.height,
      lofiEnvironmentReady
    });
  }

  if (!sheetObj || !sheetObj.image) {
    // Sheet not ready - return a COLORED fallback material so tiles are visible
    if (!lofiEnvironmentReady) {
      // Schedule a refresh when sheet loads
      pendingTileRefresh = true;
    }
    console.warn(`[FirstPerson] lofi_environment not loaded! Using colored fallback.`);
    // Return colored fallback using MeshBasicMaterial (doesn't need lighting)
    const fallbackColor = FALLBACK_COLORS[tileType] || 0x808080;
    return new THREE.MeshBasicMaterial({
      color: fallbackColor,
      side: THREE.DoubleSide
    });
  }

  try {
    // lofi_environment uses 8x8 sprites
    const spriteW = 8;
    const spriteH = 8;

    const canvas = document.createElement('canvas');
    canvas.width = spriteW;
    canvas.height = spriteH;
    const ctx = canvas.getContext('2d');

    // Draw the sprite from the lofi_environment sheet
    ctx.drawImage(sheetObj.image, spriteX, spriteY, spriteW, spriteH, 0, 0, spriteW, spriteH);

    // Check if the canvas is completely transparent/empty (invalid sprite coords)
    const imageData = ctx.getImageData(0, 0, spriteW, spriteH);
    let hasContent = false;
    for (let i = 3; i < imageData.data.length; i += 4) {
      if (imageData.data[i] > 0) {
        hasContent = true;
        break;
      }
    }

    if (!hasContent) {
      // Sprite coords are invalid/empty - use colored fallback
      console.warn(`[FirstPerson] Empty sprite at (${spriteX}, ${spriteY}), using colored fallback`);
      const fallbackColor = FALLBACK_COLORS[tileType] || 0x808080;
      return new THREE.MeshBasicMaterial({
        color: fallbackColor,
        side: THREE.DoubleSide
      });
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;

    // DEBUG: Log successful texture creation
    if (!createBiomeTileMaterial._texDebugCount) createBiomeTileMaterial._texDebugCount = 0;
    if (createBiomeTileMaterial._texDebugCount < 3) {
      createBiomeTileMaterial._texDebugCount++;
      console.log(`[FP-DEBUG] Created texture for biome (${spriteX},${spriteY}):`, {
        textureId: texture.id,
        hasImage: !!texture.image,
        imageWidth: texture.image?.width,
        imageHeight: texture.image?.height
      });
    }

    // Use MeshBasicMaterial for reliability - doesn't depend on lighting
    return new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      side: THREE.DoubleSide
    });
  } catch (error) {
    console.error(`[FirstPerson] Error creating biome material for coords (${spriteX}, ${spriteY}):`, error);
    // Return colored fallback on error
    const fallbackColor = FALLBACK_COLORS[tileType] || 0x808080;
    return new THREE.MeshBasicMaterial({
      color: fallbackColor,
      side: THREE.DoubleSide
    });
  }
}

/**
 * Fallback function to create and add basic colored materials if texture loading fails.
 * @param {THREE.Scene} scene - The Three.js scene to add fallback meshes to.
 */
function useFallbackMaterials(scene) {
  console.log('[FirstPerson] Using fallback materials for first-person view.');

  try {
    // Define fallback materials
    const floorMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.FLOOR] || 0x808080 });
    const wallMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.WALL] || 0x303030 });
    const obstacleMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.OBSTACLE] || 0xFF0000 });
    const waterMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.WATER] || 0x0000FF });
    const mountainMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.MOUNTAIN] || 0x00FF00 });

    // Define geometry for floor and walls
    const floorGeometry = new THREE.PlaneGeometry(SCALING_3D, SCALING_3D);
    floorGeometry.rotateX(-Math.PI / 2); // Rotate the plane to face upwards

    const wallGeometry = new THREE.BoxGeometry(SCALING_3D, SCALING_3D* 2, SCALING_3D);

    // Initialize InstancedMeshes for each tile type with fallback materials
    floorInstancedMesh = new THREE.InstancedMesh(floorGeometry, floorMaterial, MAX_INSTANCES);
    floorInstancedMesh.receiveShadow = true;
    floorInstancedMesh.name = 'fallbackFloorInstancedMesh';
    scene.add(floorInstancedMesh);
    console.log('[FirstPerson] Added fallbackFloorInstancedMesh to the scene');

    wallInstancedMesh = new THREE.InstancedMesh(wallGeometry, wallMaterial, MAX_INSTANCES);
    wallInstancedMesh.castShadow = true;
    wallInstancedMesh.receiveShadow = true;
    wallInstancedMesh.name = 'fallbackWallInstancedMesh';
    scene.add(wallInstancedMesh);
    console.log('[FirstPerson] Added fallbackWallInstancedMesh to the scene');

    obstacleInstancedMesh = new THREE.InstancedMesh(wallGeometry, obstacleMaterial, MAX_INSTANCES);
    obstacleInstancedMesh.castShadow = true;
    obstacleInstancedMesh.receiveShadow = true;
    obstacleInstancedMesh.name = 'fallbackObstacleInstancedMesh';
    scene.add(obstacleInstancedMesh);
    console.log('[FirstPerson] Added fallbackObstacleInstancedMesh to the scene');

    waterInstancedMesh = new THREE.InstancedMesh(floorGeometry, waterMaterial, MAX_INSTANCES);
    waterInstancedMesh.receiveShadow = true;
    waterInstancedMesh.name = 'fallbackWaterInstancedMesh';
    scene.add(waterInstancedMesh);
    console.log('[FirstPerson] Added fallbackWaterInstancedMesh to the scene');

    mountainInstancedMesh = new THREE.InstancedMesh(wallGeometry, mountainMaterial, MAX_INSTANCES);
    mountainInstancedMesh.castShadow = true;
    mountainInstancedMesh.receiveShadow = true;
    mountainInstancedMesh.name = 'fallbackMountainInstancedMesh';
    scene.add(mountainInstancedMesh);
    console.log('[FirstPerson] Added fallbackMountainInstancedMesh to the scene');

    if (!billboardSpriteGroup) {
      billboardSpriteGroup = new THREE.Group();
      scene.add(billboardSpriteGroup);
    }

    // Initial render of tiles around the character
    updateVisibleTiles();
    console.log('[FirstPerson] Initial fallback tiles rendered around the character.');
  } catch (error) {
    console.error('[FirstPerson] Error creating fallback InstancedMeshes:', error);
  }
}

/**
 * Updates the visible tiles around the character's position.
 * Only renders tiles within the VIEW_RADIUS of the character.
 */
function updateVisibleTiles() {
  // CRITICAL: Check if this function is being called at all
  if (!updateVisibleTiles._callCount) updateVisibleTiles._callCount = 0;
  updateVisibleTiles._callCount++;
  if (updateVisibleTiles._callCount <= 3) {
    console.log(`[FP] updateVisibleTiles called (call #${updateVisibleTiles._callCount})`);
    console.log('[FP] Prerequisites check:', {
      floorInstancedMesh: !!floorInstancedMesh,
      sceneGlobalRef: !!sceneGlobalRef,
      bulletInstancedMesh: !!bulletInstancedMesh,
      enemySpriteGroup: !!enemySpriteGroup,
      billboardSpriteGroup: !!billboardSpriteGroup
    });
  }

  if (!floorInstancedMesh || !sceneGlobalRef || !bulletInstancedMesh || !enemySpriteGroup || !billboardSpriteGroup) {
    console.error('[FP] updateVisibleTiles early return - missing prerequisites!');
    return;
  }

  const character = gameState.character;
  if (!character) {
    console.warn('[FirstPerson] Cannot update visible tiles: character not found');
    return;
  }

  const mapManager = gameState.map || map; // Use gameState.map if available, otherwise fall back to map
  if (!mapManager) {
    console.warn('[FirstPerson] Cannot update visible tiles: map manager not available');
    return;
  }

  // CRITICAL DEBUG: Log first tile to verify data flow
  if (!updateVisibleTiles._dataFlowChecked) {
    updateVisibleTiles._dataFlowChecked = true;
    const testX = Math.floor(character.x);
    const testY = Math.floor(character.y);
    const testTile = mapManager.getTile ? mapManager.getTile(testX, testY) : null;
    console.log('=== CRITICAL FIRST-PERSON DATA FLOW CHECK ===');
    console.log('Character position:', testX, testY);
    console.log('Test tile exists:', !!testTile);
    if (testTile) {
      console.log('Test tile full dump:', JSON.stringify(testTile, null, 2));
      console.log('spriteX:', testTile.spriteX, '(type:', typeof testTile.spriteX, ')');
      console.log('spriteY:', testTile.spriteY, '(type:', typeof testTile.spriteY, ')');
      console.log('spriteName:', testTile.spriteName);
      console.log('type:', testTile.type);
      console.log('biome:', testTile.biome);
    }
    console.log('lofi_environment sheet:', !!spriteManager.getSpriteSheet('lofi_environment'));
    console.log('=== END DATA FLOW CHECK ===');
  }

  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
  const meta = (window.gameState?.map?.mapMetadata) || { width: 0, height: 0, chunkSize: 16 };
  const cameraTileX = clamp(Math.floor(character.x), 0, Math.max(0, (meta.width  || 0) - 1));
  const cameraTileY = clamp(Math.floor(character.y), 0, Math.max(0, (meta.height || 0) - 1));

  // Update map chunks if ClientMapManager is used
  if (mapManager.updateVisibleChunksLocally) {
    mapManager.updateVisibleChunksLocally(cameraTileX, cameraTileY);
  }
  // Also ensure we request new chunks from the server so geometry never "goes black"
  if (mapManager.updateVisibleChunks) {
    mapManager.updateVisibleChunks(cameraTileX, cameraTileY);
  }

  // Store matrices per InstancedMesh so we can support many meshes (one per sprite)
  const matricesByMesh = new Map();
  let pushCount = 0;
  const pushMatrix = (mesh, mat) => {
    if (!mesh) {
      if (pushCount < 5) console.error('[FP] pushMatrix called with NULL mesh!');
      return;
    }
    if (!matricesByMesh.has(mesh)) matricesByMesh.set(mesh, []);
    matricesByMesh.get(mesh).push(mat);
    pushCount++;
  };
  const rampMatrices = []; // ramps unchanged for now

  if (DEBUG_RENDERING && Math.random() < DEBUG_FREQUENCY) {
    console.log(`[FirstPerson] Updating visible tiles around position (${cameraTileX}, ${cameraTileY}).`);
  }

  // Clear the tile cache periodically
  if (Math.random() < 0.01) { // Clear cache every ~100 frames
    fpsTileCache.clear();
  }

  for (let dy = -VIEW_RADIUS; dy <= VIEW_RADIUS; dy++) {
    for (let dx = -VIEW_RADIUS; dx <= VIEW_RADIUS; dx++) {
      const tileX = cameraTileX + dx;
      const tileY = cameraTileY + dy;
      
      // Use cache to avoid repeated getTile calls
      const tileKey = `${tileX},${tileY}`;
      let tile = fpsTileCache.get(tileKey);
      
      if (!tile) {
        // Try to get tile from map manager (may return null while chunk pending)
        tile = mapManager.getTile ? mapManager.getTile(tileX, tileY) : null;
        if (!tile) {
          // Placeholder floor tile when chunk not loaded
          tile = { type: TILE_IDS.FLOOR, height: 0 };
        } else {
          fpsTileCache.set(tileKey, tile);
        }
      }

      if (tile) {
        // Calculate height - use tile height if available, otherwise default to 0
        const heightOffset = tile.height || 0;
        
        // Create position vector for 3D world coordinates
        const position = new THREE.Vector3(
          tileX * SCALING_3D,
          heightOffset * SCALING_3D * 0.5, // Scale height for visual effect
          tileY * SCALING_3D
        );
        
        let matrix;
        const quatIdentity = new THREE.Quaternion();

        // Priority 1: Check for biome system sprite coordinates ---------------------------
        let instMesh  = null;

        // DEBUG: Track tile priority selection (first 10 tiles only)
        if (!updateVisibleTiles._tileDebugCount) updateVisibleTiles._tileDebugCount = 0;
        const shouldLogTile = updateVisibleTiles._tileDebugCount < 10;

        const hasBiomeData = tile.spriteX !== null && tile.spriteX !== undefined &&
                             tile.spriteY !== null && tile.spriteY !== undefined;

        if (shouldLogTile) {
          console.log(`[FP-TILE-DEBUG] Tile(${tileX},${tileY}):`, {
            type: tile.type,
            hasBiomeData,
            spriteX: tile.spriteX,
            spriteY: tile.spriteY,
            spriteName: tile.spriteName,
            biome: tile.biome
          });
        }

        if (hasBiomeData) {
          // Server already sends pixel coordinates (TileRegistry.js converts row/col to pixels)
          // No conversion needed - use values directly
          const pixelX = tile.spriteX;
          const pixelY = tile.spriteY;

          // Use biome-specific InstancedMesh with PIXEL coordinates
          instMesh = getInstancedMeshForBiome(tile.type, pixelX, pixelY);

          if (shouldLogTile) {
            console.log(`[FP-TILE-DEBUG] Priority 1 result:`, {
              gotMesh: !!instMesh,
              meshName: instMesh?.name,
              hasMaterial: !!instMesh?.material,
              materialType: instMesh?.material?.type,
              hasMap: !!instMesh?.material?.map,
              materialColor: instMesh?.material?.color?.getHexString?.()
            });
            updateVisibleTiles._tileDebugCount++;
          }

          // Note: Colored fallback materials don't have .map but are still valid
          // Only reject if material itself is missing
          if (instMesh && !instMesh.material) {
            console.warn(`[FirstPerson] Biome mesh for pixels (${pixelX},${pixelY}) has no material`);
            instMesh = null;  // Force fallback to Priority 2/3
          }
        }

        // Priority 2: Map-specific sprite override detection ---------------------------
        if (!instMesh) {
          let spriteOverride = null;
          if (tile.spriteName) {
            spriteOverride = tile.spriteName;
          } else if (tile.properties?.sprite) {
            spriteOverride = tile.properties.sprite;
          }

          // If we have a per-tile sprite override, pick / create a dedicated
          // InstancedMesh for it so we never change the material of other
          // tiles already rendered.
          if (spriteOverride) {
            instMesh = getInstancedMesh(tile.type, spriteOverride);
          }
        }

        // Priority 3: Default material cache & instanced mesh for this tile type
        if (!instMesh) {
          let matCache = null;
          switch (tile.type) {
            case TILE_IDS.FLOOR:     matCache = floorMaterialCache;     instMesh = floorInstancedMesh;     break;
            case TILE_IDS.WALL:      matCache = wallMaterialCache;      instMesh = wallInstancedMesh;      break;
            case TILE_IDS.OBSTACLE:  matCache = obstacleMaterialCache;  instMesh = obstacleInstancedMesh;  break;
            case TILE_IDS.WATER:     matCache = waterMaterialCache;     instMesh = waterInstancedMesh;     break;
            case TILE_IDS.MOUNTAIN:  matCache = mountainMaterialCache;  instMesh = mountainInstancedMesh;  break;
          }
        }

        // ------------------------------------------------------------------

        switch (tile.type) {
          case TILE_IDS.FLOOR: {
            matrix = new THREE.Matrix4().makeTranslation(position.x, position.y, position.z);
            // Use the InstancedMesh selected earlier (per-sprite if applicable)
            pushMatrix(instMesh, matrix);
            break;
          }
          case TILE_IDS.WALL:
          case TILE_IDS.OBSTACLE:
          case TILE_IDS.MOUNTAIN: {
            // Stretch vertically according to tile height (makes hills / cliffs)
            const baseScaleY = 1 + heightOffset * 0.4; // tweak factor
            const scale = new THREE.Vector3(1, baseScaleY, 1);
            // BoxGeometry is centred, so raise it so its bottom sits on the ground + heightOffset
            const halfHeightWorld = (SCALING_3D * 1.5) * baseScaleY; // original box half-height * scale
            const posY = heightOffset * SCALING_3D * 0.5 + halfHeightWorld;
            const composed = new THREE.Matrix4();
            composed.compose(new THREE.Vector3(position.x, posY, position.z), quatIdentity, scale);
            matrix = composed;

            // Push into whichever InstancedMesh represents this tile's
            // concrete sprite.  For default art this is the canonical mesh;
            // for map-specific sprites it is a dedicated per-sprite mesh.
            pushMatrix(instMesh, matrix);
            break;
          }
          case TILE_IDS.WATER: {
            matrix = new THREE.Matrix4().makeTranslation(position.x, position.y, position.z);
            pushMatrix(instMesh, matrix);
            break;
          }
          case undefined: {
            if (tile.slope) {
              // Determine rotation based on slope direction
              let rotY = 0;
              switch (tile.slope) {
                case 'N': rotY = 0; break; // slope up northwards
                case 'E': rotY = Math.PI / 2; break;
                case 'S': rotY = Math.PI; break;
                case 'W': rotY = -Math.PI / 2; break;
              }
              const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, rotY, 0));
              const scale = new THREE.Vector3(1,1,1);
              matrix = new THREE.Matrix4();
              matrix.compose(position, quat, scale);
              pushMatrix(rampInstancedMesh, matrix);
            }
            break;
          }
          default:
            if (DEBUG_RENDERING && Math.random() < DEBUG_FREQUENCY) {
              console.warn(`[FirstPerson] Unknown tile type: ${tile.type} at (${tileX}, ${tileY})`);
            }
        }
      }
    }
  }

  // Function to update InstancedMesh with given matrices
  const updateInstancedMesh = (instancedMesh, matrices) => {
    if (!instancedMesh) return;
    
    const count = Math.min(matrices.length, MAX_INSTANCES);
    instancedMesh.count = count;
    
    for (let i = 0; i < count; i++) {
      instancedMesh.setMatrixAt(i, matrices[i]);
    }
    
    instancedMesh.instanceMatrix.needsUpdate = true;
    
    if (DEBUG_RENDERING && Math.random() < DEBUG_FREQUENCY) {
      console.log(`[FirstPerson] Updated ${instancedMesh.name} with ${count} instances.`);
    }
  };

  // Update each InstancedMesh
  updateInstancedMesh(floorInstancedMesh, matricesByMesh.get(floorInstancedMesh) || []);
  updateInstancedMesh(wallInstancedMesh, matricesByMesh.get(wallInstancedMesh) || []);
  updateInstancedMesh(obstacleInstancedMesh, matricesByMesh.get(obstacleInstancedMesh) || []);
  updateInstancedMesh(waterInstancedMesh, matricesByMesh.get(waterInstancedMesh) || []);
  updateInstancedMesh(mountainInstancedMesh, matricesByMesh.get(mountainInstancedMesh) || []);
  updateInstancedMesh(rampInstancedMesh, rampMatrices);

  // Update any dynamically-created per-sprite meshes not covered above
  let dynamicMeshCount = 0;
  matricesByMesh.forEach((matArr, mesh) => {
    if (mesh === floorInstancedMesh || mesh === wallInstancedMesh || mesh === obstacleInstancedMesh || mesh === waterInstancedMesh || mesh === mountainInstancedMesh) return;
    updateInstancedMesh(mesh, matArr);
    dynamicMeshCount++;
  });

  // CRITICAL: Log mesh update summary (always, not just debug mode)
  if (!updateVisibleTiles._meshLogCount) updateVisibleTiles._meshLogCount = 0;
  if (updateVisibleTiles._meshLogCount < 3) {
    updateVisibleTiles._meshLogCount++;
    console.log('=== MESH UPDATE SUMMARY ===');
    console.log('Total pushMatrix calls:', pushCount);
    console.log('matricesByMesh size:', matricesByMesh.size);
    console.log('Dynamic meshes updated:', dynamicMeshCount);
    console.log('Base mesh counts after update:', {
      floor: floorInstancedMesh?.count,
      wall: wallInstancedMesh?.count,
      obstacle: obstacleInstancedMesh?.count,
      water: waterInstancedMesh?.count,
      mountain: mountainInstancedMesh?.count
    });
    console.log('Base meshes in scene:', {
      floor: floorInstancedMesh?.parent === sceneGlobalRef,
      wall: wallInstancedMesh?.parent === sceneGlobalRef,
      obstacle: obstacleInstancedMesh?.parent === sceneGlobalRef
    });
    console.log('=== END MESH SUMMARY ===');
  }

  if (DEBUG_RENDERING && Math.random() < DEBUG_FREQUENCY) {
    console.log('[FirstPerson] Visible tiles updated:', {
      floors: matricesByMesh.get(floorInstancedMesh)?.length || 0,
      walls: matricesByMesh.get(wallInstancedMesh)?.length || 0,
      obstacles: matricesByMesh.get(obstacleInstancedMesh)?.length || 0,
      water: matricesByMesh.get(waterInstancedMesh)?.length || 0,
      mountains: matricesByMesh.get(mountainInstancedMesh)?.length || 0,
      ramps: rampMatrices.length
    });
  }

  /* ---------- BILLBOARD SPRITES (decor layer >=2) ---------- */
  if (billboardSpriteGroup && window.currentObjects) {
    const objs = window.currentObjects.filter(o=>o.type==='billboard');
    const activeIds = new Set();
    objs.forEach(obj=>{
      activeIds.add(obj.id);
      let spr = billboardSpriteMap.get(obj.id);
      if(!spr){
        const tex = obj.sprite ? getSpriteTexture(obj.sprite) : null;
        const mat = new THREE.SpriteMaterial({ map: tex||null, color: tex?0xffffff:0x00ff00, transparent:true });
        spr = new THREE.Sprite(mat);
        spr.scale.set(SCALING_3D, SCALING_3D, 1);
        billboardSpriteGroup.add(spr);
        billboardSpriteMap.set(obj.id, spr);
      }
      spr.position.set(obj.x*SCALING_3D, (SCALING_3D*0.5), obj.y*SCALING_3D);
    });
    // remove obsolete
    billboardSpriteMap.forEach((sprite,id)=>{
      if(!activeIds.has(id)){
        billboardSpriteGroup.remove(sprite);
        sprite.material.map?.dispose(); sprite.material.dispose();
        billboardSpriteMap.delete(id);
      }
    });
  }
}

/**
 * Updates the camera's position and rotation based on the character's state.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera to update.
 */
export function updateFirstPerson(camera) {
  // Log first 3 calls to prove this function is being called
  if (!updateFirstPerson._logCount) updateFirstPerson._logCount = 0;
  if (updateFirstPerson._logCount < 3) {
    updateFirstPerson._logCount++;
    console.log(`!!!! updateFirstPerson CALLED (call #${updateFirstPerson._logCount}) !!!!`);
    console.log('floorInstancedMesh exists:', !!floorInstancedMesh);
    console.log('sceneGlobalRef exists:', !!sceneGlobalRef);
  }

  const character = gameState.character;
  if (!character) return;

  // DEBUG TEST: Force render a single floor tile at camera position
  // This tests if basic InstancedMesh rendering works AT ALL
  if (!updateFirstPerson._testTileAdded && floorInstancedMesh) {
    updateFirstPerson._testTileAdded = true;
    console.log('!!!!! ADDING TEST FLOOR TILE DIRECTLY UNDER CAMERA !!!!!');
    const testMatrix = new THREE.Matrix4().makeTranslation(
      character.x * SCALING_3D,
      0, // Ground level
      character.y * SCALING_3D
    );
    floorInstancedMesh.count = 1;
    floorInstancedMesh.setMatrixAt(0, testMatrix);
    floorInstancedMesh.instanceMatrix.needsUpdate = true;
    console.log('Test tile added at:', character.x * SCALING_3D, 0, character.y * SCALING_3D);
    console.log('floorInstancedMesh.count is now:', floorInstancedMesh.count);
    console.log('floorInstancedMesh.parent:', floorInstancedMesh.parent?.type);
  }

  // Position camera according to tile coordinates
  camera.position.set(
    character.x * SCALING_3D,
    (character.z || CAMERA_HEIGHT) * SCALING_3D, // Apply scaling to height
    character.y * SCALING_3D
  );

  // Set camera rotation based on character's yaw
  camera.rotation.y = character.rotation?.yaw || 0;
  camera.rotation.x = character.rotation?.pitch || 0; // Add pitch control if available
  camera.rotation.z = 0;

  // Update tiles only if character moves a sufficient distance
  if (
    Math.abs(character.x - lastUpdateX) >= UPDATE_DISTANCE ||
    Math.abs(character.y - lastUpdateY) >= UPDATE_DISTANCE
  ) {
    updateVisibleTiles();
    lastUpdateX = character.x;
    lastUpdateY = character.y;
    
    // Clear cache when moving into a new tile to force fresh lookups
    fpsTileCache.clear();
    
    if (DEBUG_RENDERING) {
      console.log(`[FirstPerson] Camera Position: (${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)})`);
      console.log(`[FirstPerson] Character Position: (${character.x.toFixed(2)}, ${character.y.toFixed(2)}, ${(character.z || CAMERA_HEIGHT).toFixed(2)})`);
      console.log(`[FirstPerson] Camera Rotation: (${camera.rotation.x.toFixed(2)}, ${camera.rotation.y.toFixed(2)}, ${camera.rotation.z.toFixed(2)})`);
    }
  }

  // If no tiles yet rendered but map is ready, force render once
  if (floorInstancedMesh && floorInstancedMesh.count === 0 && (gameState.map || map)) {
    updateVisibleTiles();
  }

  /* ---------- BULLET UPDATE (every frame) ---------- */
  if (bulletInstancedMesh && gameState.bulletManager) {
    const bm = gameState.bulletManager;
    const matrices = [];
    const max = Math.min(bm.bulletCount, BULLET_MAX_INSTANCES);
    for (let i = 0; i < max; i++) {
      // Skip friendly bullets for clarity
      if (bm.ownerId[i] === gameState.character?.id) continue;

      const worldX = bm.x[i] * SCALING_3D;
      const worldZ = bm.y[i] * SCALING_3D;
      const worldY = CAMERA_HEIGHT * SCALING_3D * 0.2;

      matrices.push(new THREE.Matrix4().makeTranslation(worldX, worldY, worldZ));
    }

    const bulletCount = Math.min(matrices.length, BULLET_MAX_INSTANCES);
    bulletInstancedMesh.count = bulletCount;
    for (let i = 0; i < bulletCount; i++) {
      bulletInstancedMesh.setMatrixAt(i, matrices[i]);
    }
    bulletInstancedMesh.instanceMatrix.needsUpdate = true;
  }

  /* ---------- ENEMY SPRITES ---------- */
  if (enemySpriteGroup && gameState.enemyManager) {
    const em = gameState.enemyManager;
    const activeIds = new Set();
    // Iterate through active enemies
    for (let i = 0; i < em.enemyCount; i++) {
      const id = em.id[i];
      activeIds.add(id);
      let spr = enemySpriteMap.get(id);
      const sName = em.spriteName ? em.spriteName[i] : (em.enemyTypes?.[em.type[i]]?.spriteName);

      if (!spr) {
        const tex = sName ? getSpriteTexture(sName) : null;
        // Use magenta for missing sprites (more visible than red)
        const mat = new THREE.SpriteMaterial({
          map: tex || null,
          color: tex ? 0xffffff : 0xff00ff, // Magenta for missing
          transparent: true
        });
        spr = new THREE.Sprite(mat);
        spr.scale.set(SCALING_3D, SCALING_3D, 1);
        spr._spriteName = sName; // Track sprite name for retry
        spr._hasTexture = !!tex;
        enemySpriteGroup.add(spr);
        enemySpriteMap.set(id, spr);
      } else if (!spr._hasTexture && sName) {
        // Try to load texture again if it was missing before
        const tex = getSpriteTexture(sName);
        if (tex) {
          spr.material.map = tex;
          spr.material.color.setHex(0xffffff);
          spr.material.needsUpdate = true;
          spr._hasTexture = true;
        }
      }

      // Scale based on enemy render scale if available
      const renderScale = em.renderScale?.[i] || 1;
      spr.scale.set(SCALING_3D * renderScale, SCALING_3D * renderScale, 1);
      spr.position.set(em.x[i] * SCALING_3D, (em.height[i]||0.5)*SCALING_3D*0.5, em.y[i] * SCALING_3D);
    }
    // Remove dead
    enemySpriteMap.forEach((sprite, id)=>{
      if (!activeIds.has(id)) {
        enemySpriteGroup.remove(sprite);
        sprite.material.map?.dispose();
        sprite.material.dispose();
        enemySpriteMap.delete(id);
      }
    });
  }

  /* ---------- BULLET SPRITES ---------- */
  if (bulletSpriteGroup && gameState.bulletManager) {
    const bm = gameState.bulletManager;
    bulletPoolIndex = 0;
    const maxB = Math.min(bm.bulletCount, BULLET_MAX_INSTANCES);
    for (let i = 0; i < maxB; i++) {
      const spriteName = bm.spriteName ? bm.spriteName[i] : null;
      let spr = bulletSpritePool[bulletPoolIndex];
      if (!spr) {
        const tex = spriteName ? getSpriteTexture(spriteName) : null;
        const mat = new THREE.SpriteMaterial({ map: tex, color: tex ? 0xffffff : 0xffff00, transparent: true });
        spr = new THREE.Sprite(mat);
        bulletSpritePool[bulletPoolIndex] = spr;
        bulletSpriteGroup.add(spr);
      }
      spr.visible = true;
      spr.scale.set(BULLET_SCALE_3D, BULLET_SCALE_3D, 1);
      spr.position.set(bm.x[i] * SCALING_3D, CAMERA_HEIGHT*SCALING_3D*0.2, bm.y[i]*SCALING_3D);
      bulletPoolIndex++;
    }
    // hide unused pool sprites
    for (let i = bulletPoolIndex; i < bulletSpritePool.length; i++) {
      bulletSpritePool[i].visible = false;
    }
  }
}

// Export a way to force update tiles (useful for debugging or when map changes)
export function forceUpdateFirstPersonView() {
  if (DEBUG_RENDERING) {
    console.log('[FirstPerson] Forcing update of first-person view');
  }
  updateVisibleTiles();
}

// ---------------------------------------------------------------------------
// Dispose meshes / materials when switching worlds to avoid leaks & texture
// inheritance.
// ---------------------------------------------------------------------------
export function disposeFirstPersonView() {
  // Dispose all cached materials & textures
  const disposeMaterial = (mat) => {
    if (!mat) return;
    if (mat.map) { mat.map.dispose?.(); }
    mat.dispose?.();
  };

  [floorMaterialCache, wallMaterialCache, obstacleMaterialCache, waterMaterialCache, mountainMaterialCache].forEach(cache => {
    cache.forEach(disposeMaterial);
    cache.clear();
  });

  // Dispose and remove the canonical base meshes (if they still exist)
  const baseMeshes = [floorInstancedMesh, wallInstancedMesh, obstacleInstancedMesh, waterInstancedMesh, mountainInstancedMesh, rampInstancedMesh];
  baseMeshes.forEach(mesh => {
    if (!mesh) return;
    if (sceneGlobalRef) sceneGlobalRef.remove(mesh);
    mesh.geometry?.dispose?.();
    disposeMaterial(mesh.material);
  });

  floorInstancedMesh = wallInstancedMesh = obstacleInstancedMesh = waterInstancedMesh = mountainInstancedMesh = rampInstancedMesh = null;

  // Dispose atlas texture
  if (tileSheetTexture) {
    tileSheetTexture.dispose?.();
    tileSheetTexture = null;
  }

  // Dispose all InstancedMeshes and geometries
  Object.values(meshesByType).forEach(map => {
    map.forEach(mesh => {
      if (sceneGlobalRef) sceneGlobalRef.remove(mesh);
      mesh.geometry?.dispose?.();
      disposeMaterial(mesh.material);
    });
    map.clear();
  });

  // Also clear texture cache
  textureCache.forEach(tex => tex.dispose?.());
  textureCache.clear();

  console.log('[FirstPerson] Disposed first-person view resources');
}

/**
 * DEBUG: Dump scene state to console for debugging rendering issues
 */
function debugDumpSceneState() {
  console.log('=== FIRST-PERSON SCENE DEBUG DUMP ===');
  console.log('sceneGlobalRef:', !!sceneGlobalRef);
  console.log('lofiEnvironmentReady:', lofiEnvironmentReady);
  console.log('pendingTileRefresh:', pendingTileRefresh);

  // Check sprite sheets
  const lofiSheet = spriteManager.getSpriteSheet('lofi_environment');
  console.log('lofi_environment sheet:', {
    exists: !!lofiSheet,
    hasImage: !!lofiSheet?.image,
    imageComplete: lofiSheet?.image?.complete,
    imageWidth: lofiSheet?.image?.width,
    imageHeight: lofiSheet?.image?.height
  });

  // List all meshes by type
  console.log('meshesByType counts:', {
    floor: meshesByType.floor.size,
    wall: meshesByType.wall.size,
    obstacle: meshesByType.obstacle.size,
    water: meshesByType.water.size,
    mountain: meshesByType.mountain.size
  });

  // Show first few biome meshes
  let shown = 0;
  meshesByType.floor.forEach((mesh, key) => {
    if (shown++ < 3) {
      console.log(`Floor mesh "${key}":`, {
        name: mesh.name,
        count: mesh.count,
        inScene: mesh.parent === sceneGlobalRef,
        hasMaterial: !!mesh.material,
        materialType: mesh.material?.type,
        hasMap: !!mesh.material?.map,
        color: mesh.material?.color?.getHexString?.()
      });
    }
  });

  // Check base meshes
  console.log('Base meshes:', {
    floor: { exists: !!floorInstancedMesh, count: floorInstancedMesh?.count, inScene: floorInstancedMesh?.parent === sceneGlobalRef },
    wall: { exists: !!wallInstancedMesh, count: wallInstancedMesh?.count, inScene: wallInstancedMesh?.parent === sceneGlobalRef },
    obstacle: { exists: !!obstacleInstancedMesh, count: obstacleInstancedMesh?.count },
    water: { exists: !!waterInstancedMesh, count: waterInstancedMesh?.count },
    mountain: { exists: !!mountainInstancedMesh, count: mountainInstancedMesh?.count }
  });

  // Check scene children
  if (sceneGlobalRef) {
    const meshChildren = sceneGlobalRef.children.filter(c => c.type === 'InstancedMesh' || c.type === 'Mesh');
    console.log(`Scene has ${sceneGlobalRef.children.length} children, ${meshChildren.length} are meshes`);
    meshChildren.slice(0, 5).forEach(m => {
      console.log(`  - ${m.name}: type=${m.type}, count=${m.count}, visible=${m.visible}`);
    });
  }

  console.log('=== END DEBUG DUMP ===');
}

// Expose for consumers like game.js world-switch handler and render.js
if (typeof window !== 'undefined') {
  window.updateFirstPerson = updateFirstPerson; // Expose update function for render.js
  window.disposeFirstPersonView = disposeFirstPersonView;
  window.debugDumpSceneState = debugDumpSceneState; // DEBUG: expose for console
  // Ensure resources are freed when the tab is closed or reloaded
  window.addEventListener('beforeunload', () => {
    try { disposeFirstPersonView(); } catch(e) {}
  });
}
