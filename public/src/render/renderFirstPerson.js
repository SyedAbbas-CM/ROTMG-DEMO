import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_IDS, SCALE, TILE_SPRITES } from '../constants/constants.js';
import { map } from '../map/map.js';
import { spriteManager } from '../assets/spriteManager.js';
import * as THREE from 'three';
import { spriteDatabase } from '../assets/SpriteDatabase.js';

// Caches for THREE.Texture per spriteName so we don't create multiple copies
const textureCache = new Map();

function getSpriteTexture(spriteName) {
  if (textureCache.has(spriteName)) return textureCache.get(spriteName);

  if (!spriteDatabase || !spriteDatabase.hasSprite(spriteName)) {
    return null;
  }

  const frame = spriteDatabase.getSprite(spriteName);
  const canvas = document.createElement('canvas');
  canvas.width = frame.width;
  canvas.height = frame.height;
  const ctx = canvas.getContext('2d');
  spriteDatabase.drawSprite(ctx, spriteName, 0, 0, frame.width, frame.height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  textureCache.set(spriteName, texture);
  return texture;
}

// Groups that hold dynamic sprites
let enemySpriteGroup;
let bulletSpriteGroup;
const enemySpriteMap = new Map(); // enemyId -> sprite
const bulletSpritePool = []; // reusable pool
let bulletPoolIndex = 0;

const VIEW_RADIUS = 16; // Reduced radius for performance - how many tiles to render around player
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
  [TILE_IDS.FLOOR]: 0x808080,      // Gray
  [TILE_IDS.WALL]: 0x303030,       // Dark Gray
  [TILE_IDS.OBSTACLE]: 0xFF0000,   // Red
  [TILE_IDS.WATER]: 0x0000FF,      // Blue
  [TILE_IDS.MOUNTAIN]: 0x00FF00,   // Green
};

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

// We capture the scene reference so getInstancedMesh() can insert meshes later
let sceneGlobalRef = null;

/**
 * Initializes and adds first-person elements to the scene.
 * @param {THREE.Scene} scene - The Three.js scene to add elements to.
 * @param {Function} callback - Function to call once elements are added.
 */
export function addFirstPersonElements(scene, callback) {
  // Keep global reference so dynamically created InstancedMeshes can attach
  sceneGlobalRef = scene;
  console.log('[FirstPerson] Adding first-person elements to the scene.');

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
  console.log('[FirstPerson] Created THREE.Texture from tile sprite sheet.');

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
  // 1) If a specific sprite name was requested (map-specific art) generate
  //    a material directly from that sprite.  We ignore the sprite sheet /
  //    TILE_SPRITES mapping in that case.
  if (spriteOverride && spriteDatabase?.hasSprite?.(spriteOverride)) {
    const tex = getSpriteTexture(spriteOverride);
    if (tex) {
      return new THREE.MeshStandardMaterial({
        map: tex,
        transparent: true,
        side: THREE.DoubleSide,
        emissive: new THREE.Color(0x000000),
        emissiveIntensity: 0
      });
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

  const cameraTileX = Math.floor(character.x);
  const cameraTileY = Math.floor(character.y);

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
  const pushMatrix = (mesh, mat) => {
    if (!mesh) return;
    if (!matricesByMesh.has(mesh)) matricesByMesh.set(mesh, []);
    matricesByMesh.get(mesh).push(mat);
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

        // Map-specific sprite override detection ---------------------------
        let spriteOverride = null;
        if (tile.spriteName) {
          spriteOverride = tile.spriteName;
        } else if (tile.properties?.sprite) {
          spriteOverride = tile.properties.sprite;
        }

        // Pick the correct material cache & instanced mesh for this tile
        let matCache = null;
        let instMesh  = null;
        switch (tile.type) {
          case TILE_IDS.FLOOR:     matCache = floorMaterialCache;     instMesh = floorInstancedMesh;     break;
          case TILE_IDS.WALL:      matCache = wallMaterialCache;      instMesh = wallInstancedMesh;      break;
          case TILE_IDS.OBSTACLE:  matCache = obstacleMaterialCache;  instMesh = obstacleInstancedMesh;  break;
          case TILE_IDS.WATER:     matCache = waterMaterialCache;     instMesh = waterInstancedMesh;     break;
          case TILE_IDS.MOUNTAIN:  matCache = mountainMaterialCache;  instMesh = mountainInstancedMesh;  break;
        }

        // If we have a per-tile sprite override, pick / create a dedicated
        // InstancedMesh for it so we never change the material of other
        // tiles already rendered.
        if (spriteOverride) {
          instMesh = getInstancedMesh(tile.type, spriteOverride);
        }

        // ------------------------------------------------------------------

        switch (tile.type) {
          case TILE_IDS.FLOOR: {
            matrix = new THREE.Matrix4().makeTranslation(position.x, position.y, position.z);
            pushMatrix(floorInstancedMesh, matrix);
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

            if (tile.type === TILE_IDS.WALL) {
              pushMatrix(wallInstancedMesh, matrix);
            } else if (tile.type === TILE_IDS.OBSTACLE) {
              pushMatrix(obstacleInstancedMesh, matrix);
            } else {
              pushMatrix(mountainInstancedMesh, matrix);
            }
            break;
          }
          case TILE_IDS.WATER: {
            matrix = new THREE.Matrix4().makeTranslation(position.x, position.y, position.z);
            pushMatrix(waterInstancedMesh, matrix);
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
  matricesByMesh.forEach((matArr, mesh) => {
    if (mesh === floorInstancedMesh || mesh === wallInstancedMesh || mesh === obstacleInstancedMesh || mesh === waterInstancedMesh || mesh === mountainInstancedMesh) return;
    updateInstancedMesh(mesh, matArr);
  });

  if (DEBUG_RENDERING && Math.random() < 0.01) {
    console.log(`[FP] mesh counts floor:${floorInstancedMesh?.count} wall:${wallInstancedMesh?.count}`);
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
}

/**
 * Updates the camera's position and rotation based on the character's state.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera to update.
 */
export function updateFirstPerson(camera) {
  const character = gameState.character;
  if (!character) return;

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
      if (!spr) {
        const sName = em.spriteName ? em.spriteName[i] : (em.enemyTypes[em.type[i]]?.spriteName);
        const tex = sName ? getSpriteTexture(sName) : null;
        const mat = new THREE.SpriteMaterial({ map: tex || null, color: tex ? 0xffffff : 0xff0000, transparent: true });
        spr = new THREE.Sprite(mat);
        spr.scale.set(SCALING_3D, SCALING_3D, 1);
        enemySpriteGroup.add(spr);
        enemySpriteMap.set(id, spr);
      }
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

// Expose for consumers like game.js world-switch handler
if (typeof window !== 'undefined') {
  window.disposeFirstPersonView = disposeFirstPersonView;
  // Ensure resources are freed when the tab is closed or reloaded
  window.addEventListener('beforeunload', () => {
    try { disposeFirstPersonView(); } catch(e) {}
  });
}
