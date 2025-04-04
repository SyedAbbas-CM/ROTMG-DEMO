// src/render/renderFirstPerson.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_IDS, SCALE, TILE_SPRITES } from '../constants/constants.js';
import { map } from '../map/map.js';
import { spriteManager } from '../assets/spriteManager.js';
import * as THREE from 'three';

const VIEW_RADIUS = 200; // Radius in tiles around the player that will be rendered
const Scaling3D = 12.8
// InstancedMeshes for different tile types
let floorInstancedMesh, wallInstancedMesh, obstacleInstancedMesh, waterInstancedMesh, mountainInstancedMesh;

// Fallback colors for different tile types
const FALLBACK_COLORS = {
  [TILE_IDS.FLOOR]: 0x808080,      // Gray
  [TILE_IDS.WALL]: 0x303030,       // Dark Gray
  [TILE_IDS.OBSTACLE]: 0xFF0000,   // Red
  [TILE_IDS.WATER]: 0x0000FF,      // Blue
  [TILE_IDS.MOUNTAIN]: 0x00FF00,   // Green
};

/**
 * Initializes and adds first-person elements to the scene.
 * @param {THREE.Scene} scene - The Three.js scene to add elements to.
 * @param {Function} callback - Function to call once elements are added.
 */
export function addFirstPersonElements(scene, callback) {
  console.log('Adding first-person elements to the scene.');



  // Create a THREE.Texture from the loaded Image
   const tileSheetObj = spriteManager.getSpriteSheet('tile_sprites');
 if (!tileSheetObj) {
   useFallbackMaterials(scene);
   if (callback) callback();
   return;
 }
 const tileTexture = new THREE.Texture(tileSheetObj.image);
 tileTexture.needsUpdate = true;
  tileTexture.needsUpdate = true; // Update the texture
  console.log('Created THREE.Texture from tile sprite sheet.');

  // Create materials for each tile type
  const floorMaterial = createTileMaterial(tileTexture, TILE_IDS.FLOOR);
  const wallMaterial = createTileMaterial(tileTexture, TILE_IDS.WALL);
  const obstacleMaterial = createTileMaterial(tileTexture, TILE_IDS.OBSTACLE);
  const waterMaterial = createTileMaterial(tileTexture, TILE_IDS.WATER);
  const mountainMaterial = createTileMaterial(tileTexture, TILE_IDS.MOUNTAIN);

  // Define geometry for floor and walls
  const floorGeometry = new THREE.PlaneGeometry( Scaling3D, Scaling3D);
  floorGeometry.rotateX(-Math.PI / 2); // Rotate the plane to face upwards

  const wallGeometry = new THREE.BoxGeometry(Scaling3D, Scaling3D*3, Scaling3D);

  // Calculate maxInstances based on square area
  const maxInstances = Math.pow((2 * VIEW_RADIUS + 1), 2); // (2*32+1)^2 = 4225

  console.log(`Creating InstancedMeshes with maxInstances: ${maxInstances}`);

  try {
    // Initialize InstancedMeshes for each tile type
    floorInstancedMesh = new THREE.InstancedMesh(floorGeometry, floorMaterial, maxInstances);
    floorInstancedMesh.receiveShadow = true;
    floorInstancedMesh.name = 'floorInstancedMesh';
    scene.add(floorInstancedMesh);
    console.log('Added floorInstancedMesh to the scene:', floorInstancedMesh);

    wallInstancedMesh = new THREE.InstancedMesh(wallGeometry, wallMaterial, maxInstances);
    wallInstancedMesh.castShadow = true;
    wallInstancedMesh.receiveShadow = true;
    wallInstancedMesh.name = 'wallInstancedMesh';
    scene.add(wallInstancedMesh);
    console.log('Added wallInstancedMesh to the scene:', wallInstancedMesh);

    obstacleInstancedMesh = new THREE.InstancedMesh(wallGeometry, obstacleMaterial, maxInstances);
    obstacleInstancedMesh.castShadow = true;
    obstacleInstancedMesh.receiveShadow = true;
    obstacleInstancedMesh.name = 'obstacleInstancedMesh';
    scene.add(obstacleInstancedMesh);
    console.log('Added obstacleInstancedMesh to the scene:', obstacleInstancedMesh);

    waterInstancedMesh = new THREE.InstancedMesh(floorGeometry, waterMaterial, maxInstances);
    waterInstancedMesh.receiveShadow = true;
    waterInstancedMesh.name = 'waterInstancedMesh';
    scene.add(waterInstancedMesh);
    console.log('Added waterInstancedMesh to the scene:', waterInstancedMesh);

    mountainInstancedMesh = new THREE.InstancedMesh(wallGeometry, mountainMaterial, maxInstances);
    mountainInstancedMesh.castShadow = true;
    mountainInstancedMesh.receiveShadow = true;
    mountainInstancedMesh.name = 'mountainInstancedMesh';
    scene.add(mountainInstancedMesh);
    console.log('Added mountainInstancedMesh to the scene:', mountainInstancedMesh);

    // Initial render of tiles around the character
    updateVisibleTiles();
    console.log('Initial tiles rendered around the character.');
  } catch (error) {
    console.error('Error creating InstancedMeshes:', error);
    useFallbackMaterials(scene);
  }

  // Call the callback function to signal that elements are added
  if (callback) callback();
}

/**
 * Creates a material for a specific tile type from a tile sprite sheet.
 * @param {THREE.Texture} texture - The loaded texture for the sprite sheet.
 * @param {number} tileType - The TILE_IDS value for which to create the material.
 * @returns {THREE.MeshStandardMaterial} - The created material.
 */
function createTileMaterial(texture, tileType) {
  const spritePos = TILE_SPRITES[tileType];
  const spriteSize = TILE_SIZE; // Assuming square tiles

  if (!spritePos) {
    console.error(`No sprite position defined for tile type ${tileType}. Using fallback color.`);
    // Use a fallback color material
    return new THREE.MeshStandardMaterial({ color: FALLBACK_COLORS[tileType] || 0xffffff, side: THREE.DoubleSide });
  }

  try {
    // Clone the texture to allow independent offsets
    const tileTexture = texture.clone();

    // Ensure texture wrapping mode is correct
    tileTexture.wrapS = THREE.ClampToEdgeWrapping;
    tileTexture.wrapT = THREE.ClampToEdgeWrapping;

    // Calculate UV offsets based on sprite position
    tileTexture.offset.set(
      spritePos.x / texture.image.width,
      1 - (spritePos.y + spriteSize) / texture.image.height
    );
    tileTexture.repeat.set(
      spriteSize / texture.image.width,
      spriteSize / texture.image.height
    );
    tileTexture.needsUpdate = true;

    // Log texture properties for debugging
    console.log(`Creating material for tileType ${tileType}:`, {
      offset: tileTexture.offset,
      repeat: tileTexture.repeat
    });

    // Create material with cloned texture
    const material = new THREE.MeshStandardMaterial({
      map: tileTexture,
      transparent: true,
      side: THREE.DoubleSide
    });

    return material;
  } catch (error) {
    console.error(`Error creating material for tileType ${tileType}:`, error);
    // Fallback to solid color
    return new THREE.MeshStandardMaterial({ color: FALLBACK_COLORS[tileType] || 0xffffff, side: THREE.DoubleSide });
  }
}

/**
 * Fallback function to create and add basic colored materials if texture loading fails.
 * @param {THREE.Scene} scene - The Three.js scene to add fallback meshes to.
 */
function useFallbackMaterials(scene) {
  console.log('Using fallback materials for first-person view.');

  try {
    // Define fallback materials
    const floorMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.FLOOR] || 0x808080 });
    const wallMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.WALL] || 0x303030 });
    const obstacleMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.OBSTACLE] || 0xFF0000 });
    const waterMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.WATER] || 0x0000FF });
    const mountainMaterial = new THREE.MeshBasicMaterial({ color: FALLBACK_COLORS[TILE_IDS.MOUNTAIN] || 0x00FF00 });

    // Define geometry for floor and walls
    const floorGeometry = new THREE.PlaneGeometry(Scaling3D, Scaling3D);
    floorGeometry.rotateX(-Math.PI / 2); // Rotate the plane to face upwards

    const wallGeometry = new THREE.BoxGeometry(Scaling3D, Scaling3D* 2,Scaling3D);

    // Calculate maxInstances based on square area
    const maxInstances = Math.pow((2 * VIEW_RADIUS + 1), 2); // (2*32+1)^2 = 4225

    console.log(`Creating fallback InstancedMeshes with maxInstances: ${maxInstances}`);

    // Initialize InstancedMeshes for each tile type with fallback materials
    floorInstancedMesh = new THREE.InstancedMesh(floorGeometry, floorMaterial, maxInstances);
    floorInstancedMesh.receiveShadow = true;
    floorInstancedMesh.name = 'fallbackFloorInstancedMesh';
    scene.add(floorInstancedMesh);
    console.log('Added fallbackFloorInstancedMesh to the scene:', floorInstancedMesh);

    wallInstancedMesh = new THREE.InstancedMesh(wallGeometry, wallMaterial, maxInstances);
    wallInstancedMesh.castShadow = true;
    wallInstancedMesh.receiveShadow = true;
    wallInstancedMesh.name = 'fallbackWallInstancedMesh';
    scene.add(wallInstancedMesh);
    console.log('Added fallbackWallInstancedMesh to the scene:', wallInstancedMesh);

    obstacleInstancedMesh = new THREE.InstancedMesh(wallGeometry, obstacleMaterial, maxInstances);
    obstacleInstancedMesh.castShadow = true;
    obstacleInstancedMesh.receiveShadow = true;
    obstacleInstancedMesh.name = 'fallbackObstacleInstancedMesh';
    scene.add(obstacleInstancedMesh);
    console.log('Added fallbackObstacleInstancedMesh to the scene:', obstacleInstancedMesh);

    waterInstancedMesh = new THREE.InstancedMesh(floorGeometry, waterMaterial, maxInstances);
    waterInstancedMesh.receiveShadow = true;
    waterInstancedMesh.name = 'fallbackWaterInstancedMesh';
    scene.add(waterInstancedMesh);
    console.log('Added fallbackWaterInstancedMesh to the scene:', waterInstancedMesh);

    mountainInstancedMesh = new THREE.InstancedMesh(wallGeometry, mountainMaterial, maxInstances);
    mountainInstancedMesh.castShadow = true;
    mountainInstancedMesh.receiveShadow = true;
    mountainInstancedMesh.name = 'fallbackMountainInstancedMesh';
    scene.add(mountainInstancedMesh);
    console.log('Added fallbackMountainInstancedMesh to the scene:', mountainInstancedMesh);

    // Initial render of tiles around the character
    updateVisibleTiles();
    console.log('Initial fallback tiles rendered around the character.');
  } catch (error) {
    console.error('Error creating fallback InstancedMeshes:', error);
  }
}

/**
 * Updates the visible tiles around the character's position.
 * Only renders tiles within the VIEW_RADIUS of the character.
 */
/**
 * Updates the visible tiles around the character's position.
 * Only renders tiles within the VIEW_RADIUS of the character.
 */
function updateVisibleTiles() {
  const character = gameState.character;
  const cameraTileX = Math.floor(character.x);
  const cameraTileY = Math.floor(character.y);

  // Temporary arrays to store matrix transformations for each tile type
  const floorMatrices = [];
  const wallMatrices = [];
  const obstacleMatrices = [];
  const waterMatrices = [];
  const mountainMatrices = [];

  console.log(`Updating visible tiles around position (${cameraTileX}, ${cameraTileY}).`);

  for (let dx = -VIEW_RADIUS; dx <= VIEW_RADIUS; dx++) {
    for (let dy = -VIEW_RADIUS; dy <= VIEW_RADIUS; dy++) {
      const tileX = cameraTileX + dx;
      const tileY = cameraTileY + dy;
      const tile = map.getTile(tileX, tileY);

      if (tile) {
        const position = new THREE.Vector3(
          tileX*Scaling3D,
          tile.height || 0,
          tileY*Scaling3D
        );
        const matrix = new THREE.Matrix4().makeTranslation(position.x, position.y, position.z);

        switch (tile.type) {
          case TILE_IDS.FLOOR:
            floorMatrices.push(matrix);
            break;
          case TILE_IDS.WALL:
            wallMatrices.push(matrix);
            break;
          case TILE_IDS.OBSTACLE:
            obstacleMatrices.push(matrix);
            break;
          case TILE_IDS.WATER:
            waterMatrices.push(matrix);
            break;
          case TILE_IDS.MOUNTAIN:
            mountainMatrices.push(matrix);
            break;
          default:
            console.warn(`Unknown tile type: ${tile.type} at (${tileX}, ${tileY})`);
        }
      }
    }
  }

  // Function to update InstancedMesh with given matrices
  const updateInstancedMesh = (instancedMesh, matrices) => {
    const count = Math.min(matrices.length, instancedMesh.count);
    instancedMesh.count = count;
    for (let i = 0; i < count; i++) {
      instancedMesh.setMatrixAt(i, matrices[i]);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
    console.log(`Updated ${instancedMesh.name} with ${count} instances.`);
  };

  // Update each InstancedMesh
  if (floorInstancedMesh) {
    updateInstancedMesh(floorInstancedMesh, floorMatrices);
  }
  if (wallInstancedMesh) {
    updateInstancedMesh(wallInstancedMesh, wallMatrices);
  }
  if (obstacleInstancedMesh) {
    updateInstancedMesh(obstacleInstancedMesh, obstacleMatrices);
  }
  if (waterInstancedMesh) {
    updateInstancedMesh(waterInstancedMesh, waterMatrices);
  }
  if (mountainInstancedMesh) {
    updateInstancedMesh(mountainInstancedMesh, mountainMatrices);
  }

  console.log('Visible tiles updated.');
}


/**
 * Updates the camera's position and rotation based on the character's state.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera to update.
 */
export function updateFirstPerson(camera) {
  const character = gameState.character;

  // Position camera according to tile coordinates
  camera.position.set(
    character.x ,
    character.z || 1.5, // Default eye height
    character.y 
  );

  // Set camera rotation based on character's yaw
  camera.rotation.y = character.rotation.yaw || 0;
  camera.rotation.x = 0; // Pitch control can be added if needed
  camera.rotation.z = 0;

  // Note: Removed camera position and rotation logs to prevent spamming

  // Update tiles only if character moves a full tile distance
  if (
    Math.abs(character.x - gameState.lastUpdateX) >= 1 ||
    Math.abs(character.y - gameState.lastUpdateY) >= 1
  ) {
    updateVisibleTiles();
    gameState.lastUpdateX = character.x;
    gameState.lastUpdateY = character.y;
  }
  console.log(`FPS View - Character Position: (${character.x}, ${character.y}, ${character.z || 1.5})`);
  console.log(`FPS View - Camera Position: (${camera.position.x}, ${camera.position.y}, ${camera.position.z})`);
}
