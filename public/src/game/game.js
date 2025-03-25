// src/game/game.js
import { spriteManager } from '../assets/spriteManager.js'; 
import { gameState } from './gamestate.js';
import { initControls } from './input.js';
import { addFirstPersonElements, updateFirstPerson } from '../render/renderFirstPerson.js';
import { EnemyManager } from '../entities/enemyManager.js';
import { updateCharacter } from './updateCharacter.js';
import { renderTopDownView } from '../render/renderTopDown.js';
import { renderStrategicView } from '../render/renderStrategic.js';
import { map } from '../map/map.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';

/* global THREE */
let renderer, scene, camera;
let lastTime = 0;
let enemyManager; // Instance of EnemyManager

// Configuration: Choose between 'procedural' or 'fixed'
const MAP_MODE = 'fixed'; // 'procedural' | 'fixed'
const FIXED_MAP_URL = 'assets/maps/fixedMap1.json'; // Path to your fixed map file

export async function initGame() {
  try {
    console.log('Starting asset loading...');
       console.log('Loading sprite sheets via spriteManager...');
    await spriteManager.loadSpriteSheet({ name: 'character_sprites', path: 'assets/images/Oryx/lofi_char.png' });
    await spriteManager.loadSpriteSheet({ name: 'enemy_sprites', path: 'assets/images/Oryx/8-Bit_Remaster_Character.png' });
    await spriteManager.loadSpriteSheet({ name: 'tile_sprites', path: 'assets/images/Oryx/8-Bit_Remaster_World.png' });
    console.log('All sprite sheets loaded.');

    // Initialize Three.js Renderer
    renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('glCanvas'), antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    console.log('Three.js Renderer initialized.');

    // Create Three.js Scene
    scene = new THREE.Scene();
    console.log('Three.js Scene created.');

    // Create Camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 1.5, 0); // Origin with eye height
    camera.rotation.order = 'YXZ'; // Rotate first around Y, then X
    console.log('Three.js Camera created and positioned:', camera.position);

    // Initialize Controls
    initControls();
    console.log('Controls initialized.');

    // Add Ambient Light
    const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.5);
    scene.add(hemisphereLight);
    console.log('Hemisphere Light added to the scene.');

    // Add Directional Light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1000, 1000, 1000); // Position the light source
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    console.log('Directional Light added to the scene.');

    if (MAP_MODE === 'fixed') {
      // Load the fixed map and disable procedural generation
      console.log(`Loading fixed map from ${FIXED_MAP_URL}...`);
      await map.loadFixedMap(FIXED_MAP_URL);
      map.disableProceduralGeneration();
      console.log('Fixed map loaded and procedural generation disabled.');
    } else {
      // Enable procedural generation and do not load a fixed map
      map.enableProceduralGeneration();
      console.log('Procedural generation enabled.');
    }

    // Add First-Person Elements to the Scene
    addFirstPersonElements(scene, () => {
      console.log('First-person elements added. Starting the game loop.');
      // Start the Game Loop after first-person elements are ready
      requestAnimationFrame(gameLoop);
    });

    // Initialize Enemy Manager
    enemyManager = new EnemyManager(scene);
    enemyManager.initializeEnemies(10); // Spawn 10 initial enemies
    console.log('Enemy Manager initialized and enemies spawned.');

    // Handle window resize
    window.addEventListener('resize', handleResize);
    console.log('Window resize event listener added.');
  } catch (error) {
    console.error('Error initializing the game:', error);
  }
}

// Handle window resize
function handleResize() {
  const canvas2D = document.getElementById('gameCanvas');

  canvas2D.width = window.innerWidth;
  canvas2D.height = window.innerHeight;

  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  console.log('Renderer and camera updated on window resize.');
}

// Main game loop
function gameLoop(time) {
  const delta = (time - lastTime) / 1000; // Convert to seconds
  lastTime = time;

  update(delta);
  render();

  requestAnimationFrame(gameLoop);
}

// Update function
function update(delta) {
  updateCharacter(delta);
  if (enemyManager) enemyManager.update(delta);

  // Update camera position based on character's position
  if (gameState.camera.viewType === 'first-person') {
    updateFirstPerson(camera);
  } else {
    gameState.camera.updatePosition({ x: gameState.character.x, y: gameState.character.y });
  }
}

// Render function
function render() {
  const viewType = gameState.camera.viewType;

  if (viewType === 'first-person') {
    renderer.render(scene, camera);
    //console.log('Rendered first-person view.');
  } else if (viewType === 'top-down') {
    renderTopDownView();
    //console.log('Rendered top-down view.');
  } else if (viewType === 'strategic') {
    renderStrategicView();
    //console.log('Rendered strategic view.');
  }
}
