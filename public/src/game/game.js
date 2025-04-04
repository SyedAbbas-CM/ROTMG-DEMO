// public/src/game/game.js
import { spriteManager } from '../assets/spriteManager.js'; 
import { gameState } from './gamestate.js';
import { initControls, getKeysPressed, getMoveSpeed } from './input.js';
import { addFirstPersonElements, updateFirstPerson } from '../render/renderFirstPerson.js';
import { updateCharacter } from './updateCharacter.js';
import { renderTopDownView } from '../render/renderTopDown.js';
import { renderStrategicView } from '../render/renderStrategic.js';
import { 
  ClientMapManager, 
  ClientNetworkManager, 
  MessageType,
  ClientBulletManager,
  ClientEnemyManager,
  ClientCollisionManager,
  debugOverlay
} from '../managers.js';
import { Player } from '../entities/player.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import * as THREE from 'three';

let renderer, scene, camera;
let lastTime = 0;

// Game managers
let networkManager;
let mapManager;
let bulletManager;
let enemyManager;
let collisionManager;
let localPlayer;

// Server connection settings
const SERVER_URL = 'ws://localhost:3000'; // Change if your server is on a different port or host

export async function initGame() {
    try {
        console.log('Starting game initialization...');
        
        // Initialize sprite sheets
        console.log('Loading sprite sheets...');
        await spriteManager.loadSpriteSheet({ 
            name: 'character_sprites', 
            path: 'assets/images/Oryx/lofi_char.png',
            defaultSpriteWidth: 8,
            defaultSpriteHeight: 8,
            spritesPerRow: 16,
            spritesPerColumn: 16
        });
        
        await spriteManager.loadSpriteSheet({ 
            name: 'enemy_sprites', 
            path: 'assets/images/Oryx/oryx_16bit_fantasy_creatures_trans.png',
            defaultSpriteWidth: 24,
            defaultSpriteHeight: 24,
            spritesPerRow: 16,
            spritesPerColumn: 8
        });
        
        await spriteManager.loadSpriteSheet({ 
            name: 'tile_sprites', 
            path: 'assets/images/Oryx/8-Bit_Remaster_World.png',
            defaultSpriteWidth: 24,
            defaultSpriteHeight: 24,
            spritesPerRow: 10,
            spritesPerColumn: 10
        });
        
        console.log('All sprite sheets loaded.');

        // Initialize the game state
        initializeGameState();
        
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
        console.log('Three.js Camera created and positioned.');

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

        // Add First-Person Elements to the Scene
        addFirstPersonElements(scene, () => {
            console.log('First-person elements added. Connecting to server...');
            
            // Connect to the server
            connectToServer().then(() => {
                console.log('Connected to server. Starting the game loop.');
                // Start the Game Loop after connection is established
                requestAnimationFrame(gameLoop);
            }).catch(error => {
                console.error('Failed to connect to server:', error);
                // Start game loop anyway for offline testing
                requestAnimationFrame(gameLoop);
            });
        });

        // Handle window resize
        window.addEventListener('resize', handleResize);
        console.log('Window resize event listener added.');
    } catch (error) {
        console.error('Error initializing the game:', error);
    }
}

/**
 * Initialize game state and managers
 */
function initializeGameState() {
    // Create local player
    localPlayer = new Player({
        name: 'Player',
        x: 50,
        y: 50,
        speed: 150,
        sprite: 'character_sprites_sprite_1'
    });
    
    // Create managers
    bulletManager = new ClientBulletManager(10000);
    enemyManager = new ClientEnemyManager(1000);
    
    // Create network manager with game reference
    networkManager = new ClientNetworkManager(SERVER_URL, {
        // Set client ID
        setClientId: (clientId) => {
            localPlayer.id = clientId;
            
            // Update player status in UI if available
            if (window.updatePlayerStatus) {
                window.updatePlayerStatus(`Player ID: ${clientId}`);
            }
        },
        
        // Initialize map
        initMap: (mapData) => {
            mapManager.initMap(mapData);
        },
        
        // Set all players
        setPlayers: (players) => {
            // TODO: Handle other players
            console.log('Players received:', players);
        },
        
        // Set initial enemies
        setEnemies: (enemies) => {
            enemyManager.setEnemies(enemies);
        },
        
        // Set initial bullets
        setBullets: (bullets) => {
            bulletManager.setBullets(bullets);
        },
        
        // Update world state
        updateWorld: (enemies, bullets, players) => {
            if (enemies) enemyManager.updateEnemies(enemies);
            if (bullets) bulletManager.updateBullets(bullets);
            // TODO: Update other players
        },
        
        // Add a player
        addPlayer: (player) => {
            // TODO: Add other player
        },
        
        // Remove a player
        removePlayer: (clientId) => {
            // TODO: Remove other player
        },
        
        // Add a bullet
        addBullet: (bullet) => {
            bulletManager.addBullet(bullet);
        },
        
        // Apply collision
        applyCollision: (collision) => {
            // Handle collision result (e.g., update bullet, enemy health)
            bulletManager.removeBulletById(collision.bulletId);
            
            if (collision.enemyId) {
                enemyManager.setEnemyHealth(collision.enemyId, collision.enemyHealth);
            }
        },
        
        // Handle enemy killed
        handleEnemyKilled: (enemyId) => {
            enemyManager.removeEnemyById(enemyId);
        },
        
        // Set chunk data
        setChunkData: (chunkX, chunkY, chunkData) => {
            mapManager.setChunkData(chunkX, chunkY, chunkData);
        },
        
        // Generate fallback chunk
        generateFallbackChunk: (chunkX, chunkY) => {
            // Let the map manager handle this with its fallback generation
        },
        
        // Handle disconnect
        handleDisconnect: () => {
            console.log('Disconnected from server. Game running in offline mode.');
            // Could show reconnect UI here
        }
    });
    
    // Create map manager with network reference
    mapManager = new ClientMapManager({
        networkManager: networkManager
    });
    
    // Create collision manager
    collisionManager = new ClientCollisionManager({
        bulletManager: bulletManager,
        enemyManager: enemyManager,
        mapManager: mapManager,
        networkManager: networkManager,
        localPlayerId: localPlayer.id
    });
    
    // Update gameState with references
    gameState.character = localPlayer;
    gameState.map = mapManager;
    gameState.networkManager = networkManager;
    gameState.bulletManager = bulletManager;
    gameState.enemyManager = enemyManager;
    gameState.collisionManager = collisionManager;
}

/**
 * Connect to the game server
 * @returns {Promise} Resolves when connected
 */
async function connectToServer() {
    try {
        await networkManager.connect();
        if (window.updateConnectionStatus) {
            window.updateConnectionStatus('Connected');
        }
        return true;
    } catch (error) {
        console.error('Connection error:', error);
        if (window.updateConnectionStatus) {
            window.updateConnectionStatus('Disconnected');
        }
        throw error;
    }
}

/**
 * Handle window resize
 */
function handleResize() {
    const canvas2D = document.getElementById('gameCanvas');

    canvas2D.width = window.innerWidth;
    canvas2D.height = window.innerHeight;

    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    console.log('Renderer and camera updated on window resize.');
}

/**
 * Main game loop
 */
function gameLoop(time) {
    const delta = (time - lastTime) / 1000; // Convert to seconds
    lastTime = time;

    update(delta);
    render();
    
    // Update debug overlay if enabled
    debugOverlay.update(gameState, time);

    requestAnimationFrame(gameLoop);
}

/**
 * Update game state
 * @param {number} delta - Time elapsed since last frame in seconds
 */
function update(delta) {
    // Update local player
    updateCharacter(delta);
    
    // Update game elements
    bulletManager.update(delta);
    enemyManager.update(delta);
    
    // Update map visible chunks based on player position
    if (mapManager) {
        mapManager.updateVisibleChunks(gameState.character.x, gameState.character.y);
    }
    
    // Update collision detection
    if (collisionManager) {
        collisionManager.update(delta);
    }
    
    // Set enemies for rendering
    gameState.enemies = enemyManager.getEnemiesForRender ? 
                         enemyManager.getEnemiesForRender() : [];
    
    // Send player update to server
    if (networkManager && networkManager.isConnected()) {
        // Handle either rotation as an object with yaw or as a simple number
        const rotation = typeof gameState.character.rotation === 'object' ?
                         gameState.character.rotation.yaw || 0 :
                         gameState.character.rotation;
                         
        networkManager.sendPlayerUpdate({
            x: gameState.character.x,
            y: gameState.character.y,
            rotation: rotation,
            health: gameState.character.health
        });
    }

    // Update camera position based on character's position
    if (gameState.camera.viewType === 'first-person') {
        updateFirstPerson(camera);
    } else {
        gameState.camera.updatePosition({ x: gameState.character.x, y: gameState.character.y });
    }
}

/**
 * Render the game
 */
function render() {
    const viewType = gameState.camera.viewType;

    if (viewType === 'first-person') {
        renderer.render(scene, camera);
    } else if (viewType === 'top-down') {
        renderTopDownView();
    } else if (viewType === 'strategic') {
        renderStrategicView();
    }
}

/**
 * Handle shooting
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 */
export function handleShoot(x, y) {
    if (!networkManager || !networkManager.isConnected()) return;
    
    // Calculate angle
    const dx = x - gameState.character.x;
    const dy = y - gameState.character.y;
    const angle = Math.atan2(dy, dx);
    
    // Create local bullet prediction
    const bulletId = bulletManager.addBullet({
        x: gameState.character.x,
        y: gameState.character.y,
        vx: Math.cos(angle) * localPlayer.projectileSpeed,
        vy: Math.sin(angle) * localPlayer.projectileSpeed,
        ownerId: localPlayer.id,
        damage: localPlayer.damage,
        lifetime: 3.0
    });
    
    // Send to server
    networkManager.sendShoot({
        x: gameState.character.x,
        y: gameState.character.y,
        angle: angle,
        speed: localPlayer.projectileSpeed,
        damage: localPlayer.damage,
        lifetime: 3.0
    });
    
    // Start cooldown
    localPlayer.startShootCooldown();
}

/**
 * Report a collision to the server
 * @param {string} bulletId - Bullet ID
 * @param {string} enemyId - Enemy ID
 */
export function reportCollision(bulletId, enemyId) {
    if (!networkManager || !networkManager.isConnected()) return;
    
    networkManager.sendCollision({
        bulletId: bulletId,
        enemyId: enemyId,
        timestamp: Date.now()
    });
}