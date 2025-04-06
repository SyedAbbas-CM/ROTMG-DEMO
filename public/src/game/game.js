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
        
        // Load bullet sprites - Add this for bullet rendering
        await spriteManager.loadSpriteSheet({ 
            name: 'bullet_sprites', 
            path: 'assets/images/Oryx/lofi_obj.png', 
            defaultSpriteWidth: 8,
            defaultSpriteHeight: 8,
            spritesPerRow: 16,
            spritesPerColumn: 16
        });
        
        console.log('All sprite sheets loaded.');

        // Initialize the game state
        initializeGameState();
        
        // The sprite editor container is already in the HTML, skip initialization
        console.log('Sprite editor container already exists in HTML');
        
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
                
                // Show debug overlay by default
                debugOverlay.show();
                debugOverlay.enabled = true;
            }).catch(error => {
                console.error('Failed to connect to server:', error);
                // Start game loop anyway for offline testing
                requestAnimationFrame(gameLoop);
                
                // Show debug overlay by default
                debugOverlay.show();
                debugOverlay.enabled = true;
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
    // Create local player with complete properties
    localPlayer = new Player({
        name: 'Player',
        x: 50,
        y: 50,
        width: 10,
        height: 10,
        speed: 150,
        projectileSpeed: 300, // Critical for shooting
        damage: 10,
        shootCooldown: 0.3,
        sprite: 'character_sprites_sprite_1'
    });
    
    console.log("Created local player:", localPlayer);
    
    // Create managers
    bulletManager = new ClientBulletManager(10000);
    enemyManager = new ClientEnemyManager(1000);
    
    // Make bulletManager available in console for debugging
    window.bulletManager = bulletManager;
    
    // Create map manager first (before network manager)
    mapManager = new ClientMapManager({});
    
    // IMPORTANT: Disable procedural generation to use server's map
    mapManager.proceduralEnabled = false;
    
    // Create network manager with proper handlers
    networkManager = new ClientNetworkManager(SERVER_URL, {
        // Set client ID
        setClientId: (clientId) => {
            localPlayer.id = clientId;
            console.log(`Received client ID: ${clientId}`);
            
            // Update player status in UI if available
            if (window.updatePlayerStatus) {
                window.updatePlayerStatus(`Player ID: ${clientId}`);
            }
        },
        
        // Initialize map
        initMap: (mapData) => {
            console.log("Received map data from server:", mapData);
            mapManager.initMap(mapData);
        },
        
        // Set all players
        setPlayers: (players) => {
            console.log('Players received:', players);
        },
        
        // Set initial enemies
        setEnemies: (enemies) => {
            console.log(`Received ${enemies.length} enemies from server`);
            enemyManager.setEnemies(enemies);
        },
        
        // Set initial bullets
        setBullets: (bullets) => {
            console.log(`Received ${bullets.length} bullets from server`);
            bulletManager.setBullets(bullets);
        },
        
        // Update world state
        updateWorld: (enemies, bullets, players) => {
            console.log(`World update: ${enemies?.length || 0} enemies, ${bullets?.length || 0} bullets`);
            if (enemies) enemyManager.updateEnemies(enemies);
            if (bullets) bulletManager.updateBullets(bullets);
            // TODO: Update other players
        },
        
        // Add a bullet
        addBullet: (bullet) => {
            console.log("Server created bullet:", bullet);
            bulletManager.addBullet(bullet);
        },
        
        // Apply collision
        applyCollision: (collision) => {
            console.log("Processing collision:", collision);
            // Handle collision result
            bulletManager.removeBulletById(collision.bulletId);
            
            if (collision.enemyId) {
                enemyManager.setEnemyHealth(collision.enemyId, collision.enemyHealth);
                
                if (collision.enemyKilled) {
                    console.log(`Enemy ${collision.enemyId} was killed`);
                }
            }
        },
        
        // Set chunk data
        setChunkData: (chunkX, chunkY, chunkData) => {
            console.log(`Received chunk data for (${chunkX}, ${chunkY})`, chunkData);
            mapManager.setChunkData(chunkX, chunkY, chunkData);
        }
    });
    
    // Now that network manager is created, set it in the map manager
    mapManager.networkManager = networkManager;
    
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
    
    console.log("Game state initialized with all managers");
}

/**
 * Connect to the game server
 * @returns {Promise} Resolves when connected
 */
async function connectToServer() {
    try {
        console.log(`Attempting to connect to server at ${SERVER_URL}`);
        await networkManager.connect();
        if (window.updateConnectionStatus) {
            window.updateConnectionStatus('Connected');
        }
        console.log("Connection to server successful");
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

    // Limit delta to avoid large jumps on tab switch or slowdown
    const cappedDelta = Math.min(delta, 0.1);
    
    update(cappedDelta);
    render();
    
    // Update debug overlay
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
        document.getElementById('gameCanvas').style.display = 'none';
        document.getElementById('glCanvas').style.display = 'block';
    } else if (viewType === 'top-down') {
        document.getElementById('gameCanvas').style.display = 'block';
        document.getElementById('glCanvas').style.display = 'none';
        renderTopDownView();
    } else if (viewType === 'strategic') {
        document.getElementById('gameCanvas').style.display = 'block';
        document.getElementById('glCanvas').style.display = 'none';
        renderStrategicView();
    }
}

/**
 * Handle shooting
 * @param {number} x - X coordinate
 * @param {number} y - Y coordinate
 */
export function handleShoot(x, y) {
    if (!networkManager || !networkManager.isConnected()) {
        console.log("Cannot shoot: network manager not connected");
        return;
    }
    
    // Check if player can shoot
    if (typeof localPlayer.canShoot === 'function' && !localPlayer.canShoot()) {
        console.log("Cannot shoot: on cooldown");
        return; // On cooldown or other restriction
    }
    
    console.log("Shooting at coordinates:", x, y);
    
    // Calculate angle from player to target
    const dx = x - gameState.character.x;
    const dy = y - gameState.character.y;
    const angle = Math.atan2(dy, dx);
    
    // Create a local bullet prediction
    const bulletData = {
        x: gameState.character.x,
        y: gameState.character.y,
        vx: Math.cos(angle) * gameState.character.projectileSpeed,
        vy: Math.sin(angle) * gameState.character.projectileSpeed,
        ownerId: gameState.character.id,
        damage: gameState.character.damage || 10,
        lifetime: 3.0,
        width: 8,
        height: 8,
        // Add sprite info
        spriteSheet: 'bullet_sprites',
        spriteX: 8 * 10, // X position in sprite sheet (col * width)
        spriteY: 8 * 11, // Y position in sprite sheet (row * height)
        spriteWidth: 8,
        spriteHeight: 8
    };
    
    console.log("Creating bullet with data:", bulletData);
    const bulletId = bulletManager.addBullet(bulletData);
    
    // Update player's last shot time if it has a cooldown
    if (typeof localPlayer.setLastShotTime === 'function') {
        localPlayer.setLastShotTime(Date.now());
    }
    
    // Send to server
    networkManager.sendShoot({
        x: gameState.character.x,
        y: gameState.character.y,
        angle,
        speed: gameState.character.projectileSpeed,
        damage: gameState.character.damage || 10
    });
    
    console.log(`Created bullet ${bulletId} with angle ${angle.toFixed(2)}`);
}

/**
 * Report a collision to the server
 * @param {string} bulletId - Bullet ID
 * @param {string} enemyId - Enemy ID
 */
export function reportCollision(bulletId, enemyId) {
    if (!networkManager || !networkManager.isConnected()) {
        console.log("Cannot report collision: network manager not connected");
        return;
    }
    
    console.log(`Reporting collision between bullet ${bulletId} and enemy ${enemyId}`);
    
    networkManager.sendCollision({
        bulletId: bulletId,
        enemyId: enemyId,
        timestamp: Date.now()
    });
}