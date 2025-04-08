// public/src/game/game.js
import { spriteManager } from '../assets/spriteManager.js'; 
import { gameState } from './gamestate.js';
import { initControls, getKeysPressed, getMoveSpeed } from './input.js';
import { addFirstPersonElements, updateFirstPerson } from '../render/renderFirstPerson.js';
import { updateCharacter } from './updateCharacter.js';
import { renderTopDownView } from '../render/renderTopDown.js';
import { renderStrategicView } from '../render/renderStrategic.js';
import { updatePlayers, playerManager, updatePlayerInterpolation } from '../entities/entities.js';
import { renderGame } from '../render/render.js';
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
        try {
            await spriteManager.loadSpriteSheet({ 
                name: 'character_sprites', 
                path: 'assets/images/Oryx/8-Bit_Remaster_Character.png',
                defaultSpriteWidth: 10,
                defaultSpriteHeight: 10
            });
        } catch (e) {
            console.error("Failed to load first character sprite path, trying fallback:", e);
            try {
                // Try lofi_char.png as a fallback
                await spriteManager.loadSpriteSheet({ 
                    name: 'character_sprites', 
                    path: 'assets/images/Oryx/lofi_char.png',
                    defaultSpriteWidth: 8,
                    defaultSpriteHeight: 8,
                    spritesPerRow: 16,
                    spritesPerColumn: 16
                });
                console.log("Successfully loaded fallback character sprite sheet");
            } catch (e2) {
                console.error("All character sprite loading attempts failed:", e2);
            }
        }
        
        // Verify character sprites loaded successfully 
        const charSheet = spriteManager.getSpriteSheet('character_sprites');
        if (!charSheet) {
            console.error('Failed to load character sprites! Players will not render correctly.');
            console.error('Verify the file exists at: assets/images/Oryx/8-Bit_Remaster_Character.png');
        } else {
            console.log('Character sprites loaded successfully:', charSheet.config);
        }
        
        await spriteManager.loadSpriteSheet({ 
            name: 'enemy_sprites', 
            path: 'assets/images/Oryx/lofi_char.png',
            defaultSpriteWidth: 8,
            defaultSpriteHeight: 8,
            spritesPerRow: 40,
            spritesPerColumn: 40
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

        // Add debug key for testing player rendering
        window.addEventListener('keydown', (e) => {
            if (e.key === 'p') {
                console.log('DEBUG: Creating test player');
                spawnTestPlayer();
            }
        });
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
        // Get client ID from server
        setClientId: (clientId) => {
            console.log(`[setClientId] Received client ID from server: ${clientId}`);
            
            // Set the ID for the local player
            localPlayer.id = clientId;
            
            // IMPORTANT: Also set this in playerManager to ensure proper filtering
            playerManager.setLocalPlayerId(clientId);
            
            console.log(`[setClientId] Local player ID set to: ${clientId}`);
        },
        
        onConnect: () => {
            console.log("Connected to server");
            gameState.isConnected = true;
        },
        
        onDisconnect: () => {
            console.log("Disconnected from server");
            gameState.isConnected = false;
        },
        [MessageType.PLAYER_LIST]: (data) => {
            // Log raw data first
            console.log(`[PLAYER_LIST] Raw data received: ${JSON.stringify(data)}`);
            console.log(`[PLAYER_LIST] Player IDs in raw data: ${Object.keys(data).join(', ')}`);
            console.log(`[PLAYER_LIST] Local player ID: ${localPlayer.id}`);
            
            // Filter out the local player to avoid the ghost sprite issue
            if (data && typeof data === 'object' && localPlayer) {
                // Create a copy of the data without the local player
                const filteredData = { ...data };
                
                // Remove local player from the data if it exists
                if (filteredData[localPlayer.id]) {
                    delete filteredData[localPlayer.id];
                    console.log(`[PLAYER_LIST] Filtered out local player (${localPlayer.id}) from player list update`);
                }
                
                // Log remaining players after filtering
                console.log(`[PLAYER_LIST] Remaining players after filtering: ${Object.keys(filteredData).join(', ')}`);
                
                // Update players with the filtered data
                updatePlayers(filteredData);
            } else {
                // If something's wrong with the data, use it as is
                updatePlayers(data);
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
            
            // Filter out the local player to avoid the ghost sprite issue
            if (players && typeof players === 'object' && localPlayer) {
                // Create a copy of the data without the local player
                const filteredPlayers = { ...players };
                
                // Remove local player from the data if it exists
                if (filteredPlayers[localPlayer.id]) {
                    delete filteredPlayers[localPlayer.id];
                    console.log(`Filtered out local player (${localPlayer.id}) from setPlayers update`);
                }
                
                // Update players with the filtered data
                updatePlayers(filteredPlayers);
            } else {
                // If something's wrong with the data, use it as is
                updatePlayers(players);
            }
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
            // Only log world updates occasionally to reduce console spam
            if (Math.random() < 0.05) {
                console.log(`World update: ${enemies?.length || 0} enemies, ${bullets?.length || 0} bullets, ${players ? Object.keys(players).length : 0} players`);
                
                // When logging, also show all player IDs
                if (players && typeof players === 'object') {
                    console.log(`[updateWorld] Player IDs in update: ${Object.keys(players).join(', ')}`);
                    console.log(`[updateWorld] Local player ID: ${localPlayer.id}`);
                }
            }
            
            // Update game entities
            if (enemies) enemyManager.updateEnemies(enemies);
            if (bullets) bulletManager.updateBullets(bullets);
            
            // Filter out local player from world updates to avoid ghost sprites
            if (players && typeof players === 'object' && localPlayer) {
                // Create a copy of players without the local player
                const filteredPlayers = { ...players };
                
                // Remove local player from the data if it exists
                if (filteredPlayers[localPlayer.id]) {
                    delete filteredPlayers[localPlayer.id];
                    
                    // Log filtering occasionally
                    if (Math.random() < 0.05) {
                        console.log(`[updateWorld] Filtered out local player (${localPlayer.id}) from update`);
                        console.log(`[updateWorld] Remaining players: ${Object.keys(filteredPlayers).join(', ')}`);
                    }
                }
                
                // Only update if we have players
                if (Object.keys(filteredPlayers).length > 0) {
                    updatePlayers(filteredPlayers);
                } else {
                    if (Math.random() < 0.05) {
                        console.log("[updateWorld] No other players to update after filtering");
                    }
                }
            } else if (players) {
                // Update with original players data if filtering isn't possible
                updatePlayers(players);
            }
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
    gameState.playerManager = playerManager;
    
    // Set the local player ID in the player manager
    // IMPORTANT: This prevents the local player from being rendered twice
    playerManager.setLocalPlayerId(localPlayer.id);
    
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
    
    // Update other players position interpolation (for smooth movement)
    updatePlayerInterpolation(delta);
    
    // Update bullet positions
    gameState.bulletManager.update(delta);
    
    // Update game elements
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
    } else {
        // For top-down and strategic views
        document.getElementById('gameCanvas').style.display = 'block';
        document.getElementById('glCanvas').style.display = 'none';
        
        // Use the main render function from render.js
        // This handles clearing the canvas and rendering all entities
        renderGame();
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
        vx: Math.cos(angle) * (gameState.character.projectileSpeed * 0.5), // Slower speed
        vy: Math.sin(angle) * (gameState.character.projectileSpeed * 0.5), // Slower speed
        ownerId: gameState.character.id,
        damage: gameState.character.damage || 10,
        lifetime: 6.0, // Double the lifetime for better visibility (was 3.0)
        width: 12, // Larger size for better visibility
        height: 12, // Larger size for better visibility
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
        speed: gameState.character.projectileSpeed * 0.5, // Match the slower speed
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

// Debug function to spawn a test player
function spawnTestPlayer() {
    if (!playerManager) {
        console.error("Cannot spawn test player: playerManager not available");
        return;
    }
    
    // Generate a random ID that won't conflict with real player IDs
    const testId = 'test-' + Math.floor(Math.random() * 9999);
    
    // Create test player data at a position offset from the local player
    const testPlayerData = {
        id: testId,
        x: gameState.character.x + (Math.random() * 10) - 5,
        y: gameState.character.y + (Math.random() * 10) - 5,
        health: 100,
        maxHealth: 100,
        name: "Test Player",
        // Match local player dimensions
        width: 10,
        height: 10,
        // Use the same sprite as the local player
        spriteX: 0,
        spriteY: 0,
        rotation: 0,
        lastUpdate: Date.now()
    };
    
    // Add to playerManager
    playerManager.players.set(testId, testPlayerData);
    console.log(`Created test player ${testId} at (${testPlayerData.x.toFixed(1)}, ${testPlayerData.y.toFixed(1)})`);
    console.log(`Player manager now has ${playerManager.players.size} other players`);
}