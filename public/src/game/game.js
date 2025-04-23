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
import { Camera } from '../camera.js';
import { EntityAnimator } from '../entities/EntityAnimator.js';
import { PlayerManager } from '../entities/PlayerManager.js';
import { initCoordinateUtils } from '../utils/coordinateUtils.js';
import { initLogger, setLogLevel, LOG_LEVELS } from '../utils/logger.js';

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
const SERVER_PORT = 3000;
const SERVER_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:${SERVER_PORT}`;// Change if your server is on a different port or host

// Collision statistics
const collisionStats = {
    reported: 0,
    validated: 0,
    rejected: 0,
    getValidationRate() {
        if (this.reported === 0) return 0;
        return ((this.validated / this.reported) * 100).toFixed(1) + '%';
    }
};

// Debug flags
window.DEBUG_COLLISIONS = false; // Can be toggled with 'C' key
window.ALLOW_CLIENT_ENEMY_BEHAVIOR = false; // Can be toggled with 'B' key

// Make collision stats available globally for UI
window.collisionStats = collisionStats;

export async function initGame() {
    try {
        console.log('Starting game initialization...');
        
        // Initialize the logger system
        const logger = initLogger();
        console.log('Logger system initialized');
        
        // Set default log levels for different modules - turning almost everything off
        setLogLevel('default', LOG_LEVELS.NONE);
        setLogLevel('player', LOG_LEVELS.ERROR);
        setLogLevel('input', LOG_LEVELS.ERROR);
        setLogLevel('camera', LOG_LEVELS.ERROR);
        setLogLevel('render', LOG_LEVELS.ERROR);
        setLogLevel('movement', LOG_LEVELS.ERROR);
        setLogLevel('network', LOG_LEVELS.ERROR);
        setLogLevel('collision', LOG_LEVELS.ERROR);
        setLogLevel('coordinate', LOG_LEVELS.ERROR);
        
        // Display current log levels
        logger.displayLogLevels();
        
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

        // Initialize coordinate utilities for world-to-screen transformations
        initCoordinateUtils();
        console.log('Coordinate utilities initialized.');

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
            
            // Debug dump of player info on 'i' key
            if (e.key === 'i') {
                debugDumpPlayerInfo();
            }
            
            // Request player list update from server on 'd' key
            if (e.key === 'd') {
                console.log('DEBUG: Requesting player list from server');
                requestPlayerList();
            }
            
            // Debug map information on 'm' key
            if (e.key === 'm') {
                console.log('DEBUG: Printing map information');
                if (mapManager) {
                    // Print basic info by default, hold shift for detailed info
                    mapManager.debugPrintMapInfo(e.shiftKey);
                } else {
                    console.log('No map manager available');
                }
            }
            
            // Toggle collision debugging on 'c' key
            if (e.key === 'c') {
                window.DEBUG_COLLISIONS = !window.DEBUG_COLLISIONS;
                console.log(`Collision debugging ${window.DEBUG_COLLISIONS ? 'enabled' : 'disabled'}`);
                
                // Create debug canvas if needed
                if (window.DEBUG_COLLISIONS && !document.getElementById('debugCanvas')) {
                    const debugCanvas = document.createElement('canvas');
                    debugCanvas.id = 'debugCanvas';
                    debugCanvas.style.position = 'absolute';
                    debugCanvas.style.top = '0';
                    debugCanvas.style.left = '0';
                    debugCanvas.style.pointerEvents = 'none';
                    debugCanvas.style.zIndex = '1000'; // Above everything
                    debugCanvas.width = window.innerWidth;
                    debugCanvas.height = window.innerHeight;
                    document.body.appendChild(debugCanvas);
                    console.log('Created debug canvas for collision visualization');
                }
            }

            // Toggle enemy behaviors on 'b' key
            if (e.key === 'b') {
                window.ALLOW_CLIENT_ENEMY_BEHAVIOR = !window.ALLOW_CLIENT_ENEMY_BEHAVIOR;
                console.log(`Enemy behaviors ${window.ALLOW_CLIENT_ENEMY_BEHAVIOR ? 'enabled' : 'disabled'}`);
                
                // Reset behavior data if disabling
                if (!window.ALLOW_CLIENT_ENEMY_BEHAVIOR && enemyManager && enemyManager.behaviorData) {
                    enemyManager.behaviorData = null;
                    console.log('Reset enemy behavior data');
                }
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
        x: 5,
        y: 5,
        width: 10,
        height: 10,
        speed: 150,
        projectileSpeed: 300, // Critical for shooting
        damage: 10,
        shootCooldown: 0.1,
        sprite: 'character_sprites_sprite_1'
    });
    
    console.log("Created local player:", localPlayer);
    
    // Create managers
    bulletManager = new ClientBulletManager(10000);
    enemyManager = new ClientEnemyManager(1000);
    
    // Verify enemy sprite data after loading sprite sheets
    if (enemyManager.verifySpriteData) {
        // Schedule this to run after all sprite sheets are loaded
        setTimeout(() => enemyManager.verifySpriteData(), 500);
    }
    
    // Make bulletManager available in console for debugging
    window.bulletManager = bulletManager;
    
    // Create map manager first (before network manager)
    mapManager = new ClientMapManager({});
    
    // IMPORTANT: Disable procedural generation to use server's map
    mapManager.proceduralEnabled = false;
    
    // Initialize collision manager
    collisionManager = new ClientCollisionManager({
        bulletManager: bulletManager,
        enemyManager: enemyManager,
        mapManager: mapManager,
        localPlayerId: localPlayer.id
    });
    
    console.log("Created collision manager:", collisionManager);
    
    // Make collision manager available in console for debugging
    window.collisionManager = collisionManager;
    
    // Create network manager with proper handlers
    networkManager = new ClientNetworkManager(SERVER_URL, {
        // Get client ID from server
        setClientId: (clientId) => {
            console.log(`[setClientId] Received client ID from server: ${clientId}`);
            
            // Set the ID for the local player
            localPlayer.id = clientId;
            
            // IMPORTANT: Store this ID in multiple places to ensure consistency
            // This is critical for proper player filtering
            playerManager.setLocalPlayerId(clientId);
            
            // If we have a character in gameState, also ensure its ID matches
            if (gameState.character) {
                gameState.character.id = clientId;
                console.log(`[setClientId] Updated gameState.character.id to ${clientId}`);
            }
            
            // Log confirmation of ID setting
            console.log(`[setClientId] Local player ID set to: ${clientId}`);
            
            // Debug check - print the player list to verify filtering will work
            console.log(`[setClientId] Current players in playerManager: ${Array.from(playerManager.players.keys()).join(', ')}`);
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
            
            // Check if data is already the players object or if it contains a players property
            let playersData = data;
            
            // Make sure we're working with the correct format
            // The server might send { players: {...} } or just {...}
            if (data.players && typeof data.players === 'object') {
                playersData = data.players;
                console.log(`[PLAYER_LIST] Found nested players object with ${Object.keys(playersData).length} players`);
            }
            
            // Filter out the local player to avoid the ghost sprite issue
            if (playersData && typeof playersData === 'object' && localPlayer) {
                // Create a copy of the data without the local player
                const filteredData = { ...playersData };
                
                // Remove local player from the data if it exists
                if (filteredData[localPlayer.id]) {
                    delete filteredData[localPlayer.id];
                    console.log(`[PLAYER_LIST] Filtered out local player (${localPlayer.id}) from player list update`);
                }
                
                // Log remaining players after filtering
                console.log(`[PLAYER_LIST] Remaining players after filtering: ${Object.keys(filteredData).join(', ')}`);
                
                // Update players with the filtered data if there are any players left
                if (Object.keys(filteredData).length > 0) {
                    updatePlayers(filteredData);
                } else {
                    console.log(`[PLAYER_LIST] No other players to update after filtering out local player`);
                }
            } else {
                // If something's wrong with the data, use it as is
                console.log(`[PLAYER_LIST] Using raw data as fallback`);
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
            //console.log('Players received:', players);
            
            // Filter out the local player to avoid the ghost sprite issue
            if (players && typeof players === 'object' && localPlayer) {
                // Create a copy of the data without the local player
                const filteredPlayers = { ...players };
                
                // Remove local player from the data if it exists
                if (filteredPlayers[localPlayer.id]) {
                    delete filteredPlayers[localPlayer.id];
                    //console.log(`Filtered out local player (${localPlayer.id}) from setPlayers update`);
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
            //console.log(`Received ${enemies.length} enemies from server`);
            enemyManager.setEnemies(enemies);
        },
        
        // Set initial bullets
        setBullets: (bullets) => {
            //console.log(`Received ${bullets.length} bullets from server`);
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
            
            // Check if we actually got player data
            if (!players || typeof players !== 'object') {
                if (Math.random() < 0.05) {
                    console.log(`[updateWorld] No valid player data received`);
                }
                return;
            }
            
            // Check if players object is empty
            if (Object.keys(players).length === 0) {
                if (Math.random() < 0.05) {
                    console.log(`[updateWorld] Received empty players object`);
                }
                return;
            }
            
            // Filter out local player from world updates to avoid ghost sprites
            if (localPlayer) {
                // Create a copy of players without the local player
                const filteredPlayers = { ...players };
                
                // Convert to strings for comparison to avoid type issues
                const localPlayerId = String(localPlayer.id);
                
                // Remove local player from the data if it exists
                for (const id of Object.keys(filteredPlayers)) {
                    if (String(id) === localPlayerId) {
                        delete filteredPlayers[id];
                        
                        // Log filtering occasionally
                        if (Math.random() < 0.05) {
                            console.log(`[updateWorld] Filtered out local player (${id}) from update`);
                        }
                        break; // Only one local player should exist
                    }
                }
                
                // Log remaining players occasionally
                if (Math.random() < 0.05) {
                    console.log(`[updateWorld] Remaining players after filtering: ${Object.keys(filteredPlayers).join(', ')}`);
                }
                
                // Only update if we have players
                if (Object.keys(filteredPlayers).length > 0) {
                    updatePlayers(filteredPlayers);
                } else {
                    if (Math.random() < 0.05) {
                        console.log("[updateWorld] No other players to update after filtering");
                    }
                }
            } else {
                // No local player reference, can't filter properly
                console.warn("[updateWorld] No localPlayer reference, sending unfiltered data");
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
            
            // Verify that we have valid chunk data before proceeding
            if (!chunkData) {
                console.error(`Received empty or invalid chunk data for (${chunkX}, ${chunkY})`);
                return;
            }
            
            mapManager.setChunkData(chunkX, chunkY, chunkData);
        }
    });
    
    // Now that network manager is created, set it in the map manager
    mapManager.networkManager = networkManager;
    
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
    //gameState.bulletManager.update(delta);
    
    // Update game elements
    enemyManager.update(delta);
    
    // Update map visible chunks based on player position
    // Use different strategies based on view type to prevent flickering
    if (mapManager) {
        const viewType = gameState.camera.viewType;
        
        if (viewType === 'strategic') {
            // For strategic view, update chunks less frequently
            // This is now handled in renderStrategic.js directly
            // Do not update chunks here to prevent flickering
        } else if (viewType === 'top-down') {
            // For top-down view, update chunks at regular rate
            // This view shows fewer chunks so network requests are less problematic
            mapManager.updateVisibleChunks(gameState.character.x, gameState.character.y);
        } else {
            // First-person view
            mapManager.updateVisibleChunks(gameState.character.x, gameState.character.y);
        }
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
    
    // Debug: Log target coordinates
    console.log(`[handleShoot] Shooting at target coordinates: (${x.toFixed(2)}, ${y.toFixed(2)})`);
    console.log(`[handleShoot] Player position: (${gameState.character.x.toFixed(2)}, ${gameState.character.y.toFixed(2)})`);
    
    // Calculate angle from player to target
    const dx = x - gameState.character.x;
    const dy = y - gameState.character.y;
    const angle = Math.atan2(dy, dx);
    
    // Debug: Log targeting details
    console.log(`[handleShoot] Delta: (${dx.toFixed(2)}, ${dy.toFixed(2)}), Angle: ${angle.toFixed(2)} rad (${(angle * 180 / Math.PI).toFixed(1)}°)`);
    
    // CRITICAL FIX: Get direction from angle BEFORE starting attack animation
    let attackDirection = 0; // default to down
    
    // Set the player's direction based on the shooting angle
    if (gameState.character.animator && typeof gameState.character.animator.setDirectionFromAngle === 'function') {
        // First, update the animator's direction without starting the animation
        gameState.character.animator.setDirectionFromAngle(angle);
        
        // Save the direction for the attack
        attackDirection = gameState.character.animator.direction;
        console.log(`[handleShoot] Attack direction set to: ${attackDirection} (${['down', 'left', 'up', 'right'][attackDirection]})`);
    } else {
        console.log(`[handleShoot] Warning: Cannot set direction, animator not available or method missing`);
    }
    
    // Update player's last shot time and trigger attack animation
    if (typeof localPlayer.setLastShotTime === 'function') {
        // Skip animation in setLastShotTime since we'll trigger it explicitly
        localPlayer.setLastShotTime(Date.now(), true);
        
        // Also explicitly start the attack animation if available
        if (localPlayer.animator && typeof localPlayer.animator.attack === 'function') {
            console.log(`[handleShoot] Explicitly triggering attack animation with direction: ${attackDirection}`);
            localPlayer.animator.attack(attackDirection);
        }
    }
    
    // Create a local bullet prediction
    const bulletData = {
        x: gameState.character.x,
        y: gameState.character.y,
        vx: Math.cos(angle) * (gameState.character.projectileSpeed * 0.3), // Slower speed (changed from 0.5 to 0.3)
        vy: Math.sin(angle) * (gameState.character.projectileSpeed * 0.3), // Slower speed (changed from 0.5 to 0.3)
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
    
    // Log the bullet data with sprite info
    console.log("Creating bullet with data:", bulletData);
    console.log("Sprite info: sheet:", bulletData.spriteSheet, 
                "coords:", bulletData.spriteX, bulletData.spriteY);
                
    const bulletId = bulletManager.addBullet(bulletData);
    
    // Send to server
    networkManager.sendShoot({
        x: gameState.character.x,
        y: gameState.character.y,
        angle,
        speed: gameState.character.projectileSpeed * 0.3, // Match the slower speed (changed from 0.5 to 0.3)
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

/**
 * Debug function to dump player information
 */
function debugDumpPlayerInfo() {
    console.log('===== PLAYER DEBUG INFO =====');
    console.log(`Local player: ID=${localPlayer?.id}, position=(${localPlayer?.x?.toFixed(1)}, ${localPlayer?.y?.toFixed(1)})`);
    console.log(`Character ID in gameState: ${gameState.character?.id}`);
    console.log(`PlayerManager local ID: ${playerManager.localPlayerId}`);
    console.log(`Number of other players: ${playerManager.players.size}`);
    
    if (playerManager.players.size > 0) {
        console.log('Other players:');
        playerManager.players.forEach((player, id) => {
            console.log(`  - ID=${id}, position=(${player.x?.toFixed(1)}, ${player.y?.toFixed(1)}), last update: ${new Date(player.lastUpdate).toISOString()}`);
        });
    } else {
        console.log('No other players currently tracked');
        
        // Try to manually create a test player to help diagnose rendering issues
        console.log('Creating test player for diagnostic purposes...');
        spawnTestPlayer();
        console.log('After adding test player, player count is now: ' + playerManager.players.size);
    }
    
    // Add test for string vs number ID comparison
    if (localPlayer && localPlayer.id && playerManager.players.size > 0) {
        console.log('ID type diagnostics:');
        console.log(`Local player ID type: ${typeof localPlayer.id}`);
        
        const firstPlayerId = Array.from(playerManager.players.keys())[0];
        console.log(`First other player ID type: ${typeof firstPlayerId}`);
        
        const stringCompareMatch = String(localPlayer.id) === String(firstPlayerId);
        const directCompareMatch = localPlayer.id === firstPlayerId;
        
        console.log(`Direct comparison: ${directCompareMatch}, String comparison: ${stringCompareMatch}`);
    }
    
    // Add rendering test
    console.log("RENDERING TEST: Forcing render of all players...");
    console.log(`Players to render: ${playerManager.getPlayersForRender().length}`);
    
    // Enable visual debugging to make other players more obvious
    playerManager.visualDebug = true;
    console.log("Visual debugging enabled - other players will have magenta borders");
    
    console.log('=============================');
    
    // Return true to indicate diagnostic ran successfully
    return true;
}

/**
 * Handle server-validated collision result
 * @param {Object} data - Collision result data from server
 */
export function applyCollision(data) {
    if (!data.valid) {
        console.warn(`Server rejected collision: ${data.reason}`);
        collisionStats.rejected++;
        collisionStats.reported++;
        console.log(`Collision stats: ${collisionStats.validated} valid / ${collisionStats.reported} total (${collisionStats.getValidationRate()})`);
        return;
    }
    
    console.log(`Server validated collision: bullet ${data.bulletId} hit enemy ${data.enemyId}, damage: ${data.damage}`);
    collisionStats.validated++;
    collisionStats.reported++;
    console.log(`Collision stats: ${collisionStats.validated} valid / ${collisionStats.reported} total (${collisionStats.getValidationRate()})`);
    
    // Remove the bullet if it still exists
    if (bulletManager && bulletManager.removeBulletById) {
        bulletManager.removeBulletById(data.bulletId);
    }
    
    // Update enemy health if it exists
    if (enemyManager && data.enemyId) {
        // Find enemy by ID
        const enemyIndex = enemyManager.findIndexById(data.enemyId);
        if (enemyIndex !== -1) {
            // Update health directly with server value
            enemyManager.health[enemyIndex] = data.enemyHealth;
            
            // Apply visual hit effect
            enemyManager.applyHitEffect(enemyIndex);
            
            // If enemy was killed, start death animation
            if (data.enemyKilled) {
                enemyManager.startDeathAnimation(enemyIndex);
            }
        }
    }
}

// Add this function to request player list from server
function requestPlayerList() {
    if (!networkManager || !networkManager.isConnected()) {
        console.log("Cannot request player list: network manager not connected");
        return;
    }
    
    console.log("Requesting current player list from server...");
    
    // Check if networkManager has a direct method to request player list
    if (networkManager.requestPlayerList) {
        networkManager.requestPlayerList();
    } else {
        // Otherwise, try to get the player list indirectly
        console.log("No direct method to request player list. Trying ping to trigger server response...");
        if (networkManager.sendPing) {
            networkManager.sendPing();
        }
    }
    
    // Show current player info
    debugDumpPlayerInfo();
}