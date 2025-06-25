// public/src/game/game.js
import { spriteManager } from '../assets/spriteManager.js'; 
// Expose for modules that expect a global handle (e.g. ClientEnemyManager)
if (typeof window !== 'undefined') {
  window.spriteManager = spriteManager;
}
import { initializeSpriteManager } from '../game.js';
import { gameState } from './gamestate.js';
import { initControls, getKeysPressed, getMoveSpeed } from './input.js';
import { addFirstPersonElements, updateFirstPerson } from '../render/renderFirstPerson.js';
import { updateCharacter, updateCollisionVisualization, addCollisionVisualizationToggle } from './updateCharacter.js';
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
  debugOverlay,
  initUIManager,
  getUIManager
} from '../managers.js';
import { Player } from '../entities/player.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import * as THREE from 'three';
import { Camera } from '../camera.js';
import { EntityAnimator } from '../entities/EntityAnimator.js';
import { PlayerManager } from '../entities/PlayerManager.js';
import { initCoordinateUtils } from '../utils/coordinateUtils.js';
import { initLogger, setLogLevel, LOG_LEVELS } from '../utils/logger.js';
import { setupDebugTools } from '../utils/debugTools.js';
import { spriteDatabase } from '../assets/SpriteDatabase.js';
import { tileDatabase } from '../assets/TileDatabase.js';
import { ClientWorld } from '../world/ClientWorld.js';
import { entityDatabase } from '../assets/EntityDatabase.js';

let renderer, scene, camera;
let lastTime = 0;

// Game managers
let networkManager;
let mapManager;
let bulletManager;
let enemyManager;
let collisionManager;
let localPlayer;
let gameUI;

// Server connection settings
const DEFAULT_SERVER_PORT = 3000;
// Allow overriding from localStorage (set in console: localStorage.setItem('DEV_SERVER_PORT', '5000'))
const OVERRIDE_PORT = localStorage.getItem('DEV_SERVER_PORT');
// If the page itself is served from the same host:port as the WebSocket server we can reuse location.port
const PAGE_PORT = location.port && location.port !== '' ? location.port : null;

const SERVER_PORT = OVERRIDE_PORT || PAGE_PORT || DEFAULT_SERVER_PORT;

const SERVER_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:${SERVER_PORT}`;

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

// Make spriteDatabase globally visible for console usage
window.spriteDatabase = spriteDatabase;

// Expose entity database as well so other modules can consume definitions early
window.entityDatabase = entityDatabase;

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
        
        // Initialize the new sprite database system
        console.log('Initializing sprite database...');
        await initializeSpriteManager();
        
        // Load consolidated entity definitions (tiles, objects, enemies, …)
        await entityDatabase.load();

        // Load tile database
        await tileDatabase.load('assets/database/tiles.json');

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
        
        // Enemy sprites now come from the provided ROTMG sheet (chars2.png)
        // Dimensions: 512×512 with 8×8 tiles → 64×64 grid
        await spriteManager.loadSpriteSheet({ 
            name: 'enemy_sprites', 
            path: 'assets/images/chars2.png',
            defaultSpriteWidth: 8,
            defaultSpriteHeight: 8,
            spritesPerRow: 64,
            spritesPerColumn: 64
        });
        
        await spriteManager.loadSpriteSheet({ 
            name: 'tile_sprites', 
            path: 'assets/images/Oryx/8-Bit_Remaster_World.png',
            defaultSpriteWidth: 12,
            defaultSpriteHeight: 12,
            spritesPerRow: 20,
            spritesPerColumn: 20
        });
        
        // Register simple aliases so renderers can ask for 'floor', 'wall', …
        // without hard-coding sprite indices everywhere.  Adjust row/col if you
        // move tiles in the sheet.
        [
          ['floor',    0, 0], // row, col
          ['wall',     0, 1],
          ['obstacle', 0, 2],
          ['water',    0, 4],
          ['mountain', 0, 5]
        ].forEach(([alias, row, col]) => {
          spriteManager.fetchGridSprite('tile_sprites', row, col, alias, 12, 12);
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
        
        // Basic placeholder aliases for enemy types until proper atlas tags are created
        [
          ['goblin', 0, 0],
          ['orc',    0, 1],
          ['troll',  0, 2],
          ['wizard', 0, 3]
        ].forEach(([alias, row, col]) => {
          spriteManager.fetchGridSprite('enemy_sprites', row, col, alias, 8, 8);
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
        window.camera = camera; // expose to movement utils
        camera.rotation.order = 'YXZ'; // Rotate first around Y, then X
        console.log('Three.js Camera created and positioned.');

        // Initialize Controls
        initControls();
        console.log('Controls initialized.');

        // Add Ambient Light
        const hemisphereLight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.5);
        scene.add(hemisphereLight);
        console.log('Hemisphere Light added to the scene.');

        // Add global ambient light to brighten scenes (especially first-person view)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        console.log('Ambient Light added to the scene.');

        // Add Directional Light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1000, 1000, 1000); // Position the light source
        directionalLight.castShadow = true;
        scene.add(directionalLight);
        console.log('Directional Light added to the scene.');

        // Add First-Person Elements to the Scene
        addFirstPersonElements(scene, () => {
            console.log('First-person elements added. Connecting to server...');
            
            // Initialize GameUI
            console.log('Initializing Game UI...');
            gameUI = initUIManager(gameState);
            
            // Connect to the server first, don't block game start on UI loading
            connectToServer().then(() => {
                console.log('Connected to server. Starting the game loop.');
                
                // Start the Game Loop after connection is established
                requestAnimationFrame(gameLoop);
                
                // Debug overlay is enabled but not shown by default
                debugOverlay.enabled = true;
                
                // Add collision visualization toggle button
                addCollisionVisualizationToggle();
                
                // Load UI after game has started
                setTimeout(() => {
                    gameUI.init().then(() => {
                        console.log('Game UI loaded successfully');
                        if (gameUI.isInitialized) {
                            gameUI.addChatMessage('Connected to server successfully!', 'system');
                        }
                    }).catch(err => {
                        console.warn('Game UI could not be loaded:', err);
                    });
                }, 1000); // Delay UI loading to prioritize game initialization
            }).catch(error => {
                console.error('Failed to connect to server:', error);
                // Start game loop anyway for offline testing
                requestAnimationFrame(gameLoop);
                
                // Debug overlay is enabled but not shown by default
                debugOverlay.enabled = true;
                
                // Add collision visualization toggle button
                addCollisionVisualizationToggle();
                
                // Try to load UI after game has started
                setTimeout(() => {
                    gameUI.init().catch(err => {
                        console.warn('Game UI could not be loaded:', err);
                    });
                }, 1000);
            });
        });

        // Handle window resize
        window.addEventListener('resize', handleResize);
        console.log('Window resize event listener added.');

        // Add debug key for testing player rendering
        window.addEventListener('keydown', (e) => {
            // DEBUG: 'p' key - removed test player functionality
            
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

            // Toggle collision coordinate debugging on 'x' key
            if (e.key === 'x') {
                if (collisionManager) {
                    collisionManager.debugCoordinates = !collisionManager.debugCoordinates;
                    console.log(`DEBUG: Coordinate debugging ${collisionManager.debugCoordinates ? 'ENABLED' : 'DISABLED'}`);
                    
                    // If enabled, also print current coordinate system info
                    if (collisionManager.debugCoordinates && gameState.character) {
                        const x = gameState.character.x;
                        const y = gameState.character.y;
                        console.log(`Player position: (${x.toFixed(2)}, ${y.toFixed(2)})`);
                        collisionManager.debugCoordinateSystem(x, y);
                    }
                } else {
                    console.log('No collision manager available');
                }
            }
            
            // Toggle player wall collision on 'w' key
            if (e.key === 'w' && e.altKey) {
                // Create the flag if it doesn't exist
                if (window.PLAYER_COLLISION_ENABLED === undefined) {
                    window.PLAYER_COLLISION_ENABLED = true; // Default to enabled
                }
                
                window.PLAYER_COLLISION_ENABLED = !window.PLAYER_COLLISION_ENABLED;
                console.log(`Player wall collision ${window.PLAYER_COLLISION_ENABLED ? 'ENABLED' : 'DISABLED'}`);
                
                // Log current player position
                if (gameState.character) {
                    const x = gameState.character.x;
                    const y = gameState.character.y;
                    console.log(`Player position: (${x.toFixed(2)}, ${y.toFixed(2)})`);
                    
                    // Check if this position would collide with a wall
                    if (gameState.map && gameState.map.isWallOrObstacle) {
                        const wouldCollide = gameState.map.isWallOrObstacle(x, y);
                        console.log(`Current position would ${wouldCollide ? '' : 'NOT '}collide with a wall`);
                    }
                }
            }
        });

        // Setup emergency teleport
        setupDebugTeleport();

        // After all managers created and attached
        setupDebugTools({
            spriteDatabase: window.spriteDatabase || spriteDatabase,
            enemyManager,
            bulletManager,
            mapManager,
            playerManager
        });

        // Hot-reload atlases when saved from editor
        window.addEventListener('storage', (e) => {
            if (e.key === 'atlasReload' && e.newValue) {
                try {
                    const payload = JSON.parse(e.newValue);
                    const atlasPath = payload.path;
                    console.log('[HOT-RELOAD] Atlas changed:', atlasPath);
                    spriteDatabase.clearCaches();
                    spriteDatabase.loadAtlases([atlasPath]).then(()=>{
                        console.log('[HOT-RELOAD] Atlas reloaded');
                        if (enemyManager?.reinitializeSprites) enemyManager.reinitializeSprites();
                    });
                } catch(err){ console.warn('Atlas reload parse error', err); }
            }
        });

        // Pointer-lock mouse look setup
        const canvas = document.getElementById('glCanvas');
        canvas.addEventListener('click', ()=>{
            if (gameState.camera.viewType === 'first-person') {
                canvas.requestPointerLock();
            }
        });

        const lookSensitivity = 0.0025;
        document.addEventListener('pointerlockchange', ()=>{
            if (document.pointerLockElement !== canvas) return;
            const onMove = (e)=>{
                const dx = e.movementX || 0;
                const dy = e.movementY || 0;
                const char = gameState.character;
                if (!char.rotation) char.rotation = { yaw:0, pitch:0 };
                char.rotation.yaw -= dx * lookSensitivity;
                char.rotation.pitch -= dy * lookSensitivity;
                // Clamp pitch (~ +/- 89°)
                const maxPitch = Math.PI/2 - 0.05;
                char.rotation.pitch = Math.max(-maxPitch, Math.min(maxPitch, char.rotation.pitch));
            };
            document.addEventListener('mousemove', onMove);
            const exit = ()=>{
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('pointerlockchange', exit);
            };
            document.addEventListener('pointerlockchange', exit);
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
        width: 1,
        height: 1,
        speed: 150,
        projectileSpeed: 300, // Critical for shooting
        damage: 10,
        shootCooldown: 0.1,
        sprite: 'character_sprites_sprite_1',
        renderScale: 10
    });
    
    console.log("Created local player:", localPlayer);
    
    // Expose player reference for UI
    gameState.player = localPlayer;

    // Create managers
    bulletManager = new ClientBulletManager(10000);
    bulletManager.bulletScale = 3.0; // Larger bullets for visibility
    enemyManager = new ClientEnemyManager(1000);
    
    // Reinitialize enemy sprites after sprite database is loaded
    if (enemyManager.reinitializeSprites) {
        // Schedule this to run after sprite database is loaded
        setTimeout(() => {
            enemyManager.reinitializeSprites();
        }, 100);
    }
    
    // Verify enemy sprite data after loading sprite sheets
    if (enemyManager.verifySpriteData) {
        // Schedule this to run after all sprite sheets are loaded
        setTimeout(() => enemyManager.verifySpriteData(), 500);
    }
    
    // Make bulletManager available in console for debugging
    window.bulletManager = bulletManager;
    
    // Create map manager once and expose globally for any legacy references
    mapManager = new ClientMapManager({});
    window.mapManager = mapManager; // legacy scripts may refer here
    
    // IMPORTANT: Disable procedural generation to use server's map
    mapManager.proceduralEnabled = false;
    
    // Create network manager with proper handlers BEFORE collision manager
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
        
        // World switch – authoritative change initiated by server after portal
        onWorldSwitch: (data) => {
            console.log(`[GAME] onWorldSwitch → ${data.mapId}`);

            // 1) Dispose previous world (if any)
            if (gameState.world && typeof gameState.world.dispose === 'function') {
                gameState.world.dispose();
            }

            // 2) Build new world container which owns its own managers
            const newWorld = new ClientWorld({ mapData: data, networkManager });
            gameState.world = newWorld;

            // 3) Redirect the top-level references so legacy code keeps working
            mapManager     = newWorld.mapManager;
            enemyManager   = newWorld.enemyManager;
            bulletManager  = newWorld.bulletManager;

            gameState.map          = mapManager;
            gameState.enemyManager = enemyManager;
            gameState.bulletManager= bulletManager;

            // 4) Snap camera to new spawn location immediately
            if (gameState.camera && data.spawnX !== undefined) {
                gameState.camera.position.x = data.spawnX;
                gameState.camera.position.y = data.spawnY;
            }

            console.log('[GAME] World switch complete → new world managers installed, waiting for chunks…');

            // Re-hook collision manager to the fresh entity managers
            if (collisionManager && typeof collisionManager.setEntityManagers==='function'){
              collisionManager.setEntityManagers(bulletManager, enemyManager);
            }
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
            const { enemyManager } = this;
            if (!enemyManager) return;
            const currentWorld = gameState.character?.worldId;
            enemyManager.setEnemies(
                currentWorld ? enemies.filter(e => e.worldId === currentWorld) : enemies
            );
        },
        
        // Set initial bullets
        setBullets: (bullets) => {
            const { bulletManager } = this;
            if (!bulletManager) return;
            const currentWorld = gameState.character?.worldId;
            bulletManager.setBullets(
                currentWorld ? bullets.filter(b => b.worldId === currentWorld) : []
            );
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
            
            const currentWorld = gameState.character?.worldId;
            if (enemies) {
                const filteredEnemies = currentWorld ? enemies.filter(e => e.worldId === currentWorld) : enemies;
                enemyManager.updateEnemies(filteredEnemies);
            }
            if (bullets) {
                const filteredBullets = currentWorld ? bullets.filter(b => b.worldId === currentWorld) : bullets;
                bulletManager.updateBullets(filteredBullets);
            }
            
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
            
            // --- Update local player state from server (health, exact position etc.) ---
            if (localPlayer) {
                const lpIdStr = String(localPlayer.id);
                const serverSelf = players[lpIdStr];
                if (serverSelf) {
                    // Sync only mutable attributes to avoid visual jitter
                    if (typeof serverSelf.health === 'number') {
                        localPlayer.health = serverSelf.health;
                        localPlayer.maxHealth = serverSelf.maxHealth || localPlayer.maxHealth;
                        // Refresh health bar in UI immediately
                        if (window.gameUI && typeof window.gameUI.updateHealth === 'function') {
                            window.gameUI.updateHealth(localPlayer.health, localPlayer.maxHealth || 100);
                        }
                    }
                    // Keep position for first-person mode accuracy but allow client prediction in top-down
                    if (window.FIRST_PERSON_VIEW_ACTIVE && serverSelf.x !== undefined && serverSelf.y !== undefined) {
                        localPlayer.x = serverSelf.x;
                        localPlayer.y = serverSelf.y;
                    }
                }
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
    
    // Expose globally for debug and input helpers
    window.networkManager = networkManager;
    
    // Now that network manager is created, set it in the map manager
    mapManager.networkManager = networkManager;
    
    // Initialize collision manager AFTER network manager
    collisionManager = new ClientCollisionManager({
        bulletManager: bulletManager,
        enemyManager: enemyManager,
        mapManager: mapManager,
        networkManager: networkManager, // IMPORTANT: Pass networkManager here
        localPlayerId: localPlayer.id
    });
    
    console.log("Created collision manager with networkManager:", collisionManager);
    
    // Make collision manager available in console for debugging
    window.collisionManager = collisionManager;
    
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
    // Add emergency fix check
    checkAndFixStuckCharacter();
    
    // Update local player
    updateCharacter(delta);
    
    // Update collision visualization when enabled
    updateCollisionVisualization();
    
    // Update other players position interpolation (for smooth movement)
    updatePlayerInterpolation(delta);
    
    // Update bullet positions
    gameState.bulletManager.update(delta);
    
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

    // Update the game UI
    if (gameUI && gameUI.isInitialized) {
        gameUI.update(gameState);
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
    
    // Send to server - let server handle bullet creation
    networkManager.sendShoot({
        x: gameState.character.x,
        y: gameState.character.y,
        angle,
        speed: gameState.character.projectileSpeed * 0.1, 
        damage: gameState.character.damage || 10
    });
    
    console.log(`Sent shoot request to server with angle ${angle.toFixed(2)}`);
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

/**
 * Emergency function to detect and fix characters stuck in walls
 * Especially targeting the 5,5 position bug
 */
function checkAndFixStuckCharacter() {
  if (!gameState.character || !gameState.map) return;
  
  const character = gameState.character;
  const worldX = character.x;
  const worldY = character.y;
  const tileSize = gameState.map.tileSize || 12;
  
  // Check if the character is stuck in a wall
  if (gameState.map.isWallOrObstacle && gameState.map.isWallOrObstacle(worldX, worldY)) {
    console.warn(`EMERGENCY FIX: Character detected in wall at position (${worldX}, ${worldY})`);
    
    // Special case for the known bug at position 5,5
    if (Math.abs(worldX - 5) < 1 && Math.abs(worldY - 5) < 1) {
      console.warn("CRITICAL: Found character at the known problematic position (5,5). Applying emergency fix...");
      
      // Try to move the character to a safe position
      // First check neighboring tiles to find a floor
      let foundSafe = false;
      
      // Try positions in increasing distance from current position
      const checkPositions = [
        { x: 12, y: 12 },   // Move to tile 1,1 in world coordinates
        { x: 24, y: 24 },   // Move to tile 2,2 in world coordinates
        { x: 36, y: 36 },   // Move to tile 3,3 in world coordinates
        { x: 48, y: 48 },   // Move to tile 4,4 in world coordinates
        { x: 60, y: 60 },   // Move to tile 5,5 in world coordinates
        { x: 120, y: 120 }, // Move further away if needed
      ];
      
      for (const pos of checkPositions) {
        if (!gameState.map.isWallOrObstacle(pos.x, pos.y)) {
          // Found a safe position! Move character there
          character.x = pos.x;
          character.y = pos.y;
          console.log(`EMERGENCY FIX: Successfully moved character to safe position (${pos.x}, ${pos.y})`);
          foundSafe = true;
          break;
        }
      }
      
      if (!foundSafe) {
        // Extreme fallback - try to move the character to (20, 20) regardless
        character.x = 20;
        character.y = 20;
        console.warn("EMERGENCY FIX: Used extreme fallback position (20, 20)");
      }
      
      // Update camera immediately
      if (gameState.camera) {
        gameState.camera.position.x = character.x;
        gameState.camera.position.y = character.y;
      }
    }
  }
}

/**
 * Setup debug teleport key to help players escape from walls
 */
function setupDebugTeleport() {
  // Add a key listener for the 'T' key to teleport
  window.addEventListener('keydown', (event) => {
    // Shift+T teleports to a safe location
    if (event.shiftKey && event.code === 'KeyT') {
      teleportToSafeLocation();
      event.preventDefault();
    }
  });
  
  // Add to the page for mobile users
  const addTeleportButton = () => {
    // Check if the button already exists
    if (document.getElementById('debug-teleport-button')) return;
    
    const button = document.createElement('button');
    button.id = 'debug-teleport-button';
    button.innerText = 'Emergency Teleport';
    button.style.position = 'fixed';
    button.style.bottom = '10px';
    button.style.right = '10px';
    button.style.padding = '10px';
    button.style.backgroundColor = 'red';
    button.style.color = 'white';
    button.style.fontWeight = 'bold';
    button.style.border = 'none';
    button.style.borderRadius = '5px';
    button.style.zIndex = '9999';
    
    button.addEventListener('click', teleportToSafeLocation);
    
    document.body.appendChild(button);
  };
  
  // Add the button
  setTimeout(addTeleportButton, 1000);
}

/**
 * Teleport the character to a safe location
 * This is an emergency function to help players who get stuck
 */
function teleportToSafeLocation() {
  if (!gameState.character || !gameState.map) {
    console.warn("Cannot teleport: Character or map is not loaded");
    return;
  }
  
  console.log("EMERGENCY TELEPORT: Attempting to move character to a safe location");
  
  // Try a series of positions until we find one that's not a wall
  const safePositions = [
    { x: 24, y: 24 },   // Try tile (2,2)
    { x: 36, y: 36 },   // Try tile (3,3)
    { x: 48, y: 48 },   // Try tile (4,4)
    { x: 60, y: 60 },   // Try tile (5,5)
    { x: 72, y: 72 },   // Try tile (6,6)
    { x: 96, y: 96 },   // Try tile (8,8)
    { x: 120, y: 120 }, // Try tile (10,10)
  ];
  
  let teleported = false;
  
  for (const pos of safePositions) {
    // Check if this position is safe
    if (!gameState.map.isWallOrObstacle(pos.x, pos.y)) {
      // Found a safe spot, teleport there
      gameState.character.x = pos.x;
      gameState.character.y = pos.y;
      
      // Update camera to follow
      if (gameState.camera) {
        gameState.camera.position.x = pos.x;
        gameState.camera.position.y = pos.y;
      }
      
      console.log(`EMERGENCY TELEPORT: Successfully teleported to (${pos.x}, ${pos.y})`);
      
      // Add a visual effect to show the teleport
      addTeleportEffect(pos.x, pos.y);
      
      teleported = true;
      break;
    }
  }
  
  if (!teleported) {
    // Extreme fallback - just move to (20, 20) regardless
    gameState.character.x = 20;
    gameState.character.y = 20;
    
    // Update camera
    if (gameState.camera) {
      gameState.camera.position.x = 20;
      gameState.camera.position.y = 20;
    }
    
    console.warn("EMERGENCY TELEPORT: Used extreme fallback position (20, 20)");
    addTeleportEffect(20, 20);
  }
  
  // Try to tell the server about our new position if possible
  if (gameState.networkManager && gameState.networkManager.sendPlayerPosition) {
    gameState.networkManager.sendPlayerPosition(gameState.character.x, gameState.character.y);
  }
}

/**
 * Add a visual effect for teleportation
 */
function addTeleportEffect(x, y) {
  // Only add if we have a canvas
  const canvas = document.getElementById('gameCanvas');
  if (!canvas || !gameState.camera) return;
  
  const ctx = canvas.getContext('2d');
  const screenPos = gameState.camera.worldToScreen(x, y, canvas.width, canvas.height);
  
  // Draw a pulsing circle
  let radius = 5;
  let alpha = 1.0;
  const interval = setInterval(() => {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(screenPos.x, screenPos.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 255, 255, 0.5)';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
    
    radius += 2;
    alpha -= 0.05;
    
    if (alpha <= 0) {
      clearInterval(interval);
    }
  }, 20);
}

/**
 * Check if position received from server might be in the wrong coordinate system
 * This fixes a bug where the server might send tile coordinates (like 5,5) instead of world coordinates
 * @param {Object} data - Position data from server
 * @returns {Object} Corrected position data
 */
function correctServerPositionCoordinates(data) {
  if (!data || !gameState.map) return data;
  
  // Deep clone the data to avoid modifying the original
  const corrected = JSON.parse(JSON.stringify(data));
  
  // Check if x and y values exist
  if (typeof corrected.x === 'number' && typeof corrected.y === 'number') {
    const tileSize = gameState.map.tileSize || 12;
    
    // If position is suspiciously small (likely tile coordinates instead of world)
    const suspiciouslySmall = 10; // Threshold to consider it might be a tile coordinate
    if (corrected.x < suspiciouslySmall && corrected.y < suspiciouslySmall && corrected.x >= 0 && corrected.y >= 0) {
      const originalX = corrected.x;
      const originalY = corrected.y;
      
      // Try to check if this position would be inside a wall
      const worldX = corrected.x * tileSize + tileSize/2; // Center of the tile
      const worldY = corrected.y * tileSize + tileSize/2;
      
      // Check if this would be a wall at the world coordinates
      if (gameState.map.isWallOrObstacle && gameState.map.isWallOrObstacle(worldX, worldY)) {
        // It's a wall in world coordinates, so assume it's a tile coordinate and convert
        corrected.x = worldX;
        corrected.y = worldY;
        
        console.warn(`COORDINATE CORRECTION: Converted probable tile coordinates (${originalX}, ${originalY}) ` +
                    `to world coordinates (${corrected.x}, ${corrected.y})`);
      } else {
        // It's not a wall, but still seems suspicious - log it
        console.log(`Small coordinates from server: (${corrected.x}, ${corrected.y}). ` + 
                   `This might be a tile coordinate, but the corresponding world position is not a wall.`);
      }
    }
  }
  
  return corrected;
}

// Make the function available globally for use in other modules
window.correctServerPositionCoordinates = correctServerPositionCoordinates;

// Add chat message handling
export function handleChatMessage(message) {
    // Forward chat message to server or handle commands
    if (message.startsWith('/')) {
        handleChatCommand(message);
    } else if (networkManager && networkManager.isConnected()) {
        // Send to all players
        // Add implementation based on your network protocol
        console.log('Sending chat message:', message);
        
        // Display in local UI
        if (gameUI && gameUI.isInitialized) {
            gameUI.addChatMessage(message, 'player', 'You');
        }
    }
}

// Process chat commands
function handleChatCommand(command) {
    const parts = command.slice(1).split(' ');
    const cmd = parts[0].toLowerCase();
    
    switch (cmd) {
        case 'help':
            if (gameUI && gameUI.isInitialized) {
                gameUI.addChatMessage('Available commands: /help, /stats, /pos', 'system');
            }
            break;
            
        case 'stats':
            if (gameUI && gameUI.isInitialized) {
                const player = gameState.player;
                if (player) {
                    gameUI.addChatMessage(`Health: ${player.health}/${player.maxHealth}, Mana: ${player.mana}/${player.maxMana}`, 'system');
                }
            }
            break;
            
        case 'pos':
            if (gameUI && gameUI.isInitialized) {
                const player = gameState.player;
                if (player) {
                    gameUI.addChatMessage(`Position: X=${Math.round(player.x)}, Y=${Math.round(player.y)}`, 'system');
                }
            }
            break;
            
        default:
            if (gameUI && gameUI.isInitialized) {
                gameUI.addChatMessage(`Unknown command: ${cmd}`, 'system');
            }
    }
}