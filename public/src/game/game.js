// public/src/game/game.js
// MUST BE FIRST - filters console output to show only BINARY/NETWORK logs
import '../utils/disableConsoleLogs.js';

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
import { updateTileEffects, getTileSpeedMultiplier } from '../utils/tileEffects.js';
import { renderTopDownView } from '../render/renderTopDown.js';
import { renderStrategicView } from '../render/renderStrategic.js';
import { updatePlayers, playerManager, updatePlayerInterpolation } from '../entities/entities.js';
import { renderGame } from '../render/render.js';
import { 
  ClientMapManager, 
  ClientNetworkManager, 
  ClientBulletManager,
  ClientEnemyManager,
  ClientCollisionManager,
  debugOverlay,
  initUIManager,
  getUIManager
} from '../managers.js';
import { MessageType } from '/common/protocol.js';
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
import { speechBubbleManager } from '../ui/SpeechBubbleManager.js';
import { ClientInventoryManager } from './ClientInventoryManager.js';
import ClientUnitManager from '../units/ClientUnitManager.js';

let renderer, scene, camera;
let lastTime = 0;

// Game managers
let networkManager;
let mapManager;
let bulletManager;
let enemyManager;
let unitManager;
let collisionManager;
let localPlayer;
let gameUI;

// Server connection settings
const DEFAULT_SERVER_PORT = 3000;
// Allow overriding from localStorage (set in console: localStorage.setItem('DEV_SERVER_PORT', '5000'))
const OVERRIDE_PORT = localStorage.getItem('DEV_SERVER_PORT');
// If the page itself is served from the same host:port as the WebSocket server we can reuse location.port
const PAGE_PORT = location.port && location.port !== '' ? location.port : null;

// For HTTPS on standard port (tunnel) OR known production domains, don't add any port suffix
// WebSocket always goes through Cloudflare (wss://), PlayIt is only for WebRTC UDP
const IS_TUNNEL = (location.protocol === 'https:' && !location.port) ||
    location.hostname.includes('eternalconquests.com') ||
    location.hostname.includes('trycloudflare.com');

const SERVER_PORT = OVERRIDE_PORT || PAGE_PORT || DEFAULT_SERVER_PORT;

const SERVER_URL = IS_TUNNEL
    ? `wss://${location.hostname}`
    : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:${SERVER_PORT}`;

console.log('[NETWORK] IS_TUNNEL:', IS_TUNNEL, 'SERVER_URL:', SERVER_URL);

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

// expose globally for network handler
if (typeof window !== 'undefined') window.speechBubbleManager = speechBubbleManager;

// --------------------------- Loading Screen -----------------------------
function showLoadingScreen(msg='Loading…') {
  let div = document.getElementById('loadingOverlay');
  if (!div) {
    div = document.createElement('div');
    div.id = 'loadingOverlay';
    div.style.position = 'fixed';
    div.style.top = 0;
    div.style.left = 0;
    div.style.width = '100%';
    div.style.height = '100%';
    div.style.background = 'rgba(0,0,0,0.8)';
    div.style.color = '#fff';
    div.style.fontSize = '24px';
    div.style.display = 'flex';
    div.style.alignItems = 'center';
    div.style.justifyContent = 'center';
    div.style.zIndex = 9999;
    document.body.appendChild(div);
  }
  div.textContent = msg;
  div.style.display = 'flex';
}

function hideLoadingScreen() {
  const div = document.getElementById('loadingOverlay');
  if (div) div.style.display = 'none';
}

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
        // TODO: Disabled until /api/entities endpoint is created
        // await entityDatabase.load();

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
        
        // 32×32 RotMG character/enemy sheet – provides vivid monsters like
        // Red Demon (row 2 col 1) and Dark Lord (row 3 col 9).  We load it
        // primarily so we can alias simpler enemy names to these sprites.
        await spriteManager.loadSpriteSheet({
          name: 'chars',
          path: 'assets/images/chars.png',
          defaultSpriteWidth: 32,
          defaultSpriteHeight: 32,
          spritesPerRow: 16,
          spritesPerColumn: 16
        });
        
        // Define key 32×32 sprites so later alias mapping works safely
        spriteManager.fetchGridSprite('chars', 2, 1, 'red_demon', 32, 32);
        spriteManager.fetchGridSprite('chars', 3, 9, 'dark_lord', 32, 32);
        
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
        
        // ------------------------------------------------------------------
        // VISIBILITY PATCH – some grid sprites are nearly blank, resulting in
        // invisible enemies when they use names like "goblin" and "orc".
        // Map those aliases to vivid 32×32 creatures from the main chars
        // atlas so they always render.
        // ------------------------------------------------------------------
        if (spriteManager.getSprite('red_demon')) {
          spriteManager.registerAlias('goblin', 'chars', 'red_demon');
        }
        if (spriteManager.getSprite('dark_lord')) {
          spriteManager.registerAlias('orc',    'chars', 'dark_lord');
        }
        
        console.log('All sprite sheets loaded.');

        // Initialize coordinate utilities for world-to-screen transformations
        initCoordinateUtils();
        console.log('Coordinate utilities initialized.');

        console.log('[DEBUG] About to call initializeGameState...');
        // Initialize the game state
        initializeGameState();
        console.log('[DEBUG] initializeGameState completed.');
        
        // The sprite editor container is already in the HTML, skip initialization
        console.log('Sprite editor container already exists in HTML');
        
        // Initialize Three.js Renderer
        renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('glCanvas'), antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        window.renderer = renderer; // Expose to render.js for first-person view
        console.log('Three.js Renderer initialized.');

        // Create Three.js Scene
        scene = new THREE.Scene();
        window.scene = scene; // Expose to render.js for first-person view
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

        // Add First-Person Elements to the Scene - now properly awaits async function
        await addFirstPersonElements(scene, null);
        console.log('First-person elements added. Connecting to server...');

        // Initialize GameUI
        console.log('Initializing Game UI...');
        gameUI = initUIManager(gameState);

        // Connect to the server
        try {
            await connectToServer();
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
        } catch (error) {
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
        }

        // Handle window resize
        window.addEventListener('resize', handleResize);
        console.log('Window resize event listener added.');

        // Add debug key for testing player rendering
        window.addEventListener('keydown', (e) => {
            // DEBUG: 'p' key - removed test player functionality
            
            // Debug dump of player info on 'i' key
            if (e.key === 'i') {
                console.log('Player:', character);
                console.log('Position:', character?.x, character?.y);
                console.log('Health:', character?.health, '/', character?.maxHealth);
            }
            
            // Request player list update from server on Shift+D key
            if (e.key === 'D' && e.shiftKey) {
                console.log('DEBUG: Requesting player list from server');
                if (window.networkManager && typeof window.networkManager.requestPlayerList === 'function') {
                    window.networkManager.requestPlayerList();
                }
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

        // Setup emergency teleport (guard if not available)
        if (typeof setupDebugTeleport === 'function') {
            setupDebugTeleport();
        }

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
    console.log('[initializeGameState] Starting game state initialization...');
    // Create local player with complete properties
    console.log('[initializeGameState] Creating local player...');
    localPlayer = new Player({
        name: 'Player',
        x: 2, // spawn slightly more to the left
        y: 35, // spawn much lower on the map
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
    unitManager = new ClientUnitManager(2000); // Match server capacity
    window.unitManager = unitManager; // Expose for debugging
    
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
    gameState.mapManager = mapManager; // Make available in gameState for tile effects system

    // IMPORTANT: Disable procedural generation to use server's map
    mapManager.proceduralEnabled = false;
    
    // Create game object with required methods for ClientNetworkManager
    const gameObject = {
        initMap: (mapData) => {
            console.log("Received map data from server:", mapData);
            mapManager.initMap(mapData);
        },
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
        
        // Player management methods
        setPlayers: (players) => {
            if (players && typeof players === 'object' && localPlayer) {
                // Create a copy of the data without the local player
                const filteredPlayers = { ...players };
                
                // Remove local player from the data if it exists
                if (filteredPlayers[localPlayer.id]) {
                    delete filteredPlayers[localPlayer.id];
                }
                
                // Update players with the filtered data
                if (typeof updatePlayers === 'function') {
                    updatePlayers(filteredPlayers);
                }
            } else {
                // If something's wrong with the data, use it as is
                if (typeof updatePlayers === 'function') {
                    updatePlayers(players);
                }
            }
        },
        
        setEnemies: (enemies) => {
            if (!enemyManager) return;
            const currentWorld = gameState.character?.worldId;
            enemyManager.setEnemies(
                currentWorld ? enemies.filter(e => e.worldId === currentWorld) : enemies
            );
        },
        
        setBullets: (bullets) => {
            if (!bulletManager) return;
            const currentWorld = gameState.character?.worldId;
            bulletManager.setBullets(
                currentWorld ? bullets.filter(b => b.worldId === currentWorld) : []
            );
        },
        
        updateWorld: (enemies, bullets, players, objects=[], units=[]) => {
            // Only log world updates occasionally to reduce console spam
            if (Math.random() < 0.05) {
                console.log(`World update: ${enemies?.length || 0} enemies, ${bullets?.length || 0} bullets, ${players ? Object.keys(players).length : 0} players`);
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
            
            // Update units if present
            if (units && unitManager) {
                const filteredUnits = currentWorld ? units.filter(u => u.worldId === currentWorld) : units;
                unitManager.spawnMany(filteredUnits);
            }
            
            // Check if we actually got player data
            if (!players || typeof players !== 'object') {
                return;
            }
            
            // Filter out local player from world updates to avoid ghost sprites
            // CRITICAL: Try multiple sources for local player ID to handle race conditions
            const localId = localPlayer?.id || gameState.character?.id || networkManager?.clientId;
            if (localId) {
                const filteredPlayers = { ...players };
                // Binary protocol uses entity_X format, but localPlayer.id might be just X
                const localPlayerId = String(localId);
                const entityLocalPlayerId = localPlayerId.startsWith('entity_') ? localPlayerId : `entity_${localPlayerId}`;

                // CRITICAL FIX: Extract server health BEFORE deleting local player
                // The server is authoritative for health, so we need to apply it!
                // Try both ID formats
                const serverPlayer = players[localPlayerId] || players[entityLocalPlayerId];
                if (serverPlayer && typeof serverPlayer.health === 'number') {
                    // Update local player health from server
                    const oldHealth = gameState.character.health;
                    gameState.character.health = serverPlayer.health;

                    // Also sync maxHealth from server to fix HP bar display
                    if (typeof serverPlayer.maxHealth === 'number') {
                        gameState.character.maxHealth = serverPlayer.maxHealth;
                    }

                    // Update UI if health changed
                    if (oldHealth !== serverPlayer.health && window.gameUI && typeof window.gameUI.updateHealth === 'function') {
                        window.gameUI.updateHealth(gameState.character.health, gameState.character.maxHealth || 100);
                    }
                }

                // Remove local player from updates - check BOTH ID formats
                Object.keys(filteredPlayers).forEach(playerId => {
                    const pid = String(playerId);
                    if (pid === localPlayerId || pid === entityLocalPlayerId) {
                        delete filteredPlayers[playerId];
                    }
                });

                // Update players with the filtered data
                if (typeof updatePlayers === 'function') {
                    updatePlayers(filteredPlayers);
                }
            } else {
                if (typeof updatePlayers === 'function') {
                    updatePlayers(players);
                }
            }
        },
        
        setChunkData: (chunkX, chunkY, chunkData) => {
            if (mapManager && typeof mapManager.setChunkData === 'function') {
                mapManager.setChunkData(chunkX, chunkY, chunkData);
            }
        }
    };

    // Create network manager with game object
    networkManager = new ClientNetworkManager(SERVER_URL, gameObject);

    // Set up connection event handlers
    networkManager.onConnect = () => {
        console.log("Connected to server");
        gameState.isConnected = true;
    };
    
    networkManager.onDisconnect = () => {
        console.log("Disconnected from server");
        gameState.isConnected = false;
    };
    
    // Set up message handlers
    networkManager.on(MessageType.PLAYER_LIST, (data) => {
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
    });
    
    // Other message handlers can be added here as needed
    
    // Expose globally for debug and input helpers
    window.networkManager = networkManager;

    // Now that network manager is created, set it in the map manager
    mapManager.networkManager = networkManager;
    console.log('[INIT] Set networkManager in mapManager:', !!networkManager);
    
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
    
    // Initialize inventory UI
    const uiMgr = getUIManager && getUIManager();
    const inventoryManager = new ClientInventoryManager();
    if(uiMgr) inventoryManager.init(uiMgr);
    if (window.gameManager) window.gameManager.inventoryManager = inventoryManager;
    // Expose for console
    window.inventoryManager = inventoryManager;
    
    // Update gameState with references
    gameState.character = localPlayer;
    gameState.map = mapManager;
    gameState.networkManager = networkManager;
    gameState.bulletManager = bulletManager;
    gameState.enemyManager = enemyManager;
    gameState.unitManager = unitManager;
    gameState.collisionManager = collisionManager;
    gameState.playerManager = playerManager;
    
    // Set the local player ID in the player manager
    // IMPORTANT: This prevents the local player from being rendered twice
    playerManager.setLocalPlayerId(localPlayer.id);
    
    console.log("Game state initialized with all managers");
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
    if (debugOverlay && debugOverlay.enabled) {
        debugOverlay.update(cappedDelta);
    }
    
    requestAnimationFrame(gameLoop);
}

/**
 * Handle window resize
 */
function handleResize() {
    const canvas2D = document.getElementById('gameCanvas');

    canvas2D.width = window.innerWidth;
    canvas2D.height = window.innerHeight;

    if (renderer) {
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    if (camera) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
    }
    console.log('Renderer and camera updated on window resize.');
}

/**
 * Update game state
 * @param {number} delta - Time elapsed since last frame in seconds
 */
function update(delta) {
    // Add emergency fix check
    if (typeof checkAndFixStuckCharacter === 'function') {
        checkAndFixStuckCharacter();
    }
    
    // Update local player
    if (typeof updateCharacter === 'function') {
        updateCharacter(delta);
    }

    // Update tile effects (speed penalties + damage from water/lava)
    if (gameState.character && typeof updateTileEffects === 'function') {
        updateTileEffects(gameState.character);
    }

    // Update collision visualization when enabled
    if (typeof updateCollisionVisualization === 'function') {
        updateCollisionVisualization();
    }
    
    // Update other players position interpolation (for smooth movement)
    if (typeof updatePlayerInterpolation === 'function') {
        updatePlayerInterpolation(delta);
    }
    
    // Update bullet positions
    if (gameState.bulletManager) {
        gameState.bulletManager.update(delta);
    }
    
    // Update game elements
    if (enemyManager) {
        enemyManager.update(delta);
    }

    // Note: Chunk updates are handled by renderTopDown.js every 2 seconds
    // Removed duplicate updateVisibleChunks call to prevent chunk spam

    // Update collision detection
    if (collisionManager) {
        collisionManager.update(delta);
    }
    
    // Send player update to server
    if (networkManager && networkManager.isConnected() && gameState.character) {
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
    if (gameState.camera && gameState.character) {
        if (gameState.camera.viewType === 'first-person') {
            if (typeof updateFirstPerson === 'function') {
                updateFirstPerson(camera);
            }
        } else {
            gameState.camera.updatePosition({ x: gameState.character.x, y: gameState.character.y });
        }
    }

    // Update the game UI
    if (gameUI && gameUI.isInitialized) {
        gameUI.update(gameState);
    }

    // Update speech bubbles lifetime
    if (window.speechBubbleManager) {
        window.speechBubbleManager.update();
    }
}

/**
 * Render the game
 */
function render() {
    const viewType = gameState.camera ? gameState.camera.viewType : 'top-down';

    if (viewType === 'first-person') {
        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
        document.getElementById('gameCanvas').style.display = 'none';
        document.getElementById('glCanvas').style.display = 'block';
    } else {
        // For top-down and strategic views
        document.getElementById('gameCanvas').style.display = 'block';
        document.getElementById('glCanvas').style.display = 'none';
        
        // Use the main render function from render.js
        if (typeof renderGame === 'function') {
            renderGame();
        }

        // After main render draw speech bubbles
        const ctx = document.getElementById('gameCanvas').getContext('2d');
        if (window.speechBubbleManager) {
            window.speechBubbleManager.render(ctx);
        }
    }
}

/**
 * Handle shooting at target coordinates
 * @param {number} targetX - World X coordinate to shoot at
 * @param {number} targetY - World Y coordinate to shoot at
 */
export function handleShoot(targetX, targetY) {
    // SUPER VISIBLE DIAGNOSTIC - Can't be cached!
    document.title = `SHOOT! ${Date.now()}`;
    console.log("=== DIAGNOSTIC: NEW CODE LOADED - handleShoot called ===", {
        timestamp: new Date().toISOString(),
        targetX,
        targetY,
        playerX: gameState.character?.x,
        playerY: gameState.character?.y
    });

    if (!gameState.character || !networkManager) {
        console.error("❌ Cannot shoot: character or network manager not available", {
            hasCharacter: !!gameState.character,
            hasNetworkManager: !!networkManager
        });
        return;
    }

    // Calculate bullet direction
    const playerX = gameState.character.x;
    const playerY = gameState.character.y;
    const dx = targetX - playerX;
    const dy = targetY - playerY;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance === 0) {
        console.warn("Cannot shoot: target is at player position");
        return;
    }

    // Normalize direction and set bullet speed
    const bulletSpeed = 10; // Adjust as needed
    const vx = (dx / distance) * bulletSpeed;
    const vy = (dy / distance) * bulletSpeed;

    // Convert to angle/speed and use sendShoot API expected by server
    const angle = Math.atan2(vy, vx);
    const speed = Math.sqrt(vx * vx + vy * vy);

    // Log shoot request
    console.log(`[CLIENT SHOOT REQUEST] Pos: (${playerX.toFixed(4)}, ${playerY.toFixed(4)}), Tile: (${Math.floor(playerX)}, ${Math.floor(playerY)}), Angle: ${angle.toFixed(2)}, Speed: ${speed.toFixed(2)}, Target: (${targetX.toFixed(2)}, ${targetY.toFixed(2)})`);

    // DIAGNOSTIC: Show gameState.character and what coordinates we're sending to server
    console.log(`[CLIENT SHOOT] gameState.character:`, {
        x: gameState.character.x,
        y: gameState.character.y,
        worldId: gameState.character.worldId
    });
    console.log(`[CLIENT SHOOT] Sending to server: x=${playerX}, y=${playerY}`);

    if (typeof networkManager.sendShoot === 'function') {
        networkManager.sendShoot({
            x: playerX,
            y: playerY,
            angle,
            speed,
            damage: 10
        });
    }
}



/**
 * Game class (for main.js)
 */
export class Game {
    constructor(serverUrl) {
        this.serverUrl = serverUrl;
        this.initialized = false;
        this.running = false;
    }

    async init() {
        try {
            if (this.initialized) return true;
            
            console.log(`Game: Initializing with server ${this.serverUrl}`);
            
            // Use the existing initGame function
            await initGame();
            
            this.initialized = true;
            this.running = true;
            
            return true;
        } catch (error) {
            console.error('Game: Failed to initialize:', error);
            return false;
        }
    }

    start() {
        if (!this.initialized) {
            console.warn('Game: Cannot start - not initialized');
            return;
        }
        console.log('Game: Starting...');
        this.running = true;
    }

    stop() {
        console.log('Game: Stopping...');
        this.running = false;
    }

    cleanup() {
        console.log('Game: Cleaning up...');
        this.running = false;
        // Add any cleanup logic here
        if (networkManager && typeof networkManager.disconnect === 'function') {
            networkManager.disconnect();
        }
    }
}

/**
 * Connect to the game server
 * @returns {Promise} Resolves when connected
 */
async function connectToServer() {
    try {
        console.log(`Attempting to connect to server at ${SERVER_URL}`);
        return new Promise((resolve, reject) => {
            if (networkManager && typeof networkManager.connect === 'function') {
                networkManager.connect()
                    .then(() => {
                        console.log('Successfully connected to server');
                        resolve();
                    })
                    .catch(reject);
            } else {
                reject(new Error('Network manager not initialized'));
            }
        });
    } catch (error) {
        console.error('Failed to connect to server:', error);
        throw error;
    }
}

// Ping test function - call from console: testPing(10)
window.testPing = function(count = 10) {
    if (!networkManager || !networkManager.ws || networkManager.ws.readyState !== WebSocket.OPEN) {
        console.error('❌ Not connected to server!');
        return;
    }

    console.log(`🏓 Starting ping test - sending ${count} pings...`);

    const stats = { sent: 0, received: 0, rtts: [], min: Infinity, max: 0 };
    const pendingPings = new Map();

    const originalOnMessage = networkManager.ws.onmessage;

    // Intercept PONG messages
    networkManager.ws.onmessage = (event) => {
        try {
            const packet = networkManager.BinaryPacket.decode(event.data);
            if (packet.type === MessageType.PONG) {
                const sentTime = packet.data.timestamp;
                if (pendingPings.has(sentTime)) {
                    const rtt = Date.now() - sentTime;
                    stats.received++;
                    stats.rtts.push(rtt);
                    stats.min = Math.min(stats.min, rtt);
                    stats.max = Math.max(stats.max, rtt);
                    pendingPings.delete(sentTime);
                    console.log(`📥 PONG #${stats.received}: RTT = ${rtt}ms`);

                    if (stats.received === count) {
                        printStats();
                    }
                }
            }
        } catch (err) {
            // Not a PONG message, pass to original handler
        }

        // Always call original handler
        if (originalOnMessage) originalOnMessage.call(networkManager.ws, event);
    };

    function printStats() {
        const avg = stats.rtts.reduce((a, b) => a + b, 0) / stats.rtts.length;
        const loss = ((stats.sent - stats.received) / stats.sent * 100).toFixed(1);

        console.log(`
╔════════════════════════════════════╗
║     PING TEST RESULTS              ║
╠════════════════════════════════════╣
║ Packets Sent:      ${stats.sent.toString().padStart(3)}            ║
║ Packets Received:  ${stats.received.toString().padStart(3)}            ║
║ Packet Loss:       ${loss}%          ║
║ Min RTT:           ${stats.min.toString().padStart(3)}ms          ║
║ Max RTT:           ${stats.max.toString().padStart(3)}ms          ║
║ Average RTT:       ${Math.round(avg).toString().padStart(3)}ms          ║
╚════════════════════════════════════╝
        `);

        // Restore original handler
        networkManager.ws.onmessage = originalOnMessage;
    }

    // Send pings
    for (let i = 0; i < count; i++) {
        setTimeout(() => {
            const timestamp = Date.now();
            const packet = networkManager.BinaryPacket.encode(MessageType.PING, { timestamp });
            networkManager.ws.send(packet);
            pendingPings.set(timestamp, true);
            stats.sent++;
            console.log(`📤 PING #${stats.sent} sent at ${timestamp}`);

            // Clean up after 5 seconds
            setTimeout(() => pendingPings.delete(timestamp), 5000);
        }, i * 100); // 100ms between each ping
    }

    // Print stats after all pings should have completed
    setTimeout(printStats, (count * 100) + 2000);
};

console.log('💡 Ping test available! Run: testPing(10) to send 10 pings');
