// public/src/game/input.js

import { gameState } from './gamestate.js';
import { openSpriteEditor } from '../screens/spriteEditor.js';
import { handleShoot as handleShootOriginal } from './game.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { Camera } from '../camera.js';
import { createLogger, LOG_LEVELS, setGlobalLogLevel } from '../utils/logger.js';
import { MessageType } from '../shared/messages.js';

// INLINED handleShoot - BYPASS CACHE COMPLETELY
function handleShoot(targetX, targetY) {
    // SUPER VISIBLE DIAGNOSTIC - Can't be cached!
    // document.title = `🔥 INLINED! ${Date.now()}`;
    // console.error("🔥🔥🔥 === INLINED handleShoot called ===", {
    //     timestamp: new Date().toISOString(),
    //     targetX,
    //     targetY,
    //     playerX: gameState.character?.x,
    //     playerY: gameState.character?.y
    // });

    // Get networkManager from window
    const networkManager = window.networkManager || window.gameState?.networkManager;

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
    const bulletSpeed = 10;
    const vx = (dx / distance) * bulletSpeed;
    const vy = (dy / distance) * bulletSpeed;

    // Convert to angle/speed
    const angle = Math.atan2(vy, vx);
    const speed = Math.sqrt(vx * vx + vy * vy);

    // Log shoot request
    // console.error(`🔥 [INLINED SHOOT] Pos: (${playerX.toFixed(4)}, ${playerY.toFixed(4)}), Angle: ${angle.toFixed(2)}, Speed: ${speed.toFixed(2)}, Target: (${targetX.toFixed(2)}, ${targetY.toFixed(2)})`);

    if (typeof networkManager.sendShoot === 'function') {
        // console.error('🔥 [INLINED] About to call networkManager.sendShoot()');
        networkManager.sendShoot({
            x: playerX,
            y: playerY,
            angle,
            speed,
            damage: 10
        });
        // console.error('🔥 [INLINED] networkManager.sendShoot() completed');
    } else {
        // console.error('❌ networkManager.sendShoot is not a function!', typeof networkManager.sendShoot);
    }
}

// Create a logger for this module
const logger = createLogger('input');

// Track currently pressed keys
let keysPressed = {};

// Mouse position
let mouseX = 0;
let mouseY = 0;

// RTS command mode state (toggled with Tab key)
let rtsCommandMode = false;

// Sensitivity and speed settings
const MOUSE_SENSITIVITY = 0.002;
const MOVE_SPEED = 6.0; // Very slow movement speed for ROTMG-like feel

/**
 * Check if RTS command mode is active (Tab toggle OR shift held temporarily)
 */
function isRTSCommandMode(event) {
    return rtsCommandMode || event?.shiftKey;
}

/**
 * Initialize game controls
 */
export function initControls() {
    // SUPER VISIBLE: Prove initControls is called
    document.title = '🎮 INPUT.JS LOADED!';
    console.error('🎮🎮🎮 [INPUT.JS] initControls() CALLED AT:', new Date().toISOString());

    // Expose isInRTSMode to window for UI rendering
    window.isInRTSMode = isInRTSMode;

    // Clear any existing key states
    Object.keys(keysPressed).forEach(key => delete keysPressed[key]);
    
    // Keyboard input
    window.addEventListener('keydown', (e) => {
        // Toggle RTS command mode with Tab key
        if (e.code === 'Tab') {
            e.preventDefault(); // Prevent default tab behavior
            rtsCommandMode = !rtsCommandMode;
            console.log(`[RTS MODE] ${rtsCommandMode ? 'ENABLED' : 'DISABLED'} (press Tab to toggle)`);

            // Clear selections when disabling RTS mode
            if (!rtsCommandMode && gameState.selectedUnits) {
                gameState.selectedUnits = [];
            }
            return;
        }

        // Check for Enter key to focus chat input
        if (e.code === 'Enter') {
            // Find the chat input element
            const chatInput = document.querySelector('.chat-input');
            if (chatInput) {
                e.preventDefault();
                chatInput.focus();
                return;
            }
        }

        // Skip game controls if chat input is active
        if (window.chatInputActive) {
            return;
        }

        keysPressed[e.code] = true;
        // Remove excessive logging
        // console.log(`Key pressed: ${e.code}`); // Debug log

        // Handle view switching with 'V' key
        if (e.code === 'KeyV') {
            switchView();
            e.preventDefault();
        }
        
        // Sprite editor with 'E' key
        if (e.code === 'KeyE') {
            e.preventDefault();
            if (e.ctrlKey) {
                // Ctrl+E keeps sprite editor shortcut
                openSpriteEditor();
            } else {
                // Plain E → interact (portal)
                if (window.networkManager?.sendPortalEnter) {
                    window.networkManager.sendPortalEnter();
                }
            }
        }
        
        // Add spacebar to also trigger shooting
        if (e.code === 'Space') {
            // Shoot in the direction the player is facing
            const rotation = typeof gameState.character.rotation === 'object' ?
                   gameState.character.rotation.yaw || 0 :
                   gameState.character.rotation;
                   
            const distance = 100; // Distance ahead of the player to aim
            const targetX = gameState.character.x + Math.cos(rotation) * distance;
            const targetY = gameState.character.y + Math.sin(rotation) * distance;
            
            handleShoot(targetX, targetY);
        }

        // Add F3 key to toggle debug mode
        if (e.code === 'F3') {
          if (gameState.camera) {
            const debugEnabled = gameState.camera.toggleDebugMode();
            logger.info(`Debug mode ${debugEnabled ? 'enabled' : 'disabled'}`);
            
            // Also toggle debug overlay if it exists
            if (window.debugOverlay && typeof window.debugOverlay.toggle === 'function') {
              window.debugOverlay.toggle();
            }
          }
        }
        
        // Add F4 key to toggle logging verbosity
        if (e.code === 'F4') {
            // If current log level is set to VERBOSE, set it to INFO
            // Otherwise, set it to VERBOSE for troubleshooting
            const currentLevel = window.gameLogger?.currentLevel || LOG_LEVELS.INFO;
            const newLevel = currentLevel === LOG_LEVELS.VERBOSE ? LOG_LEVELS.INFO : LOG_LEVELS.VERBOSE;

            setGlobalLogLevel(newLevel);
            window.gameLogger.currentLevel = newLevel;

            logger.info(`Logging verbosity set to ${newLevel === LOG_LEVELS.INFO ? 'NORMAL' : 'VERBOSE'}`);
        }

        // Unit spawning controls
        if (e.code === 'Digit1' && !e.ctrlKey && !e.shiftKey) {
            // Spawn Light Infantry
            if (window.networkManager) {
                window.networkManager.send(MessageType.UNIT_SPAWN, {
                    unitType: 'INFANTRY_LIGHT',
                    x: gameState.character.x + 2,
                    y: gameState.character.y
                });
                console.log('Spawning Light Infantry');
            }
        }
        if (e.code === 'Digit2' && !e.ctrlKey && !e.shiftKey) {
            // Spawn Archer
            if (window.networkManager) {
                window.networkManager.send(MessageType.UNIT_SPAWN, {
                    unitType: 'ARCHER_LIGHT',
                    x: gameState.character.x + 2,
                    y: gameState.character.y
                });
                console.log('Spawning Archer');
            }
        }
        if (e.code === 'Digit3' && !e.ctrlKey && !e.shiftKey) {
            // Spawn Light Cavalry
            if (window.networkManager) {
                window.networkManager.send(MessageType.UNIT_SPAWN, {
                    unitType: 'CAVALRY_LIGHT',
                    x: gameState.character.x + 2,
                    y: gameState.character.y
                });
                console.log('Spawning Light Cavalry');
            }
        }
        if (e.code === 'Digit4' && !e.ctrlKey && !e.shiftKey) {
            // Spawn Heavy Infantry
            if (window.networkManager) {
                window.networkManager.send(MessageType.UNIT_SPAWN, {
                    unitType: 'INFANTRY_HEAVY',
                    x: gameState.character.x + 2,
                    y: gameState.character.y
                });
                console.log('Spawning Heavy Infantry');
            }
        }
    });

    window.addEventListener('keyup', (e) => {
        // Skip game controls if chat input is active
        if (window.chatInputActive) {
            return;
        }

        keysPressed[e.code] = false;
    });

    // Mouse movement for camera rotation in first-person view
    window.addEventListener('click', (e) => {
        // console.error('🎯 [CLICK LISTENER #1] Fired!', {
        //     viewType: gameState.camera?.viewType,
        //     targetId: e.target.id,
        //     isFirstPerson: gameState.camera?.viewType === 'first-person'
        // });

        if (gameState.camera.viewType === 'first-person') {
            const canvas = document.getElementById('glCanvas');
            if (document.pointerLockElement !== canvas) {
                canvas.requestPointerLock();
            }
        } else {
            // Handle shooting in top-down view
            // console.error('🎯 [CLICK LISTENER #1] Calling handleMouseClick');
            handleMouseClick(e);
        }
    });

    // Track mouse position
    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        // Only update rotation when in first-person view
        if (gameState.camera.viewType === 'first-person') {
            // This will be handled by the onMouseMove function
        }
        // Removed automatic rotation for top-down view
    });

    // Handle pointer lock for first-person mode
    document.addEventListener('pointerlockchange', () => {
        const canvas = document.getElementById('glCanvas');
        if (document.pointerLockElement === canvas) {
            document.addEventListener('mousemove', onMouseMove, false);
        } else {
            document.removeEventListener('mousemove', onMouseMove, false);
        }
    });

    // Add mouse click event for shooting (first-person only – top-down is handled by handleMouseClick above)
    window.addEventListener('click', (e) => {
        const targetId = e.target.id;
        // console.error('🎯 [CLICK LISTENER #2] Fired!', {
        //     targetId,
        //     viewType: gameState.camera?.viewType,
        //     isCanvas: targetId === 'gameCanvas' || targetId === 'glCanvas'
        // });

        if (targetId !== 'gameCanvas' && targetId !== 'glCanvas') {
            // console.warn('⚠️ [CLICK LISTENER #2] Ignored: not canvas');
            return;
        }

        // If we're not in first-person, let handleMouseClick() (bound earlier) handle shooting
        if (gameState.camera.viewType !== 'first-person') {
            // console.warn('⚠️ [CLICK LISTENER #2] Ignored: not first-person');
            return;
        }
        
        const chatInput = document.querySelector('.chat-input');
        if (chatInput && document.activeElement === chatInput) return;
        
        const cameraDir = gameState.camera.getDirection();
        const worldX = gameState.character.x + cameraDir.x * 100;
        const worldY = gameState.character.y + cameraDir.y * 100;
        handleShoot(worldX, worldY);
    });

    // Add touch event listener for trackpad support
    window.addEventListener('touchstart', (e) => {
        // Prevent default to avoid scrolling
        e.preventDefault();

        // Get touch position
        const touch = e.touches[0];
        const targetId = e.target.id;
        if (targetId !== 'gameCanvas' && targetId !== 'glCanvas') {
            return;
        }

        // Find and blur the chat input if it exists and is focused
        const chatInput = document.querySelector('.chat-input');
        if (chatInput && document.activeElement === chatInput) {
            chatInput.blur();
            return; // Don't process the touch further if we were in chat
        }

        // Get click position in game world
        const rect = e.target.getBoundingClientRect();
        const touchX = touch.clientX - rect.left;
        const touchY = touch.clientY - rect.top;

        // Convert to world coordinates
        let worldX, worldY;

        if (gameState.camera.viewType === 'first-person') {
            // In first-person, shoot forward
            const cameraDir = gameState.camera.getDirection();
            worldX = gameState.character.x + cameraDir.x * 100;
            worldY = gameState.character.y + cameraDir.y * 100;
        } else {
            // In other views, convert screen coordinates to world coordinates
            worldX = touchX + gameState.camera.x;
            worldY = touchY + gameState.camera.y;
        }

        // Handle shooting
        handleShoot(worldX, worldY);
    }, { passive: false });

    // Add right-click handler for unit commands
    window.addEventListener('contextmenu', (e) => {
        const targetId = e.target.id;
        if (targetId !== 'gameCanvas' && targetId !== 'glCanvas') {
            return; // Allow normal context menu for non-game elements
        }

        // Check if in RTS command mode
        const rtsMode = isRTSCommandMode(e);

        // Only prevent default context menu if in RTS mode
        if (rtsMode) {
            e.preventDefault();
        } else {
            return; // Allow normal context menu in non-RTS mode
        }

        // Only handle in top-down or strategic view
        if (gameState.camera.viewType === 'first-person') {
            return;
        }

        // Check if we have selected units
        if (!gameState.selectedUnits || gameState.selectedUnits.length === 0) {
            return;
        }

        try {
            // Convert screen position to world position
            const canvas = document.getElementById('gameCanvas');
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            // Get camera scale factor for current view
            const scaleFactor = gameState.camera.getViewScaleFactor();

            // Convert screen coordinates to world coordinates
            const targetX = (e.clientX - centerX) / (TILE_SIZE * scaleFactor) + gameState.camera.position.x;
            const targetY = (e.clientY - centerY) / (TILE_SIZE * scaleFactor) + gameState.camera.position.y;

            // Send move command to server
            if (window.networkManager) {
                window.networkManager.send(MessageType.UNIT_COMMAND, {
                    unitIds: gameState.selectedUnits,
                    command: 'move',
                    targetX: targetX,
                    targetY: targetY
                });
                console.log(`Commanding ${gameState.selectedUnits.length} units to move to (${targetX.toFixed(2)}, ${targetY.toFixed(2)})`);
            }
        } catch (error) {
            console.error('Error in contextmenu handler:', error);
        }
    });
}

/**
 * Handle mouse movement for first-person camera
 * @param {MouseEvent} event - Mouse event
 */
function onMouseMove(event) {
    const character = gameState.character;
    if (!character) return;
    // Ensure rotation object exists – it may still be 0 right after connect
    if (typeof character.rotation !== 'object' || character.rotation === null) {
        character.rotation = { yaw: 0, pitch: 0 };
    }
    const sensitivity = MOUSE_SENSITIVITY;
    character.rotation.yaw -= event.movementX * sensitivity;
    character.rotation.pitch -= event.movementY * sensitivity;

    // Clamp the pitch to prevent flipping
    character.rotation.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, character.rotation.pitch));
}

/**
 * Update player rotation to face mouse cursor in top-down view
 * @param {MouseEvent} event - Mouse event
 */

/**
 * Handle mouse click
 * @param {MouseEvent} event - Mouse event
 */
function handleMouseClick(event) {
    // Only allow actions if not in first-person view
    if (gameState.camera.viewType === 'first-person') {
        console.warn('⚠️ Click ignored: first-person view');
        return;
    }

    try {
        // Convert screen position to world position
        const canvas = document.getElementById('gameCanvas');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;

        // Get camera scale factor for current view
        const scaleFactor = gameState.camera.getViewScaleFactor();

        // Convert screen coordinates to world coordinates
        const worldX = (event.clientX - centerX) / (TILE_SIZE * scaleFactor) + gameState.camera.position.x;
        const worldY = (event.clientY - centerY) / (TILE_SIZE * scaleFactor) + gameState.camera.position.y;

        // Check if in RTS command mode (Tab toggle or shift held)
        const rtsMode = isRTSCommandMode(event);

        // Debug logging
        if (Math.random() < 0.1) { // Log occasionally to avoid spam
            console.log(`[CLICK] RTS Mode: ${rtsMode}, Tab Toggle: ${rtsCommandMode}, Shift: ${event.shiftKey}`);
        }

        // Only allow unit selection in RTS mode
        if (rtsMode && gameState.unitManager && event.button === 0) {
            const clickRadius = 0.5; // Click tolerance in world units
            const nearbyUnits = gameState.unitManager.getUnitsInRadius(worldX, worldY, clickRadius);

            if (nearbyUnits.length > 0) {
                // Sort by distance and select the closest
                nearbyUnits.sort((a, b) => a.distance - b.distance);
                const selectedUnit = nearbyUnits[0];

                // Toggle selection (shift = add to selection, normal = replace selection)
                if (event.shiftKey) {
                    const idx = gameState.selectedUnits.indexOf(selectedUnit.id);
                    if (idx === -1) {
                        gameState.selectedUnits.push(selectedUnit.id);
                    } else {
                        gameState.selectedUnits.splice(idx, 1);
                    }
                } else {
                    gameState.selectedUnits = [selectedUnit.id];
                }

                console.log(`Selected units: ${gameState.selectedUnits.join(', ')}`);
                return; // Don't shoot if we selected a unit
            }
        }

        // In RTS mode with no unit clicked, deselect all
        if (rtsMode && event.button === 0) {
            if (gameState.selectedUnits.length > 0) {
                gameState.selectedUnits = [];
                console.log('Deselected all units');
            }
            return; // Don't shoot in RTS mode
        }

        // Normal mode - allow shooting
        // Check if player can shoot
        if (gameState.character.canShoot && !gameState.character.canShoot()) {
            console.warn('⚠️ Click ignored: canShoot() returned false');
            return;
        }

        // Handle shooting
        handleShoot(worldX, worldY);
    } catch (error) {
        console.error('Error in handleMouseClick:', error);
        console.error('Error stack:', error.stack);
    }
}

/**
 * Switch between view modes (top-down, first-person, strategic)
 */
function switchView() {
    // Call the enhanced debugging function we defined
    toggleViewMode();
    
    // Toggle the canvas visibility
    toggleViews();
    
    // Log available render functions for debugging
    logger.debug(`Render functions available:
    - Top-down: ${typeof window.renderTopDownView === 'function'}
    - Strategic: ${typeof window.renderStrategicView === 'function'}
    Current view: ${gameState.camera.viewType}`);
}

/**
 * Toggle between 2D and 3D canvases
 */
function toggleViews() {
    const canvas2D = document.getElementById('gameCanvas');
    const canvas3D = document.getElementById('glCanvas');

    if (gameState.camera.viewType === 'first-person') {
        canvas2D.style.display = 'none';
        canvas3D.style.display = 'block';
    } else {
        canvas2D.style.display = 'block';
        canvas3D.style.display = 'none';
    }
}

/**
 * Get currently pressed keys
 * @returns {Object} Keys pressed state
 */
export function getKeysPressed() {
    return keysPressed;
}

/**
 * Get movement speed
 * @returns {number} Movement speed
 */
export function getMoveSpeed() {
    return MOVE_SPEED;
}

/**
 * Get current mouse position
 * @returns {Object} Mouse position {x, y}
 */
export function getMousePosition() {
    return { x: mouseX, y: mouseY };
}

/**
 * Check if RTS mode is currently active
 * @returns {boolean} True if Tab toggle is enabled
 */
export function isInRTSMode() {
    return rtsCommandMode;
}

/**
 * Handle view mode switching
 * Most likely toggles between first-person, top-down, and strategic views
 * Add debug logging to help diagnose view switching issues
 */
function toggleViewMode() {
  if (!gameState.camera) {
    logger.error("Cannot toggle view: gameState.camera is not defined");
    return;
  }
  
  const currentView = gameState.camera.viewType;
  logger.info(`Toggling view from: ${currentView}`);
  
  // Cycle through view types
  if (gameState.camera.viewType === 'first-person') {
    gameState.camera.viewType = 'top-down';
    
    logger.debug(`Switched to top-down view. render function available: ${typeof window.renderTopDownView === 'function'}`);
  } else if (gameState.camera.viewType === 'top-down') {
    gameState.camera.viewType = 'strategic';
    logger.debug(`Switched to strategic view. render function available: ${typeof window.renderStrategicView === 'function'}`);
  } else {
    gameState.camera.viewType = 'first-person';
    logger.debug(`Switched to first-person view`);
  }
  
  logger.info(`New view type: ${gameState.camera.viewType}`);
}