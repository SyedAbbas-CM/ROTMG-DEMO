// public/src/game/input.js

import { gameState } from './gamestate.js';
import { openSpriteEditor } from '../screens/spriteEditor.js';
import { handleShoot } from './game.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { Camera } from '../camera.js';

// Track currently pressed keys
export const keysPressed = {};

// Mouse position
let mouseX = 0;
let mouseY = 0;

// Sensitivity and speed settings
const MOUSE_SENSITIVITY = 0.002;
const MOVE_SPEED = 100.0; // Increased movement speed for better responsiveness

/**
 * Initialize game controls
 */
export function initControls() {
    // Clear any existing key states
    Object.keys(keysPressed).forEach(key => delete keysPressed[key]);
    
    // Keyboard input
    window.addEventListener('keydown', (e) => {
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
            openSpriteEditor();
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
    });

    window.addEventListener('keyup', (e) => {
        keysPressed[e.code] = false;
    });

    // Mouse movement for camera rotation in first-person view
    window.addEventListener('click', (e) => {
        if (gameState.camera.viewType === 'first-person') {
            const canvas = document.getElementById('glCanvas');
            if (document.pointerLockElement !== canvas) {
                canvas.requestPointerLock();
            }
        } else {
            // Handle shooting in top-down view
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

    // Add mouse click event for shooting
    window.addEventListener('click', (e) => {
        // Only handle clicks on canvas
        const targetId = e.target.id;
        if (targetId !== 'gameCanvas' && targetId !== 'glCanvas') {
            return;
        }
        
        console.log("Mouse click detected");
        
        // Get click position in game world
        const rect = e.target.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        // Convert to world coordinates based on view type
        let worldX, worldY;
        
        if (gameState.camera.viewType === 'first-person') {
            // In first-person view, shoot forward
            const cameraDir = gameState.camera.getDirection();
            worldX = gameState.character.x + cameraDir.x * 100;
            worldY = gameState.character.y + cameraDir.y * 100;
        } else {
            // In other views, convert screen coordinates to world coordinates
            worldX = clickX + gameState.camera.x;
            worldY = clickY + gameState.camera.y;
        }
        
        // Handle shooting
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
}

/**
 * Handle mouse movement for first-person camera
 * @param {MouseEvent} event - Mouse event
 */
function onMouseMove(event) {
    const sensitivity = MOUSE_SENSITIVITY;
    gameState.character.rotation.yaw -= event.movementX * sensitivity;
    gameState.character.rotation.pitch -= event.movementY * sensitivity;

    // Clamp the pitch to prevent flipping
    gameState.character.rotation.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, gameState.character.rotation.pitch));
}

/**
 * Update player rotation to face mouse cursor in top-down view
 * @param {MouseEvent} event - Mouse event
 */
function updatePlayerRotation(event) {
    // Get viewport dimensions
    const canvas = document.getElementById('gameCanvas');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Calculate angle between player and mouse
    const dx = event.clientX - centerX;
    const dy = event.clientY - centerY;
    const angle = Math.atan2(dy, dx);
    
    // Update player rotation - use rotateTo method if available
    if (gameState.character.rotateTo) {
        // Use the Player class method
        const targetX = gameState.character.x + dx;
        const targetY = gameState.character.y + dy;
        gameState.character.rotateTo(targetX, targetY);
    } else if (typeof gameState.character.rotation === 'number') {
        // Character has simple rotation as number
        gameState.character.rotation = angle;
    } else if (gameState.character.rotation && typeof gameState.character.rotation === 'object') {
        // Try to create a new rotation object to avoid readonly property error
        gameState.character.rotation = { yaw: angle };
    }
}

/**
 * Handle mouse click
 * @param {MouseEvent} event - Mouse event
 */
function handleMouseClick(event) {
    // Only allow shooting if not in first-person view
    if (gameState.camera.viewType === 'first-person') return;
    
    // Check if player can shoot
    if (gameState.character.canShoot && !gameState.character.canShoot()) {
        return;
    }
    
    // Convert screen position to world position
    const canvas = document.getElementById('gameCanvas');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Calculate target position using camera position and mouse offset
    const targetX = gameState.character.x + (event.clientX - centerX) / (TILE_SIZE * SCALE);
    const targetY = gameState.character.y + (event.clientY - centerY) / (TILE_SIZE * SCALE);
    
    // Call shoot handler
    handleShoot(targetX, targetY);
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
    console.log(`[switchView] Render functions available:
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
 * Handle view mode switching
 * Most likely toggles between first-person, top-down, and strategic views
 * Add debug logging to help diagnose view switching issues
 */
function toggleViewMode() {
  if (!gameState.camera) {
    console.error("Cannot toggle view: gameState.camera is not defined");
    return;
  }
  
  const currentView = gameState.camera.viewType;
  console.log(`[input] Toggling view from: ${currentView}`);
  
  // Cycle through view types
  if (gameState.camera.viewType === 'first-person') {
    gameState.camera.viewType = 'top-down';
    console.log(`[input] Switched to top-down view. render function available: ${typeof window.renderTopDownView === 'function'}`);
  } else if (gameState.camera.viewType === 'top-down') {
    gameState.camera.viewType = 'strategic';
    console.log(`[input] Switched to strategic view. render function available: ${typeof window.renderStrategicView === 'function'}`);
  } else {
    gameState.camera.viewType = 'first-person';
    console.log(`[input] Switched to first-person view`);
  }
  
  console.log(`[input] New view type: ${gameState.camera.viewType}`);
}