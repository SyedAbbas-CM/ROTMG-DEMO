// src/game/input.js

import { gameState } from './gamestate.js';
import { openSpriteEditor } from '../screens/spriteEditor.js';
export const keysPressed = {};

// Sensitivity and speed settings
const MOUSE_SENSITIVITY = 0.002;
const MOVE_SPEED = 40.0; // Units per second

export function initControls() {
  // Keyboard input
  window.addEventListener('keydown', (e) => {
    keysPressed[e.code] = true;

    // Handle view switching with 'V' key
    if (e.code === 'KeyV') {
      switch (gameState.camera.viewType) {
        case 'top-down':
          gameState.camera.viewType = 'strategic';
          toggleViews();
          break;
        case 'first-person':
          gameState.camera.viewType = 'strategic';
          toggleViews();
          break;
        case 'strategic':
          gameState.camera.viewType = 'top-down';
          toggleViews();
          break;
        default:
          gameState.camera.viewType = 'top-down';
          toggleViews();
          break;
      }
      e.preventDefault(); // Prevent default only for the 'V' key
    }
    if (e.code === 'KeyE') {
      e.preventDefault();
      openSpriteEditor();
    }
    // Do not prevent default for other keys
  });

  window.addEventListener('keyup', (e) => {
    keysPressed[e.code] = false;
    // Do not prevent default for keyup events
  });

  // Mouse movement for camera rotation in first-person view
  window.addEventListener('click', () => {
    if (gameState.camera.viewType === 'first-person') {
      const canvas = document.getElementById('glCanvas');
      if (document.pointerLockElement !== canvas) {
        canvas.requestPointerLock();
      }
    }
  });

  document.addEventListener('pointerlockchange', () => {
    const canvas = document.getElementById('glCanvas');
    if (document.pointerLockElement === canvas) {
      document.addEventListener('mousemove', onMouseMove, false);
    } else {
      document.removeEventListener('mousemove', onMouseMove, false);
    }
  });
}

function onMouseMove(event) {
  const sensitivity = MOUSE_SENSITIVITY;
  gameState.character.rotation.yaw -= event.movementX * sensitivity;
  gameState.character.rotation.pitch -= event.movementY * sensitivity;

  // Clamp the pitch to prevent flipping
  gameState.character.rotation.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, gameState.character.rotation.pitch));

  // Optional: Log yaw and pitch for debugging
  // console.log(`Yaw: ${gameState.character.rotation.yaw.toFixed(2)}, Pitch: ${gameState.character.rotation.pitch.toFixed(2)}`);
}

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

export function getKeysPressed() {
  return keysPressed;
}

export function getMoveSpeed() {
  return MOVE_SPEED;
}
