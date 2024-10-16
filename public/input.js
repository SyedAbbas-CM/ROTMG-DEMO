// input.js
import { player } from './gamestate.js';
import {
  offCenter,
  scan1,
  scan2,
  scanC1,
  scanC2,
  scanOC1,
  scanOC2,
  yCenter,
} from './constants.js';
import { updateScanValues } from './constants.js';

// === INPUT HANDLING ===
export let mouseX = 0,
  mouseY = 0;
export let mouseDown = false;
export const key = {
  W: false,
  A: false,
  S: false,
  D: false,
  Q: false,
  E: false,
  ' ': false,
};

// Function to get mouse position
function getMousePos(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

// Initialize input event listeners
export function initializeInput(canvas) {
  console.log("INPUT: INIT INPUT")
  // Mouse movement handling
  canvas.addEventListener('mousemove', event => {
    const pos = getMousePos(canvas, event);
    mouseX = pos.x;
    mouseY = pos.y;
  });

  // Mouse click handling
  canvas.addEventListener('mousedown', () => {
    mouseDown = true;
  });
  canvas.addEventListener('mouseup', () => {
    mouseDown = false;
  });

  // Scroll handling
  canvas.addEventListener('wheel', e => {
    // Handle scrolling for zooming or block selection
  });

  // Keyboard input handling
  document.addEventListener('keydown', e => {
    const k = e.key.toUpperCase();
    if (key.hasOwnProperty(k)) {
      key[k] = true;
    }
  });

  document.addEventListener('keyup', e => {
    const k = e.key.toUpperCase();

    // Update key state
    if (key.hasOwnProperty(k)) {
      key[k] = false;
    }

    // Handle game state transitions
    switch (k) {
      case 'T':
        alert('pause.');
        break;
      case 'Z':
        player.r = 0.0001;
        break;

      case 'X':
        offCenter = !offCenter;

        // Toggle scan and yCenter values based on offCenter
        if (offCenter) {
          scan1 = scanOC1;
          scan2 = scanOC2;
          yCenter = 450;
        } else {
          scan1 = scanC1;
          scan2 = scanC2;
          yCenter = 324;
        }
        break;
    }
  });
}
