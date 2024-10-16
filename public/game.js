// game.js

import {
  tSize,
  xCenter,
  yCenter,
  offCenter,
  offCenterDiff,
  scan1,
  scan2,
  WALL_SIZE,
  screen,
  min,
  max,
} from './constants.js';

import { player as p, enemies, bullets } from './gamestate.js';
import { key, mouseX, mouseY, mouseDown } from './input.js';
import { map, mapSize, texMap, adjacentNWES } from './map.js';
import { speed, updateEnemies, updateBullets } from './entities.js';
import { rankArray } from './utils.js';
import { renderEnemies, renderBullets, renderMap, gctx } from './render.js';

// === GAME LOOP ===
export function gameLoop() {
  console.log("main game loop")
  // Update player rotation based on input
  if (key.E) p.r -= 0.0233;
  if (key.Q) p.r += 0.0233;

  const sin = Math.sin(p.r);
  const cos = Math.cos(p.r);

  // Determine speed based on diagonal movement
  let currentSpeed = (key.W + key.A + key.S + key.D) > 1 ? speed * 0.707 : speed;

  // Update player position based on input
  if (key.W) {
    let dx = Math.cos(-1.5708 - p.r);
    let dy = Math.sin(-1.5708 - p.r);
    mapCollision(dx, dy, currentSpeed);
  }
  if (key.S) {
    let dx = Math.cos(1.5708 - p.r);
    let dy = Math.sin(1.5708 - p.r);
    mapCollision(dx, dy, currentSpeed);
  }
  if (key.D) mapCollision(cos, -sin, currentSpeed);
  if (key.A) mapCollision(-cos, sin, currentSpeed);

  // Update game entities
  updateEnemies();
  updateBullets();

  // Render the game
  renderGame();
  ctx.drawImage(gameCanvas, 0, 0);
  // Continue the game loop
  requestAnimationFrame(gameLoop);
}

export function mapCollision(dx, dy, currentSpeed) {
  let x = dx * currentSpeed;
  let y = dy * currentSpeed;
  let sizeX = x > 0 ? 0.25 : -0.25;
  let sizeY = y > 0 ? 0.25 : -0.25;

  // Collision detection logic...
  if (
    !texMap.get(map[Math.round(p.x + x + sizeX) + mapSize * Math.round(p.y)]).solid &&
    p.x + x > 0 &&
    p.x + x < mapSize - 1
  ) {
    p.x += x;
  }

  if (
    !texMap.get(map[Math.round(p.x) + mapSize * Math.round(p.y + y + sizeY)]).solid &&
    p.y + y > 0 &&
    p.y + y < mapSize - 1
  ) {
    p.y += y;
  }
}

function renderGame() {
  // Clear the canvas
  gctx.clearRect(0, 0, screen, screen);

  // Render the map
  renderMap();

  // Render enemies and bullets
  renderEnemies();
  renderBullets();

  // Render the player (assuming you have a function for this)
  // renderPlayer(); // Implement this function if necessary

  // Additional rendering (e.g., UI elements)
}

// === INITIALIZE GAME ===
export function initializeGame() {
  gctx.font = '25px agj';
  console.log("Game initialized");
}
