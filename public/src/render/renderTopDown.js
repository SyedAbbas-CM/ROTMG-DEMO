// src/render/renderTopDown.js

import { gameState } from '../game/gamestate.js';
import { map } from '../map/map.js';
import { UNIT_SIZE, TILE_SPRITES, SPRITE_SIZE } from '../constants/constants.js';
import { assets } from '../assets/assets.js';
import { renderCharacter, renderEnemies } from './render.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

// Rendering parameters
const TILE_PIXEL_SIZE = 32; // Size of each tile in pixels on the screen


export function renderTopDownView() {
  const camera = gameState.camera;

  // Clear the canvas
  ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);

  // Determine visible tiles based on camera position
  const tilesInViewX = Math.ceil(canvas2D.width / TILE_PIXEL_SIZE);
  const tilesInViewY = Math.ceil(canvas2D.height / TILE_PIXEL_SIZE);

  const startX = Math.floor(camera.position.x  - tilesInViewX / 2);
  const startY = Math.floor(camera.position.y  - tilesInViewY / 2);
  const endX = startX + tilesInViewX;
  const endY = startY + tilesInViewY;

  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      const tile = map.getTile(x, y);
      if (tile) {
        const spritePos = TILE_SPRITES[tile.type];
        // Draw tile
        ctx.drawImage(
          assets.tileSpriteSheet,
          spritePos.x, spritePos.y, SPRITE_SIZE, SPRITE_SIZE, // Source rectangle
          screenX,
          screenY,
          TILE_PIXEL_SIZE,
          TILE_PIXEL_SIZE
        );
      }
    }
  }

  // Draw character
  renderCharacter();

  // Draw enemies
  renderEnemies();

  console.log(`Top-Down View - Character Position: (${gameState.character.x}, ${gameState.character.y})`);
  console.log(`Top-Down View - Camera Position: (${gameState.camera.position.x}, ${gameState.camera.position.y})`);
}
