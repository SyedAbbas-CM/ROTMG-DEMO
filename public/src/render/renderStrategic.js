// src/render/renderStrategic.js

import { gameState } from '../game/gamestate.js';
import { map } from '../map/map.js';
import { SPRITE_SIZE, TILE_SPRITES } from '../constants/constants.js';
import { assets } from '../assets/assets.js';
import { renderCharacter, renderEnemies } from './render.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

// Rendering parameters
const scaleFactor = 0.5; // Reduce tile size for strategic view

export function renderStrategicView() {
  const camera = gameState.camera;

  // Clear the canvas
  ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);

  // Determine visible tiles based on camera position
  const tilesInViewX = Math.ceil(canvas2D.width / (SPRITE_SIZE * scaleFactor));
  const tilesInViewY = Math.ceil(canvas2D.height / (SPRITE_SIZE * scaleFactor));

  const startX = Math.floor(camera.position.x / SPRITE_SIZE - tilesInViewX / 2);
  const startY = Math.floor(camera.position.y / SPRITE_SIZE - tilesInViewY / 2);
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
          (x * SPRITE_SIZE - camera.position.x) * scaleFactor + canvas2D.width / 2,
          (y * SPRITE_SIZE - camera.position.y) * scaleFactor + canvas2D.height / 2,
          SPRITE_SIZE * scaleFactor,
          SPRITE_SIZE * scaleFactor
        );
      }
    }
  }

  // Draw character
  renderCharacter();

  // Draw enemies
  renderEnemies();
}
