// src/render/renderTopDown.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_SPRITES } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';

// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

// Rendering parameters
const scaleFactor = 4; // Adjust scale as needed

export function renderTopDownView() {
  const camera = gameState.camera;
  const mapManager = gameState.map;

  if (!mapManager) {
    console.warn("Cannot render map: map manager not available");
    return;
  }

  // Clear handled in renderGame, don't do it here
  // ctx.clearRect(0, 0, canvas2D.width, canvas2D.height);

  // Determine visible tiles based on camera position
  const tilesInViewX = Math.ceil(canvas2D.width / TILE_SIZE);
  const tilesInViewY = Math.ceil(canvas2D.height / TILE_SIZE);

  const startX = Math.floor(camera.position.x / TILE_SIZE - tilesInViewX / 2);
  const startY = Math.floor(camera.position.y / TILE_SIZE - tilesInViewY / 2);
  const endX = startX + tilesInViewX;
  const endY = startY + tilesInViewY;
  const tileSheetObj = spriteManager.getSpriteSheet("tile_sprites"); // Name as defined in your config
  if (!tileSheetObj) return;
  const tileSpriteSheet = tileSheetObj.image;

  //console.log(`Rendering map from camera position (${Math.floor(camera.position.x)}, ${Math.floor(camera.position.y)})`);

  for (let y = startY; y <= endY; y++) {
    for (let x = startX; x <= endX; x++) {
      const tile = mapManager.getTile ? mapManager.getTile(x, y) : null;
      
      if (tile) {
        const spritePos = TILE_SPRITES[tile.type];
        // Draw tile
        ctx.drawImage(
          tileSpriteSheet,
          spritePos.x, spritePos.y, TILE_SIZE, TILE_SIZE, // Source rectangle
          (x * TILE_SIZE - camera.position.x) * scaleFactor + canvas2D.width / 2,
          (y * TILE_SIZE - camera.position.y) * scaleFactor + canvas2D.height / 2,
          TILE_SIZE * scaleFactor,
          TILE_SIZE * scaleFactor
        );
      }
    }
  }

  // Entities are rendered in the main renderGame function
  // No need to render them here
}

// Export to window object to avoid circular references
window.renderTopDownView = renderTopDownView;
