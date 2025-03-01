// src/render/render.js

import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { spriteManager } from '../assets/spriteManager.js';
// Get 2D Canvas Context
const canvas2D = document.getElementById('gameCanvas');
const ctx = canvas2D.getContext('2d');

// Resize canvas to match window size
canvas2D.width = window.innerWidth;
canvas2D.height = window.innerHeight;

// Rendering parameters
const scaleFactor = SCALE;

export function renderCharacter() {
  const character = gameState.character;

   const charSheetObj = spriteManager.getSpriteSheet('character_sprites');
 if (!charSheetObj) return; // fallback or early return
 const characterSpriteSheet = charSheetObj.image;
  const width = character.width * scaleFactor;
  const height = character.height * scaleFactor;
  
  // Calculate screen position in top-down view (pixels)
  const x = (character.x * TILE_SIZE * scaleFactor) - (gameState.camera.position.x * TILE_SIZE * scaleFactor) + canvas2D.width / 2 - width / 2;
  const y = (character.y * TILE_SIZE * scaleFactor) - (gameState.camera.position.y * TILE_SIZE * scaleFactor) + canvas2D.height / 2 - height / 2;

  ctx.drawImage(
    characterSpriteSheet,
    character.spriteX, character.spriteY, character.width, character.height, // Source rectangle
    x,
    y,
    width,
    height
  );
}

export function renderEnemies() {
  const enemies = gameState.enemies;
   const enemySheetObj = spriteManager.getSpriteSheet('enemy_sprites');
 if (!enemySheetObj) return;
 const enemySpriteSheet = enemySheetObj.image;

  enemies.forEach(enemy => {
    const width = enemy.width * scaleFactor;
    const height = enemy.height * scaleFactor;

    const x = (enemy.x * TILE_SIZE * scaleFactor) - (gameState.camera.position.x * TILE_SIZE * scaleFactor) + canvas2D.width / 2 - width / 2;
    const y = (enemy.y * TILE_SIZE * scaleFactor) - (gameState.camera.position.y * TILE_SIZE * scaleFactor) + canvas2D.height / 2 - height / 2;

    ctx.drawImage(
      enemySpriteSheet,
      enemy.spriteX, enemy.spriteY, enemy.width, enemy.height, // Source rectangle
      x,
      y,
      width,
      height
    );
  });
}
