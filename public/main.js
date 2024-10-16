// main.js

import { loadAssets } from './assets.js';
import { initializeGame, gameLoop } from './game.js';
import { initializeInput } from './input.js';
import { canvas } from './render.js';

loadAssets()
  .then(() => {
    initializeInput(canvas);
    initializeGame(); // Now performs actual initialization
    gameLoop();
  })
  .catch(error => {
    console.error('Error loading assets:', error);
  });
