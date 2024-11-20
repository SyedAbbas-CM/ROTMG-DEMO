// main.js

import { initGame } from './src/game/game.js';

async function init() {
  try {
    await initGame(); // Initialize the game
  } catch (error) {
    console.error('Error during game initialization:', error);
  }
}

window.addEventListener('DOMContentLoaded', init);
