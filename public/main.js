// main.js
import { Game } from './src/game/Game.js';

// Server URL (change for production)
const SERVER_URL = 'ws://localhost:3000';

// Create and initialize game
const game = new Game(SERVER_URL);

// Wait for DOM content to load
window.addEventListener('DOMContentLoaded', async () => {
  try {
    // Initialize game
    const success = await game.init();
    
    if (success) {
      console.log('Game initialized successfully');
    } else {
      console.error('Failed to initialize game');
      showConnectionError();
    }
  } catch (error) {
    console.error('Error initializing game:', error);
    showConnectionError();
  }
});

// Show connection error message
function showConnectionError() {
  const errorMessage = document.createElement('div');
  errorMessage.style.position = 'fixed';
  errorMessage.style.top = '50%';
  errorMessage.style.left = '50%';
  errorMessage.style.transform = 'translate(-50%, -50%)';
  errorMessage.style.padding = '20px';
  errorMessage.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
  errorMessage.style.color = 'white';
  errorMessage.style.borderRadius = '10px';
  errorMessage.style.zIndex = '1000';
  errorMessage.innerHTML = `
    <h2>Connection Error</h2>
    <p>Could not connect to the game server.</p>
    <p>Please check your connection and try again.</p>
    <button id="retryButton" style="padding: 8px 16px; margin-top: 10px;">Retry</button>
  `;
  
  document.body.appendChild(errorMessage);
  
  // Add retry button functionality
  document.getElementById('retryButton').addEventListener('click', async () => {
    document.body.removeChild(errorMessage);
    try {
      const success = await game.init();
      if (!success) {
        showConnectionError();
      }
    } catch (error) {
      console.error('Error reconnecting:', error);
      showConnectionError();
    }
  });
}

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // Page is hidden, pause game
    game.stop();
  } else {
    // Page is visible again, resume game
    game.start();
  }
});

// Handle window unload
window.addEventListener('beforeunload', () => {
  game.cleanup();
});