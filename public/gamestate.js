// gamestate.js

// Shared game state for enemies, bullets, and player.
export let enemies = [];
export let bullets = [];
export const player = {
  id: null,
  name: 'Player',
  x: 50,
  y: 50,
  r: 0.003,
  tx: 0,
  ty: 0,
};

// Utility function to reset game state (optional for future use).
export function resetGameState() {
  enemies = [];
  bullets = [];
}

// Utility function to update player position based on input (modularized).
export function updatePlayerPosition(dx, dy) {
  player.x += dx;
  player.y += dy;
}
