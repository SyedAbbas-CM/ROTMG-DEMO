// src/game/gamestate.js

import { Camera } from '../camera.js';
import { character } from '../entities/character.js';
import { map } from '../map/map.js'; // Import the map instance

export const gameState = {
  character: character,
  camera: new Camera('top-down', { x: character.x, y: character.y }, 1),
  enemies: [], // Initialize as an empty array
  projectiles: [],
  map: map, // Assign the map instance to gameState
  lastUpdateX: character.x, // Initialize lastUpdateX
  lastUpdateY: character.y, // Initialize lastUpdateY
};
