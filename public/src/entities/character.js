// src/entities/character.js

import { TILE_SIZE, CHARACTER_SPRITE_POSITIONS } from '../constants/constants.js';

export const character = {
  x: 12, // Starting X position in tile coordinates
  y: 12, // Starting Y position in tile coordinates
  z: 5.0,  // Increased Player height in the 3D world for first-person view
  speed: 5000, // Tile units per second
  spriteX: CHARACTER_SPRITE_POSITIONS.SPRITE_1.x,
  spriteY: CHARACTER_SPRITE_POSITIONS.SPRITE_1.y,
  width: TILE_SIZE,
  height: TILE_SIZE,
  rotation: {
    yaw: 0, // Rotation around Y-axis in radians
  },
  health: 100,
};
