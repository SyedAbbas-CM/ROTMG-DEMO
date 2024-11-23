// src/entities/character.js

import { SPRITE_SIZE, CHARACTER_SPRITE_POSITIONS } from '../constants/constants.js';

export const character = {
  x: 62.5, // Starting X position in tile coordinates
  y: 62.5, // Starting Y position in tile coordinates
  z: 5.0,  // Increased Player height in the 3D world for first-person view
  speed: 50, // Tile units per second
  spriteX: CHARACTER_SPRITE_POSITIONS.SPRITE_1.x,
  spriteY: CHARACTER_SPRITE_POSITIONS.SPRITE_1.y,
  width: SPRITE_SIZE,
  height: SPRITE_SIZE,
  rotation: {
    yaw: 0, // Rotation around Y-axis in radians
  },
  health: 100,
};
