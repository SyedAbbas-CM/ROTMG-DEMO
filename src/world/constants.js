// Managers/world/constants.js

export const TILE_SIZE = 12; // Size of each tile in pixels (Oryx tiles are 12×12)
export const CHARACTER_SPRITE_SIZE = 10; // Size of each character sprite in pixels
export const SCALE = 3; // Scale factor for sprites
export const CANVAS_WIDTH = 600;
export const CANVAS_HEIGHT = 600;


export const TILE_IDS = {
  FLOOR: 0,
  WALL: 1,
  OBSTACLE: 2,
  WATER: 3,
  MOUNTAIN: 4,
};

// Sprite grid dimensions for tile sprite sheet
export const TILE_SPRITES_PER_ROW = 24;
export const TILE_SPRITES_PER_COLUMN = 11;

// Generate array for all tile sprites (no spacing)
export const ALL_TILE_SPRITES = [];
for (let row = 0; row < TILE_SPRITES_PER_COLUMN; row++) {
  for (let col = 0; col < TILE_SPRITES_PER_ROW; col++) {
    ALL_TILE_SPRITES.push({
      x: col * TILE_SIZE,
      y: row * TILE_SIZE,
    });
  }
}

// Map TILE_SPRITES with specific indices for each TILE_ID
export const TILE_SPRITES = {
  [TILE_IDS.FLOOR]: ALL_TILE_SPRITES[0],       // Example: Use the first sprite for FLOOR
  [TILE_IDS.WALL]: ALL_TILE_SPRITES[1],        // Use the second sprite for WALL
  [TILE_IDS.OBSTACLE]: ALL_TILE_SPRITES[2],    // Use the third sprite for OBSTACLE
  [TILE_IDS.WATER]: ALL_TILE_SPRITES[3],       // Use the fourth sprite for WATER
  [TILE_IDS.MOUNTAIN]: ALL_TILE_SPRITES[4],    // Use the fifth sprite for MOUNTAIN
};

// Character sprite sheet configuration
const CHARACTER_SPRITES_PER_ROW = 16;
const CHARACTER_SPRITES_PER_COLUMN = 15;
const CHARACTER_HORIZONTAL_SPACING = 0;
const CHARACTER_VERTICAL_SPACING = 0;

// Generate array for all character sprites with spacing
export const ALL_CHARACTER_SPRITES = [];
for (let row = 0; row < CHARACTER_SPRITES_PER_COLUMN; row++) {
  for (let col = 0; col < CHARACTER_SPRITES_PER_ROW; col++) {
    ALL_CHARACTER_SPRITES.push({
      x: col * (CHARACTER_SPRITE_SIZE + CHARACTER_HORIZONTAL_SPACING),
      y: row * (CHARACTER_SPRITE_SIZE + CHARACTER_VERTICAL_SPACING),
    });
  }
}

// Numbered CHARACTER_SPRITE_POSITIONS
export const CHARACTER_SPRITE_POSITIONS = {};
ALL_CHARACTER_SPRITES.forEach((sprite, index) => {
  CHARACTER_SPRITE_POSITIONS[`SPRITE_${index + 1}`] = sprite;
});

// Enemy sprite positions within enemySpriteSheet
export const ENEMY_SPRITE_POSITIONS = {
  DEFAULT: { x: 0, y: 0 },
};

// Wall sprite positions within wallSpriteSheet
export const WALL_SPRITE_POSITIONS = {
  DEFAULT: { x: 0, y: 0 },
};

// Chunking Constants
export const CHUNK_SIZE = 16; // Size of each chunk (e.g., 16x16 tiles)
