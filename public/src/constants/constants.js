// src/constants/constants.js

export const TILE_SIZE = 12; // Size of each tile in pixels (Oryx tiles are 12×12)
export const CHARACTER_SPRITE_SIZE = 12; // Size of each character sprite in pixels
export const SCALE = 5; // Scale factor for sprites
export const SCALE_S = 2;


export const CANVAS_WIDTH = window.innerWidth;
export const CANVAS_HEIGHT = window.innerHeight;


export const UNIT_MODE = 'tiles'; 
export const toPixels   = v => v * TILE_SIZE;
export const fromPixels = v => v / TILE_SIZE;
// hit-box scaling
export const HITBOX_SCALE_BULLET = 0.5;
export const HITBOX_SCALE_ENEMY  = 0.6;

// ENHANCED: Expanded tile type IDs with more descriptive names and additional types
export const TILE_IDS = {
  // Base types
  FLOOR: 0,
  WALL: 1,
  OBSTACLE: 2,
  WATER: 3,
  MOUNTAIN: 4,
  
  // Extended types for more variety
  SAND: 5,
  LAVA: 6,
  ICE: 7,
  GRASS: 8,
  FOREST: 9,
  ROAD: 10,
  STONE: 11,
  
  // Add additional tile types here as needed
  // Every type represents a different collision behavior and appearance
};

// ENHANCED: Properties for each tile type for consistent behavior
export const TILE_PROPERTIES = {
  [TILE_IDS.FLOOR]: { isWalkable: true, isTransparent: true, movementCost: 1.0 },
  [TILE_IDS.WALL]: { isWalkable: false, isTransparent: false, movementCost: Infinity },
  [TILE_IDS.OBSTACLE]: { isWalkable: false, isTransparent: true, movementCost: Infinity },
  [TILE_IDS.WATER]: { isWalkable: false, isTransparent: true, movementCost: 2.0 },
  [TILE_IDS.MOUNTAIN]: { isWalkable: false, isTransparent: true, movementCost: Infinity },
  [TILE_IDS.SAND]: { isWalkable: true, isTransparent: true, movementCost: 1.5 },
  [TILE_IDS.LAVA]: { isWalkable: false, isTransparent: true, movementCost: Infinity },
  [TILE_IDS.ICE]: { isWalkable: true, isTransparent: true, movementCost: 0.8 },
  [TILE_IDS.GRASS]: { isWalkable: true, isTransparent: true, movementCost: 1.2 },
  [TILE_IDS.FOREST]: { isWalkable: true, isTransparent: true, movementCost: 1.8 },
  [TILE_IDS.ROAD]: { isWalkable: true, isTransparent: true, movementCost: 0.9 },
  [TILE_IDS.STONE]: { isWalkable: true, isTransparent: true, movementCost: 1.3 },
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

// ENHANCED: More flexible sprite mapping with variations
// Maps TILE_IDS to sprite indices, now supporting variations
export const TILE_SPRITES = {
  // Base types - primary sprites
  [TILE_IDS.FLOOR]: ALL_TILE_SPRITES[0],
  [TILE_IDS.WALL]: ALL_TILE_SPRITES[1],
  [TILE_IDS.OBSTACLE]: ALL_TILE_SPRITES[2],
  [TILE_IDS.WATER]: ALL_TILE_SPRITES[3],
  [TILE_IDS.MOUNTAIN]: ALL_TILE_SPRITES[4],
  
  // New types
  [TILE_IDS.SAND]: ALL_TILE_SPRITES[5],
  [TILE_IDS.LAVA]: ALL_TILE_SPRITES[6],
  [TILE_IDS.ICE]: ALL_TILE_SPRITES[7],
  [TILE_IDS.GRASS]: ALL_TILE_SPRITES[8],
  [TILE_IDS.FOREST]: ALL_TILE_SPRITES[9],
  [TILE_IDS.ROAD]: ALL_TILE_SPRITES[10],
  [TILE_IDS.STONE]: ALL_TILE_SPRITES[11],
  
  // Variations for base types (using specific sprite indices)
  // Format: "TYPE_VARIATION"
  [`${TILE_IDS.FLOOR}_1`]: ALL_TILE_SPRITES[24], // Alternative floor (row 1, col 0)
  [`${TILE_IDS.WALL}_1`]: ALL_TILE_SPRITES[25],  // Alternative wall
  [`${TILE_IDS.WATER}_1`]: ALL_TILE_SPRITES[27], // Alternative water
};

// Helper function to get sprite for a tile type and variation
export function getTileSpriteForVariation(tileType, variation = 0) {
  if (variation === 0) {
    return TILE_SPRITES[tileType];
  }
  
  const variationKey = `${tileType}_${variation}`;
  return TILE_SPRITES[variationKey] || TILE_SPRITES[tileType]; // Fallback to base sprite
}

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
