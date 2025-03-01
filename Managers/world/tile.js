// Managers/world/tile.js

export class Tile {
    constructor(type, height = 0, properties = {}) {
      this.type = type; // e.g., TILE_IDS.FLOOR, TILE_IDS.WALL, etc.
      this.height = height; // Height value for 3D rendering
      this.properties = properties; // Additional properties like textures
    }
  }
  