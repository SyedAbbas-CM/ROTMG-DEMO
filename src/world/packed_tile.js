// Managers/world/packedTile.js

export class PackedTile {
    constructor(packedValue) {
      this.packedValue = packedValue;
    }
  
    static pack(type, height) {
      // Pack type (5 bits) and height (3 bits) into a single 8-bit number
      return ((height & 0b111) << 5) | (type & 0b11111);
    }
  
    static unpack(packedValue) {
      const type = packedValue & 0b11111;          // lower 5 bits
      const height = (packedValue >> 5) & 0b111;    // upper 3 bits
      return { type, height };
    }
  
    getType() {
      return this.packedValue & 0b11111;
    }
  
    getHeight() {
      return (this.packedValue >> 5) & 0b111;
    }
  
    toTile(properties = {}) {
      // Convert back to a normal Tile class if needed
      const { type, height } = PackedTile.unpack(this.packedValue);
      return new Tile(type, height, properties);
    }
  }
  