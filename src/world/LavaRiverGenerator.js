/**
 * LavaRiverGenerator.js
 * Generates flowing lava rivers in volcanic biomes using Perlin worm paths
 */

import { PerlinNoise } from './PerlinNoise.js';

export class LavaRiverGenerator {
  constructor(seed) {
    // Use a different seed offset for river generation
    this.perlin = new PerlinNoise(seed + 12345);
    this.riverCache = new Map(); // Cache generated rivers by region key
  }

  /**
   * Get a region key for caching
   * @param {number} regionX - Region X coordinate (world coords / 128)
   * @param {number} regionY - Region Y coordinate (world coords / 128)
   * @returns {string} Region key
   */
  getRegionKey(regionX, regionY) {
    return `${regionX},${regionY}`;
  }

  /**
   * Generate lava rivers for a region
   * @param {number} regionX - Region X coordinate (world coords / 128)
   * @param {number} regionY - Region Y coordinate (world coords / 128)
   * @returns {Set<string>} Set of tile keys "x,y" that should be lava
   */
  generateRiversForRegion(regionX, regionY) {
    const regionKey = this.getRegionKey(regionX, regionY);

    // Check cache
    if (this.riverCache.has(regionKey)) {
      return this.riverCache.get(regionKey);
    }

    const lavaSet = new Set();
    const regionSize = 128;
    const baseX = regionX * regionSize;
    const baseY = regionY * regionSize;

    // Determine number of rivers (1-2 per region)
    const numRivers = this.perlin.get(regionX * 7.89, regionY * 11.23) > 0 ? 2 : 1;

    // Generate each river
    for (let i = 0; i < numRivers; i++) {
      const riverSeed = regionX * 1000 + regionY * 100 + i * 10;
      this.generateRiver(baseX, baseY, regionSize, riverSeed, lavaSet);
    }

    // Add sparse random lava tiles (5% of river tiles)
    const riverTiles = Array.from(lavaSet);
    const sparseCount = Math.floor(riverTiles.length * 0.05);

    for (let i = 0; i < sparseCount; i++) {
      // Pick random tiles near river
      const baseTile = riverTiles[Math.floor(this.seededRandom(regionX + i) * riverTiles.length)];
      const [tx, ty] = baseTile.split(',').map(Number);

      // Add 1-3 tiles in random directions
      const numSparseTiles = Math.floor(this.seededRandom(regionY + i) * 3) + 1;
      for (let j = 0; j < numSparseTiles; j++) {
        const offsetX = Math.floor(this.seededRandom(tx + j) * 5) - 2;
        const offsetY = Math.floor(this.seededRandom(ty + j) * 5) - 2;
        lavaSet.add(`${tx + offsetX},${ty + offsetY}`);
      }
    }

    // Cache and return
    this.riverCache.set(regionKey, lavaSet);
    return lavaSet;
  }

  /**
   * Generate a single river using Perlin worm algorithm
   * @param {number} baseX - Base X coordinate of region
   * @param {number} baseY - Base Y coordinate of region
   * @param {number} regionSize - Size of region (128)
   * @param {number} seed - Seed for this river
   * @param {Set<string>} lavaSet - Set to add lava tiles to
   */
  generateRiver(baseX, baseY, regionSize, seed, lavaSet) {
    // River parameters
    const length = 30 + Math.floor(this.seededRandom(seed) * 51); // 30-80 tiles
    const width = 2 + Math.floor(this.seededRandom(seed + 1) * 4); // 2-5 tiles

    // Start position (random within region, biased toward center)
    let x = baseX + regionSize * (0.2 + this.seededRandom(seed + 2) * 0.6);
    let y = baseY + regionSize * (0.2 + this.seededRandom(seed + 3) * 0.6);

    // Initial direction (random angle)
    let angle = this.seededRandom(seed + 4) * Math.PI * 2;

    // Generate river path
    for (let i = 0; i < length; i++) {
      // Use Perlin noise to adjust direction (creates winding effect)
      const noiseX = this.perlin.get(x / 20, y / 20);
      const noiseY = this.perlin.get(x / 20 + 100, y / 20 + 100);

      // Gradually change angle based on noise
      angle += (noiseX * 0.5 - 0.25); // Adjust by ±0.25 radians

      // Move forward
      x += Math.cos(angle) * 1.0;
      y += Math.sin(angle) * 1.0;

      // Add width to river (mark tiles around center point)
      const tileX = Math.floor(x);
      const tileY = Math.floor(y);

      for (let dx = -width; dx <= width; dx++) {
        for (let dy = -width; dy <= width; dy++) {
          // Use circular pattern for river width
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist <= width) {
            lavaSet.add(`${tileX + dx},${tileY + dy}`);
          }
        }
      }

      // Optional: prevent river from leaving region too early
      // Bounce back if getting close to edges
      if (i < length * 0.3) { // Only enforce boundaries for first 30% of river
        if (x < baseX + 10) angle = this.constrainAngle(angle, -Math.PI/2, Math.PI/2);
        if (x > baseX + regionSize - 10) angle = this.constrainAngle(angle, Math.PI/2, Math.PI*1.5);
        if (y < baseY + 10) angle = this.constrainAngle(angle, 0, Math.PI);
        if (y > baseY + regionSize - 10) angle = this.constrainAngle(angle, Math.PI, Math.PI*2);
      }
    }
  }

  /**
   * Constrain angle to a range
   * @param {number} angle - Current angle
   * @param {number} min - Min angle
   * @param {number} max - Max angle
   * @returns {number} Constrained angle
   */
  constrainAngle(angle, min, max) {
    // Normalize angle to 0-2π
    while (angle < 0) angle += Math.PI * 2;
    while (angle >= Math.PI * 2) angle -= Math.PI * 2;

    // If outside range, steer toward center of range
    if (angle < min || angle > max) {
      return (min + max) / 2;
    }
    return angle;
  }

  /**
   * Check if a tile should be lava (for real-time generation)
   * @param {number} x - World X coordinate
   * @param {number} y - World Y coordinate
   * @returns {boolean} True if tile should be lava
   */
  isLavaTile(x, y) {
    const regionX = Math.floor(x / 128);
    const regionY = Math.floor(y / 128);

    const lavaSet = this.generateRiversForRegion(regionX, regionY);
    return lavaSet.has(`${x},${y}`);
  }

  /**
   * Seeded random number generator (simple LCG)
   * @param {number} seed - Seed value
   * @returns {number} Random value between 0 and 1
   */
  seededRandom(seed) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }

  /**
   * Clear cache (useful for debugging or regenerating rivers)
   */
  clearCache() {
    this.riverCache.clear();
  }
}
