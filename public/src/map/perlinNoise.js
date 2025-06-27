// src/map/perlinNoise.js

export class PerlinNoise {
  constructor(seed = Math.random()) {
    this.seed = seed;
    this.gradients = {};
    this.memory = {};
  }

  // Generate a random gradient vector
  randomGradient(ix, iy) {
    const random = 2920 * Math.sin(ix * 21942 + iy * 171324 + this.seed * 8912) *
                  Math.cos(ix * 23157 * iy * 217832 + this.seed * 9758);
    return { x: Math.cos(random), y: Math.sin(random) };
  }

  // Dot product of the distance and gradient vectors
  dotGridGradient(ix, iy, x, y) {
    const gradient = this.gradients[[ix, iy]] || (this.gradients[[ix, iy]] = this.randomGradient(ix, iy));

    const dx = x - ix;
    const dy = y - iy;

    return dx * gradient.x + dy * gradient.y;
  }

  // Interpolation function
  lerp(a0, a1, w) {
    return (1 - w) * a0 + w * a1;
  }

  // Compute Perlin noise at coordinates x, y
  get(x, y) {
    const x0 = Math.floor(x);
    const x1 = x0 + 1;
    const y0 = Math.floor(y);
    const y1 = y0 + 1;

    // Interpolation weights
    const sx = x - x0;
    const sy = y - y0;

    // Interpolate between grid point gradients
    const n0 = this.dotGridGradient(x0, y0, x, y);
    const n1 = this.dotGridGradient(x1, y0, x, y);
    const ix0 = this.lerp(n0, n1, sx);

    const n2 = this.dotGridGradient(x0, y1, x, y);
    const n3 = this.dotGridGradient(x1, y1, x, y);
    const ix1 = this.lerp(n2, n3, sx);

    const value = this.lerp(ix0, ix1, sy);
    return value;
  }
}

export const perlin = new PerlinNoise(); // Instantiate with a random seed
