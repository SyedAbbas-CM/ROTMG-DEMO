// src/utils/prng.js
// Deterministic Xorshift32 PRNG helper so replays are reproducible.
// Usage:
//   import { createPRNG } from '../utils/prng.js';
//   const rng = createPRNG(123);
//   const v = rng();   // 0…1 float

export function createPRNG(seed = 1) {
  let x = seed | 0;
  if (x === 0) x = 0xCAFEBABE; // avoid zero state
  return function rng() {
    // Xorshift32
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    // Convert to [0,1)
    return ((x >>> 0) / 0xFFFFFFFF);
  };
} 