// EXTREME test: Make size differences VERY obvious
// Change line 119-121 to:

// Size: EXTREME scaling - 10x difference instead of 4x
const sizeMultiplier = intensity < 0.5
  ? 0.2  // LOW intensity = 0.2x (TINY: 3.2px with 0.4 base * 16px/tile = 1.3px!)
  : 5.0; // HIGH intensity = 5.0x (HUGE: 32px)
const bulletWidth = config.bulletWidth * sizeMultiplier;
const bulletHeight = config.bulletHeight * sizeMultiplier;

// This gives:
// Low (0.1): 0.4 * 0.2 = 0.08 tiles = 1.3 pixels  (TINY DOT)
// High (0.9): 0.4 * 5.0 = 2.0 tiles = 32 pixels   (HUGE)
// Ratio: 25x size difference - IMPOSSIBLE to miss!
