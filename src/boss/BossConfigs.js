/**
 * BossConfigs - Pre-defined boss configurations for AI Pattern system
 *
 * Each config defines:
 * - styles: which pattern styles to use (and in what order)
 * - attackInterval: seconds between attacks
 * - compileOpts: how to compile patterns (bullets per burst, duration, etc.)
 * - playerConfig: bullet properties (damage, lifetime)
 * - rageMode: what happens at low HP
 */

export const BossConfigs = {

  // ============================================================
  // BOSS 1: Bloom Guardian (Easy/Tutorial)
  // Slow flower patterns, forgiving timing, low damage
  // ============================================================
  bloom_guardian: {
    name: "Bloom Guardian",
    styles: ['flower', 'flower', 'spiral'],  // Mostly flowers
    attackInterval: 4.0,  // Slow attacks

    compileOpts: {
      bulletsPerEvent: 4,
      maxEvents: 12,
      duration: 3.0,        // Slow bloom over 3 seconds
      directionMode: 'pixel',
      timeMode: 'spiral',
      spawnThreshold: 0.15,
    },

    playerConfig: {
      damage: 8,
      lifetime: 4.0,
    },

    // Rage mode at <30% HP
    rageMode: {
      attackInterval: 2.5,
      compileOpts: {
        bulletsPerEvent: 6,
        maxEvents: 16,
        duration: 2.0,
      },
      playerConfig: {
        damage: 12,
        rotationSpeed: 0.2,
      }
    }
  },

  // ============================================================
  // BOSS 2: Storm Caller (Medium - Sweep attacks)
  // Sweeping arcs that rotate around, requires positioning
  // ============================================================
  storm_caller: {
    name: "Storm Caller",
    styles: ['sweep', 'sweep', 'wave'],
    attackInterval: 2.5,

    compileOpts: {
      bulletsPerEvent: 10,
      maxEvents: 8,
      duration: 1.2,
      directionMode: 'pixel',
      timeMode: 'angle',      // Sweeps around like clock hand
      baseSpread: 0.3,
    },

    playerConfig: {
      damage: 12,
      lifetime: 5.0,
      rotationSpeed: 0.4,     // Pattern rotates while firing
    },

    rageMode: {
      attackInterval: 1.5,
      compileOpts: {
        bulletsPerEvent: 14,
        maxEvents: 12,
        duration: 0.8,
      },
      playerConfig: {
        damage: 16,
        rotationSpeed: 0.8,
      }
    }
  },

  // ============================================================
  // BOSS 3: Void Burst (Hard - Rapid bursts)
  // Quick successive bursts, tests reaction time
  // ============================================================
  void_burst: {
    name: "Void Burst",
    styles: ['burst', 'burst', 'flower'],
    attackInterval: 2.0,

    compileOpts: {
      bulletsPerEvent: 8,
      maxEvents: 6,
      duration: 0.4,          // Very fast burst
      directionMode: 'pixel',
      timeMode: 'radius',     // Expands outward quickly
      baseSpread: 0.5,
    },

    playerConfig: {
      damage: 15,
      lifetime: 3.5,
    },

    rageMode: {
      attackInterval: 1.0,    // Rapid fire
      compileOpts: {
        bulletsPerEvent: 12,
        maxEvents: 8,
        duration: 0.3,
      },
      playerConfig: {
        damage: 20,
      }
    }
  },

  // ============================================================
  // BOSS 4: Serpent King (Expert - Wavy curved bullets)
  // Curved, wavy bullets that are hard to predict
  // ============================================================
  serpent_king: {
    name: "Serpent King",
    styles: ['wave', 'spiral', 'wave'],
    attackInterval: 3.0,

    compileOpts: {
      bulletsPerEvent: 6,
      maxEvents: 15,
      duration: 2.5,
      directionMode: 'mixed',
      mixDir: 0.5,            // Half ML, half pixel
      timeMode: 'spiral',
      baseSpread: 0.4,
    },

    playerConfig: {
      damage: 14,
      lifetime: 6.0,          // Long-lived bullets
    },

    // Override bullet physics for wavy motion
    bulletOverrides: {
      waveAmpMultiplier: 2.0,   // Extra wavy
      curveMultiplier: 1.5,     // Extra curvy
    },

    rageMode: {
      attackInterval: 1.8,
      compileOpts: {
        bulletsPerEvent: 10,
        maxEvents: 20,
        duration: 1.5,
      },
      playerConfig: {
        damage: 18,
        rotationSpeed: 0.3,
      }
    }
  },

};

export default BossConfigs;
