/**
 * PatternToBulletAdapter - Converts AI-generated pattern fields into actual bullet spawns
 *
 * Supports two pattern formats:
 *
 * V1 Format (2 channels): [32][32][2]
 *   [0] = intensity (0-1): spawn probability
 *   [1] = direction (0-1): maps to 0-2π radians
 *
 * V2 Format (8 channels): [32][32][8]
 *   [0] = spawn (0-1): spawn probability
 *   [1] = direction (0-1): maps to 0-2π radians
 *   [2] = size (0-1): bullet size multiplier
 *   [3] = speed (0-1): bullet speed multiplier
 *   [4] = accel (0-1): acceleration (0.5 = none, <0.5 = decel, >0.5 = accel)
 *   [5] = curve (0-1): angular velocity (0.5 = straight)
 *   [6] = wave_amp (0-1): wave amplitude
 *   [7] = wave_freq (0-1): wave frequency
 */

export class PatternToBulletAdapter {
  constructor(bulletManager) {
    this.bulletManager = bulletManager;

    // Configuration (tunable per boss/phase)
    this.config = {
      // Spawn threshold - pixels below this intensity are skipped
      spawnThreshold: 0.05,  // Low threshold for testing visual variety

      // Spatial mapping - how far from boss center to spawn bullets (in TILE UNITS)
      spawnRadius: 0.0,  // 0 tiles = spawn from boss center

      // Velocity scaling
      baseSpeed: 4.0,    // TILES/SEC (4 tiles/sec = 48 pixels/sec)
      speedVariation: 0.3, // ±30% speed based on intensity

      // Bullet properties
      baseDamage: 12,
      bulletWidth: 0.4,   // TILE UNITS (0.4 tiles ≈ 5 pixels)
      bulletHeight: 0.4,
      lifetime: 5.0,      // seconds

      // Density control - spawn every Nth pixel to avoid spam
      sparsity: 2,  // 1=every pixel, 2=every other pixel, 3=every 3rd, etc.

      // Sequential spawning - spawn bullets one at a time with delay
      sequentialSpawn: true,
      bulletSpawnDelay: 0.1,  // Delay in seconds between each bullet (0.1s = 100ms)
    };

    // Queue for sequential bullet spawning
    this.spawnQueue = [];
    this.queueTimer = 0;
  }

  /**
   * Update spawning queue - call this every frame
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (this.spawnQueue.length === 0) return;

    this.queueTimer -= dt;
    if (this.queueTimer <= 0) {
      // Spawn next bullet (one at a time)
      const bulletData = this.spawnQueue.shift();
      this.bulletManager.addBullet(bulletData);

      // Reset timer for next bullet
      if (this.spawnQueue.length > 0) {
        this.queueTimer = this.config.bulletSpawnDelay;
      }
    }
  }

  /**
   * Spawn bullets from a pattern field
   * @param {Array} pattern - [32][32][N] pattern array from ML model (N=2 for v1, N=8 for v2)
   * @param {Object} bossData - {x, y, ownerId, worldId, faction}
   * @param {Object} customConfig - Optional config overrides
   * @returns {number} Number of bullets spawned
   */
  spawnPattern(pattern, bossData, customConfig = {}) {
    // Merge custom config
    const config = { ...this.config, ...customConfig };

    const { x: bossX, y: bossY, ownerId, worldId, faction } = bossData;

    if (!worldId) {
      console.warn('[PatternToBulletAdapter] No worldId provided, bullets will be rejected');
      return 0;
    }

    let spawnCount = 0;
    const gridSize = pattern.length; // Should be 32
    const bulletsToSpawn = [];

    // Detect pattern version (v1=2 channels, v2=8 channels)
    const numChannels = pattern[0]?.[0]?.length || 2;
    const isV2 = numChannels >= 8;

    // Iterate through pattern grid and collect bullets
    for (let row = 0; row < gridSize; row += config.sparsity) {
      for (let col = 0; col < gridSize; col += config.sparsity) {
        const pixel = pattern[row][col];

        // V2: Read all 8 channels directly
        // V1: Read 2 channels and derive the rest
        const spawn = pixel[0];
        const directionNorm = pixel[1];

        // Skip low spawn probability cells
        if (spawn < config.spawnThreshold) continue;

        // Read or derive properties based on version
        let sizeNorm, speedNorm, accelNorm, curveNorm, waveAmpNorm, waveFreqNorm;

        if (isV2) {
          // V2: Direct from model
          sizeNorm = pixel[2];
          speedNorm = pixel[3];
          accelNorm = pixel[4];
          curveNorm = pixel[5];
          waveAmpNorm = pixel[6];
          waveFreqNorm = pixel[7];
        } else {
          // V1: Derive from intensity (legacy behavior)
          sizeNorm = spawn;
          speedNorm = 1 - spawn * 0.5;
          accelNorm = 0.5;  // No acceleration
          curveNorm = 0.5;  // Straight
          waveAmpNorm = 0;
          waveFreqNorm = 0;
        }

        // Map grid position to world offset from boss
        const offsetX = ((col - gridSize / 2) / (gridSize / 2)) * config.spawnRadius;
        const offsetY = ((row - gridSize / 2) / (gridSize / 2)) * config.spawnRadius;

        const spawnX = bossX + offsetX;
        const spawnY = bossY + offsetY;

        // Convert direction (0-1) to angle (0-2π)
        const angle = directionNorm * Math.PI * 2;

        // Map normalized values to actual bullet properties
        // Size: 0.2 to 1.5 tiles
        const bulletSize = 0.2 + sizeNorm * 1.3;

        // Speed: 2 to 12 tiles/sec
        const speed = 2 + speedNorm * 10;

        // Acceleration: -3 to +3 tiles/sec^2 (0.5 = no accel)
        const accel = (accelNorm - 0.5) * 6;

        // Curve: -2 to +2 rad/sec (0.5 = straight)
        const angularVel = (curveNorm - 0.5) * 4;

        // Wave amplitude: 0 to 1.5 tiles
        const waveAmp = waveAmpNorm * 1.5;

        // Wave frequency: 0 to 4 Hz
        const waveFreq = waveFreqNorm * 4;

        // Lifetime based on spawn probability
        const lifetime = config.lifetime * (0.8 + spawn * 0.8);

        // Velocity components
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed;

        // Acceleration components (same direction as velocity)
        const ax = Math.cos(angle) * accel;
        const ay = Math.sin(angle) * accel;

        // Create bullet data with all v2 properties
        const bulletData = {
          x: spawnX,
          y: spawnY,
          vx: vx,
          vy: vy,
          ax: ax,
          ay: ay,
          angularVel: angularVel,
          waveAmp: waveAmp,
          waveFreq: waveFreq,
          wavePhase: Math.random() * Math.PI * 2,  // Random phase for variety
          lifetime: lifetime,
          width: bulletSize,
          height: bulletSize,
          damage: this.calculateDamage(spawn, config),
          ownerId: ownerId,
          spriteName: this.selectSprite(spawn),
          worldId: worldId,
          faction: faction !== undefined ? faction : 0
        };

        bulletsToSpawn.push(bulletData);
        spawnCount++;
      }
    }

    // Either queue for sequential spawning or spawn immediately
    if (config.sequentialSpawn && bulletsToSpawn.length > 1) {
      this.spawnQueue.push(...bulletsToSpawn);
      if (this.queueTimer <= 0) {
        this.queueTimer = config.bulletSpawnDelay;
      }
    } else {
      bulletsToSpawn.forEach(bulletData => {
        this.bulletManager.addBullet(bulletData);
      });
    }

    return spawnCount;
  }

  /**
   * Calculate bullet damage based on intensity
   * @param {number} intensity - Spawn intensity (0-1)
   * @param {Object} config - Current config
   * @returns {number} Damage value
   */
  calculateDamage(intensity, config) {
    // High intensity = higher damage
    // Scale damage between 50% and 150% of base
    const damageScale = 0.5 + intensity;
    return Math.floor(config.baseDamage * damageScale);
  }

  /**
   * Select bullet sprite based on intensity
   * @param {number} intensity - Spawn intensity (0-1)
   * @returns {string} Sprite name
   */
  selectSprite(intensity) {
    // You can customize this based on your sprite assets
    if (intensity > 0.7) {
      return 'bullet_large'; // High intensity = bigger bullets
    } else if (intensity > 0.5) {
      return 'bullet_medium';
    } else {
      return 'bullet_small';
    }
  }

  /**
   * Update configuration for different boss phases/styles
   * @param {string} style - Style preset name
   */
  setStyle(style) {
    const styles = {
      'dense': {
        spawnThreshold: 0.2,
        sparsity: 1,
        baseSpeed: 3.0,
        baseDamage: 8
      },
      'sparse_deadly': {
        spawnThreshold: 0.5,
        sparsity: 3,
        baseSpeed: 6.0,
        baseDamage: 20
      },
      'fast_chaos': {
        spawnThreshold: 0.3,
        sparsity: 2,
        baseSpeed: 8.0,
        baseDamage: 10,
        lifetime: 3.0
      },
      'slow_wall': {
        spawnThreshold: 0.25,
        sparsity: 1,
        baseSpeed: 2.0,
        baseDamage: 15,
        lifetime: 8.0
      }
    };

    if (styles[style]) {
      this.config = { ...this.config, ...styles[style] };
      console.log(`[PatternAdapter] Style set to: ${style}`);
    } else {
      console.warn(`[PatternAdapter] Unknown style: ${style}`);
    }
  }

  /**
   * Helper: Spawn pattern with phase-based scaling
   * @param {Array} pattern - Pattern field
   * @param {Object} bossData - Boss data
   * @param {number} phase - Boss phase (1, 2, 3...)
   */
  spawnPatternForPhase(pattern, bossData, phase) {
    const phaseConfig = {
      1: { spawnThreshold: 0.4, baseSpeed: 3.5, baseDamage: 10 },
      2: { spawnThreshold: 0.35, baseSpeed: 4.5, baseDamage: 14 },
      3: { spawnThreshold: 0.3, baseSpeed: 5.5, baseDamage: 18 }
    };

    return this.spawnPattern(pattern, bossData, phaseConfig[phase] || phaseConfig[1]);
  }
}

export default PatternToBulletAdapter;
