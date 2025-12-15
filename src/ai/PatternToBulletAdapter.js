/**
 * PatternToBulletAdapter - Converts AI-generated pattern fields into actual bullet spawns
 *
 * Takes a 32×32×2 pattern from the ML model and converts it into bullet spawns
 * for the BulletManager.
 *
 * Pattern format:
 *   [32][32][0] = intensity (0-1): spawn probability/density
 *   [32][32][1] = direction (0-1): maps to 0-2π radians
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
   * @param {Array} pattern - [32][32][2] pattern array from ML model
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

    // Iterate through pattern grid and collect bullets
    for (let row = 0; row < gridSize; row += config.sparsity) {
      for (let col = 0; col < gridSize; col += config.sparsity) {
        const intensity = pattern[row][col][0];
        const directionNorm = pattern[row][col][1];

        // Skip low-intensity cells
        if (intensity < config.spawnThreshold) continue;

        // Map grid position to world offset from boss
        // Grid center (16, 16) → boss position
        // Grid edges → ±spawnRadius tiles from boss
        const offsetX = ((col - gridSize / 2) / (gridSize / 2)) * config.spawnRadius;
        const offsetY = ((row - gridSize / 2) / (gridSize / 2)) * config.spawnRadius;

        const spawnX = bossX + offsetX;
        const spawnY = bossY + offsetY;

        // Convert direction (0-1) to angle (0-2π)
        const angle = directionNorm * Math.PI * 2;

        // Calculate visual properties from intensity for better pattern visibility
        // HIGH intensity = LARGE, SLOW, LONG-LASTING (heavy projectiles)
        // LOW intensity = SMALL, FAST, SHORT-LASTING (light projectiles)

        // Speed: Inverse relationship with intensity (heavy bullets slower)
        const speedMultiplier = 1.5 - (intensity * 0.8); // Range: 0.7x to 1.5x
        const speed = config.baseSpeed * speedMultiplier;

        // Size: EXTREME scaling for testing - 25x difference!
        const sizeMultiplier = intensity < 0.5
          ? 0.2   // LOW intensity = TINY (1-2 pixels)
          : 5.0;  // HIGH intensity = HUGE (32 pixels)
        const bulletWidth = config.bulletWidth * sizeMultiplier;
        const bulletHeight = config.bulletHeight * sizeMultiplier;

        // Lifetime: Longer for high-intensity bullets (makes pattern visible longer)
        const lifetimeMultiplier = 0.8 + (intensity * 1.4); // Range: 0.8x to 2.2x
        const lifetime = config.lifetime * lifetimeMultiplier;

        // Velocity components
        const vx = Math.cos(angle) * speed;
        const vy = Math.sin(angle) * speed;

        // Create bullet data with varied visual properties
        const bulletData = {
          x: spawnX,
          y: spawnY,
          vx: vx,
          vy: vy,
          lifetime: lifetime,
          width: bulletWidth,
          height: bulletHeight,
          damage: this.calculateDamage(intensity, config),
          ownerId: ownerId,
          spriteName: this.selectSprite(intensity),
          worldId: worldId,
          faction: faction !== undefined ? faction : 0 // Default to enemy faction
        };

        bulletsToSpawn.push(bulletData);
        spawnCount++;
      }
    }

    // Either queue for sequential spawning or spawn immediately
    if (config.sequentialSpawn && bulletsToSpawn.length > 1) {
      // Add to queue and start timer
      this.spawnQueue.push(...bulletsToSpawn);
      if (this.queueTimer <= 0) {
        this.queueTimer = config.bulletSpawnDelay; // Start with first delay
      }
    } else {
      // Spawn all immediately (original behavior for single bullets)
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
