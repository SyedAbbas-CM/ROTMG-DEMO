// File: src/entities/LagCompensation.js
// Lag compensation system for rewinding player positions
// Used to make hit detection fair for players with high latency

export class LagCompensation {
  constructor(config = {}) {
    this.enabled = config.enabled !== false;
    this.maxRewindMs = config.maxRewindMs || 200; // Maximum time to rewind (milliseconds)
    this.minRTT = config.minRTT || 50; // Don't compensate for very low latencies
    this.debug = config.debug || false;
    this.fileLogger = config.fileLogger || null;

    // Statistics
    this.stats = {
      rewindsPerformed: 0,
      avgRewindAmount: 0,
      maxRewindAmount: 0,
      hitsCompensated: 0
    };
  }

  /**
   * Calculate how far back to rewind for a given RTT
   * @param {number} rtt - Round-trip time in milliseconds
   * @returns {number} - Rewind amount in milliseconds
   */
  calculateRewindAmount(rtt) {
    // If RTT is below threshold, don't compensate
    if (rtt < this.minRTT) {
      return 0;
    }

    // Rewind by half RTT (one-way latency)
    let rewindMs = rtt / 2;

    // Clamp to max rewind amount
    rewindMs = Math.min(rewindMs, this.maxRewindMs);

    return rewindMs;
  }

  /**
   * Rewind a player's position to a specific time in the past
   * @param {object} player - Player object with positionHistory
   * @param {number} rewindMs - How far back to rewind (milliseconds)
   * @param {number} currentTime - Current server timestamp
   * @returns {object} - Rewound position {x, y, found, originalX, originalY}
   */
  rewindPlayerPosition(player, rewindMs, currentTime) {
    if (!this.enabled || rewindMs === 0) {
      return {
        x: player.x,
        y: player.y,
        found: true,
        originalX: player.x,
        originalY: player.y,
        rewound: false
      };
    }

    // Calculate target timestamp
    const targetTimestamp = currentTime - rewindMs;

    // Get position from history
    const result = player.positionHistory.getPositionAt(targetTimestamp);

    // Track statistics
    if (result.found) {
      this.stats.rewindsPerformed++;
      this.stats.avgRewindAmount =
        (this.stats.avgRewindAmount * (this.stats.rewindsPerformed - 1) + rewindMs) /
        this.stats.rewindsPerformed;
      this.stats.maxRewindAmount = Math.max(this.stats.maxRewindAmount, rewindMs);
    }

    return {
      x: result.x,
      y: result.y,
      found: result.found,
      originalX: player.x,
      originalY: player.y,
      rewound: true,
      rewindAmount: rewindMs
    };
  }

  /**
   * Rewind all players in a list
   * Returns original positions for restoration
   * @param {Array} players - Array of player objects
   * @param {number} rewindMs - How far back to rewind (milliseconds)
   * @param {number} currentTime - Current server timestamp
   * @returns {Map} - Map of playerId -> {originalX, originalY}
   */
  rewindAllPlayers(players, rewindMs, currentTime) {
    const originalPositions = new Map();

    if (!this.enabled || rewindMs === 0) {
      return originalPositions;
    }

    for (const player of players) {
      // Store original position
      originalPositions.set(player.id, {
        x: player.x,
        y: player.y
      });

      // Rewind position
      const rewound = this.rewindPlayerPosition(player, rewindMs, currentTime);

      // Update player object (TEMPORARY - must restore later)
      player.x = rewound.x;
      player.y = rewound.y;

      if (this.debug && rewound.found) {
        const distance = Math.sqrt(
          Math.pow(rewound.x - rewound.originalX, 2) +
          Math.pow(rewound.y - rewound.originalY, 2)
        );

        if (distance > 0.1) { // Only log if meaningful movement
          console.log(`[LAG_COMP] Rewound player ${player.id}: ` +
            `(${rewound.originalX.toFixed(2)}, ${rewound.originalY.toFixed(2)}) -> ` +
            `(${rewound.x.toFixed(2)}, ${rewound.y.toFixed(2)}) ` +
            `[${rewindMs}ms, ${distance.toFixed(2)} tiles]`);
        }
      }
    }

    return originalPositions;
  }

  /**
   * Restore players to their original positions
   * @param {Array} players - Array of player objects
   * @param {Map} originalPositions - Map from rewindAllPlayers()
   */
  restoreAllPlayers(players, originalPositions) {
    if (originalPositions.size === 0) {
      return; // Nothing was rewound
    }

    for (const player of players) {
      const original = originalPositions.get(player.id);
      if (original) {
        player.x = original.x;
        player.y = original.y;
      }
    }

    if (this.debug) {
      console.log(`[LAG_COMP] Restored ${originalPositions.size} players to original positions`);
    }
  }

  /**
   * Log a compensated hit for analytics
   * @param {object} bullet - Bullet that hit
   * @param {object} target - Target that was hit
   * @param {number} rewindAmount - How far back we rewound (ms)
   * @param {object} position - Position where hit occurred
   */
  logCompensatedHit(bullet, target, rewindAmount, position) {
    this.stats.hitsCompensated++;

    if (this.debug) {
      console.log(`[LAG_COMP] Hit registered: Bullet ${bullet.id} -> Target ${target.id} ` +
        `at (${position.x.toFixed(2)}, ${position.y.toFixed(2)}) ` +
        `[rewind: ${rewindAmount}ms]`);
    }

    if (this.fileLogger) {
      this.fileLogger.info('LAG_COMPENSATION', 'Compensated hit', {
        bulletId: bullet.id,
        bulletOwnerId: bullet.ownerId,
        targetId: target.id,
        rewindAmount,
        hitPosition: {
          x: position.x.toFixed(2),
          y: position.y.toFixed(2)
        },
        targetCurrentPosition: {
          x: target.x.toFixed(2),
          y: target.y.toFixed(2)
        }
      });
    }
  }

  /**
   * Get current statistics
   * @returns {object} - Statistics object
   */
  getStats() {
    return { ...this.stats };
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.stats = {
      rewindsPerformed: 0,
      avgRewindAmount: 0,
      maxRewindAmount: 0,
      hitsCompensated: 0
    };
  }

  /**
   * Log statistics summary
   */
  logStats() {
    console.log('\n========== LAG COMPENSATION STATS ==========');
    console.log(`Enabled: ${this.enabled}`);
    console.log(`Rewinds Performed: ${this.stats.rewindsPerformed}`);
    console.log(`Avg Rewind Amount: ${this.stats.avgRewindAmount.toFixed(2)}ms`);
    console.log(`Max Rewind Amount: ${this.stats.maxRewindAmount.toFixed(2)}ms`);
    console.log(`Hits Compensated: ${this.stats.hitsCompensated}`);
    console.log('==========================================\n');

    if (this.fileLogger) {
      this.fileLogger.info('LAG_COMPENSATION', 'Statistics summary', this.stats);
    }
  }
}

export default LagCompensation;
