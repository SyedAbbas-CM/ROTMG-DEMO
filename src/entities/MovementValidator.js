// File: src/entities/MovementValidator.js
// Movement validation system for detecting suspicious player movement
// SOFT VALIDATION: Logs only, never blocks gameplay

export class MovementValidator {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.maxSpeedTilesPerSec = options.maxSpeedTilesPerSec || 7.2; // 6.0 + 20% tolerance
    this.teleportThresholdTiles = options.teleportThresholdTiles || 3.0;
    this.logInterval = options.logInterval || 5000; // Log summary every 5 seconds
    this.fileLogger = options.fileLogger || null;

    // Tracking state per player
    this.playerStates = new Map(); // playerId -> { lastPosition, lastTimestamp, violations }

    // Aggregated stats for periodic logging
    this.stats = {
      totalUpdates: 0,
      speedViolations: 0,
      teleportViolations: 0,
      lastLogTime: Date.now()
    };

    // Start periodic logging
    if (this.enabled && this.logInterval > 0) {
      this.logIntervalId = setInterval(() => this.logStats(), this.logInterval);
    }
  }

  /**
   * Validate a player movement update
   * @param {string} playerId - Player identifier
   * @param {number} newX - New X position (tiles)
   * @param {number} newY - New Y position (tiles)
   * @param {number} timestamp - Current timestamp (milliseconds)
   * @returns {object} - Validation result with warnings (non-blocking)
   */
  validate(playerId, newX, newY, timestamp) {
    if (!this.enabled) {
      return { valid: true, warnings: [] };
    }

    this.stats.totalUpdates++;

    const warnings = [];
    const playerState = this.playerStates.get(playerId);

    // First update for this player
    if (!playerState) {
      this.playerStates.set(playerId, {
        lastPosition: { x: newX, y: newY },
        lastTimestamp: timestamp,
        violations: {
          speed: 0,
          teleport: 0
        }
      });
      return { valid: true, warnings: [] };
    }

    // Calculate distance and time delta
    const dx = newX - playerState.lastPosition.x;
    const dy = newY - playerState.lastPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const timeDelta = (timestamp - playerState.lastTimestamp) / 1000; // seconds

    // Ignore if time delta is too small (< 16ms / 60fps) or negative (clock skew)
    // Small timeDelta causes false positives: 0.01 tiles / 0.001s = 10 tiles/sec
    const MIN_TIME_DELTA = 0.016; // 16ms minimum (~60fps)
    if (timeDelta <= MIN_TIME_DELTA || timeDelta > 1.0) {
      playerState.lastPosition = { x: newX, y: newY };
      playerState.lastTimestamp = timestamp;
      return { valid: true, warnings: [] }; // Don't warn on normal frame times
    }

    // Check for teleportation (instant large distance)
    if (distance > this.teleportThresholdTiles) {
      playerState.violations.teleport++;
      this.stats.teleportViolations++;

      warnings.push({
        type: 'TELEPORT',
        distance: distance.toFixed(2),
        threshold: this.teleportThresholdTiles,
        from: { x: playerState.lastPosition.x, y: playerState.lastPosition.y },
        to: { x: newX, y: newY }
      });

      // Log immediately for teleports (suspicious)
      if (this.fileLogger) {
        this.fileLogger.warn('MOVEMENT_VALIDATOR', `Teleport detected for player ${playerId}`, {
          distance: distance.toFixed(2),
          threshold: this.teleportThresholdTiles,
          from: playerState.lastPosition,
          to: { x: newX, y: newY }
        });
      }
    }

    // Check speed (tiles per second)
    const speed = distance / timeDelta;
    if (speed > this.maxSpeedTilesPerSec) {
      playerState.violations.speed++;
      this.stats.speedViolations++;

      warnings.push({
        type: 'SPEED',
        speed: speed.toFixed(2),
        maxSpeed: this.maxSpeedTilesPerSec,
        distance: distance.toFixed(2),
        timeDelta: timeDelta.toFixed(3)
      });

      // Only log if significantly over threshold (reduce noise)
      if (speed > this.maxSpeedTilesPerSec * 1.5 && this.fileLogger) {
        this.fileLogger.warn('MOVEMENT_VALIDATOR', `High speed detected for player ${playerId}`, {
          speed: speed.toFixed(2),
          maxSpeed: this.maxSpeedTilesPerSec,
          distance: distance.toFixed(2),
          timeDelta: timeDelta.toFixed(3)
        });
      }
    }

    // Update tracking state
    playerState.lastPosition = { x: newX, y: newY };
    playerState.lastTimestamp = timestamp;

    // Always return valid: true (soft validation)
    return {
      valid: true,
      warnings
    };
  }

  /**
   * Log aggregate statistics
   */
  logStats() {
    if (this.stats.totalUpdates === 0) return;

    const now = Date.now();
    const elapsed = (now - this.stats.lastLogTime) / 1000; // seconds

    const speedViolationRate = (this.stats.speedViolations / this.stats.totalUpdates * 100).toFixed(2);
    const teleportViolationRate = (this.stats.teleportViolations / this.stats.totalUpdates * 100).toFixed(2);

    // Only log to file, not console (reduces terminal spam)
    // console.log stats disabled - use FILE_LOGGER_ENABLED=true to see movement validation stats

    if (this.fileLogger) {
      this.fileLogger.info('MOVEMENT_VALIDATOR', 'Periodic stats', {
        period: elapsed.toFixed(1),
        totalUpdates: this.stats.totalUpdates,
        speedViolations: this.stats.speedViolations,
        teleportViolations: this.stats.teleportViolations,
        activePlayers: this.playerStates.size,
        speedViolationRate,
        teleportViolationRate
      });
    }

    // Reset stats for next period
    this.stats.totalUpdates = 0;
    this.stats.speedViolations = 0;
    this.stats.teleportViolations = 0;
    this.stats.lastLogTime = now;
  }

  /**
   * Remove a player from tracking (on disconnect)
   * @param {string} playerId - Player identifier
   */
  removePlayer(playerId) {
    this.playerStates.delete(playerId);
  }

  /**
   * Get violation count for a specific player
   * @param {string} playerId - Player identifier
   * @returns {object} - Violation counts
   */
  getPlayerViolations(playerId) {
    const state = this.playerStates.get(playerId);
    if (!state) {
      return { speed: 0, teleport: 0 };
    }
    return { ...state.violations };
  }

  /**
   * Clean up (stop interval)
   */
  destroy() {
    if (this.logIntervalId) {
      clearInterval(this.logIntervalId);
      this.logIntervalId = null;
    }
  }
}

export default MovementValidator;
