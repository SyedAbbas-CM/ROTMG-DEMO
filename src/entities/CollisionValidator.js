/**
 * CollisionValidator.js
 * Validates collision positions based on player latency and server position
 * Detects impossible collisions (e.g., player too far from server-known position)
 */

import { COLLISION_VALIDATION } from '../../common/constants.js';

export default class CollisionValidator {
  constructor(options = {}) {
    this.enabled = options.enabled ?? COLLISION_VALIDATION.ENABLED;
    this.mode = options.mode ?? COLLISION_VALIDATION.MODE;
    this.maxDistanceBase = options.maxDistanceBase ?? COLLISION_VALIDATION.MAX_DISTANCE_BASE_TILES;
    this.distancePer100msRTT = options.distancePer100msRTT ?? COLLISION_VALIDATION.DISTANCE_PER_100MS_RTT;
    this.suspiciousThreshold = options.suspiciousThreshold ?? COLLISION_VALIDATION.SUSPICIOUS_THRESHOLD;
    this.fileLogger = options.fileLogger || null;

    // Stats tracking
    this.stats = {
      totalChecks: 0,
      validCollisions: 0,
      suspiciousCollisions: 0,
      rejectedCollisions: 0
    };
  }

  /**
   * Calculate maximum allowed distance based on player RTT
   * @param {number} rtt - Round trip time in milliseconds
   * @returns {number} Maximum allowed distance in tiles
   */
  calculateMaxDistance(rtt = 0) {
    // Base distance + extra per 100ms of latency
    const latencyBonus = (rtt / 100) * this.distancePer100msRTT;
    return this.maxDistanceBase + latencyBonus;
  }

  /**
   * Validate a collision position against server-known player position
   * @param {Object} params - Validation parameters
   * @param {Object} params.serverPosition - Server's known player position {x, y}
   * @param {Object} params.clientPosition - Client-reported collision position {x, y}
   * @param {number} params.rtt - Player's round trip time in ms
   * @param {string} params.playerId - Player ID for logging
   * @param {string} params.collisionType - Type of collision (bullet_hit, contact, etc.)
   * @returns {Object} Validation result {valid, suspicious, distance, maxAllowed, reason}
   */
  validatePosition(params) {
    if (!this.enabled) {
      return { valid: true, suspicious: false, skipped: true };
    }

    const { serverPosition, clientPosition, rtt = 0, playerId, collisionType } = params;
    this.stats.totalChecks++;

    // Calculate distance between server and client positions
    const dx = clientPosition.x - serverPosition.x;
    const dy = clientPosition.y - serverPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Calculate max allowed distance based on latency
    const maxAllowed = this.calculateMaxDistance(rtt);
    const suspiciousDistance = maxAllowed * this.suspiciousThreshold;

    const result = {
      valid: true,
      suspicious: false,
      distance,
      maxAllowed,
      serverPosition,
      clientPosition,
      rtt,
      playerId,
      collisionType
    };

    if (distance > maxAllowed) {
      // Collision is invalid - too far from server position
      result.valid = this.mode !== 'strict'; // Only invalid in strict mode
      result.suspicious = true;
      result.reason = 'distance_exceeded';

      if (this.mode === 'strict') {
        this.stats.rejectedCollisions++;
      } else {
        this.stats.suspiciousCollisions++;
      }

      if (this.fileLogger) {
        this.fileLogger.collisionValidation(
          this.mode === 'strict' ? 'rejected' : 'suspicious',
          {
            reason: 'distance_exceeded',
            distance: distance.toFixed(2),
            maxAllowed: maxAllowed.toFixed(2),
            rtt,
            playerId,
            collisionType,
            serverPos: serverPosition,
            clientPos: clientPosition
          }
        );
      }
    } else if (distance > suspiciousDistance) {
      // Collision is suspicious but not invalid
      result.suspicious = true;
      result.reason = 'suspicious_distance';
      this.stats.suspiciousCollisions++;

      if (this.fileLogger) {
        this.fileLogger.collisionValidation('suspicious', {
          reason: 'suspicious_distance',
          distance: distance.toFixed(2),
          maxAllowed: maxAllowed.toFixed(2),
          threshold: suspiciousDistance.toFixed(2),
          rtt,
          playerId,
          collisionType,
          serverPos: serverPosition,
          clientPos: clientPosition
        });
      }
    } else {
      this.stats.validCollisions++;
    }

    return result;
  }

  /**
   * Get current validation statistics
   * @returns {Object} Stats object
   */
  getStats() {
    return { ...this.stats };
  }

  /**
   * Reset statistics
   */
  resetStats() {
    this.stats = {
      totalChecks: 0,
      validCollisions: 0,
      suspiciousCollisions: 0,
      rejectedCollisions: 0
    };
  }
}
