// src/boss/addons/GameplayHistoryRecorder.js
// Records gameplay history for strategic analysis batching

/**
 * Records important game moments for batch strategic analysis
 */
export class GameplayHistoryRecorder {
  constructor(config = {}) {
    this.maxSize = config.maxSize || 100;
    this.history = [];
    this.sessionStart = Date.now();
  }

  /**
   * Record a snapshot if it's significant enough
   */
  record(snapshot, eventType = 'normal') {
    const entry = {
      timestamp: Date.now(),
      eventType,
      boss: {
        hp: snapshot.boss?.hp,
        maxHp: snapshot.boss?.maxHp,
        position: { x: snapshot.boss?.x, y: snapshot.boss?.y }
      },
      players: snapshot.players?.map(p => ({
        id: p.id,
        distance: p.distance,
        hp: p.hp
      })) || [],
      bullets: {
        active: snapshot.bullets?.active || 0
      }
    };

    this.history.push(entry);

    // Keep only most recent entries
    if (this.history.length > this.maxSize) {
      this.history.shift();
    }
  }

  /**
   * Get aggregate metrics from recorded history
   */
  getAggregateMetrics() {
    if (this.history.length === 0) return null;

    const sessionDuration = (Date.now() - this.sessionStart) / 1000;

    let totalDamageDealt = 0;
    let totalDamageTaken = 0;
    let playerDistances = [];

    for (let i = 1; i < this.history.length; i++) {
      const prev = this.history[i - 1];
      const curr = this.history[i];

      // Calculate damage taken by boss
      if (prev.boss.hp > curr.boss.hp) {
        totalDamageTaken += prev.boss.hp - curr.boss.hp;
      }

      // Record player distances for pattern analysis
      curr.players.forEach(p => {
        if (p.distance) playerDistances.push(p.distance);
      });
    }

    return {
      sessionDuration,
      totalDamageDealt,
      totalDamageTaken,
      playerPatterns: {
        averageDistance: playerDistances.length > 0
          ? playerDistances.reduce((a, b) => a + b, 0) / playerDistances.length
          : 0,
        minDistance: Math.min(...playerDistances),
        maxDistance: Math.max(...playerDistances)
      },
      snapshotCount: this.history.length
    };
  }

  /**
   * Get key moments (important events) for strategic analysis
   */
  getKeyMoments(limit = 10) {
    // Prioritize significant events
    const priorityEvents = this.history.filter(h => h.eventType !== 'normal');

    // If not enough priority events, fill with recent normal events
    if (priorityEvents.length < limit) {
      const recentNormal = this.history
        .filter(h => h.eventType === 'normal')
        .slice(-Math.max(0, limit - priorityEvents.length));
      return [...priorityEvents, ...recentNormal];
    }

    return priorityEvents.slice(-limit);
  }

  /**
   * Clear history (after strategic analysis completes)
   */
  clear() {
    this.history = [];
    this.sessionStart = Date.now();
  }
}
