/**
 * PatternPlayer - Executes compiled pattern events over time
 *
 * This replaces the "queue drip" approach with proper burst emission.
 * Instead of 1 bullet every 0.1s, we emit N bullets per burst at scheduled times.
 *
 * Usage:
 *   const compiler = new PatternCompiler();
 *   const events = compiler.compile(pattern);
 *   const player = new PatternPlayer(bulletManager);
 *   player.start(events, bossData);
 *   // In game loop:
 *   player.update(dt);
 */

export class PatternPlayer {
  constructor(bulletManager) {
    this.bulletManager = bulletManager;

    // Playback state
    this.events = [];
    this.eventIndex = 0;
    this.time = 0;
    this.active = false;

    // Boss/source data
    this.bossData = null;

    // Config
    this.config = {
      lifetime: 5.0,
      damage: 12,
      // Optional: global rotation applied to all events (for rotating attacks)
      rotationSpeed: 0,  // rad/sec - pattern rotates while playing
    };

    // For rotating patterns
    this.rotationOffset = 0;
  }

  /**
   * Start playing a compiled event list
   * @param {Array} events - From PatternCompiler.compile()
   * @param {Object} bossData - { x, y, ownerId, worldId, faction }
   * @param {Object} config - Optional overrides
   */
  start(events, bossData, config = {}) {
    this.events = events.slice().sort((a, b) => a.t - b.t);
    this.eventIndex = 0;
    this.time = 0;
    this.active = true;
    this.bossData = bossData;
    this.rotationOffset = 0;

    this.config = {
      lifetime: 5.0,
      damage: 12,
      rotationSpeed: 0,
      ...config
    };

    if (this.events.length > 0) {
      console.log(`[PatternPlayer] Starting pattern with ${this.events.length} events over ${this.events[this.events.length - 1].t.toFixed(2)}s`);
    }
  }

  /**
   * Stop playback
   */
  stop() {
    this.active = false;
    this.events = [];
    this.eventIndex = 0;
  }

  /**
   * Check if still playing
   */
  isPlaying() {
    return this.active;
  }

  /**
   * Get playback progress [0, 1]
   */
  getProgress() {
    if (this.events.length === 0) return 1;
    const maxT = this.events[this.events.length - 1].t;
    return maxT > 0 ? Math.min(1, this.time / maxT) : 1;
  }

  /**
   * Update playback - call every frame
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (!this.active) return;

    this.time += dt;

    // Update rotation offset for rotating patterns
    if (this.config.rotationSpeed !== 0) {
      this.rotationOffset += this.config.rotationSpeed * dt;
    }

    // Execute all events that should have fired by now
    while (this.eventIndex < this.events.length &&
           this.events[this.eventIndex].t <= this.time) {
      this._spawnBurst(this.events[this.eventIndex]);
      this.eventIndex++;
    }

    // Check if done
    if (this.eventIndex >= this.events.length) {
      this.active = false;
    }
  }

  /**
   * Spawn a burst of bullets from an event
   * @param {Object} event - Burst event from compiler
   */
  _spawnBurst(event) {
    if (!this.bossData) {
      console.warn('[PatternPlayer] No boss data');
      return;
    }

    const { x: bossX, y: bossY, ownerId, worldId, faction } = this.bossData;

    // DEBUG: Log ownerId once per pattern play
    if (this.eventIndex === 0) {
      console.log(`[PatternPlayer DEBUG] Spawning bullets with ownerId=${ownerId} worldId=${worldId}`);
    }

    // Apply rotation offset to base angle
    const baseAngle = event.angle + this.rotationOffset;

    // Emit 'count' bullets spread across the cone
    for (let i = 0; i < event.count; i++) {
      // Distribute bullets across spread angle
      // u ranges from -0.5 to 0.5
      const u = event.count === 1 ? 0 : (i / (event.count - 1) - 0.5);
      const angle = baseAngle + u * event.spread;

      // Velocity from angle and speed
      const vx = Math.cos(angle) * event.speed;
      const vy = Math.sin(angle) * event.speed;

      // Acceleration in same direction as velocity
      const ax = Math.cos(angle) * event.accel;
      const ay = Math.sin(angle) * event.accel;

      this.bulletManager.addBullet({
        x: bossX,
        y: bossY,
        vx, vy,
        ax, ay,
        angularVel: event.angularVel,
        waveAmp: event.waveAmp,
        waveFreq: event.waveFreq,
        wavePhase: Math.random() * Math.PI * 2,
        lifetime: this.config.lifetime,
        width: event.size,
        height: event.size,
        damage: this.config.damage,
        ownerId,
        worldId,
        faction: faction !== undefined ? faction : 0
      });
    }
  }
}

/**
 * PatternQueue - Manages multiple patterns playing in sequence or parallel
 */
export class PatternQueue {
  constructor(bulletManager) {
    this.bulletManager = bulletManager;
    this.players = [];
    this.maxConcurrent = 3; // Max patterns playing at once
  }

  /**
   * Queue a pattern to play
   * @param {Array} events - Compiled events
   * @param {Object} bossData - Boss data
   * @param {Object} config - Player config
   */
  queue(events, bossData, config = {}) {
    // Clean up finished players
    this.players = this.players.filter(p => p.isPlaying());

    // If at max, skip (or could wait)
    if (this.players.length >= this.maxConcurrent) {
      console.warn('[PatternQueue] Max concurrent patterns reached');
      return null;
    }

    const player = new PatternPlayer(this.bulletManager);
    player.start(events, bossData, config);
    this.players.push(player);

    return player;
  }

  /**
   * Update all active players
   */
  update(dt) {
    for (const player of this.players) {
      player.update(dt);
    }

    // Clean up finished
    this.players = this.players.filter(p => p.isPlaying());
  }

  /**
   * Stop all patterns
   */
  stopAll() {
    for (const player of this.players) {
      player.stop();
    }
    this.players = [];
  }

  /**
   * Get count of active patterns
   */
  getActiveCount() {
    return this.players.filter(p => p.isPlaying()).length;
  }
}

export default PatternPlayer;
