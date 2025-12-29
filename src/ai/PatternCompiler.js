/**
 * PatternCompiler - Converts static 32x32 grid into timed burst events
 *
 * This is the "bridge" that makes ML patterns look like bullet hell attacks.
 * Instead of spawning one bullet per pixel, we compile to an event schedule.
 *
 * Key insight: The grid is NOT a spatial spawn map (bullets spawn at boss center).
 * The grid is a parameter lookup table that we convert into timed emission events.
 */

export class PatternCompiler {
  constructor() {
    this.defaultConfig = {
      // Spawn filtering
      spawnThreshold: 0.08,

      // Event structure
      maxEvents: 20,           // Total bursts to emit
      bulletsPerEvent: 6,      // Bullets per burst
      duration: 2.5,           // Seconds for full pattern to play

      // Time mapping - how grid position maps to emission time
      // "radius" = expand outward, "angle" = sweep around, "spiral" = bloom, "brightness" = stroke order
      timeMode: "spiral",
      spiralA: 1.0,            // Radius weight for spiral
      spiralB: 0.8,            // Angle weight for spiral

      // Direction interpretation
      // "ml" = trust model output, "pixel" = use pixel-to-center angle, "mixed" = blend
      directionMode: "mixed",
      mixDir: 0.3,             // 0 = pure pixel geometry, 1 = pure ML direction

      // Spread for each burst
      baseSpread: 0.2,         // Radians - how wide each burst fans out
      spawnVariance: 0.5,      // How much spawn value affects spread (higher spawn = tighter)
    };
  }

  /**
   * Compile a pattern grid into timed burst events
   * @param {Array} pattern - [32][32][8] channel array
   * @param {Object} opts - Override default config
   * @returns {Array} List of BulletEvent objects sorted by time
   */
  compile(pattern, opts = {}) {
    const gridSize = pattern.length;
    const config = { ...this.defaultConfig, ...opts };

    // Grid center
    const cx = (gridSize - 1) * 0.5;
    const cy = (gridSize - 1) * 0.5;
    const maxR = Math.sqrt(cx * cx + cy * cy);

    // Collect candidate pixels above spawn threshold
    const candidates = [];

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const p = pattern[row][col];
        const spawn = p[0];

        if (spawn < config.spawnThreshold) continue;

        // Compute pixel geometry
        const dx = col - cx;
        const dy = row - cy;
        const r = Math.sqrt(dx * dx + dy * dy);
        const rNorm = maxR > 0 ? (r / maxR) : 0;

        // Angle from center (radial outward direction)
        const theta = Math.atan2(dy, dx);
        const thetaNorm = (theta + Math.PI) / (2 * Math.PI); // [0, 1]

        candidates.push({
          row, col,
          spawn,
          rNorm,
          theta,        // Pixel's geometric angle (radians)
          thetaNorm,    // Normalized [0, 1]
          pixel: p
        });
      }
    }

    if (candidates.length === 0) {
      console.warn('[PatternCompiler] No pixels above threshold');
      return [];
    }

    // Score each candidate with time assignment
    const scored = candidates.map(c => {
      const p = c.pixel;

      // Compute emission time based on mode
      let tNorm = 0;
      switch (config.timeMode) {
        case "radius":
          // Expand outward from center
          tNorm = c.rNorm;
          break;
        case "angle":
          // Sweep around like a clock hand
          tNorm = c.thetaNorm;
          break;
        case "brightness":
          // High spawn = early, low spawn = late (stroke drawing)
          tNorm = 1 - c.spawn;
          break;
        case "spiral":
        default:
          // Combination creates blooming spiral effect
          const v = config.spiralA * c.rNorm + config.spiralB * c.thetaNorm;
          tNorm = v - Math.floor(v); // frac() to wrap to [0,1]
          break;
      }

      // Direction computation
      const mlAngle = (p[1] ?? 0) * Math.PI * 2;  // ML output (absolute)
      const pixelAngle = c.theta;                  // Geometric (radial outward)

      let baseAngle;
      switch (config.directionMode) {
        case "ml":
          baseAngle = mlAngle;
          break;
        case "pixel":
          baseAngle = pixelAngle;
          break;
        case "mixed":
        default:
          baseAngle = this._lerpAngle(pixelAngle, mlAngle, config.mixDir);
          break;
      }

      // Read other channels with safe defaults
      const sizeNorm = p[2] ?? c.spawn;
      const speedNorm = p[3] ?? (1 - c.spawn * 0.5);
      const accelNorm = p[4] ?? 0.5;
      const curveNorm = p[5] ?? 0.5;
      const waveAmpNorm = p[6] ?? 0;
      const waveFreqNorm = p[7] ?? 0;

      // Map normalized values to game units (match PatternToBulletAdapter ranges)
      const size = 0.2 + sizeNorm * 1.3;           // 0.2 - 1.5 tiles
      const speed = 2 + speedNorm * 10;            // 2 - 12 tiles/sec
      const accel = (accelNorm - 0.5) * 6;         // -3 to +3 tiles/sec²
      const angularVel = (curveNorm - 0.5) * 4;    // -2 to +2 rad/sec
      const waveAmp = waveAmpNorm * 1.5;           // 0 - 1.5 tiles
      const waveFreq = waveFreqNorm * 4;           // 0 - 4 Hz

      return {
        t: tNorm * config.duration,
        spawn: c.spawn,
        angle: baseAngle,
        rNorm: c.rNorm,
        size, speed, accel, angularVel, waveAmp, waveFreq
      };
    });

    // Sort by time
    scored.sort((a, b) => a.t - b.t);

    // Sample representative events across time
    const events = [];
    const n = Math.min(config.maxEvents, scored.length);

    for (let i = 0; i < n; i++) {
      // Evenly sample across the sorted list
      const idx = Math.floor((i / Math.max(1, n - 1)) * (scored.length - 1));
      const s = scored[idx];

      // Spread: high spawn = tight cone, low spawn = wide spread
      const spread = config.baseSpread + (1 - s.spawn) * config.spawnVariance;

      events.push({
        t: s.t,
        count: config.bulletsPerEvent,
        angle: s.angle,
        spread: spread,
        size: s.size,
        speed: s.speed,
        accel: s.accel,
        angularVel: s.angularVel,
        waveAmp: s.waveAmp,
        waveFreq: s.waveFreq,
      });
    }

    return events;
  }

  /**
   * Compile with preset style
   * @param {Array} pattern - Pattern grid
   * @param {string} style - "flower", "spiral", "sweep", "burst", "wave"
   * @returns {Array} Events
   */
  compileWithStyle(pattern, style) {
    const styles = {
      flower: {
        timeMode: "spiral",
        spiralA: 1.2,
        spiralB: 0.6,
        directionMode: "pixel",
        bulletsPerEvent: 5,
        maxEvents: 16,
        duration: 2.0,
      },
      spiral: {
        timeMode: "spiral",
        spiralA: 0.8,
        spiralB: 1.5,
        directionMode: "mixed",
        mixDir: 0.4,
        bulletsPerEvent: 4,
        maxEvents: 20,
        duration: 3.0,
      },
      sweep: {
        timeMode: "angle",
        directionMode: "pixel",
        bulletsPerEvent: 8,
        maxEvents: 12,
        duration: 1.5,
        baseSpread: 0.1,
      },
      burst: {
        timeMode: "radius",
        directionMode: "pixel",
        bulletsPerEvent: 12,
        maxEvents: 6,
        duration: 0.5,
      },
      wave: {
        timeMode: "brightness",
        directionMode: "mixed",
        mixDir: 0.6,
        bulletsPerEvent: 6,
        maxEvents: 15,
        duration: 2.5,
      },
    };

    return this.compile(pattern, styles[style] || {});
  }

  /**
   * Shortest-path angle interpolation
   */
  _lerpAngle(a, b, t) {
    let d = b - a;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return a + d * t;
  }
}

export default PatternCompiler;
