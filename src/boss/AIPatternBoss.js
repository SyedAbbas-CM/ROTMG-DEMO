/**
 * AIPatternBoss - Enhanced boss with ML-generated bullet patterns
 *
 * Now uses PatternCompiler + PatternPlayer for proper bullet hell emergence:
 * - Grid is compiled to timed burst events
 * - Bursts emit over time (flower bloom, spiral unfold, sweep)
 * - Not "one bullet at a time" drip feed
 */

import { PatternCompiler } from '../ai/PatternCompiler.js';
import { PatternPlayer, PatternQueue } from '../ai/PatternPlayer.js';
import { PatternLibrary } from '../ai/PatternLibrary.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class AIPatternBoss {
  constructor(bossManager, bulletManager) {
    this.bossManager = bossManager;
    this.bulletManager = bulletManager;

    // Pattern library (loads pre-generated patterns)
    this.patternLibrary = new PatternLibrary();

    // NEW: Compiler converts grid → timed events
    this.patternCompiler = new PatternCompiler();

    // NEW: Queue manages multiple patterns playing simultaneously
    this.patternQueue = new PatternQueue(bulletManager);

    // Load patterns
    const jsonPath = path.join(__dirname, '../../ml/visualizations/pattern_library.json');
    const loaded = this.patternLibrary.loadFromJSON(jsonPath);

    if (loaded === 0) {
      console.warn('[AIPatternBoss] No patterns loaded - using fallback');
      this.aiEnabled = false;
    } else {
      console.log(`[AIPatternBoss] Loaded ${loaded} AI patterns`);
      this.aiEnabled = true;
    }

    // AI attack cooldowns (per boss)
    this.aiAttackCooldown = new Float32Array(bossManager.maxBosses).fill(0);
    this.aiAttackInterval = 3.0; // Fire pattern every 3 seconds (patterns take ~2s to play)

    // Phase timing (per boss)
    this.phaseTimer = new Float32Array(bossManager.maxBosses).fill(0);
    this.phaseInterval = 8.0; // 8 seconds per phase (longer to let patterns play out)

    // Pattern style rotation for variety
    this.styleRotation = ['flower', 'spiral', 'sweep', 'burst', 'wave'];
    this.styleIndex = new Uint8Array(bossManager.maxBosses).fill(0);

    // Track last pattern to avoid repetition
    this.lastPatternIds = new Array(bossManager.maxBosses).fill(-1);
  }

  /**
   * Update AI pattern attacks
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (!this.aiEnabled) return;

    // Update all active pattern players
    this.patternQueue.update(dt);

    const bossCount = this.bossManager.bossCount;

    for (let i = 0; i < bossCount; i++) {
      // Update phase timer
      this.phaseTimer[i] += dt;
      if (this.phaseTimer[i] >= this.phaseInterval) {
        this.phaseTimer[i] = 0;
        this.bossManager.phase[i] = (this.bossManager.phase[i] + 1) % 3;
        console.log(`[AIPatternBoss] Boss ${i} entered phase ${this.bossManager.phase[i]}`);
      }

      // Update attack cooldown
      this.aiAttackCooldown[i] -= dt;

      if (this.aiAttackCooldown[i] <= 0) {
        this.fireAIPattern(i);
        this.aiAttackCooldown[i] = this.aiAttackInterval;
      }
    }
  }

  /**
   * Fire an AI pattern for a specific boss
   * @param {number} bossIndex - Index in BossManager arrays
   */
  fireAIPattern(bossIndex) {
    const phase = this.bossManager.phase[bossIndex];
    const hp = this.bossManager.hp[bossIndex];
    const worldId = this.bossManager.worldId[bossIndex];

    // Get pattern for phase
    let pattern = this.patternLibrary.getPatternForPhase(phase + 1);

    // Avoid repeating same pattern
    if (pattern && pattern.id === this.lastPatternIds[bossIndex]) {
      pattern = this.patternLibrary.getPatternForPhase(phase + 1);
    }

    if (!pattern) {
      console.warn(`[AIPatternBoss] No pattern available for boss ${bossIndex}`);
      return;
    }

    this.lastPatternIds[bossIndex] = pattern.id;

    // Convert to array format
    const patternArray = this.patternLibrary.toPatternArray(pattern);

    // Boss data
    const ei = this.bossManager.enemyIndex[bossIndex];
    const ownerId = ei >= 0 && this.bossManager.enemyMgr
      ? this.bossManager.enemyMgr.id[ei]
      : this.bossManager.id[bossIndex];

    const bossData = {
      x: this.bossManager.x[bossIndex],
      y: this.bossManager.y[bossIndex],
      ownerId: ownerId,
      worldId: worldId,
      faction: 0
    };

    // Choose compilation style based on phase and HP
    let style = this.styleRotation[this.styleIndex[bossIndex]];
    this.styleIndex[bossIndex] = (this.styleIndex[bossIndex] + 1) % this.styleRotation.length;

    // Rage mode at low HP
    let compileOpts = {};
    let playerConfig = { lifetime: 5.0, damage: 12 };

    if (hp < 0.3) {
      // Rage: faster, denser, rotating
      style = 'burst';
      compileOpts = {
        bulletsPerEvent: 10,
        maxEvents: 12,
        duration: 1.0,
      };
      playerConfig.rotationSpeed = 0.3;
      playerConfig.damage = 18;
    } else if (hp < 0.6) {
      // Mid: moderate increase
      compileOpts = {
        bulletsPerEvent: 8,
        maxEvents: 18,
        duration: 2.0,
      };
      playerConfig.damage = 14;
    }

    // COMPILE: Convert grid to timed events
    const events = this.patternCompiler.compileWithStyle(patternArray, style);

    // Override with HP-based config if set
    const finalEvents = Object.keys(compileOpts).length > 0
      ? this.patternCompiler.compile(patternArray, compileOpts)
      : events;

    // PLAY: Start the pattern
    const player = this.patternQueue.queue(finalEvents, bossData, playerConfig);

    if (player) {
      const totalBullets = finalEvents.reduce((sum, e) => sum + e.count, 0);
      const duration = finalEvents.length > 0 ? finalEvents[finalEvents.length - 1].t : 0;
      console.log(`[AIPatternBoss] Boss ${bossIndex} (HP: ${(hp*100).toFixed(0)}%, Phase: ${phase}) ` +
                  `fired pattern ${pattern.id} style=${style} → ${finalEvents.length} bursts, ` +
                  `~${totalBullets} bullets over ${duration.toFixed(1)}s`);
    }
  }

  /**
   * Manually trigger pattern with specific style
   * @param {number} bossIndex - Boss index
   * @param {string} style - "flower", "spiral", "sweep", "burst", "wave"
   */
  triggerPattern(bossIndex, style = 'flower') {
    if (!this.aiEnabled) {
      console.warn('[AIPatternBoss] AI patterns not loaded');
      return 0;
    }

    const pattern = this.patternLibrary.getRandomPattern();
    if (!pattern) return 0;

    const patternArray = this.patternLibrary.toPatternArray(pattern);

    const ei = this.bossManager.enemyIndex[bossIndex];
    const ownerId = ei >= 0 && this.bossManager.enemyMgr
      ? this.bossManager.enemyMgr.id[ei]
      : this.bossManager.id[bossIndex];

    const bossData = {
      x: this.bossManager.x[bossIndex],
      y: this.bossManager.y[bossIndex],
      ownerId: ownerId,
      worldId: this.bossManager.worldId[bossIndex],
      faction: 0
    };

    const events = this.patternCompiler.compileWithStyle(patternArray, style);
    const player = this.patternQueue.queue(events, bossData);

    return events.reduce((sum, e) => sum + e.count, 0);
  }

  /**
   * Adjust AI attack frequency
   * @param {number} interval - Seconds between AI attacks
   */
  setAttackInterval(interval) {
    this.aiAttackInterval = interval;
    console.log(`[AIPatternBoss] AI attack interval set to ${interval}s`);
  }

  /**
   * Get number of patterns currently playing
   */
  getActivePatternsCount() {
    return this.patternQueue.getActiveCount();
  }

  /**
   * Stop all playing patterns
   */
  stopAllPatterns() {
    this.patternQueue.stopAll();
  }
}

export default AIPatternBoss;
