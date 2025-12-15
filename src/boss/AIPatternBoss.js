/**
 * AIPatternBoss - Enhanced boss with ML-generated bullet patterns
 * Extends the baseline boss behavior with AI pattern attacks
 */

import { PatternToBulletAdapter } from '../ai/PatternToBulletAdapter.js';
import { PatternLibrary } from '../ai/PatternLibrary.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class AIPatternBoss {
  constructor(bossManager, bulletManager) {
    this.bossManager = bossManager;
    this.bulletManager = bulletManager;

    // Initialize AI pattern system
    this.patternLibrary = new PatternLibrary();
    this.patternAdapter = new PatternToBulletAdapter(bulletManager);

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
    this.aiAttackInterval = 1.5; // Fire AI pattern every 1.5 seconds (cycle faster)

    // Phase timing (per boss) - cycle through phases every 5 seconds
    this.phaseTimer = new Float32Array(bossManager.maxBosses).fill(0);
    this.phaseInterval = 5.0; // 5 seconds per phase

    // Track pattern history to avoid repetition
    this.lastPatternIds = new Array(bossManager.maxBosses).fill(-1);
  }

  /**
   * Update AI pattern attacks
   * Call this from your main boss tick loop
   * @param {number} dt - Delta time in seconds
   */
  update(dt) {
    if (!this.aiEnabled) return;

    // Update pattern adapter's spawn queue
    this.patternAdapter.update(dt);

    const bossCount = this.bossManager.bossCount;

    for (let i = 0; i < bossCount; i++) {
      // Update phase timer - cycle through phases every 5 seconds
      this.phaseTimer[i] += dt;
      if (this.phaseTimer[i] >= this.phaseInterval) {
        this.phaseTimer[i] = 0;
        // Cycle through phases 0, 1, 2
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

    // Get appropriate pattern for phase
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

    // Boss data for adapter
    const ei = this.bossManager.enemyIndex[bossIndex];
    const ownerId = ei >= 0 && this.bossManager.enemyMgr
      ? this.bossManager.enemyMgr.id[ei]
      : this.bossManager.id[bossIndex];

    const bossData = {
      x: this.bossManager.x[bossIndex],
      y: this.bossManager.y[bossIndex],
      ownerId: ownerId,
      worldId: worldId,
      faction: 0  // Enemy faction
    };

    // Adapt config based on HP (rage mode at low HP)
    let customConfig = {};
    if (hp < 0.3) {
      // Rage mode: dense, fast, deadly
      customConfig = {
        spawnThreshold: 0.25,
        baseSpeed: 6.0,
        baseDamage: 16,
        sparsity: 1
      };
    } else if (hp < 0.6) {
      // Mid HP: moderate
      customConfig = {
        spawnThreshold: 0.30,
        baseSpeed: 5.0,
        baseDamage: 14,
        sparsity: 2
      };
    }

    // Spawn the pattern!
    const bulletCount = this.patternAdapter.spawnPattern(
      patternArray,
      bossData,
      customConfig
    );

    console.log(`[AIPatternBoss] Boss ${bossIndex} (HP: ${(hp*100).toFixed(0)}%, Phase: ${phase}) fired pattern ${pattern.id} → ${bulletCount} bullets`);
  }

  /**
   * Manually trigger AI pattern (useful for testing or special attacks)
   * @param {number} bossIndex - Boss index
   * @param {string} style - Pattern style ('sparse', 'medium', 'dense', 'chaotic')
   */
  triggerPattern(bossIndex, style = null) {
    if (!this.aiEnabled) {
      console.warn('[AIPatternBoss] AI patterns not loaded');
      return 0;
    }

    const pattern = style
      ? this.patternLibrary.getPatternByStyle(style)
      : this.patternLibrary.getRandomPattern();

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

    return this.patternAdapter.spawnPattern(patternArray, bossData);
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
   * Change adapter style preset
   * @param {string} style - Style name
   */
  setAdapterStyle(style) {
    this.patternAdapter.setStyle(style);
  }
}

export default AIPatternBoss;
