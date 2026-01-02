/**
 * AIPatternBoss - Enhanced boss with ML-generated bullet patterns
 *
 * Uses PatternCompiler + PatternPlayer for proper bullet hell emergence.
 * Now supports per-boss configurations via BossConfigs.
 */

import { PatternCompiler } from '../ai/PatternCompiler.js';
import { PatternPlayer, PatternQueue } from '../ai/PatternPlayer.js';
import { PatternLibrary } from '../ai/PatternLibrary.js';
import { BossConfigs } from './BossConfigs.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class AIPatternBoss {
  constructor(bossManager, bulletManager) {
    this.bossManager = bossManager;
    this.bulletManager = bulletManager;

    // Pattern library
    this.patternLibrary = new PatternLibrary();
    this.patternCompiler = new PatternCompiler();
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

    // Per-boss state
    this.bossConfigs = new Array(bossManager.maxBosses).fill(null);  // Config name per boss
    this.aiAttackCooldown = new Float32Array(bossManager.maxBosses).fill(0);
    this.styleIndex = new Uint8Array(bossManager.maxBosses).fill(0);
    this.lastPatternIds = new Array(bossManager.maxBosses).fill(-1);

    // Default config for bosses without explicit config
    this.defaultConfigName = 'bloom_guardian';
  }

  /**
   * Assign a config to a boss
   * @param {number} bossIndex - Index in BossManager
   * @param {string} configName - Key from BossConfigs (e.g., 'bloom_guardian')
   */
  setBossConfig(bossIndex, configName) {
    if (!BossConfigs[configName]) {
      console.warn(`[AIPatternBoss] Unknown config: ${configName}, using default`);
      configName = this.defaultConfigName;
    }
    this.bossConfigs[bossIndex] = configName;
    console.log(`[AIPatternBoss] Boss ${bossIndex} set to config: ${configName}`);
  }

  /**
   * Get the config for a boss
   */
  getConfig(bossIndex) {
    const configName = this.bossConfigs[bossIndex] || this.defaultConfigName;
    return BossConfigs[configName] || BossConfigs[this.defaultConfigName];
  }

  /**
   * Update AI pattern attacks
   */
  update(dt) {
    if (!this.aiEnabled) return;

    this.patternQueue.update(dt);

    const bossCount = this.bossManager.bossCount;

    for (let i = 0; i < bossCount; i++) {
      const config = this.getConfig(i);
      const hp = this.bossManager.hp[i];

      // Determine attack interval (normal or rage mode)
      const isRage = hp < 0.3 && config.rageMode;
      const attackInterval = isRage
        ? config.rageMode.attackInterval
        : config.attackInterval;

      this.aiAttackCooldown[i] -= dt;

      if (this.aiAttackCooldown[i] <= 0) {
        this.fireAIPattern(i);
        this.aiAttackCooldown[i] = attackInterval;
      }
    }
  }

  /**
   * Fire an AI pattern for a specific boss
   */
  fireAIPattern(bossIndex) {
    const config = this.getConfig(bossIndex);
    const hp = this.bossManager.hp[bossIndex];
    const worldId = this.bossManager.worldId[bossIndex];
    const isRage = hp < 0.3 && config.rageMode;

    // Get pattern
    const phase = this.bossManager.phase[bossIndex];
    let pattern = this.patternLibrary.getPatternForPhase(phase + 1);

    if (pattern && pattern.id === this.lastPatternIds[bossIndex]) {
      pattern = this.patternLibrary.getPatternForPhase(phase + 1);
    }

    if (!pattern) {
      console.warn(`[AIPatternBoss] No pattern for boss ${bossIndex}`);
      return;
    }

    this.lastPatternIds[bossIndex] = pattern.id;
    const patternArray = this.patternLibrary.toPatternArray(pattern);

    // Boss data
    const ei = this.bossManager.enemyIndex[bossIndex];
    const ownerId = ei >= 0 && this.bossManager.enemyMgr
      ? this.bossManager.enemyMgr.id[ei]
      : this.bossManager.id[bossIndex];

    const bossData = {
      x: this.bossManager.x[bossIndex],
      y: this.bossManager.y[bossIndex],
      ownerId,
      worldId,
      faction: 0
    };

    // Choose style from config rotation
    const styles = config.styles;
    const style = styles[this.styleIndex[bossIndex] % styles.length];
    this.styleIndex[bossIndex] = (this.styleIndex[bossIndex] + 1) % styles.length;

    // Get compile options (normal or rage)
    const compileOpts = isRage
      ? { ...config.compileOpts, ...config.rageMode.compileOpts }
      : config.compileOpts;

    // Get player config (normal or rage)
    const playerConfig = isRage
      ? { ...config.playerConfig, ...config.rageMode.playerConfig }
      : config.playerConfig;

    // Compile pattern
    const events = this.patternCompiler.compile(patternArray, {
      ...compileOpts,
      timeMode: compileOpts.timeMode || this.getTimeModeForStyle(style),
    });

    // Apply bullet overrides if present
    if (config.bulletOverrides) {
      for (const event of events) {
        if (config.bulletOverrides.waveAmpMultiplier) {
          event.waveAmp *= config.bulletOverrides.waveAmpMultiplier;
        }
        if (config.bulletOverrides.curveMultiplier) {
          event.angularVel *= config.bulletOverrides.curveMultiplier;
        }
      }
    }

    // Play pattern
    const player = this.patternQueue.queue(events, bossData, playerConfig);

    if (player) {
      const totalBullets = events.reduce((sum, e) => sum + e.count, 0);
      const duration = events.length > 0 ? events[events.length - 1].t : 0;
      const configName = this.bossConfigs[bossIndex] || this.defaultConfigName;
      console.log(`[AIPatternBoss] ${config.name} (Boss ${bossIndex}, HP: ${(hp*100).toFixed(0)}%${isRage ? ' RAGE' : ''}) ` +
                  `→ ${style} pattern, ${events.length} bursts, ~${totalBullets} bullets over ${duration.toFixed(1)}s`);
    }
  }

  /**
   * Get default timeMode for a style
   */
  getTimeModeForStyle(style) {
    const modes = {
      flower: 'spiral',
      spiral: 'spiral',
      sweep: 'angle',
      burst: 'radius',
      wave: 'brightness',
    };
    return modes[style] || 'spiral';
  }

  /**
   * Manually trigger pattern
   */
  triggerPattern(bossIndex, style = null) {
    if (!this.aiEnabled) return 0;

    const config = this.getConfig(bossIndex);
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
      ownerId,
      worldId: this.bossManager.worldId[bossIndex],
      faction: 0
    };

    const useStyle = style || config.styles[0];
    const events = this.patternCompiler.compileWithStyle(patternArray, useStyle);
    this.patternQueue.queue(events, bossData, config.playerConfig);

    return events.reduce((sum, e) => sum + e.count, 0);
  }

  /**
   * Set attack interval for a boss
   */
  setAttackInterval(bossIndex, interval) {
    const config = this.getConfig(bossIndex);
    config.attackInterval = interval;
  }

  getActivePatternsCount() {
    return this.patternQueue.getActiveCount();
  }

  stopAllPatterns() {
    this.patternQueue.stopAll();
  }
}

export default AIPatternBoss;
