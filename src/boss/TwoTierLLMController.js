/**
 * TwoTierLLMController - Dual-mode AI decision system
 *
 * TIER 1 (Tactical): Fast, frequent decisions using existing attacks (10-30s)
 * TIER 2 (Strategic): Slow, rare decisions creating new attacks (5-10 min)
 *
 * Benefits:
 * - Stays within free tier limits (1,000-14,400 RPD per model)
 * - Boss adapts in real-time (tactical) AND learns (strategic)
 * - Efficient token usage via batching
 */

import { createProvider } from './llm/ProviderFactory.js';
import { registry } from '../registry/index.js';
import ScriptBehaviourRunner from '../ScriptBehaviourRunner.js';
import { logLLM } from './llm/llmLogger.js';
import { trace } from '@opentelemetry/api';
import xxhash32 from 'xxhash-wasm';

const HASH_SEED = 0xABCD1234;
const tracer = trace.getTracer('game');

export default class TwoTierLLMController {
  constructor(bossMgr, bulletMgr, mapMgr, enemyMgr, config = {}) {
    this.bossMgr = bossMgr;
    this.bulletMgr = bulletMgr;
    this.mapMgr = mapMgr;
    this.enemyMgr = enemyMgr;

    // Script runner for executing actions
    this.runner = new ScriptBehaviourRunner(bossMgr, bulletMgr, enemyMgr);

    // Configuration with smart defaults
    this.config = {
      // Tier 1: Tactical (immediate response)
      tacticalModel: config.tacticalModel || 'models/gemini-2.5-flash-lite',
      tacticalMinInterval: config.tacticalMinInterval || 10,  // seconds
      tacticalMaxInterval: config.tacticalMaxInterval || 30,  // seconds
      tacticalEnabled: config.tacticalEnabled ?? true,

      // Tier 2: Strategic (batched learning)
      strategicModel: config.strategicModel || 'models/gemini-2.5-pro',
      strategicInterval: config.strategicInterval || 300,  // 5 minutes
      strategicEnabled: config.strategicEnabled ?? true,

      // Advanced
      cacheEnabled: config.cacheEnabled ?? true,
      maxHistorySize: config.maxHistorySize || 100,
      ...config
    };

    // Tier 1: Tactical state
    this.tacticalProvider = null;
    this.tacticalCooldown = 0;
    this.lastTacticalHash = null;
    this.pendingTactical = false;

    // Tier 2: Strategic state
    this.strategicProvider = null;
    this.gameplayHistory = [];
    this.lastStrategicCall = Date.now();
    this.pendingStrategic = false;

    // Metrics
    this.metrics = {
      tacticalCalls: 0,
      strategicCalls: 0,
      cacheHits: 0,
      tacticalFailures: 0,
      strategicFailures: 0
    };

    console.log('[TwoTierLLM] Initialized with config:', {
      tactical: this.config.tacticalModel,
      strategic: this.config.strategicModel,
      tacticalInterval: `${this.config.tacticalMinInterval}-${this.config.tacticalMaxInterval}s`,
      strategicInterval: `${this.config.strategicInterval}s`
    });
  }

  /**
   * Main update loop - called every server frame
   */
  async tick(dt, players) {
    // Execute queued actions from previous decisions
    this._drainActionQueue(dt);

    // Tick the script runner (handles ongoing actions)
    this.runner.tick(dt);

    // Tier 1: Tactical decisions (frequent)
    if (this.config.tacticalEnabled) {
      await this._tickTactical(dt, players);
    }

    // Tier 2: Strategic analysis (infrequent)
    if (this.config.strategicEnabled) {
      await this._tickStrategic(dt, players);
    }

    // Record history for batching (only key moments)
    this._recordHistory(players);
  }

  /**
   * Tier 1: Tactical decisions (every 10-30 seconds)
   */
  async _tickTactical(dt, players) {
    this.tacticalCooldown -= dt;

    // Wait for cooldown
    if (this.tacticalCooldown > 0) return;

    // Skip if already processing
    if (this.pendingTactical) return;

    // Build current game snapshot
    const snapshot = this._buildTacticalSnapshot(players);
    if (!snapshot) return;

    // Check if situation changed significantly (via hash)
    const hapi = await xxhash32();
    const snapshotHash = hapi.h32(JSON.stringify(snapshot), HASH_SEED);

    if (snapshotHash === this.lastTacticalHash) {
      // No significant change, postpone
      this.tacticalCooldown = 1.0;
      return;
    }

    this.lastTacticalHash = snapshotHash;

    // Set adaptive cooldown based on game state
    this.tacticalCooldown = this._getAdaptiveTacticalInterval(snapshot);

    // Make tactical decision
    try {
      this.pendingTactical = true;

      if (!this.tacticalProvider) {
        this.tacticalProvider = createProvider({
          model: this.config.tacticalModel,
          temperature: 0.7,
          maxTokens: 1024
        });
      }

      const span = tracer.startSpan('llm.tactical');
      const startTime = Date.now();

      const result = await this.tacticalProvider.generate(snapshot);

      const latency = Date.now() - startTime;
      span.setAttribute('latency_ms', latency);
      span.setAttribute('tokens', result.tokens || 0);
      span.end();

      this.metrics.tacticalCalls++;

      // Log decision
      logLLM({
        type: 'tactical',
        ts: startTime,
        snapshot,
        result: result.json,
        latency,
        tokens: result.tokens
      });

      // Execute the tactical plan
      if (result.json?.actions) {
        await this._executeTacticalPlan(result.json);
      }

      console.log(`[TwoTierLLM] Tactical decision: ${result.json?.intent || 'no intent'} (${latency}ms, ${result.tokens || 0} tokens)`);

    } catch (err) {
      console.warn('[TwoTierLLM] Tactical call failed:', err.message);
      this.metrics.tacticalFailures++;
      this.tacticalCooldown = 5; // Back off for 5 seconds
    } finally {
      this.pendingTactical = false;
    }
  }

  /**
   * Tier 2: Strategic learning (every 5-10 minutes)
   */
  async _tickStrategic(dt, players) {
    const now = Date.now();
    const timeSinceLastCall = (now - this.lastStrategicCall) / 1000;

    // Wait for interval
    if (timeSinceLastCall < this.config.strategicInterval) return;

    // Skip if already processing
    if (this.pendingStrategic) return;

    // Need enough history to analyze
    if (this.gameplayHistory.length < 5) {
      console.log('[TwoTierLLM] Not enough history for strategic call yet');
      this.lastStrategicCall = now;
      return;
    }

    try {
      this.pendingStrategic = true;
      this.lastStrategicCall = now;

      if (!this.strategicProvider) {
        this.strategicProvider = createProvider({
          model: this.config.strategicModel,
          temperature: 0.9,
          maxTokens: 4096
        });
      }

      const batch = this._buildStrategicBatch();

      const span = tracer.startSpan('llm.strategic');
      const startTime = Date.now();

      const result = await this.strategicProvider.generate(batch);

      const latency = Date.now() - startTime;
      span.setAttribute('latency_ms', latency);
      span.setAttribute('tokens', result.tokens || 0);
      span.setAttribute('history_size', this.gameplayHistory.length);
      span.end();

      this.metrics.strategicCalls++;

      // Log decision
      logLLM({
        type: 'strategic',
        ts: startTime,
        batch,
        result: result.json,
        latency,
        tokens: result.tokens
      });

      // Process strategic decision (might create new capabilities!)
      if (result.json) {
        await this._processStrategicDecision(result.json);
      }

      console.log(`[TwoTierLLM] Strategic decision: ${result.json?.analysis?.substring(0, 80) || 'no analysis'} (${latency}ms, ${result.tokens || 0} tokens)`);

      // Clear history after processing
      this.gameplayHistory = [];

    } catch (err) {
      console.warn('[TwoTierLLM] Strategic call failed:', err.message);
      this.metrics.strategicFailures++;
    } finally {
      this.pendingStrategic = false;
    }
  }

  /**
   * Build snapshot for tactical decisions (lightweight)
   */
  _buildTacticalSnapshot(players) {
    const boss = this.bossMgr;
    if (!boss || boss.count <= 0) return null;

    const bossIdx = 0;
    const bossX = boss.x[bossIdx];
    const bossY = boss.y[bossIdx];
    const bossHp = boss.hp[bossIdx];
    const maxHp = boss.maxHp[bossIdx];

    // Process players - only active ones
    const activePlayers = players
      .filter(p => p && p.health > 0)
      .map(p => {
        const dx = p.x - bossX;
        const dy = p.y - bossY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        return {
          id: p.id,
          distance: Math.round(distance * 10) / 10,
          direction: { dx: Math.sign(dx), dy: Math.sign(dy) },
          hp: p.health || 100
        };
      })
      .sort((a, b) => a.distance - b.distance)  // Closest first
      .slice(0, 5);  // Max 5 players to keep snapshot small

    // Available capabilities from registry
    const capabilities = Object.keys(registry.validators)
      .map(id => {
        const [group, name] = id.split(':');
        return name.split('@')[0]
          .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
          .toLowerCase();
      });

    return {
      boss: {
        hp: bossHp,
        maxHp,
        hpPercent: Math.round((bossHp / maxHp) * 100) / 100,
        position: { x: Math.round(bossX), y: Math.round(bossY) },
        recentActions: boss.recentActions || []
      },
      players: activePlayers,
      bullets: {
        active: this.bulletMgr.count || 0
      },
      capabilities,
      timestamp: Date.now()
    };
  }

  /**
   * Build batch data for strategic analysis
   */
  _buildStrategicBatch() {
    // Aggregate metrics from history
    const metrics = this._aggregateMetrics();

    return {
      sessionDuration: this.gameplayHistory.length > 0
        ? (Date.now() - this.gameplayHistory[0].timestamp) / 1000
        : 0,
      snapshotCount: this.gameplayHistory.length,
      keyMoments: this.gameplayHistory.slice(0, 10),  // First 10 key moments
      metrics,
      currentCapabilities: Object.keys(registry.validators).length,
      goal: 'Analyze gameplay and suggest improvements or new attack patterns'
    };
  }

  /**
   * Calculate aggregate metrics from gameplay history
   */
  _aggregateMetrics() {
    if (this.gameplayHistory.length === 0) {
      return { noData: true };
    }

    let totalDamageDealt = 0;
    let totalDamageTaken = 0;
    const attackUsage = {};
    let playerDeaths = 0;

    for (const moment of this.gameplayHistory) {
      if (moment.damageDealt) totalDamageDealt += moment.damageDealt;
      if (moment.damageTaken) totalDamageTaken += moment.damageTaken;
      if (moment.bossAction) {
        attackUsage[moment.bossAction] = (attackUsage[moment.bossAction] || 0) + 1;
      }
      if (moment.event === 'player_death') playerDeaths++;
    }

    return {
      totalDamageDealt,
      totalDamageTaken,
      damageRatio: totalDamageDealt / (totalDamageTaken || 1),
      attackUsage,
      playerDeaths,
      avgPlayersAlive: this.gameplayHistory.reduce((sum, m) => sum + (m.playerCount || 0), 0) / this.gameplayHistory.length
    };
  }

  /**
   * Get adaptive interval based on game state
   */
  _getAdaptiveTacticalInterval(snapshot) {
    const { boss, players } = snapshot;

    // Critical HP - more frequent decisions
    if (boss.hpPercent < 0.25) return this.config.tacticalMinInterval;

    // Many players - more combat
    if (players.length > 2) return this.config.tacticalMinInterval + 5;

    // No players - idle
    if (players.length === 0) return this.config.tacticalMaxInterval * 2;

    // Default
    return (this.config.tacticalMinInterval + this.config.tacticalMaxInterval) / 2;
  }

  /**
   * Execute tactical plan from LLM
   */
  async _executeTacticalPlan(plan) {
    // High priority clears existing queue
    if (plan.priority === 'high') {
      this.runner.clear();
    }

    const compiled = [];

    for (const action of (plan.actions || [])) {
      // Map ability name to capability type
      const capType = this._findCapabilityType(action.ability);
      if (!capType) {
        console.warn('[TwoTierLLM] Unknown ability:', action.ability);
        continue;
      }

      // Build brick
      const brick = { type: capType, ...action.args };

      // Validate
      const { ok, errors } = registry.validate(brick);
      if (!ok) {
        console.warn('[TwoTierLLM] Invalid brick:', errors);
        continue;
      }

      // Compile and add to runner
      try {
        const node = registry.compile(brick);
        compiled.push(node);
        this.runner.add(node);
      } catch (err) {
        console.warn('[TwoTierLLM] Compile failed:', err.message);
      }
    }

    if (compiled.length > 0) {
      console.log('[TwoTierLLM] Queued', compiled.length, 'actions');
    }
  }

  /**
   * Process strategic decision (might define new capabilities!)
   */
  async _processStrategicDecision(decision) {
    // If LLM wants to create a new capability
    if (decision.define_component) {
      console.log('[TwoTierLLM] LLM proposes new capability!');

      try {
        const { defineComponent } = await import('./llm/defineComponent.js');
        const result = await defineComponent(
          decision.define_component.manifest,
          decision.define_component.impl
        );

        if (result.ok) {
          console.log('[TwoTierLLM] ✅ New capability registered:', result.capabilityId);
        } else {
          console.warn('[TwoTierLLM] ❌ Capability rejected:', result.error);
        }
      } catch (err) {
        console.warn('[TwoTierLLM] Failed to define component:', err.message);
      }
    }

    // If LLM wants to adjust tactical behavior
    if (decision.adjustments) {
      console.log('[TwoTierLLM] Applying strategic adjustments:', decision.adjustments);

      // Could update config.tacticalMinInterval, etc.
      // For now, just log
    }

    // If LLM provides a full script
    if (decision.script) {
      this.runner.load(decision.script);
      console.log('[TwoTierLLM] Loaded strategic script');
    }
  }

  /**
   * Map ability name (snake_case) to capability type
   */
  _findCapabilityType(abilityName) {
    const search = abilityName.toLowerCase().replace(/_/g, '');

    for (const capId of Object.keys(registry.validators)) {
      const [group, name] = capId.split(':');
      const capName = name.split('@')[0].toLowerCase();

      if (capName === search || capName.replace(/[^a-z]/g, '') === search) {
        return capId;
      }
    }

    return null;
  }

  /**
   * Record key moments for strategic analysis
   */
  _recordHistory(players) {
    // Only record key moments to save memory
    if (!this._isKeyMoment(players)) return;

    const boss = this.bossMgr;
    if (!boss || boss.count <= 0) return;

    const moment = {
      timestamp: Date.now(),
      playerCount: players.filter(p => p && p.health > 0).length,
      bossHp: boss.hp[0],
      bossHpPercent: boss.hp[0] / boss.maxHp[0],
      bossAction: boss.lastAction || null,
      bulletCount: this.bulletMgr.count || 0
    };

    this.gameplayHistory.push(moment);

    // Limit history size
    if (this.gameplayHistory.length > this.config.maxHistorySize) {
      this.gameplayHistory.shift();
    }
  }

  /**
   * Determine if this moment is worth recording
   */
  _isKeyMoment(players) {
    const boss = this.bossMgr;
    if (!boss || boss.count <= 0) return false;

    const bossHpPercent = boss.hp[0] / boss.maxHp[0];

    // Record if:
    // - Boss HP changed significantly (>10%)
    // - Player joined/left
    // - Boss used a new action
    // - Random 5% sampling

    // For now, simple: record every 10th frame
    return Math.random() < 0.1;
  }

  /**
   * Execute queued actions from runner
   */
  _drainActionQueue(dt) {
    const q = this.bossMgr.actionQueue[0];
    while (q.length) {
      const node = q[0];
      const finished = registry.invoke(node, {
        dt,
        bossMgr: this.bossMgr,
        bulletMgr: this.bulletMgr,
        mapMgr: this.mapMgr,
        enemyMgr: this.enemyMgr
      });
      if (finished) q.shift();
      else break;
    }
  }

  /**
   * Get current metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      historySize: this.gameplayHistory.length,
      tacticalCooldown: this.tacticalCooldown,
      timeSinceStrategic: (Date.now() - this.lastStrategicCall) / 1000
    };
  }

  /**
   * Force a tactical decision now (for testing)
   */
  async forceTacticalDecision(players) {
    this.tacticalCooldown = 0;
    this.lastTacticalHash = null;
    await this._tickTactical(0, players);
  }

  /**
   * Force a strategic decision now (for testing)
   */
  async forceStrategicDecision(players) {
    this.lastStrategicCall = 0;
    await this._tickStrategic(0, players);
  }
}
