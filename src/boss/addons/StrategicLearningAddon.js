// src/boss/addons/StrategicLearningAddon.js
// Strategic tier - long-interval batch analysis and capability generation

import { createProvider } from '../llm/ProviderFactory.js';
import { GameplayHistoryRecorder } from './GameplayHistoryRecorder.js';
import { trace } from '@opentelemetry/api';

const tracer = trace.getTracer('game');

/**
 * Strategic learning addon - handles long-interval batch analysis
 * Works alongside existing tactical LLM controller
 */
export class StrategicLearningAddon {
  constructor(tacticalController, config = {}) {
    this.tactical = tacticalController; // Reference to existing controller

    // Configuration
    this.strategicModel = config.strategicModel || process.env.STRATEGIC_MODEL || 'models/gemini-2.0-flash';
    this.interval = config.strategicInterval || 300; // 5 minutes default
    this.enabled = config.strategicEnabled !== false;

    // State
    this.history = new GameplayHistoryRecorder({ maxSize: 100 });
    this.lastStrategicCall = Date.now();
    this.strategicProvider = null;
    this.pendingStrategic = false;

    console.log('[StrategicAddon] Initialized', {
      model: this.strategicModel,
      interval: this.interval,
      enabled: this.enabled
    });
  }

  /**
   * Tick - called alongside tactical controller
   */
  async tick(dt, players) {
    if (!this.enabled) return;

    // Record current snapshot
    const snapshot = this.tactical.bossMgr.buildSnapshot(
      players,
      this.tactical.bulletMgr,
      this.tactical.tickCount
    );

    if (snapshot) {
      this._recordHistory(snapshot);
    }

    // Check if it's time for strategic analysis
    const timeSinceLastCall = (Date.now() - this.lastStrategicCall) / 1000;
    if (timeSinceLastCall >= this.interval && !this.pendingStrategic) {
      await this._performStrategicAnalysis(snapshot);
    }
  }

  /**
   * Record gameplay history for batching
   */
  _recordHistory(snapshot) {
    // Determine if this is a significant event
    let eventType = 'normal';

    const boss = snapshot.boss;
    if (boss) {
      const hpPercent = boss.hp / boss.maxHp;

      // Mark significant events
      if (hpPercent < 0.25) {
        eventType = 'low_hp';
      } else if (snapshot.players && snapshot.players.length > 2) {
        eventType = 'multi_player';
      }
    }

    this.history.record(snapshot, eventType);
  }

  /**
   * Perform strategic batch analysis
   */
  async _performStrategicAnalysis(currentSnapshot) {
    console.log('[StrategicAddon] Starting strategic analysis...');

    try {
      this.pendingStrategic = true;

      // Gather historical data
      const metrics = this.history.getAggregateMetrics();
      const keyMoments = this.history.getKeyMoments(10);

      if (!metrics || keyMoments.length === 0) {
        console.log('[StrategicAddon] Insufficient history, skipping');
        this.lastStrategicCall = Date.now();
        return;
      }

      // Build strategic prompt with batched history
      const strategicSnapshot = {
        type: 'strategic_batch',
        currentState: currentSnapshot,
        sessionMetrics: metrics,
        keyMoments: keyMoments,
        tacticalFeedback: this.tactical.feedback.slice(-10), // Last 10 tactical decisions
        timestamp: Date.now()
      };

      // Create strategic provider if not exists
      if (!this.strategicProvider) {
        this.strategicProvider = createProvider({
          backend: 'gemini',
          model: this.strategicModel,
          temperature: 0.9, // Higher temperature for creativity
          maxTokens: 4096   // More tokens for complex analysis
        });
      }

      const span = tracer.startSpan('boss.strategic_plan');
      const sentAt = Date.now();

      console.log('[StrategicAddon] Calling LLM with batched history...', {
        keyMoments: keyMoments.length,
        sessionDuration: metrics.sessionDuration
      });

      const { json: res, deltaMs, tokens } = await this.strategicProvider.generate(strategicSnapshot);

      span.setAttribute('latency_ms', deltaMs);
      span.setAttribute('tokens', tokens);
      span.setAttribute('history_size', keyMoments.length);
      span.end();

      console.log('[StrategicAddon] Strategic analysis complete', {
        latencyMs: deltaMs,
        tokens
      });

      // Process strategic response
      if (res) {
        await this._processStrategicResponse(res);
      }

      // Clear history after successful analysis
      this.history.clear();
      this.lastStrategicCall = Date.now();

    } catch (err) {
      console.warn('[StrategicAddon] Strategic analysis failed', err.message);
      // Don't reset timer on failure - retry sooner
      this.lastStrategicCall = Date.now() - (this.interval * 0.5);
    } finally {
      this.pendingStrategic = false;
    }
  }

  /**
   * Process strategic response - typically new capability definitions
   */
  async _processStrategicResponse(response) {
    console.log('[StrategicAddon] Processing strategic response:', response);

    // Strategic tier should focus on define_component (new capabilities)
    if (response.define_component) {
      console.log('[StrategicAddon] New capability suggested by strategic tier');

      // Inject into tactical controller's ingestion pipeline
      // This reuses ALL existing validation, compilation, and safety checks
      await this.tactical._ingestPlan({
        define_component: response.define_component,
        actions: [] // No immediate actions, just capability definition
      });
    }

    // Strategic tier can also suggest high-level patterns
    if (response.strategic_guidance) {
      console.log('[StrategicAddon] Strategic guidance:', response.strategic_guidance);
      // Could be used to influence future tactical decisions
      // For now, just log it
    }
  }
}
