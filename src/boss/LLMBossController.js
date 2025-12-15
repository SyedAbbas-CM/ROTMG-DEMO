// src/LLMBossController.js
// Coordinates BossManager with remote LLM planner.

import { registry } from '../registry/index.js';
import ScriptBehaviourRunner from '../ScriptBehaviourRunner.js';
import { createProvider } from './llm/ProviderFactory.js';
import { logLLM } from './llm/llmLogger.js';
import xxhash32 from 'xxhash-wasm';
import { trace } from '@opentelemetry/api';
import { evaluate } from './critic/DifficultyCritic.js';
import llmConfig from './config/llmConfig.js';
import { getAdaptiveCooldown } from './addons/AdaptiveFrequency.js';
import { StrategicLearningAddon } from './addons/StrategicLearningAddon.js';

const HASH_SEED = 0xABCD1234;
const PLAN_PERIOD = llmConfig.planPeriodSec;
const BACKOFF_SEC = llmConfig.backoffSec;

let provider; // lazy singleton
const hashApiPromise = xxhash32();
const tracer = trace.getTracer('game');


export default class LLMBossController {
  constructor(bossMgr, bulletMgr, mapMgr, enemyMgr, config = {}) {
    this.bossMgr   = bossMgr;
    this.bulletMgr = bulletMgr;
    this.mapMgr    = mapMgr;
    this.enemyMgr  = enemyMgr;

    this.runner    = new ScriptBehaviourRunner(bossMgr, bulletMgr, enemyMgr);
    this.lastHash  = null;
    this.cooldown  = 0;           // seconds until next allowed LLM call
    this.tickCount = 0;
    this.pendingLLM = false; // guard to avoid overlapping provider calls
    this.feedback = [];

    // NEW: Optional configuration for enhancements
    this.config = {
      adaptiveFrequency: config.adaptiveFrequency !== false, // Default: enabled
      tacticalMinInterval: config.tacticalMinInterval || 10,
      tacticalMaxInterval: config.tacticalMaxInterval || 30,
      strategicEnabled: config.strategicEnabled || false,    // Default: disabled (opt-in)
      strategicModel: config.strategicModel,
      strategicInterval: config.strategicInterval || 300,
      tacticalModel: config.tacticalModel
    };

    // NEW: Initialize strategic learning addon if enabled
    if (this.config.strategicEnabled) {
      this.strategic = new StrategicLearningAddon(this, this.config);
      console.log('[LLMBoss] Strategic learning addon enabled');
    } else {
      this.strategic = null;
    }
  }

  /**
   * Main update, called every server frame.
   */
  async tick(dt, players) {
    this.tickCount++;
    // 1. run queued mutators originating from runner
    this._drainActionQueue(dt);

    // 1b. tick script runner (adds to queue)
    this.runner.tick(dt);

    // 1c. NEW: Tick strategic learning addon if enabled
    if (this.strategic) {
      await this.strategic.tick(dt, players);
    }

    // 2. Call LLM provider if cooldown elapsed and snapshot changed
    this.cooldown -= dt;
    if (this.pendingLLM) return; // Only one in-flight call allowed
    if (this.cooldown <= 0) {
      const snap = this.bossMgr.buildSnapshot(players, this.bulletMgr, this.tickCount);
      import('./llm/llmLogger.js').then(m=>m.logLLM({type:'snapshot',tick:this.tickCount,data:snap})).catch(()=>{});
      if (!snap) return;
      // Attach recent feedback memory (last 5 rated decisions)
      snap.feedback = this.feedback.slice(-5);

      const hapi = await hashApiPromise;
      const newHash = hapi.h32(JSON.stringify(snap), HASH_SEED);
      if (newHash !== this.lastHash) {
        this.lastHash = newHash;

        // NEW: Use adaptive cooldown if enabled, otherwise use fixed period
        if (this.config.adaptiveFrequency) {
          this.cooldown = getAdaptiveCooldown(snap, this.config);
        } else {
          this.cooldown = PLAN_PERIOD;
        }
        try {
          // NEW: Support tactical model override
          if (!provider) {
            if (this.config.tacticalModel) {
              provider = createProvider({
                backend: 'gemini',
                model: this.config.tacticalModel,
                temperature: 0.7,
                maxTokens: 1024
              });
            } else {
              provider = createProvider();
            }
          }

          this.pendingLLM = true;
          const span = tracer.startSpan('boss.plan');
          const sentAt = Date.now();
          const { json: res, deltaMs, tokens } = await provider.generate(snap);
          // Log raw LLM actions/explanation (if any)
          import('./llm/llmLogger.js').then(m=>m.logLLM({type:'llm_res',res})).catch(()=>{});

          // Persist raw output for later analysis
          logLLM({ ts: sentAt, snapshot: snap, result: res, deltaMs, tokens });

          span.setAttribute('latency_ms', deltaMs);
          span.setAttribute('tokens', tokens);
          span.end();

          if (res?.script) {
            this.runner.load(res.script);
          } else if (res?.actions) {
            await this._ingestPlan(res);
          }

          // Live safety check via DifficultyCritic (optional metrics)
          if (res?.metrics) {
            const { ok, reasons } = evaluate(res.metrics, { tier: 'mid' });
            if (!ok) {
              console.warn('[LLMBoss] DifficultyCritic veto', reasons.join(','));
            }
          }
        } catch (err) {
          console.warn('[LLMBoss] LLM call failed', err.message);
          this.cooldown = BACKOFF_SEC;
        } finally {
          // Make sure the flag is cleared even on errors
          this.pendingLLM = false;
        }
      } else {
        // unchanged snapshot – postpone
        this.cooldown = 1.0;
      }
    }
  }

  async _ingestPlan(plan) {
    if (!plan) return;
    if (plan.explain) {
      import('./llm/llmLogger.js').then(m=>m.logLLM({type:'explain',text:plan.explain})).catch(()=>{});
      delete plan.explain;
    }

    const planId = 'p' + Date.now().toString(36);

    // --- Rate the plan simple heuristic / critic ---
    let rating = 0.5;
    if (plan.metrics) {
      const { ok } = evaluate(plan.metrics, { tier: 'mid' });
      rating = ok ? 1 : 0;
    } else if (plan.actions && plan.actions.length) {
      rating = Math.min(1, plan.actions.length / 8);
    }
    this.feedback.push({ planId, ts: Date.now(), rating, plan: plan.actions?.map(a=>a.ability) });

    // Persist rating for RLHF dataset
    import('./llm/llmLogger.js').then(m=>m.logLLMRating(planId, rating));

    // Handle dynamic capability creation first
    if (plan.define_component && plan.define_component.manifest && plan.define_component.impl) {
      try {
        const { defineComponent } = await import('../llm/defineComponent.js');
        const res = await defineComponent(plan.define_component.manifest, plan.define_component.impl);
        if (!res.ok) {
          console.warn('[LLMBoss] define_component rejected', res.error);
        } else {
          console.log('[LLMBoss] New capability installed');
        }
      } catch(err) {
        console.warn('[LLMBoss] define_component failed', err.message);
      }
    }

    // Build dynamic ability→capId map from registry for quick lookup
    const abilityMap = {};
    for (const capId of Object.keys(registry.validators)) {
      // capId pattern: Group:Name@version  -> produce snake_case name
      const [, name] = capId.split(':'); // Name@version
      const snake = name.split('@')[0]
        .replace(/([a-z0-9])([A-Z])/g, '$1_$2') // camel→snake
        .toLowerCase();
      abilityMap[snake] = capId;
    }

    if (plan.priority === 'high') this.runner.clear();

    const compiled = [];
    for (const act of (plan.actions || [])) {
      // Allow LLM to supply full capability id via act.type
      const capType = act.type || abilityMap[act.ability];
      if (!capType) {
        console.warn('[LLMBoss] Unknown ability from LLM', act.ability || act.type);
        continue;
      }
      const brick = { type: capType, ...act.args };
      const { ok, errors } = registry.validate(brick);
      if (!ok) {
        console.warn('[LLMBoss] Brick validation failed', errors);
        continue;
      }
      try {
        const node = registry.compile(brick);
        compiled.push(node);
        this.runner.add(node);
      } catch (err) {
        console.warn('[LLMBoss] Compile failed', err.message);
      }
    }

    console.dir({ fromLLM: plan, compiled }, { depth: 6 });
  }

  _drainActionQueue(dt) {
    const q = this.bossMgr.actionQueue[0];
    while (q.length) {
      const node = q[0];
      const finished = registry.invoke(node, { dt, bossMgr: this.bossMgr, bulletMgr: this.bulletMgr, mapMgr: this.mapMgr, enemyMgr: this.enemyMgr });
      if (finished) q.shift(); else break;
    }
  }
} 