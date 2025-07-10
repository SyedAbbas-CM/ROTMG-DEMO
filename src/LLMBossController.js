// src/LLMBossController.js
// Coordinates BossManager with remote LLM planner.

import { runMutator } from './mutators/index.js';
import ScriptBehaviourRunner from './ScriptBehaviourRunner.js';
import { createProvider } from './llm/ProviderFactory.js';
import { logLLM } from './llm/llmLogger.js';
import xxhash32 from 'xxhash-wasm';
import { trace } from '@opentelemetry/api';
import { evaluate } from './critic/DifficultyCritic.js';

const HASH_SEED = 0xABCD1234;
const PLAN_PERIOD = 3.0;   // seconds between LLM calls when state changed
const BACKOFF_SEC = 6.0;  // cooldown after error

let provider; // lazy singleton
const hashReady = xxhash32();
const tracer = trace.getTracer('game');


export default class LLMBossController {
  constructor(bossMgr, bulletMgr, mapMgr, enemyMgr) {
    this.bossMgr   = bossMgr;
    this.bulletMgr = bulletMgr;
    this.mapMgr    = mapMgr;
    this.enemyMgr  = enemyMgr;

    this.runner    = new ScriptBehaviourRunner(bossMgr, bulletMgr, enemyMgr);
    this.lastHash  = null;
    this.cooldown  = 0;           // seconds until next allowed LLM call
    this.tickCount = 0;
    this.pendingLLM = false; // guard to avoid overlapping provider calls
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

    // 2. Call LLM provider if cooldown elapsed and snapshot changed
    this.cooldown -= dt;
    if (this.pendingLLM) return; // Only one in-flight call allowed
    if (this.cooldown <= 0) {
      const snap = this.bossMgr.buildSnapshot(players, this.tickCount);
      if (!snap) return;

      const hapi = await hashReady;
      const newHash = hapi.h32(JSON.stringify(snap), HASH_SEED);
      if (newHash !== this.lastHash) {
        this.lastHash = newHash;
        this.cooldown = PLAN_PERIOD;
        try {
          if (!provider) provider = createProvider();

          this.pendingLLM = true;
          const span = tracer.startSpan('boss.plan');
          const sentAt = Date.now();
          const { json: res, deltaMs, tokens } = await provider.generate(snap);
          this.pendingLLM = false;

          // Persist raw output for later analysis
          logLLM({ ts: sentAt, snapshot: snap, result: res, deltaMs, tokens });

          span.setAttribute('latency_ms', deltaMs);
          span.setAttribute('tokens', tokens);
          span.end();

          if (res?.script) {
            this.runner.load(res.script);
          } else if (res?.actions) {
            this._ingestPlan(res);
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
          this.pendingLLM = false;
        }
      } else {
        // unchanged snapshot – postpone
        this.cooldown = 1.0;
      }
    }
  }

  _ingestPlan(plan) {
    if (!plan) return;
    if (plan.priority === 'high') this.runner.clear();
    (plan.actions || []).forEach(a => this.runner.add(a));
    console.log(`[LLMBoss] Ingested ${plan.actions?.length||0} actions`);
  }

  _drainActionQueue(dt) {
    const q = this.bossMgr.actionQueue[0];
    while (q.length) {
      const node = q[0];
      const finished = runMutator(node, dt, this.bossMgr, this.bulletMgr, this.mapMgr, this.enemyMgr);
      if (finished) q.shift(); else break;
    }
  }
} 