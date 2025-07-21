// src/ScriptBehaviourRunner.js
// Executes declarative boss behaviour scripts (DSL v1)
// fs/path imports no longer needed after DSL removal

import { registry } from './registry/index.js';
import DslInterpreter from './DslInterpreter.js';

// OLD DSL imports removed (Mutators, Ajv)

// Utility: deep-clone helper used for state action queues
const clone = obj => JSON.parse(JSON.stringify(obj));

export default class ScriptBehaviourRunner {
  constructor(bossMgr, bulletMgr, enemyMgr, { bossIdx = 0 } = {}) {
    this.bossMgr = bossMgr;
    this.bulletMgr = bulletMgr;
    this.enemyMgr = enemyMgr;
    this.bossIdx = bossIdx;
    this.interpreter = new DslInterpreter();
    this.reset();
  }

  reset() {
    this.script = null;
    this.currentState = null;
    this.stateClock = 0;
    this.pending = [];
    this.afterTimer = null;
  }

  load(scriptObj) {
    this.script = scriptObj;
    if (Array.isArray(scriptObj?.nodes)) {
      // Switch to interpreter mode
      this.interpreter.load(scriptObj.nodes);
      console.log('[ScriptRunner] Script(nodes) loaded');
    } else {
      this._gotoState(scriptObj.entry);
      console.log('[ScriptRunner] Script loaded', scriptObj.meta?.name || 'unnamed');
    }
    return true;
  }

  tick(dt) {
    // Interpreter path
    if (this.script?.nodes) {
      const nodes = this.interpreter.tick(dt);
      const q = this.bossMgr.actionQueue[this.bossIdx];
      for (const n of nodes) q.push(n);
      return;
    }
    if (!this.script) return;
    this.stateClock += dt;

    for (let i = 0; i < this.pending.length; i++) {
      const ta = this.pending[i];
      if (ta.at !== undefined && this.stateClock >= ta.at) {
        this._emit(ta.do);
        this.pending.splice(i--, 1);
        continue;
      }
      if (ta.every !== undefined) {
        ta._accum = (ta._accum || 0) + dt;
        const untilOk = ta.until === undefined || this.stateClock <= ta.until;
        if (untilOk && ta._accum >= ta.every) {
          ta._accum -= ta.every;
          this._emit(ta.do);
        }
      }
    }

    if (this.afterTimer !== null) {
      this.afterTimer -= dt;
      if (this.afterTimer <= 0) {
        const tgt = this._firstTransitionTarget(this.currentState);
        this._gotoState(tgt);
      }
    }
  }

  // ---------- helpers ----------
  _emit(prim) {
    const nodeOrNodes = this._primitiveToNode(prim);
    if (!nodeOrNodes) return;
    const q = this.bossMgr.actionQueue[this.bossIdx];
    if (Array.isArray(nodeOrNodes)) {
      for (const n of nodeOrNodes) q.push(n);
    } else {
      q.push(nodeOrNodes);
    }
  }

  _primitiveToNode(p) {
    if (!p || typeof p !== 'object' || typeof p.type !== 'string') {
      console.warn('[ScriptRunner] Expected Brick JSON with "type", got', p);
      return null;
    }

    const { ok, errors } = registry.validate(p);
    if (!ok) {
      console.warn('[ScriptRunner] Brick validation failed', errors);
      return null;
    }
    try {
      return registry.compile(p);
    } catch (err) {
      console.warn('[ScriptRunner] Brick compile failed', err.message);
      return null;
    }
  }

  _cloneActionsOf(state) {
    const st = this.script.states[state];
    return (st?.actions || []).map(clone);
  }

  _computeAfter(state) {
    const aft = (this.script.states[state]?.transitions || []).find(t => t.after !== undefined);
    return aft ? aft.after : null;
  }

  _firstTransitionTarget(state) {
    const aft = (this.script.states[state]?.transitions || []).find(t => t.after !== undefined);
    return aft?.to || this.script.entry;
  }

  _gotoState(state) {
    if (!this.script.states[state]) {
      console.warn('[ScriptRunner] Unknown state', state);
      return;
    }
    this.currentState = state;
    this.stateClock = 0;
    this.pending = this._cloneActionsOf(state);
    this.afterTimer = this._computeAfter(state);
    console.log('[ScriptRunner] → state', state);
  }

  /** external push of single node (deprecated) */
  add(action) {
    // convert old-style action into pending queue for immediate emit
    this.bossMgr.actionQueue[this.bossIdx].push(action);
  }

  clear() { this.bossMgr.actionQueue[this.bossIdx].length = 0; }
} 