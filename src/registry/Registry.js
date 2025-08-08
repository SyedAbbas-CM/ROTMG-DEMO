// src/registry/Registry.js
// Lightweight Capability Registry MVP.
//  – Stores JSON Schemas for each capability (brick type)
//  – Validates a single brick instance
//  – Compiles a brick into an engine-specific action-node understood by the existing Mutator layer.
//
// This is intentionally minimal; the full hot-reload, dynamic directory walk and
// version negotiation will arrive in a later sprint.

import Ajv from 'ajv';
import { trace } from '@opentelemetry/api';
import { EventEmitter } from 'events';
const tracer = trace.getTracer('game');

/* =====================================================================
   Built-in capability schemas (hard-coded for now)
   ===================================================================== */

const SCHEMAS = {
  // Built-ins that do NOT exist on disk (keep minimal).

  'Core:Wait@1.0.0': {
    $id: 'Core:Wait@1.0.0',
    title: 'Core:Wait',
    type: 'object',
    required: ['type'],
    properties: {
      type: { const: 'Core:Wait@1.0.0' },
      duration: { type: 'number', minimum: 0, maximum: 10, default: 1 }
    },
    additionalProperties: false,
  },

  // Minimal built-ins for tests and fallback when dynamic load is disabled
  'Emitter:RadialBurst@1.0.0': {
    $id: 'Emitter:RadialBurst@1.0.0',
    title: 'Emitter:RadialBurst',
    type: 'object',
    required: ['type', 'projectiles'],
    properties: {
      type: { const: 'Emitter:RadialBurst@1.0.0' },
      projectiles: { type: 'number', minimum: 1, maximum: 400, default: 6 },
    },
    additionalProperties: true,
  },
  'Movement:Dash@1.0.0': {
    $id: 'Movement:Dash@1.0.0',
    title: 'Movement:Dash',
    type: 'object',
    required: ['type'],
    properties: {
      type: { const: 'Movement:Dash@1.0.0' },
      dx: { type: 'number', default: 3 },
      dy: { type: 'number', default: 0 },
      duration: { type: 'number', minimum: 0, maximum: 5, default: 0.3 },
    },
    additionalProperties: true,
  },
};

/* =====================================================================
   Registry implementation
   ===================================================================== */

class CapabilityRegistry {
  constructor() {
    this.events = new EventEmitter();
    this.ajv = new Ajv({ strict: false });
    this.validators = {};
    // Compilers that originate from dynamically discovered capabilities
    this.dynamicCompilers = {};
    this.invokers = {};

    // Compile validators eagerly (cheap for the handful of built-ins)
    for (const [name, schema] of Object.entries(SCHEMAS)) {
      this.validators[name] = this.ajv.compile(schema);
    }

    // Built-in invokers for core capabilities
    this.invokers['Core:Wait@1.0.0'] = (node, state = {}, { dt }) => {
      state.elapsed = (state.elapsed || 0) + dt;
      return state.elapsed >= (node.duration ?? node.args?.duration ?? 1);
    };

    // Minimal no-op invokers to satisfy interpreter contracts in tests
    this.invokers['Emitter:RadialBurst@1.0.0'] = (_node, _state, _ctx) => true;
    this.invokers['Movement:Dash@1.0.0'] = (node, state = {}, { dt }) => {
      state.elapsed = (state.elapsed || 0) + dt;
      return state.elapsed >= (node.args?.duration ?? node.duration ?? 0.3);
    };
  }

  /**
   * Validate a single brick instance.
   * @param {object} brick – JSON object emitted by LLM / designer.
   * @return {{ ok: boolean, errors?: any[] }}
   */
  validate(brick) {
    const fn = this.validators[brick?.type];
    if (!fn) return { ok: false, errors: [`Unknown capability type '${brick?.type}'`] };
    const res = fn(brick);
    return res ? { ok: true } : { ok: false, errors: fn.errors };
  }

  /**
   * Compile a validated brick into the action-node that the engine's Mutator
   * system understands today.  In the future this will dynamically import a
   * capability-specific JS module; for now we encode the mapping inline.
   *
   * @param {object} brick – previously validated brick JSON
   * @return {object} node – { ability, args }
   */
  compile(brick) {
    const span = tracer.startSpan('registry.compile');
    const start = Date.now();
    // Prefer dynamically discovered compiler if present
    if (brick?.type && this.dynamicCompilers[brick.type]) {
      const res = this.dynamicCompilers[brick.type](brick);
      span.setAttribute('duration_ms', Date.now() - start);
      span.end();
      return res;
    }

    switch (brick.type) {
      case 'Core:Wait@1.0.0':
        const n3 = { ability: 'wait', args: {} };
        n3._capType = brick.type;
        return n3;

      case 'Emitter:RadialBurst@1.0.0':
        return {
          ability: 'emit_radial',
          args: { projectiles: brick.projectiles ?? 6 },
          _capType: brick.type,
        };

      case 'Movement:Dash@1.0.0':
        return {
          ability: 'dash',
          args: {
            dx: brick.dx ?? 3,
            dy: brick.dy ?? 0,
            duration: brick.duration ?? 0.3,
          },
          _capType: brick.type,
        };

      default:
        throw new Error(`[Registry] Unimplemented compile for '${brick.type}'`);
    }
  }

  /** Execute compiled node. ctx = { dt, bossMgr, bulletMgr, mapMgr, enemyMgr } */
  invoke(node, ctx) {
    const type = node._capType;
    const inv = this.invokers[type];
    if (!inv) {
      console.warn('[Registry] No invoker for', type);
      return true;
    }
    node._state = node._state || {};
    return inv(node, node._state, ctx);
  }

  /** Merge new validators & compilers (hot-reload). */
  merge({ validators = {}, compilers = {}, invokers = {} } = {}) {
    Object.assign(this.validators, validators);
    Object.assign(this.dynamicCompilers, compilers);
    Object.assign(this.invokers, invokers);
    this.events.emit('update');
  }
}

export const registry = new CapabilityRegistry();

// At module init, attempt to discover capabilities under src/capabilities and merge.
// Skip dynamic loading during tests to prevent Jest teardown warnings.
if (process.env.NODE_ENV !== 'test') {
  (async () => {
    try {
      const { loadCapabilities } = await import('./DirectoryLoader.js');
      const { validators, compilers, invokers } = await loadCapabilities();
      registry.merge({ validators, compilers, invokers });
      console.log('[Registry] Loaded', Object.keys(validators).length, 'dynamic capabilities');

      // Start watcher for hot reloads once on startup
      const { watchCapabilities } = await import('./DirectoryLoader.js');
      const watcher = watchCapabilities();
      watcher.on('change', payload => {
        registry.merge(payload);
        console.log('[Registry] Hot-reloaded capabilities', Object.keys(payload.validators).length);
        // Optionally regenerate TS types
        import('../../ci-tools/generateTypes.js').catch(()=>{});
      });
    } catch (err) {
      console.warn('[Registry] Dynamic capability load failed', err.message);
    }
  })();
}

/**
 * Build a registry that is pre-populated with capabilities found on disk.
 * @param {{baseDir?:string}} opts
 * @returns {Promise<CapabilityRegistry>}
 */
export async function buildRegistry(opts = {}) {
  const reg = new CapabilityRegistry();
  try {
    const { loadCapabilities } = await import('./DirectoryLoader.js');
    const { validators, compilers } = await loadCapabilities(opts.baseDir);

    Object.assign(reg.validators, validators);
    Object.assign(reg.dynamicCompilers, compilers);
  } catch (err) {
    console.warn('[Registry] Failed to load dynamic capabilities:', err);
  }
  return reg;
} 