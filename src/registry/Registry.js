// src/registry/Registry.js
// Lightweight Capability Registry MVP.
//  – Stores JSON Schemas for each capability (brick type)
//  – Validates a single brick instance
//  – Compiles a brick into an engine-specific action-node understood by the existing Mutator layer.
//
// This is intentionally minimal; the full hot-reload, dynamic directory walk and
// version negotiation will arrive in a later sprint.

import Ajv from 'ajv';

/* =====================================================================
   Built-in capability schemas (hard-coded for now)
   ===================================================================== */

const SCHEMAS = {
  // --------------------------------------------------------------------------------
  // Emitter:RadialBurst@1.0.0 – radial projectile fan around the boss
  // --------------------------------------------------------------------------------
  'Emitter:RadialBurst@1.0.0': {
    $id: 'Emitter:RadialBurst@1.0.0',
    title: 'Emitter:RadialBurst',
    type: 'object',
    required: ['type'],
    properties: {
      type: { const: 'Emitter:RadialBurst@1.0.0' },
      projectiles: {
        type: 'integer',
        minimum: 1,
        maximum: 360,
        default: 12,
      },
      speed: {
        type: 'number',
        minimum: 0.1,
        maximum: 100,
        default: 8,
      },
    },
    additionalProperties: false,
  },

  // --------------------------------------------------------------------------------
  // Movement:Dash@1.0.0 – short, fast positional burst
  // --------------------------------------------------------------------------------
  'Movement:Dash@1.0.0': {
    $id: 'Movement:Dash@1.0.0',
    title: 'Movement:Dash',
    type: 'object',
    required: ['type'],
    properties: {
      type: { const: 'Movement:Dash@1.0.0' },
      dx: { type: 'number', default: 1 },
      dy: { type: 'number', default: 0 },
      speed: { type: 'number', minimum: 0.1, maximum: 100, default: 10 },
      duration: { type: 'number', minimum: 0, maximum: 10, default: 0.5 },
    },
    additionalProperties: false,
  },

  // --------------------------------------------------------------------------------
  // Core:Wait@1.0.0 – no-op spacer used for timing control
  // --------------------------------------------------------------------------------
  'Core:Wait@1.0.0': {
    $id: 'Core:Wait@1.0.0',
    title: 'Core:Wait',
    type: 'object',
    required: ['type'],
    properties: {
      type: { const: 'Core:Wait@1.0.0' },
      // No params yet – reserved for future extension (e.g., conditional wait)
    },
    additionalProperties: false,
  },
};

/* =====================================================================
   Registry implementation
   ===================================================================== */

class CapabilityRegistry {
  constructor() {
    this.ajv = new Ajv({ strict: false });
    this.validators = {};
    // Compilers that originate from dynamically discovered capabilities
    this.dynamicCompilers = {};

    // Compile validators eagerly (cheap for the handful of built-ins)
    for (const [name, schema] of Object.entries(SCHEMAS)) {
      this.validators[name] = this.ajv.compile(schema);
    }
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
    // Prefer dynamically discovered compiler if present
    if (brick?.type && this.dynamicCompilers[brick.type]) {
      return this.dynamicCompilers[brick.type](brick);
    }

    switch (brick.type) {
      case 'Emitter:RadialBurst@1.0.0':
        return {
          ability: 'radial_burst',
          args: {
            projectiles: brick.projectiles ?? 12,
            speed: brick.speed ?? 8,
          },
        };

      case 'Movement:Dash@1.0.0':
        return {
          ability: 'dash',
          args: {
            dx: brick.dx ?? 0,
            dy: brick.dy ?? 0,
            speed: brick.speed ?? 10,
            duration: brick.duration ?? 0.5,
          },
        };

      case 'Core:Wait@1.0.0':
        return { ability: 'wait', args: {} };

      default:
        throw new Error(`[Registry] Unimplemented compile for '${brick.type}'`);
    }
  }
}

export const registry = new CapabilityRegistry();

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