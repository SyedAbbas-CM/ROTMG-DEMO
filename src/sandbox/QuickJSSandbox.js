// src/sandbox/QuickJSSandbox.js
// Minimal stub for QuickJS-based script execution.

import { EventEmitter } from 'events';

export class QuickJSSandbox extends EventEmitter {
  static _cache = new Map();

  /**
   * Retrieve or create a sandbox instance associated with a unique code hash.
   * @param {string} hash – Content hash used for caching.
   * @param {string} code – Source code to execute inside the sandbox.
   * @param {{cpuMs?:number,memKB?:number}} opts – Budget hints (ignored for now).
   */
  static async get(hash, code = '', opts = {}) {
    if (!QuickJSSandbox._cache.has(hash)) {
      QuickJSSandbox._cache.set(hash, new QuickJSSandbox(hash, code, opts));
    }
    return QuickJSSandbox._cache.get(hash);
  }

  constructor(hash, code, opts) {
    super();
    this.hash = hash;
    this.code = code;
    this.opts = opts;
    // Future: instantiate QuickJS runtime & evaluate code here.
  }

  /**
   * Step the sandboxed script. Currently a no-op placeholder.
   * @param {object} entity – The entity context provided by the engine.
   * @param {number} dt – Delta-time in seconds.
   */
  update(entity, dt) {
    // Intentionally empty – replace with real bridge to QuickJS once implemented.
  }

  dispose() {
    QuickJSSandbox._cache.delete(this.hash);
  }
} 