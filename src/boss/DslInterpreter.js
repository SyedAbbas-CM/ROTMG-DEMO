// src/DslInterpreter.js
// Minimal stateless interpreter for DSL v1 nodes.
// Supported node types: sequence, parallel, wait, plus any capability brick.
// Each tick returns an array of compiled mutator nodes ready to enqueue.

import { registry } from '../registry/index.js';
import { trace } from '@opentelemetry/api';

const tracer = trace.getTracer('game');

export default class DslInterpreter {
  constructor() {
    this.reset();
  }

  reset() {
    this._root = null;    // original nodes tree
    this._stack = [];     // execution stack for sequence/parallel
    this._waitTime = 0;   // seconds remaining for current wait
  }

  load(nodes) {
    this.reset();
    this._root = nodes;
    // Start with top-level sequence over provided nodes
    this._stack.push({ idx: 0, list: nodes, parallel: false });
  }

  /**
   * Tick interpreter and collect mutator nodes.
   * @param {number} dt seconds
   * @returns {object[]} compiledNodes
   */
  tick(dt) {
    if (!this._stack.length) return [];

    let out = [];
    const span = tracer.startSpan('dsl.tick');
    try {
      this._process(dt, out);
    } finally {
      span.end();
    }
    return out;
  }

  _process(dt, out) {
    // Simple model: depth-first sequence execution; parallel lists tracked separately.
    // Handle wait timer first
    if (this._waitTime > 0) {
      this._waitTime -= dt;
      return;
    }

    // Work top of stack
    while (this._stack.length) {
      const frame = this._stack[this._stack.length - 1];
      if (frame.idx >= frame.list.length) {
        // finished this frame
        this._stack.pop();
        continue;
      }
      const node = frame.list[frame.idx];

      // Advance idx for next pass unless parallel collects its own
      frame.idx += 1;

      if (!node || typeof node !== 'object') continue;

      if (node.type === 'wait' || node.type?.startsWith('Core:Wait')) {
        this._waitTime = node.duration ?? node.t ?? 0;
        return; // defer rest until wait done
      }
      if (node.type === 'sequence') {
        this._stack.push({ idx: 0, list: node.steps || [], parallel: false });
        continue; // loop to process inner sequence
      }
      if (node.type === 'parallel') {
        // push each step into its own stack frame but mark parallel
        for (const step of node.steps || []) {
          this._stack.push({ idx: 0, list: [step], parallel: true });
        }
        // parallel frames will process next loop iteration
        continue;
      }

      // shorthand form { parallel:[...] }
      if (Array.isArray(node.parallel)) {
        for (const step of node.parallel) {
          this._stack.push({ idx: 0, list: [step], parallel: true });
        }
        continue;
      }

      // shorthand { sequence:[...] }
      if (Array.isArray(node.sequence)) {
        this._stack.push({ idx: 0, list: node.sequence, parallel: false });
        continue;
      }

      // Otherwise treat as capability brick
      const { ok, errors } = registry.validate(node);
      if (!ok) {
        console.warn('[DslInterpreter] Brick validation failed', errors);
        continue;
      }
      try {
        const compiled = registry.compile(node);
        if (compiled) out.push(compiled);
      } catch (err) {
        console.warn('[DslInterpreter] Compile error', err.message);
      }

      if (!frame.parallel) {
        // for sequential frame, process only one leaf per tick
        break;
      }
    }
  }
} 