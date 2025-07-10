// src/mutators/index.js
// Registry of low-level mutator functions that perform concrete actions each frame.
// Each mutator returns true when finished so the queue node can be removed.

import { dash } from './dash.js';
import { radial_burst } from './radial_burst.js';
import { wait } from './mutator_wait.js';
import { spawn_minions } from './spawn_minions.js';
import { cone_aoe } from './cone_aoe.js';
import { reposition } from './reposition.js';
import { taunt } from './taunt.js';
import { trace } from '@opentelemetry/api';
const tracer = trace.getTracer('game');

export const Mutators = {
  'dash': dash,
  'radial_burst': radial_burst,
  'wait': wait,
  'spawn_minions': spawn_minions,
  'cone_aoe': cone_aoe,
  'reposition': reposition,
  'taunt': taunt
};

/**
 * Execute a single action node.
 * @param {Object} node – { ability, args, _state }
 * @param {number} dt  – deltaTime
 * @param {*} bossMgr
 * @param {*} bulletMgr
 * @param {*} mapMgr
 * @param {*} enemyMgr
 * @returns {boolean} finished?
 */
export function runMutator(node, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  const fn = Mutators[node.ability];
  if (!fn) {
    console.warn(`[Mutators] Unknown ability '${node.ability}'`);
    return true;
  }
  if (!node._state) node._state = {}; // per-action scratch data

  return tracer.startActiveSpan(`mutator.${node.ability}`, span => {
    try {
      return fn(node._state, node.args || {}, dt, bossMgr, bulletMgr, mapMgr, enemyMgr);
    } finally {
      span.end();
    }
  });
} 