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

// Ultra-flexible mutators for complex behavior composition
import { spawn_formation } from './spawn_formation.js';
import { teleport_to_player } from './teleport_to_player.js';
import { charge_attack } from './charge_attack.js';
import { heal_self } from './heal_self.js';
import { shield_phase } from './shield_phase.js';
import { summon_orbitals } from './summon_orbitals.js';
import { pattern_shoot } from './pattern_shoot.js';
import { dynamic_movement } from './dynamic_movement.js';
import { conditional_trigger } from './conditional_trigger.js';
import { effect_aura } from './effect_aura.js';
import { environment_control } from './environment_control.js';

export const Mutators = {
  // Basic mutators
  'dash': dash,
  'radial_burst': radial_burst,
  'wait': wait,
  'spawn_minions': spawn_minions,
  'cone_aoe': cone_aoe,
  'reposition': reposition,
  'taunt': taunt,
  
  // Advanced formation and positioning
  'spawn_formation': spawn_formation,
  'teleport_to_player': teleport_to_player,
  'dynamic_movement': dynamic_movement,
  
  // Combat abilities
  'charge_attack': charge_attack,
  'pattern_shoot': pattern_shoot,
  'summon_orbitals': summon_orbitals,
  
  // Status and effects
  'heal_self': heal_self,
  'shield_phase': shield_phase,
  'effect_aura': effect_aura,
  
  // Behavior control
  'conditional_trigger': conditional_trigger,
  
  // Environment manipulation
  'environment_control': environment_control
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