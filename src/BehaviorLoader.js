import { BehaviorState } from './BehaviorState.js';
import * as Behaviors from './Behaviors.js';
import * as Transitions from './Transitions.js';
import BehaviorSystem from './BehaviorSystem.js';

/**
 * Convert a plain-object behaviour template into a live BehaviourState tree and
 * register it under the supplied name.
 *
 * Template format example (JSON):
 * {
 *   "name": "basicShooter",
 *   "initialState": "idle",
 *   "states": {
 *     "idle": {
 *       "behaviors": [ { "type": "Wander", "speed": 0.6, "duration": 2 } ],
 *       "transitions": [ { "type": "PlayerWithinRange", "range": 60, "to": "attack" } ]
 *     },
 *     "attack": {
 *       "behaviors": [
 *         { "type": "Chase", "speed": 1.0, "minDistance": 30 },
 *         { "type": "Shoot", "cooldownMultiplier": 1.0, "projectileCount": 1, "spread": 0 }
 *       ],
 *       "transitions": [ { "type": "NoPlayerWithinRange", "range": 80, "to": "idle" } ]
 *     }
 *   }
 * }
 */
export function registerBehaviorTemplateFromJSON(template) {
  if (!template || !template.name || !template.states) {
    console.error('[BehaviorLoader] Invalid template', template);
    return null;
  }

  const stateObjMap = new Map();

  // First pass: create empty BehaviorState objects so transitions can resolve
  Object.entries(template.states).forEach(([stateName]) => {
    stateObjMap.set(stateName, new BehaviorState(stateName));
  });

  // Second pass: populate behaviors & transitions
  Object.entries(template.states).forEach(([stateName, def]) => {
    const state = stateObjMap.get(stateName);
    if (!state) return;

    /* -------- Behaviors -------- */
    if (Array.isArray(def.behaviors)) {
      for (const bDef of def.behaviors) {
        const { type, ...args } = bDef;
        if (!type || !Behaviors[type]) {
          console.warn(`[BehaviorLoader] Unknown behavior type '${type}' in state '${stateName}'`);
          continue;
        }
        try {
          // eslint-disable-next-line new-cap
          const behaviorInstance = new Behaviors[type](...Object.values(args));
          state.addBehavior(behaviorInstance);
        } catch (err) {
          console.error('[BehaviorLoader] Failed to instantiate behavior', type, err);
        }
      }
    }

    /* -------- Transitions -------- */
    if (Array.isArray(def.transitions)) {
      for (const tDef of def.transitions) {
        const { type, to: targetName, ...args } = tDef;
        if (!type || !Transitions[type]) {
          console.warn(`[BehaviorLoader] Unknown transition type '${type}' in state '${stateName}'`);
          continue;
        }
        const targetState = stateObjMap.get(targetName);
        if (!targetState) {
          console.warn(`[BehaviorLoader] Target state '${targetName}' not found for transition in '${stateName}'`);
          continue;
        }
        try {
          // eslint-disable-next-line new-cap
          const transitionInstance = new Transitions[type](...Object.values(args), targetState);
          state.addTransition(transitionInstance);
        } catch (err) {
          console.error('[BehaviorLoader] Failed to instantiate transition', type, err);
        }
      }
    }
  });

  const rootState = stateObjMap.get(template.initialState) || stateObjMap.values().next().value;
  if (!rootState) {
    console.error('[BehaviorLoader] Could not determine root state for template', template.name);
    return null;
  }

  // Register in the global BehaviorSystem so EnemyManager can reference by name
  BehaviorSystem.registerBehaviorTemplate(template.name, rootState);
  console.log(`[BehaviorLoader] Registered behaviour template '${template.name}' with root state '${rootState.name}'`);
  return rootState;
} 