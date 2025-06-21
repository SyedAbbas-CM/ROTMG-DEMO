// src/BehaviorTree.js – Minimal runtime to mirror RotMG-style state machine trees
// This file is server-side only (Node.js). It does NOT import any DOM code.

import { Shoot as ShootBehaviour, Wander as WanderBehaviour, Chase as ChaseBehaviour } from './Behaviors.js';

/**
 * Leaf behaviour registry – map JSON "type" strings to the Behaviour classes we already have.
 */
const BEHAVIOUR_REGISTRY = {
  Shoot: ShootBehaviour,
  Wander: WanderBehaviour,
  Chase: ChaseBehaviour,
  // Add more as needed
};

// ---------------------------------------------------------------------------
// Core classes
// ---------------------------------------------------------------------------

export class StateNode {
  constructor({ name, behaviours = [], transitions = [], children = [] }) {
    this.name = name;
    // Instantiate behaviour leaves immediately
    this.behaviours = behaviours.map(b => new (BEHAVIOUR_REGISTRY[b.type] || WanderBehaviour)(...(b.args || [])));
    this.transitions = transitions.map(t => new Transition(t));
    this.children = children.map(c => new StateNode(c));
  }

  /**
   * Return child by name (1-level lookup).
   */
  getChild(name) {
    return this.children.find(c => c.name === name);
  }
}

export class Transition {
  constructor({ type, value, to }) {
    this.type = type;
    this.value = value;
    this.to = to; // target state name
  }

  /**
   * Evaluate whether this transition should fire.
   * @param {Object} ctx – runtime helper (enemyManager, index, timers, target, etc.)
   */
  evaluate(ctx) {
    switch (this.type) {
      case 'Timed':
        return ctx.stateTimer >= this.value;
      case 'PlayerWithin': {
        const dx = ctx.target.x - ctx.enemyManager.x[ctx.index];
        const dy = ctx.target.y - ctx.enemyManager.y[ctx.index];
        return dx * dx + dy * dy <= this.value * this.value;
      }
      default:
        return false;
    }
  }
}

// ---------------------------------------------------------------------------
// Parser helper – converts JSON (already JS object) to StateNode tree
// ---------------------------------------------------------------------------

export function parseBehaviourTree(rootJson) {
  return new StateNode(rootJson);
}

/**
 * Runtime helper that executes a tree for an individual enemy.
 * Stores currentStateName and timer in the EnemyManager arrays.
 */
export class BehaviourTreeRunner {
  constructor(rootNode) {
    this.root = rootNode;
  }

  /**
   * @param {number} index – enemy index
   * @param {EnemyManager} enemyManager
   * @param {BulletManager} bulletManager
   * @param {Object} target – usually player
   * @param {number} deltaTime – seconds
   */
  tick(index, enemyManager, bulletManager, target, deltaTime) {
    // Initialise per-enemy storage on first run
    if (!enemyManager._btState) {
      enemyManager._btState = {};
    }
    const store = enemyManager._btState;
    if (!store[index]) {
      store[index] = { current: this.root, timer: 0 };
    }

    const s = store[index];
    const ctx = { enemyManager, index, target, stateTimer: s.timer, bulletManager };

    // Execute behaviours of current state
    for (const beh of s.current.behaviours) {
      beh.execute(index, enemyManager, bulletManager, target, deltaTime, {});
    }

    // Check transitions
    for (const tr of s.current.transitions) {
      if (tr.evaluate(ctx)) {
        const next = s.current.getChild(tr.to) || this.root.getChild(tr.to);
        if (next) {
          s.current = next;
          s.timer = 0;
        }
        break;
      }
    }

    // Update timer
    s.timer += deltaTime;
  }
} 