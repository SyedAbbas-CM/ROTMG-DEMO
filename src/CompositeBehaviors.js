// src/CompositeBehaviors.js
// Collection of higher-level composite / decorator behaviours inspired by the
// original RotMG C# server code.  They are intentionally lightweight – enough
// for port-parity with existing enemy XMLs but not yet feature-complete.
//
// All classes extend the common Behavior base and therefore have access to
// the per-behaviour private storage via _getStore(stateData).

import { Behavior } from './BehaviorState.js';

/**
 * Sequence – run an array of behaviours every tick, in order.
 * In this simplified port we do not wait for children to “finish”; instead all
 * children are executed sequentially each frame.
 */
export class Sequence extends Behavior {
  constructor(children = []) {
    super();
    this.children = children;
  }
  execute(index, enemyManager, bulletManager, target, dt, stateData) {
    for (const child of this.children) {
      child.execute(index, enemyManager, bulletManager, target, dt, stateData);
    }
  }
}

/**
 * Prioritize – execute child behaviours top-to-bottom until one of them performs
 * an action (best-effort heuristic: first one that moves the enemy).  Because
 * our Behaviour API does not expose explicit status, we approximate by checking
 * for a position delta larger than a small epsilon.
 */
export class Prioritize extends Behavior {
  constructor(children = []) {
    super();
    this.children = children;
  }
  execute(index, enemyManager, bulletManager, target, dt, stateData) {
    const beforeX = enemyManager.x[index];
    const beforeY = enemyManager.y[index];
    for (const child of this.children) {
      child.execute(index, enemyManager, bulletManager, target, dt, stateData);
      const moved = (enemyManager.x[index] !== beforeX) || (enemyManager.y[index] !== beforeY);
      if (moved) break; // give priority to first behaviour that moved the enemy
    }
  }
}

/**
 * Timed – run a child behaviour for a fixed duration, then become a no-op.
 */
export class Timed extends Behavior {
  constructor(duration = 1.0, child) {
    super();
    this.duration = duration;
    this.child = child;
  }
  init(stateData) {
    const store = this._getStore(stateData);
    store.elapsed = 0;
  }
  execute(index, enemyManager, bulletManager, target, dt, stateData) {
    const store = this._getStore(stateData);
    if (store.elapsed >= this.duration) return;
    this.child.execute(index, enemyManager, bulletManager, target, dt, stateData);
    store.elapsed += dt;
  }
}

/**
 * ConditionalBehavior – wrapper that only executes the child when `predicate`
 * returns true.
 */
export class ConditionalBehavior extends Behavior {
  constructor(predicateFn, child) {
    super();
    this.predicateFn = predicateFn;
    this.child = child;
  }
  execute(index, enemyManager, bulletManager, target, dt, stateData) {
    if (this.predicateFn(index, enemyManager, target)) {
      this.child.execute(index, enemyManager, bulletManager, target, dt, stateData);
    }
  }
}

/**
 * WhileEntityWithin – execute child while the target is within a given range.
 */
export class WhileEntityWithin extends Behavior {
  constructor(range = 100, child) {
    super();
    this.rangeSq = range * range;
    this.child = child;
  }
  execute(index, enemyManager, bulletManager, target, dt, stateData) {
    if (!target) return;
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    if ((dx * dx + dy * dy) <= this.rangeSq) {
      this.child.execute(index, enemyManager, bulletManager, target, dt, stateData);
    }
  }
}

/**
 * WhileEntityNotWithin – execute child while the target is outside a given range.
 */
export class WhileEntityNotWithin extends Behavior {
  constructor(range = 100, child) {
    super();
    this.rangeSq = range * range;
    this.child = child;
  }
  execute(index, enemyManager, bulletManager, target, dt, stateData) {
    if (!target) return;
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    if ((dx * dx + dy * dy) > this.rangeSq) {
      this.child.execute(index, enemyManager, bulletManager, target, dt, stateData);
    }
  }
}

/**
 * Suicide – instantly kills the enemy (used by some RotMG mobs).
 */
export class Suicide extends Behavior {
  execute(index, enemyManager /*, bulletManager, target, dt, stateData */) {
    if (enemyManager && typeof enemyManager.applyDamage === 'function') {
      enemyManager.applyDamage(index, enemyManager.health[index]);
    } else {
      // Fallback: flag as zero HP so EnemyManager update treats it as dead
      enemyManager.health[index] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// Helper to register these behaviours with existing factories, if needed.
// For now the BehaviourSystem / BehaviourTree import them explicitly.
// --------------------------------------------------------------------------- 