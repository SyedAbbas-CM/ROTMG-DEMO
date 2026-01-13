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
  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    for (const child of this.children) {
      child.execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem);
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
  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    const beforeX = enemyManager.x[index];
    const beforeY = enemyManager.y[index];
    for (const child of this.children) {
      child.execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem);
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
  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    const store = this._getStore(stateData);
    if (store.elapsed >= this.duration) return;
    this.child.execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem);
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
  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    if (this.predicateFn(index, enemyManager, target)) {
      this.child.execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem);
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
  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    if (!target) return;
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    if ((dx * dx + dy * dy) <= this.rangeSq) {
      this.child.execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem);
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
  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    if (!target) return;
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    if ((dx * dx + dy * dy) > this.rangeSq) {
      this.child.execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem);
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

/**
 * TeleportToTarget – instantly move to within range of target
 * Based on RotMG's teleport behaviors for assassin-type enemies
 */
export class TeleportToTarget extends Behavior {
  /**
   * @param {number} range - Distance from target to teleport to
   * @param {number} cooldown - Cooldown between teleports (seconds)
   */
  constructor(range = 8, cooldown = 2) {
    super();
    this.range = range;
    this.cooldown = cooldown;
  }

  init(stateData) {
    const store = this._getStore(stateData);
    store.timer = this.cooldown; // Ready to teleport immediately
  }

  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    if (!target) return;

    const store = this._getStore(stateData);
    store.timer = (store.timer || 0) + dt;

    if (store.timer >= this.cooldown) {
      store.timer = 0;

      // Calculate random position within range of target
      const angle = Math.random() * Math.PI * 2;
      const dist = this.range * (0.5 + Math.random() * 0.5);

      enemyManager.x[index] = target.x + Math.cos(angle) * dist;
      enemyManager.y[index] = target.y + Math.sin(angle) * dist;
    }
  }
}

/**
 * TossObject – periodically spawn child entities around the enemy
 * Based on RotMG's spawner behaviors
 */
export class TossObject extends Behavior {
  /**
   * @param {number|string} childType - Type/ID of enemy to spawn
   * @param {number} range - Range to spawn within
   * @param {number} cooldown - Cooldown between spawns (seconds)
   */
  constructor(childType = 0, range = 5, cooldown = 3) {
    super();
    this.childType = childType;
    this.range = range;
    this.cooldown = cooldown;
  }

  init(stateData) {
    const store = this._getStore(stateData);
    store.timer = 0;
  }

  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    const store = this._getStore(stateData);
    store.timer = (store.timer || 0) + dt;

    if (store.timer >= this.cooldown) {
      store.timer = 0;

      // Calculate spawn position
      const angle = Math.random() * Math.PI * 2;
      const dist = Math.random() * this.range;
      const spawnX = enemyManager.x[index] + Math.cos(angle) * dist;
      const spawnY = enemyManager.y[index] + Math.sin(angle) * dist;

      // Spawn child enemy
      if (typeof this.childType === 'string') {
        enemyManager.spawnEnemyById(this.childType, spawnX, spawnY, enemyManager.worldId[index]);
      } else {
        enemyManager.spawnEnemy(this.childType, spawnX, spawnY, enemyManager.worldId[index]);
      }
    }
  }
}

/**
 * Protect – stay close to and guard another entity type
 * Based on RotMG's protection behaviors
 */
export class Protect extends Behavior {
  /**
   * @param {number|string} protectType - Type/ID of entity to protect
   * @param {number} speed - Movement speed multiplier
   * @param {number} acquireRange - Range to detect protected entity
   * @param {number} protectionRange - Desired distance from protected entity
   * @param {number} reprotectRange - Range to re-acquire protection target
   */
  constructor(protectType = 0, speed = 1, acquireRange = 10, protectionRange = 2, reprotectRange = 1) {
    super();
    this.protectType = protectType;
    this.speed = speed;
    this.acquireRange = acquireRange;
    this.protectionRange = protectionRange;
    this.reprotectRange = reprotectRange;
  }

  execute(index, enemyManager, bulletManager, target, dt, stateData, behaviorSystem) {
    const store = this._getStore(stateData);

    // Find entity to protect
    let protectTarget = null;
    let minDist = this.acquireRange;

    for (let i = 0; i < enemyManager.enemyCount; i++) {
      if (i === index || enemyManager.health[i] <= 0) continue;

      const matches = (typeof this.protectType === 'string')
        ? enemyManager.id[i] === this.protectType
        : enemyManager.type[i] === this.protectType;

      if (matches) {
        const dx = enemyManager.x[i] - enemyManager.x[index];
        const dy = enemyManager.y[i] - enemyManager.y[index];
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < minDist) {
          minDist = dist;
          protectTarget = { x: enemyManager.x[i], y: enemyManager.y[i] };
        }
      }
    }

    if (!protectTarget) return;

    // Move toward protected entity if too far
    const dx = protectTarget.x - enemyManager.x[index];
    const dy = protectTarget.y - enemyManager.y[index];
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > this.protectionRange) {
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
      const moveAmount = Math.min(moveSpeed * dt, dist - this.reprotectRange);

      if (dist > 0) {
        enemyManager.x[index] += (dx / dist) * moveAmount;
        enemyManager.y[index] += (dy / dist) * moveAmount;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Helper to register these behaviours with existing factories, if needed.
// For now the BehaviourSystem / BehaviourTree import them explicitly.
// --------------------------------------------------------------------------- 