// BehaviorsCore.js – minimal, self-contained behaviour stubs needed for the enemy BehaviourSystem

import { Behavior } from './BehaviorState.js';

/*
 * NOTE:  These are intentionally very lightweight implementations – enough for the
 *        server to boot and for BehaviourSystem to instantiate the objects it
 *        expects.  They can be fleshed out later or replaced by capability-based
 *        behaviours once the old BehaviourSystem is fully retired.
 */

export class Wander extends Behavior {
  constructor(speed = 1.0, duration = 3.0) {
    super();
    this.speed = speed;
    this.duration = duration;
  }
  execute(/* index, enemyManager, bulletManager, target, deltaTime, stateData */) {
    // Placeholder – no-op on purpose.
  }
}

export class Chase extends Behavior {
  constructor(speed = 1.0, minDistance = 0) {
    super();
    this.speed = speed;
    this.minDistance = minDistance;
  }
  execute() { /* no-op */ }
}

export class RunAway extends Behavior {
  constructor(speed = 1.0, maxDistance = 500) {
    super();
    this.speed = speed;
    this.maxDistance = maxDistance;
  }
  execute() { /* no-op */ }
}

export class Orbit extends Behavior {
  constructor(speed = 1.0, radius = 100) {
    super();
    this.speed = speed;
    this.radius = radius;
  }
  execute() { /* no-op */ }
}

export class Swirl extends Behavior {
  constructor(speed = 1.0, radius = 8) {
    super();
    this.speed = speed;
    this.radius = radius;
  }
  execute() { /* no-op */ }
}

export class Shoot extends Behavior {
  constructor(cooldownMultiplier = 1.0, projectileCount = 1, spread = 0, inaccuracy = 0) {
    super();
    this.cooldownMultiplier = cooldownMultiplier;
    this.projectileCount = projectileCount;
    this.spread = spread;
    this.inaccuracy = inaccuracy;
  }
  execute() { /* no-op */ }
} 