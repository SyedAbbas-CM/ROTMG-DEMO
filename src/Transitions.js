/**
 * Transitions.js - Collection of state transitions for the behavior system
 * These define the conditions for switching between states
 */

import { Transition } from './BehaviorState.js';

/**
 * Transition when player comes within a certain range
 */
export class PlayerWithinRange extends Transition {
  /**
   * @param {number} range - Detection range
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(range, targetState) {
    super(targetState);
    this.range = range;
    this.rangeSquared = range * range;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    if (!target) return false;
    
    // Calculate distance to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distanceSquared = dx * dx + dy * dy;
    
    // Transition if within range
    return distanceSquared <= this.rangeSquared;
  }
}

/**
 * Transition when no player is within a certain range
 */
export class NoPlayerWithinRange extends Transition {
  /**
   * @param {number} range - Detection range
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(range, targetState) {
    super(targetState);
    this.range = range;
    this.rangeSquared = range * range;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    if (!target) return true; // No player, so condition is met
    
    // Calculate distance to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distanceSquared = dx * dx + dy * dy;
    
    // Transition if outside range
    return distanceSquared > this.rangeSquared;
  }
}

/**
 * Transition when health falls below a percentage
 */
export class HealthBelow extends Transition {
  /**
   * @param {number} threshold - Health threshold (0.0 - 1.0)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(threshold, targetState) {
    super(targetState);
    this.threshold = threshold;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    const healthRatio = enemyManager.health[index] / enemyManager.maxHealth[index];
    return healthRatio < this.threshold;
  }
}

/**
 * Transition when health rises above a percentage
 */
export class HealthAbove extends Transition {
  /**
   * @param {number} threshold - Health threshold (0.0 - 1.0)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(threshold, targetState) {
    super(targetState);
    this.threshold = threshold;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    const healthRatio = enemyManager.health[index] / enemyManager.maxHealth[index];
    return healthRatio > this.threshold;
  }
}

/**
 * Transition after a random time within a range
 */
export class RandomTimer extends Transition {
  /**
   * @param {number} minTime - Minimum time before transition (seconds)
   * @param {number} maxTime - Maximum time before transition (seconds)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(minTime, maxTime, targetState) {
    super(targetState);
    this.minTime = minTime;
    this.maxTime = maxTime;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Initialize timer if needed
    if (!stateData.randomTimer) {
      stateData.randomTimer = 0;
      stateData.randomTimerTarget = this.minTime + Math.random() * (this.maxTime - this.minTime);
    }
    
    // Update timer
    stateData.randomTimer += deltaTime;
    
    // Check if timer is complete
    if (stateData.randomTimer >= stateData.randomTimerTarget) {
      // Reset timer for next use
      stateData.randomTimer = 0;
      stateData.randomTimerTarget = this.minTime + Math.random() * (this.maxTime - this.minTime);
      return true;
    }
    
    return false;
  }
}

/**
 * Transition after a fixed time
 */
export class TimedTransition extends Transition {
  /**
   * @param {number} time - Time before transition (seconds)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(time, targetState) {
    super(targetState);
    this.time = time;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Initialize timer if needed
    if (!stateData.timedTransitionTimer) {
      stateData.timedTransitionTimer = 0;
    }
    
    // Update timer
    stateData.timedTransitionTimer += deltaTime;
    
    // Check if timer is complete
    if (stateData.timedTransitionTimer >= this.time) {
      // Reset timer for next use
      stateData.timedTransitionTimer = 0;
      return true;
    }
    
    return false;
  }
}

/**
 * Transition with a random chance each tick
 */
export class RandomChance extends Transition {
  /**
   * @param {number} chance - Chance per second (0.0 - 1.0)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(chance, targetState) {
    super(targetState);
    this.chance = chance;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Scale chance by delta time (chance per second)
    const scaledChance = this.chance * deltaTime;
    return Math.random() < scaledChance;
  }
}

/**
 * Transition when damage taken exceeds threshold since state entry
 */
export class DamageTaken extends Transition {
  /**
   * @param {number} threshold - Damage threshold to trigger transition
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(threshold, targetState) {
    super(targetState);
    this.threshold = threshold;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Initialize damage counter if needed
    if (stateData.initialHealth === undefined) {
      stateData.initialHealth = enemyManager.health[index];
    }
    
    // Calculate damage taken
    const damageTaken = stateData.initialHealth - enemyManager.health[index];
    
    // Check if damage threshold is met
    return damageTaken >= this.threshold;
  }
}

/**
 * Composite AND transition - all conditions must be true
 */
export class AndTransition extends Transition {
  /**
   * @param {Array<Transition>} transitions - Array of transitions to check
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(transitions, targetState) {
    super(targetState);
    this.transitions = transitions;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // All sub-transitions must return true
    for (const transition of this.transitions) {
      if (!transition.check(index, enemyManager, target, deltaTime, stateData)) {
        return false;
      }
    }
    return true;
  }
}

/**
 * Composite OR transition - any condition can be true
 */
export class OrTransition extends Transition {
  /**
   * @param {Array<Transition>} transitions - Array of transitions to check
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(transitions, targetState) {
    super(targetState);
    this.transitions = transitions;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Any sub-transition can return true
    for (const transition of this.transitions) {
      if (transition.check(index, enemyManager, target, deltaTime, stateData)) {
        return true;
      }
    }
    return false;
  }
} 