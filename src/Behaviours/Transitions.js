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

/**
 * Transition when a specific entity exists
 * Based on the C# RotMG EntityExists transition
 */
export class EntityExists extends Transition {
  /**
   * @param {string|number} entityId - ID or type of entity to check for
   * @param {number} range - Range to search for entity (0 = anywhere)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(entityId, range = 0, targetState) {
    super(targetState);
    this.entityId = entityId;
    this.range = range;
    this.rangeSquared = range * range;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    const ex = enemyManager.x[index];
    const ey = enemyManager.y[index];
    
    // Search for entity
    for (let i = 0; i < enemyManager.enemyCount; i++) {
      if (i === index) continue; // Skip self
      if (enemyManager.health[i] <= 0) continue; // Skip dead entities
      
      // Check if entity matches
      const matches = (typeof this.entityId === 'string') 
        ? enemyManager.id[i] === this.entityId
        : enemyManager.type[i] === this.entityId;
      
      if (!matches) continue;
      
      // Check range if specified
      if (this.range > 0) {
        const dx = enemyManager.x[i] - ex;
        const dy = enemyManager.y[i] - ey;
        const distanceSquared = dx * dx + dy * dy;
        
        if (distanceSquared <= this.rangeSquared) {
          return true;
        }
      } else {
        return true; // Found entity, no range restriction
      }
    }
    
    return false;
  }
}

/**
 * Transition when a specific entity does not exist
 * Based on the C# RotMG EntityNotExists transition
 */
export class EntityNotExists extends Transition {
  /**
   * @param {string|number} entityId - ID or type of entity to check for
   * @param {number} range - Range to search for entity (0 = anywhere)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(entityId, range = 0, targetState) {
    super(targetState);
    this.entityId = entityId;
    this.range = range;
    this.rangeSquared = range * range;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Use EntityExists logic and invert result
    const entityExists = new EntityExists(this.entityId, this.range, null);
    return !entityExists.check(index, enemyManager, target, deltaTime, stateData);
  }
}

/**
 * Transition when any entity is within range
 * Based on the C# RotMG AnyEntityWithin transition
 */
export class AnyEntityWithin extends Transition {
  /**
   * @param {number} range - Detection range
   * @param {Array<number>} entityTypes - Array of entity types to check (empty = all types)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(range, entityTypes = [], targetState) {
    super(targetState);
    this.range = range;
    this.rangeSquared = range * range;
    this.entityTypes = entityTypes;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    const ex = enemyManager.x[index];
    const ey = enemyManager.y[index];
    
    // Check for any entity within range
    for (let i = 0; i < enemyManager.enemyCount; i++) {
      if (i === index) continue; // Skip self
      if (enemyManager.health[i] <= 0) continue; // Skip dead entities
      
      // Check entity type filter
      if (this.entityTypes.length > 0 && !this.entityTypes.includes(enemyManager.type[i])) {
        continue;
      }
      
      // Check distance
      const dx = enemyManager.x[i] - ex;
      const dy = enemyManager.y[i] - ey;
      const distanceSquared = dx * dx + dy * dy;
      
      if (distanceSquared <= this.rangeSquared) {
        return true;
      }
    }
    
    // Also check for players if target exists
    if (target) {
      const dx = target.x - ex;
      const dy = target.y - ey;
      const distanceSquared = dx * dx + dy * dy;
      
      if (distanceSquared <= this.rangeSquared) {
        return true;
      }
    }
    
    return false;
  }
}

/**
 * Transition when no entity is within range
 * Based on the C# RotMG NoEntityWithin transition
 */
export class NoEntityWithin extends Transition {
  /**
   * @param {number} range - Detection range
   * @param {Array<number>} entityTypes - Array of entity types to check (empty = all types)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(range, entityTypes = [], targetState) {
    super(targetState);
    this.range = range;
    this.entityTypes = entityTypes;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Use AnyEntityWithin logic and invert result
    const anyEntityWithin = new AnyEntityWithin(this.range, this.entityTypes, null);
    return !anyEntityWithin.check(index, enemyManager, target, deltaTime, stateData);
  }
}

/**
 * Transition when another entity's health is below threshold
 * Based on the C# RotMG EntityHpLess transition
 */
export class EntityHpLess extends Transition {
  /**
   * @param {string|number} entityId - ID or type of entity to check
   * @param {number} threshold - Health threshold (0.0 - 1.0)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(entityId, threshold, targetState) {
    super(targetState);
    this.entityId = entityId;
    this.threshold = threshold;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Find target entity
    for (let i = 0; i < enemyManager.enemyCount; i++) {
      if (i === index) continue; // Skip self
      if (enemyManager.health[i] <= 0) continue; // Skip dead entities
      
      // Check if entity matches
      const matches = (typeof this.entityId === 'string') 
        ? enemyManager.id[i] === this.entityId
        : enemyManager.type[i] === this.entityId;
      
      if (matches) {
        const healthRatio = enemyManager.health[i] / enemyManager.maxHealth[i];
        return healthRatio < this.threshold;
      }
    }
    
    return false;
  }
}

/**
 * Transition when entity is not moving
 * Based on the C# RotMG NotMoving transition
 */
export class NotMoving extends Transition {
  /**
   * @param {number} threshold - Movement threshold (speed below this = not moving)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(threshold = 0.1, targetState) {
    super(targetState);
    this.threshold = threshold;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Initialize previous position if needed
    if (!stateData.prevPosition) {
      stateData.prevPosition = {
        x: enemyManager.x[index],
        y: enemyManager.y[index],
        timestamp: Date.now()
      };
      return false;
    }
    
    // Calculate movement speed
    const dx = enemyManager.x[index] - stateData.prevPosition.x;
    const dy = enemyManager.y[index] - stateData.prevPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const timeElapsed = (Date.now() - stateData.prevPosition.timestamp) / 1000;
    
    if (timeElapsed > 0) {
      const speed = distance / timeElapsed;
      
      // Update previous position
      stateData.prevPosition = {
        x: enemyManager.x[index],
        y: enemyManager.y[index],
        timestamp: Date.now()
      };
      
      return speed < this.threshold;
    }
    
    return false;
  }
}

/**
 * Transition based on player text/chat
 * Based on the C# RotMG PlayerText transition
 */
export class PlayerText extends Transition {
  /**
   * @param {string} text - Text to match (case insensitive)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(text, targetState) {
    super(targetState);
    this.text = text.toLowerCase();
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // This would need integration with the chat system
    // For now, return false - implementation would depend on chat message tracking
    // TODO: Implement chat message integration
    return false;
  }
}

/**
 * Pure random transition with specified probability
 * Based on the C# RotMG RandomTransition
 */
export class RandomTransition extends Transition {
  /**
   * @param {number} probability - Probability per check (0.0 - 1.0)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(probability, targetState) {
    super(targetState);
    this.probability = probability;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    return Math.random() < this.probability;
  }
}

/**
 * Random transition with time-based probability
 * Based on the C# RotMG TimedRandomTransition
 */
export class TimedRandomTransition extends Transition {
  /**
   * @param {number} time - Time window for random check (seconds)
   * @param {number} probability - Probability within time window
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(time, probability, targetState) {
    super(targetState);
    this.time = time;
    this.probability = probability;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Initialize timer if needed
    if (!stateData.timedRandomTimer) {
      stateData.timedRandomTimer = 0;
    }
    
    // Update timer
    stateData.timedRandomTimer += deltaTime;
    
    // Check if time window has elapsed
    if (stateData.timedRandomTimer >= this.time) {
      stateData.timedRandomTimer = 0;
      return Math.random() < this.probability;
    }
    
    return false;
  }
}

/**
 * Transition when multiple entities don't exist
 * Based on the C# RotMG EntitiesNotExist transition
 */
export class EntitiesNotExist extends Transition {
  /**
   * @param {Array<string|number>} entityIds - Array of entity IDs or types to check
   * @param {number} range - Range to search (0 = anywhere)
   * @param {BehaviorState} targetState - State to transition to
   */
  constructor(entityIds, range = 0, targetState) {
    super(targetState);
    this.entityIds = entityIds;
    this.range = range;
  }
  
  check(index, enemyManager, target, deltaTime, stateData) {
    // Check that none of the specified entities exist
    for (const entityId of this.entityIds) {
      const entityExists = new EntityExists(entityId, this.range, null);
      if (entityExists.check(index, enemyManager, target, deltaTime, stateData)) {
        return false; // At least one entity exists
      }
    }
    
    return true; // None of the entities exist
  }
} 