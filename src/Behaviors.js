/**
 * Behaviors.js - Collection of behavior components for the behavior system
 * These components define what enemies do in each state
 */

import { Behavior } from './BehaviorState.js';

/**
 * Wander behavior - random movement in an area
 */
export class Wander extends Behavior {
  /**
   * @param {number} speed - Movement speed multiplier
   * @param {number} duration - How long to move in each direction (seconds)
   */
  constructor(speed = 1.0, duration = 3.0) {
    super();
    this.speed = speed;
    this.duration = duration;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize direction if needed
    if (!stateData.wanderDir) {
      stateData.wanderDir = { 
        x: Math.random() * 2 - 1, 
        y: Math.random() * 2 - 1 
      };
      stateData.wanderTimer = 0;
    }
    
    // Update timer
    stateData.wanderTimer += deltaTime;
    
    // Change direction if needed
    if (stateData.wanderTimer >= this.duration) {
      stateData.wanderDir = { 
        x: Math.random() * 2 - 1, 
        y: Math.random() * 2 - 1 
      };
      stateData.wanderTimer = 0;
    }
    
    // Normalize direction
    const length = Math.sqrt(
      stateData.wanderDir.x * stateData.wanderDir.x + 
      stateData.wanderDir.y * stateData.wanderDir.y
    );
    
    if (length > 0) {
      stateData.wanderDir.x /= length;
      stateData.wanderDir.y /= length;
    }
    
    // Calculate move amount based on enemy's base move speed
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const moveAmount = moveSpeed * deltaTime;
    
    // Apply movement
    enemyManager.x[index] += stateData.wanderDir.x * moveAmount;
    enemyManager.y[index] += stateData.wanderDir.y * moveAmount;
  }
}

/**
 * Chase behavior - move toward target
 */
export class Chase extends Behavior {
  /**
   * @param {number} speed - Speed multiplier (1.0 = normal speed)
   * @param {number} minDistance - Minimum distance to maintain from target
   */
  constructor(speed = 1.0, minDistance = 0) {
    super();
    this.speed = speed;
    this.minDistance = minDistance;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;
    
    // Calculate direction to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distanceSquared = dx * dx + dy * dy;
    
    // Skip if already at min distance
    if (distanceSquared <= this.minDistance * this.minDistance) {
      return;
    }
    
    // Normalize direction
    const distance = Math.sqrt(distanceSquared);
    const dirX = dx / distance;
    const dirY = dy / distance;
    
    // Calculate move amount based on enemy's base move speed
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const moveAmount = moveSpeed * deltaTime;
    
    // Apply movement
    enemyManager.x[index] += dirX * moveAmount;
    enemyManager.y[index] += dirY * moveAmount;
  }
}

/**
 * RunAway behavior - move away from target
 */
export class RunAway extends Behavior {
  /**
   * @param {number} speed - Speed multiplier (1.0 = normal speed)
   * @param {number} maxDistance - Maximum distance to run away
   */
  constructor(speed = 1.0, maxDistance = 500) {
    super();
    this.speed = speed;
    this.maxDistance = maxDistance;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;
    
    // Calculate direction from target (reversed)
    const dx = enemyManager.x[index] - target.x;
    const dy = enemyManager.y[index] - target.y;
    const distanceSquared = dx * dx + dy * dy;
    
    // Skip if already at max distance
    if (distanceSquared >= this.maxDistance * this.maxDistance) {
      return;
    }
    
    // Normalize direction
    const distance = Math.sqrt(distanceSquared);
    const dirX = dx / distance;
    const dirY = dy / distance;
    
    // Calculate move amount based on enemy's base move speed
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const moveAmount = moveSpeed * deltaTime;
    
    // Apply movement
    enemyManager.x[index] += dirX * moveAmount;
    enemyManager.y[index] += dirY * moveAmount;
  }
}

/**
 * Orbit behavior - circle around target
 */
export class Orbit extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for orbiting
   * @param {number} radius - Desired orbit radius
   * @param {number} acquireRange - Range to acquire target
   * @param {string} target - Target type to orbit
   * @param {number} speedVariance - Speed variance percentage
   * @param {number} radiusVariance - Radius variance percentage
   * @param {boolean} orbitClockwise - Whether to orbit clockwise
   */
  constructor(speed = 1.0, radius = 100, acquireRange = 10, target = null, speedVariance = null, radiusVariance = null, orbitClockwise = false) {
    super();
    this.speed = speed;
    this.radius = radius;
    this.acquireRange = acquireRange;
    this.targetType = target;
    this.speedVariance = speedVariance !== null ? speedVariance : speed * 0.1;
    this.radiusVariance = radiusVariance !== null ? radiusVariance : speed * 0.1;
    this.orbitClockwise = orbitClockwise;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize orbit state
    if (!stateData.orbitState) {
      const orbitDir = this.orbitClockwise === null ?
        (Math.random() < 0.5 ? 1 : -1) :
        (this.orbitClockwise ? 1 : -1);
        
      stateData.orbitState = {
        speed: this.speed + this.speedVariance * (Math.random() * 2 - 1),
        radius: this.radius + this.radiusVariance * (Math.random() * 2 - 1),
        direction: orbitDir
      };
    }
    
    const state = stateData.orbitState;
    
    // Check for paralysis or similar conditions
    if (enemyManager.paralyzed && enemyManager.paralyzed[index]) {
      return;
    }
    
    const orbitTarget = target || this.findNearestTarget(index, enemyManager);
    
    if (orbitTarget) {
      let angle;
      if (enemyManager.y[index] === orbitTarget.y && enemyManager.x[index] === orbitTarget.x) {
        // Small offset to avoid division by zero
        angle = Math.atan2(
          enemyManager.y[index] - orbitTarget.y + (Math.random() * 2 - 1),
          enemyManager.x[index] - orbitTarget.x + (Math.random() * 2 - 1)
        );
      } else {
        angle = Math.atan2(
          enemyManager.y[index] - orbitTarget.y,
          enemyManager.x[index] - orbitTarget.x
        );
      }
      
      const angularSpeed = state.direction * enemyManager.moveSpeed[index] * state.speed / state.radius;
      angle += angularSpeed * deltaTime;
      
      const targetX = orbitTarget.x + Math.cos(angle) * state.radius;
      const targetY = orbitTarget.y + Math.sin(angle) * state.radius;
      
      // Calculate movement vector
      const dx = targetX - enemyManager.x[index];
      const dy = targetY - enemyManager.y[index];
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance > 0) {
        const normalizedX = dx / distance;
        const normalizedY = dy / distance;
        const moveSpeed = enemyManager.moveSpeed[index] * state.speed * deltaTime;
        
        enemyManager.x[index] += normalizedX * moveSpeed;
        enemyManager.y[index] += normalizedY * moveSpeed;
      }
    }
  }
  
  findNearestTarget(index, enemyManager) {
    // Implementation depends on your enemy manager's target finding system
    // This is a simplified version
    return null; // Will use passed target parameter instead
  }
}

/**
 * Enhanced Swirl behavior - circular movement around target or center
 */
export class Swirl extends Behavior {
  /**
   * @param {number} speed - Movement speed multiplier
   * @param {number} radius - Swirl radius
   * @param {number} acquireRange - Range to acquire target
   * @param {boolean} targeted - Whether to target player or use fixed center
   */
  constructor(speed = 1.0, radius = 8, acquireRange = 10, targeted = true) {
    super();
    this.speed = speed;
    this.radius = radius;
    this.acquireRange = acquireRange;
    this.targeted = targeted;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize swirl state
    if (!stateData.swirlState) {
      stateData.swirlState = {
        centerX: this.targeted ? 0 : enemyManager.x[index],
        centerY: this.targeted ? 0 : enemyManager.y[index],
        acquired: !this.targeted,
        remainingTime: 0
      };
    }
    
    const state = stateData.swirlState;
    
    // Check for paralysis
    if (enemyManager.paralyzed && enemyManager.paralyzed[index]) {
      return;
    }
    
    const period = (1000 * this.radius / (enemyManager.moveSpeed[index] * this.speed) * (2 * Math.PI));
    
    if (!state.acquired && state.remainingTime <= 0 && this.targeted) {
      const swirlTarget = target;
      if (swirlTarget && swirlTarget.x !== enemyManager.x[index] && swirlTarget.y !== enemyManager.y[index]) {
        // Calculate circle that passes through host and player position
        const distance = Math.sqrt(
          Math.pow(swirlTarget.x - enemyManager.x[index], 2) +
          Math.pow(swirlTarget.y - enemyManager.y[index], 2)
        );
        
        const halfX = (enemyManager.x[index] + swirlTarget.x) / 2;
        const halfY = (enemyManager.y[index] + swirlTarget.y) / 2;
        const c = Math.sqrt(Math.abs(this.radius * this.radius - distance * distance / 4));
        
        state.centerX = halfX + c * (enemyManager.y[index] - swirlTarget.y) / distance;
        state.centerY = halfY + c * (swirlTarget.x - enemyManager.x[index]) / distance;
        
        state.remainingTime = period;
        state.acquired = true;
      } else {
        state.acquired = false;
      }
    } else if (state.remainingTime <= 0) {
      if (this.targeted) {
        state.acquired = false;
        state.remainingTime = target ? 0 : 5000;
      } else {
        state.remainingTime = 5000;
      }
    } else {
      state.remainingTime -= deltaTime * 1000;
    }
    
    // Calculate movement
    let angle;
    if (enemyManager.y[index] === state.centerY && enemyManager.x[index] === state.centerX) {
      // Small offset to avoid division by zero
      angle = Math.atan2(
        enemyManager.y[index] - state.centerY + (Math.random() * 2 - 1),
        enemyManager.x[index] - state.centerX + (Math.random() * 2 - 1)
      );
    } else {
      angle = Math.atan2(
        enemyManager.y[index] - state.centerY,
        enemyManager.x[index] - state.centerX
      );
    }
    
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed * (state.acquired ? 1 : 0.2);
    const angularSpeed = moveSpeed / this.radius;
    angle += angularSpeed * deltaTime;
    
    const targetX = state.centerX + Math.cos(angle) * this.radius;
    const targetY = state.centerY + Math.sin(angle) * this.radius;
    
    // Apply movement
    const dx = targetX - enemyManager.x[index];
    const dy = targetY - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > 0) {
      const normalizedX = dx / distance;
      const normalizedY = dy / distance;
      const frameSpeed = moveSpeed * deltaTime;
      
      enemyManager.x[index] += normalizedX * frameSpeed;
      enemyManager.y[index] += normalizedY * frameSpeed;
    }
  }
}

/**
 * Enhanced Grenade behavior - throws grenades with AOE damage and effects
 */
export class Grenade extends Behavior {
  /**
   * @param {number} radius - Explosion radius
   * @param {number} damage - Damage amount
   * @param {number} range - Throw range
   * @param {number} fixedAngle - Fixed angle in degrees (null for target-based)
   * @param {Object} cooldown - Cooldown configuration
   * @param {string} effect - Condition effect to apply
   * @param {number} effectDuration - Effect duration in ms
   * @param {number} color - Grenade color
   */
  constructor(radius = 2, damage = 100, range = 5, fixedAngle = null, cooldown = {min: 1000, max: 2000}, effect = null, effectDuration = 0, color = 0xffff0000) {
    super();
    this.radius = radius;
    this.damage = damage;
    this.range = range;
    this.fixedAngle = fixedAngle !== null ? (fixedAngle * Math.PI / 180) : null;
    this.cooldown = cooldown;
    this.effect = effect;
    this.effectDuration = effectDuration;
    this.color = color;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize cooldown state
    if (!stateData.grenadeCooldown) {
      stateData.grenadeCooldown = 0;
    }
    
    // Update cooldown
    if (stateData.grenadeCooldown > 0) {
      stateData.grenadeCooldown -= deltaTime * 1000;
      return;
    }
    
    // Check for stun
    if (enemyManager.stunned && enemyManager.stunned[index]) {
      return;
    }
    
    const grenadeTarget = target;
    if (grenadeTarget || this.fixedAngle !== null) {
      let targetX, targetY;
      
      if (this.fixedAngle !== null) {
        targetX = this.range * Math.cos(this.fixedAngle) + enemyManager.x[index];
        targetY = this.range * Math.sin(this.fixedAngle) + enemyManager.y[index];
      } else {
        targetX = grenadeTarget.x;
        targetY = grenadeTarget.y;
      }
      
      // Create grenade projectile with delayed explosion
      this.throwGrenade(index, enemyManager, bulletManager, targetX, targetY);
      
      // Set cooldown
      const cooldownRange = this.cooldown.max - this.cooldown.min;
      stateData.grenadeCooldown = this.cooldown.min + Math.random() * cooldownRange;
    }
  }
  
  throwGrenade(index, enemyManager, bulletManager, targetX, targetY) {
    // Add grenade as special projectile
    bulletManager.addBullet({
      x: enemyManager.x[index],
      y: enemyManager.y[index],
      vx: (targetX - enemyManager.x[index]) / 1.5, // 1.5 second flight time
      vy: (targetY - enemyManager.y[index]) / 1.5,
      ownerId: enemyManager.id[index],
      damage: 0, // No direct damage, only AOE
      lifetime: 1.5,
      width: 0.6,
      height: 0.6,
      isEnemy: true,
      isGrenade: true,
      explosionRadius: this.radius,
      explosionDamage: this.damage,
      explosionEffect: this.effect,
      explosionEffectDuration: this.effectDuration,
      spriteName: 'grenade',
      worldId: enemyManager.worldId[index]
    });
  }
}

/**
 * Shoot behavior - fire projectiles at target
 */
export class Shoot extends Behavior {
  /**
   * @param {number} cooldownMultiplier - Multiplier for cooldown (lower = faster shots)
   * @param {number} projectileCount - Number of projectiles to fire at once
   * @param {number} spread - Angular spread between projectiles (radians)
   * @param {number} inaccuracy - Random angle variation (radians)
   */
  constructor(cooldownMultiplier = 1.0, projectileCount = 1, spread = 0, inaccuracy = 0) {
    super();
    this.cooldownMultiplier = cooldownMultiplier;
    this.projectileCount = projectileCount;
    this.spread = spread;
    this.inaccuracy = inaccuracy;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target || !bulletManager) return;
    
    // Skip if enemy can't shoot
    if (!enemyManager.canShoot[index]) return;
    
    // Skip if on cooldown
    if (enemyManager.currentCooldown[index] > 0) return;
    
    // Calculate direction to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distanceSquared = dx * dx + dy * dy;
    
    // Skip if out of range
    if (distanceSquared > enemyManager.shootRange[index] * enemyManager.shootRange[index]) {
      return;
    }
    
    // Calculate angle to target
    const baseAngle = Math.atan2(dy, dx);
    
    // Reset cooldown
    enemyManager.currentCooldown[index] = enemyManager.cooldown[index] * this.cooldownMultiplier;

    // Fire projectiles
    this.fireProjectiles(index, enemyManager, bulletManager, baseAngle);
  }
  
  fireProjectiles(index, enemyManager, bulletManager, baseAngle) {
    // Push bullet so it starts just outside the enemy's own hit-box (enemyWidth/2 + small margin)
    const spawnOffset = (enemyManager.width[index] * 0.5) + 0.3; // tile-units

    // Single projectile case
    if (this.projectileCount <= 1) {
      // Apply inaccuracy if specified
      const angle = baseAngle + (Math.random() * 2 - 1) * this.inaccuracy;
      
      const spawnX = enemyManager.x[index] + Math.cos(angle) * spawnOffset;
      const spawnY = enemyManager.y[index] + Math.sin(angle) * spawnOffset;
      
      bulletManager.addBullet({
        x: spawnX,
        y: spawnY,
        vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
        vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: 3.0,
        width: 0.4,
        height: 0.4,
        isEnemy: true,
        spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
        worldId: enemyManager.worldId[index]
      });
      
      return;
    }
    
    // Multiple projectiles case
    const startAngle = baseAngle - (this.spread * (this.projectileCount - 1) / 2);
    
    for (let i = 0; i < this.projectileCount; i++) {
      // Base angle plus spread offset plus inaccuracy
      const angle = startAngle + (this.spread * i) + (Math.random() * 2 - 1) * this.inaccuracy;
      
      const spawnX = enemyManager.x[index] + Math.cos(angle) * spawnOffset;
      const spawnY = enemyManager.y[index] + Math.sin(angle) * spawnOffset;
      
      bulletManager.addBullet({
        x: spawnX,
        y: spawnY,
        vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
        vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: 3.0,
        width: 0.4,
        height: 0.4,
        isEnemy: true,
        spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
        worldId: enemyManager.worldId[index]
      });
    }
  }
}

/**
 * Charge behavior - rush quickly toward target
 */
export class Charge extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for charging
   * @param {number} duration - How long to charge (seconds)
   * @param {number} cooldown - Time between charges (seconds)
   */
  constructor(speed = 3.0, duration = 1.0, cooldown = 5.0) {
    super();
    this.speed = speed;
    this.duration = duration;
    this.cooldown = cooldown;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;

    // Initialize state data if needed
    if (!stateData.chargeState) {
      stateData.chargeState = 'ready';
      stateData.chargeTimer = 0;
      stateData.chargeCooldown = 0;
      stateData.chargeDir = { x: 0, y: 0 };
    }
    
    // Update timers
    if (stateData.chargeState === 'charging') {
      stateData.chargeTimer += deltaTime;
    } else if (stateData.chargeState === 'cooldown') {
      stateData.chargeCooldown += deltaTime;
    }
    
    // State machine for charge behavior
    switch (stateData.chargeState) {
      case 'ready':
        // Prepare to charge - calculate direction to target
        const dx = target.x - enemyManager.x[index];
        const dy = target.y - enemyManager.y[index];
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 0) {
          stateData.chargeDir = { 
            x: dx / distance, 
            y: dy / distance 
          };
        }
        
        stateData.chargeTimer = 0;
        stateData.chargeState = 'charging';
        break;
        
      case 'charging':
        // Execute charge movement
        const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
        const moveAmount = moveSpeed * deltaTime;
        
        enemyManager.x[index] += stateData.chargeDir.x * moveAmount;
        enemyManager.y[index] += stateData.chargeDir.y * moveAmount;
        
        // Check if charge duration is complete
        if (stateData.chargeTimer >= this.duration) {
          stateData.chargeState = 'cooldown';
          stateData.chargeCooldown = 0;
        }
        break;
        
      case 'cooldown':
        // Wait for cooldown to finish
        if (stateData.chargeCooldown >= this.cooldown) {
          stateData.chargeState = 'ready';
        }
        break;
    }
  }
}

/**
 * Flash behavior - temporarily become invulnerable/visible
 */
export class Flash extends Behavior {
  /**
   * @param {number} duration - How long to flash (seconds)
   * @param {number} cooldown - Time between flashes (seconds)
   */
  constructor(duration = 0.5, cooldown = 5.0) {
    super();
    this.duration = duration;
    this.cooldown = cooldown;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize state data if needed
    if (!stateData.flashState) {
      stateData.flashState = 'idle';
      stateData.flashTimer = 0;
      stateData.flashCooldown = 0;
    }
    
    // Update timers
    if (stateData.flashState === 'flashing') {
      stateData.flashTimer += deltaTime;
    } else if (stateData.flashState === 'cooldown') {
      stateData.flashCooldown += deltaTime;
    }
    
    // State machine for flash behavior
    switch (stateData.flashState) {
      case 'idle':
        // Randomly decide to start flashing
        if (Math.random() < 0.1 * deltaTime) {
          stateData.flashState = 'flashing';
          stateData.flashTimer = 0;
          
          // Set invulnerability flag (would be implemented in EnemyManager)
          // enemyManager.setInvulnerable(index, true);
        }
        break;
        
      case 'flashing':
        // Check if flash duration is complete
        if (stateData.flashTimer >= this.duration) {
          stateData.flashState = 'cooldown';
          stateData.flashCooldown = 0;
          
          // Remove invulnerability
          // enemyManager.setInvulnerable(index, false);
        }
        break;
        
      case 'cooldown':
        // Wait for cooldown to finish
        if (stateData.flashCooldown >= this.cooldown) {
          stateData.flashState = 'idle';
        }
        break;
    }
  }
}

/**
 * Follow behavior - Advanced following with states (DontKnowWhere, Acquired, Resting)
 * Based on the C# RotMG Follow behavior with state tracking
 */
export class Follow extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for following
   * @param {number} acquireRange - Range to detect and start following target
   * @param {number} loseRange - Range at which to lose target
   * @param {number} stopDistance - Distance to stop at when following
   */
  constructor(speed = 1.0, acquireRange = 120, loseRange = 160, stopDistance = 20) {
    super();
    this.speed = speed;
    this.acquireRange = acquireRange;
    this.loseRange = loseRange;
    this.stopDistance = stopDistance;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;
    
    // Initialize state data
    if (!stateData.followState) {
      stateData.followState = 'DontKnowWhere';
      stateData.lastKnownPos = null;
      stateData.restTimer = 0;
    }
    
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    switch (stateData.followState) {
      case 'DontKnowWhere':
        if (distance <= this.acquireRange) {
          stateData.followState = 'Acquired';
          stateData.lastKnownPos = { x: target.x, y: target.y };
        }
        break;
        
      case 'Acquired':
        if (distance > this.loseRange) {
          stateData.followState = 'DontKnowWhere';
          stateData.lastKnownPos = null;
        } else if (distance <= this.stopDistance) {
          stateData.followState = 'Resting';
          stateData.restTimer = 0;
        } else {
          // Move toward target
          const dirX = dx / distance;
          const dirY = dy / distance;
          const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
          const moveAmount = moveSpeed * deltaTime;
          
          enemyManager.x[index] += dirX * moveAmount;
          enemyManager.y[index] += dirY * moveAmount;
          
          stateData.lastKnownPos = { x: target.x, y: target.y };
        }
        break;
        
      case 'Resting':
        stateData.restTimer += deltaTime;
        
        if (distance > this.stopDistance * 1.5) {
          stateData.followState = 'Acquired';
        } else if (stateData.restTimer > 2.0) { // Rest for 2 seconds before potentially moving again
          if (distance > this.stopDistance) {
            stateData.followState = 'Acquired';
          }
        }
        break;
    }
  }
}

/**
 * MoveLine behavior - Move in a straight line with specified direction
 */
export class MoveLine extends Behavior {
  /**
   * @param {number} speed - Speed multiplier
   * @param {number} direction - Direction in degrees (0 = right, 90 = up)
   */
  constructor(speed = 1.0, direction = 0) {
    super();
    this.speed = speed;
    this.direction = direction * Math.PI / 180; // Convert to radians
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Check for paralysis
    if (enemyManager.paralyzed && enemyManager.paralyzed[index]) {
      return;
    }
    
    // Calculate movement vector
    const dirX = Math.cos(this.direction);
    const dirY = Math.sin(this.direction);
    
    // Apply movement
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const distance = moveSpeed * deltaTime;
    
    enemyManager.x[index] += dirX * distance;
    enemyManager.y[index] += dirY * distance;
  }
}

/**
 * MoveTo behavior - Move to a specific coordinate
 */
export class MoveTo extends Behavior {
  /**
   * @param {number} speed - Speed multiplier
   * @param {number} x - Target X coordinate
   * @param {number} y - Target Y coordinate
   */
  constructor(speed = 1.0, x = 0, y = 0) {
    super();
    this.speed = speed;
    this.targetX = x;
    this.targetY = y;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Check for paralysis
    if (enemyManager.paralyzed && enemyManager.paralyzed[index]) {
      return;
    }
    
    // Initialize completion state
    if (!stateData.moveToCompleted) {
      stateData.moveToCompleted = false;
    }
    
    // Skip if already completed
    if (stateData.moveToCompleted) {
      return;
    }
    
    // Calculate path to target
    const dx = this.targetX - enemyManager.x[index];
    const dy = this.targetY - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const frameDistance = moveSpeed * deltaTime;
    
    // Check if we can reach target this frame
    if (distance <= frameDistance) {
      // Move to exact target and mark completed
      enemyManager.x[index] = this.targetX;
      enemyManager.y[index] = this.targetY;
      stateData.moveToCompleted = true;
    } else {
      // Move toward target
      const dirX = dx / distance;
      const dirY = dy / distance;
      
      enemyManager.x[index] += dirX * frameDistance;
      enemyManager.y[index] += dirY * frameDistance;
    }
  }
}

/**
 * StayAbove behavior - Move toward center when below specified altitude
 */
export class StayAbove extends Behavior {
  /**
   * @param {number} speed - Speed multiplier
   * @param {number} altitude - Minimum altitude to maintain
   */
  constructor(speed = 1.0, altitude = 0) {
    super();
    this.speed = speed;
    this.altitude = altitude;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Check for paralysis
    if (enemyManager.paralyzed && enemyManager.paralyzed[index]) {
      return;
    }
    
    // Get current tile elevation (simplified - assumes flat world if no elevation data)
    const currentElevation = enemyManager.elevation ? enemyManager.elevation[index] : 0;
    
    if (currentElevation !== 0 && currentElevation < this.altitude) {
      // Calculate direction toward world center (simplified)
      const worldCenterX = 500; // Adjust based on your world size
      const worldCenterY = 500;
      
      const dx = worldCenterX - enemyManager.x[index];
      const dy = worldCenterY - enemyManager.y[index];
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance > 0) {
        const dirX = dx / distance;
        const dirY = dy / distance;
        
        const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
        const frameDistance = moveSpeed * deltaTime;
        
        enemyManager.x[index] += dirX * frameDistance;
        enemyManager.y[index] += dirY * frameDistance;
      }
    }
  }
}

/**
 * Swirl behavior - Complex circular movement around dynamic center points
 * Based on the C# RotMG Swirl behavior
 */
export class OldSwirl extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for swirling
   * @param {number} radius - Radius of the swirl
   * @param {boolean} clockwise - Direction of swirl
   * @param {boolean} useTargetAsCenter - Whether to swirl around target or spawn point
   */
  constructor(speed = 1.0, radius = 60, clockwise = true, useTargetAsCenter = false) {
    super();
    this.speed = speed;
    this.radius = radius;
    this.direction = clockwise ? 1 : -1;
    this.useTargetAsCenter = useTargetAsCenter;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize state data
    if (!stateData.swirlState) {
      stateData.swirlState = {
        centerX: enemyManager.x[index],
        centerY: enemyManager.y[index],
        angle: 0,
        phase: 0
      };
      
      // If spawning, store spawn position as center
      if (!this.useTargetAsCenter) {
        stateData.swirlState.centerX = enemyManager.x[index];
        stateData.swirlState.centerY = enemyManager.y[index];
      }
    }
    
    const state = stateData.swirlState;
    
    // Update center position if using target
    if (this.useTargetAsCenter && target) {
      state.centerX = target.x;
      state.centerY = target.y;
    }
    
    // Calculate angular speed based on enemy's move speed
    const baseSpeed = enemyManager.moveSpeed[index] * this.speed;
    const angularSpeed = (baseSpeed / this.radius) * this.direction;
    
    // Update angle and phase for complex patterns
    state.angle += angularSpeed * deltaTime;
    state.phase += deltaTime * 0.5; // Slower phase change for complexity
    
    // Add some variation to radius based on phase
    const radiusVariation = Math.sin(state.phase) * (this.radius * 0.2);
    const currentRadius = this.radius + radiusVariation;
    
    // Calculate new position
    const newX = state.centerX + Math.cos(state.angle) * currentRadius;
    const newY = state.centerY + Math.sin(state.angle) * currentRadius;
    
    // Move toward calculated position (smooth movement)
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const maxMove = moveSpeed * deltaTime;
    
    const moveX = newX - enemyManager.x[index];
    const moveY = newY - enemyManager.y[index];
    const moveLength = Math.sqrt(moveX * moveX + moveY * moveY);
    
    if (moveLength > maxMove) {
      const scaleFactor = maxMove / moveLength;
      enemyManager.x[index] += moveX * scaleFactor;
      enemyManager.y[index] += moveY * scaleFactor;
    } else {
      enemyManager.x[index] = newX;
      enemyManager.y[index] = newY;
    }
  }
}

/**
 * BackAndForth behavior - Move back and forth between two points
 * Based on the C# RotMG BackAndForth behavior
 */
export class BackAndForth extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for movement
   * @param {number} distance - Distance to travel in each direction
   */
  constructor(speed = 1.0, distance = 100) {
    super();
    this.speed = speed;
    this.distance = distance;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize state data
    if (!stateData.backForthState) {
      stateData.backForthState = {
        startX: enemyManager.x[index],
        startY: enemyManager.y[index],
        direction: Math.random() * Math.PI * 2, // Random initial direction
        movingForward: true,
        currentDistance: 0
      };
    }
    
    const state = stateData.backForthState;
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const moveAmount = moveSpeed * deltaTime;
    
    // Calculate movement direction
    const dirX = Math.cos(state.direction);
    const dirY = Math.sin(state.direction);
    
    if (state.movingForward) {
      // Move forward
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
      state.currentDistance += moveAmount;
      
      if (state.currentDistance >= this.distance) {
        state.movingForward = false;
        state.direction += Math.PI; // Reverse direction
      }
    } else {
      // Move backward
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
      state.currentDistance -= moveAmount;
      
      if (state.currentDistance <= 0) {
        state.movingForward = true;
        state.direction += Math.PI; // Reverse direction again
        state.currentDistance = 0;
      }
    }
  }
}

/**
 * Grenade behavior - Throw projectiles with delayed explosion and AOE
 * Based on the C# RotMG Grenade behavior
 */
