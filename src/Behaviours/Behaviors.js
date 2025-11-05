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

    // Calculate new position
    const newX = enemyManager.x[index] + dirX * moveAmount;
    const newY = enemyManager.y[index] + dirY * moveAmount;

    // Get map bounds from enemyManager's mapManager reference (if available)
    // Otherwise assume a safe boundary of 5 tiles from edge
    const mapWidth = enemyManager.mapWidth || 512;
    const mapHeight = enemyManager.mapHeight || 512;
    const margin = 5; // Stay 5 tiles away from map edge

    // Enforce boundaries - prevent fleeing off map
    if (newX >= margin && newX < mapWidth - margin &&
        newY >= margin && newY < mapHeight - margin) {
      enemyManager.x[index] = newX;
      enemyManager.y[index] = newY;
    }
    // If at boundary, don't move (enemy is stuck at edge)
  }
}

/**
 * CavalryCharge behavior - charge with velocity phases
 * When slow: can turn freely and shoot
 * When charging: moves fast, limited turning, no shooting
 */
export class CavalryCharge extends Behavior {
  /**
   * @param {number} slowSpeed - Speed when in slow mode (default 0.5)
   * @param {number} chargeSpeed - Speed when charging (default 2.5)
   * @param {number} acceleration - How quickly to accelerate (default 3.0)
   * @param {number} chargeDuration - How long to charge before slowing (default 2.0s)
   * @param {number} slowDuration - How long to stay slow (default 1.5s)
   */
  constructor(slowSpeed = 0.5, chargeSpeed = 2.5, acceleration = 3.0, chargeDuration = 2.0, slowDuration = 1.5) {
    super();
    this.slowSpeed = slowSpeed;
    this.chargeSpeed = chargeSpeed;
    this.acceleration = acceleration;
    this.chargeDuration = chargeDuration;
    this.slowDuration = slowDuration;
  }

  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;

    // Initialize charge state
    if (!stateData.chargeState) {
      stateData.chargeState = {
        currentSpeed: this.slowSpeed,
        isCharging: false,
        timer: 0,
        lastDirX: 0,
        lastDirY: 0
      };
    }

    const state = stateData.chargeState;

    // Update timer and phase
    state.timer += deltaTime;

    if (state.isCharging && state.timer >= this.chargeDuration) {
      // Switch to slow mode
      state.isCharging = false;
      state.timer = 0;
    } else if (!state.isCharging && state.timer >= this.slowDuration) {
      // Switch to charge mode
      state.isCharging = true;
      state.timer = 0;
    }

    // Calculate direction to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance === 0) return;

    let dirX = dx / distance;
    let dirY = dy / distance;

    // Apply turn rate limiting when charging
    if (state.isCharging && state.lastDirX !== 0 && state.lastDirY !== 0) {
      // Calculate max turn angle per frame based on speed
      // Fast = can't turn much, slow = can turn more
      const maxTurnRate = 1.5 / (state.currentSpeed / this.slowSpeed); // Inversely proportional to speed, increased from 0.5 to 1.5
      const maxTurnAngle = maxTurnRate * deltaTime;

      // Get current heading
      const currentAngle = Math.atan2(state.lastDirY, state.lastDirX);
      const targetAngle = Math.atan2(dirY, dirX);

      // Calculate angle difference
      let angleDiff = targetAngle - currentAngle;
      // Normalize to -PI to PI
      while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
      while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

      // Limit turn angle
      const limitedAngleDiff = Math.max(-maxTurnAngle, Math.min(maxTurnAngle, angleDiff));
      const newAngle = currentAngle + limitedAngleDiff;

      dirX = Math.cos(newAngle);
      dirY = Math.sin(newAngle);
    }

    // Accelerate/decelerate to target speed
    const targetSpeed = state.isCharging ? this.chargeSpeed : this.slowSpeed;
    const speedDiff = targetSpeed - state.currentSpeed;
    const speedChange = Math.sign(speedDiff) * Math.min(Math.abs(speedDiff), this.acceleration * deltaTime);
    state.currentSpeed += speedChange;

    // Apply movement
    const moveSpeed = enemyManager.moveSpeed[index] * state.currentSpeed;
    const moveAmount = moveSpeed * deltaTime;

    enemyManager.x[index] += dirX * moveAmount;
    enemyManager.y[index] += dirY * moveAmount;

    // Store direction for next frame
    state.lastDirX = dirX;
    state.lastDirY = dirY;
  }
}

/**
 * DirectionalShoot behavior - only shoots if target is within a cone in front
 * Used for cavalry to prevent shooting sideways/backwards while charging
 */
export class DirectionalShoot extends Behavior {
  /**
   * @param {number} cooldownMultiplier - Multiplier for cooldown (lower = faster shots)
   * @param {number} projectileCount - Number of projectiles to fire at once
   * @param {number} spread - Angular spread between projectiles (radians)
   * @param {number} maxAngle - Maximum angle from facing direction to allow shooting (radians, default PI/4 = 45°)
   */
  constructor(cooldownMultiplier = 1.0, projectileCount = 1, spread = 0, maxAngle = Math.PI / 4) {
    super();
    this.cooldownMultiplier = cooldownMultiplier;
    this.projectileCount = projectileCount;
    this.spread = spread;
    this.maxAngle = maxAngle;
  }

  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target || !bulletManager) return;

    // Skip if enemy can't shoot
    if (!enemyManager.canShoot[index]) return;

    // Skip if on cooldown
    if (enemyManager.currentCooldown[index] > 0) return;

    // Get cavalry's facing direction from charge state
    const chargeState = stateData.chargeState;
    if (!chargeState || chargeState.lastDirX === 0 && chargeState.lastDirY === 0) {
      return; // No facing direction yet
    }

    // Calculate direction to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distanceSquared = dx * dx + dy * dy;

    // Skip if out of range
    if (distanceSquared > enemyManager.shootRange[index] * enemyManager.shootRange[index]) {
      return;
    }

    // Calculate angle to target
    const targetAngle = Math.atan2(dy, dx);

    // Calculate cavalry's facing angle
    const facingAngle = Math.atan2(chargeState.lastDirY, chargeState.lastDirX);

    // Calculate angle difference
    let angleDiff = targetAngle - facingAngle;
    // Normalize to -PI to PI
    while (angleDiff > Math.PI) angleDiff -= Math.PI * 2;
    while (angleDiff < -Math.PI) angleDiff += Math.PI * 2;

    // Only shoot if target is within the forward cone
    if (Math.abs(angleDiff) > this.maxAngle) {
      return; // Target is too far to the side or behind
    }

    // Reset cooldown
    enemyManager.currentCooldown[index] = enemyManager.cooldown[index] * this.cooldownMultiplier;

    // Debug: Log enemy shooting
    console.log(`🔫 [CAVALRY SHOOT] Index ${index} at (${enemyManager.x[index].toFixed(2)}, ${enemyManager.y[index].toFixed(2)}) firing ${this.projectileCount} bullet(s), angle ${(targetAngle * 180 / Math.PI).toFixed(1)}°, angleDiff ${(angleDiff * 180 / Math.PI).toFixed(1)}°`);

    // Fire projectiles (reuse from Shoot behavior)
    this.fireProjectiles(index, enemyManager, bulletManager, targetAngle);
  }

  fireProjectiles(index, enemyManager, bulletManager, baseAngle) {
    // Spawn bullet at enemy center (no offset)
    const spawnOffset = 0;
    const spawnX = enemyManager.x[index];
    const spawnY = enemyManager.y[index];

    // Fire multiple projectiles with spread
    for (let i = 0; i < this.projectileCount; i++) {
      // Calculate angle offset for this projectile
      let angleOffset = 0;
      if (this.projectileCount > 1) {
        // Spread projectiles evenly
        const spreadRange = this.spread * (this.projectileCount - 1);
        angleOffset = -spreadRange / 2 + (i * this.spread);
      }

      // Apply inaccuracy (random variation)
      const inaccuracyOffset = (Math.random() - 0.5) * this.inaccuracy;
      const finalAngle = baseAngle + angleOffset + inaccuracyOffset;

      // Calculate velocity
      const speed = enemyManager.bulletSpeed[index];
      const vx = Math.cos(finalAngle) * speed;
      const vy = Math.sin(finalAngle) * speed;

      // Create bullet
      const bulletId = bulletManager.addBullet({
        x: spawnX,
        y: spawnY,
        vx: vx,
        vy: vy,
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: enemyManager.bulletLifetime[index],
        width: 0.3,
        height: 0.3,
        isEnemy: true,
        spriteName: enemyManager.bulletSpriteName[index] || null,
        worldId: enemyManager.worldId[index]
      });

      console.log(`  ↳ Created bullet ${bulletId} from ${enemyManager.id[index]}, lifetime=${enemyManager.bulletLifetime[index].toFixed(2)}s, damage=${enemyManager.damage[index]}`);
    }
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

    // Debug: Log enemy shooting
    console.log(`🔫 [ENEMY SHOOT] Index ${index} at (${enemyManager.x[index].toFixed(2)}, ${enemyManager.y[index].toFixed(2)}) firing ${this.projectileCount} bullet(s), angle ${(baseAngle * 180 / Math.PI).toFixed(1)}°`);

    // Fire projectiles
    this.fireProjectiles(index, enemyManager, bulletManager, baseAngle);
  }
  
  fireProjectiles(index, enemyManager, bulletManager, baseAngle) {
    // Spawn bullet at enemy center (no offset)
    const spawnOffset = 0; // tile-units

    // Single projectile case
    if (this.projectileCount <= 1) {
      // Apply inaccuracy if specified
      const angle = baseAngle + (Math.random() * 2 - 1) * this.inaccuracy;
      
      const spawnX = enemyManager.x[index] + Math.cos(angle) * spawnOffset;
      const spawnY = enemyManager.y[index] + Math.sin(angle) * spawnOffset;
      
      const bulletId = bulletManager.addBullet({
        x: spawnX,
        y: spawnY,
        vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
        vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: enemyManager.bulletLifetime[index],
        width: 0.3,
        height: 0.3,
        isEnemy: true,
        spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
        worldId: enemyManager.worldId[index]
      });

      console.log(`  ↳ Created bullet ${bulletId} from ${enemyManager.id[index]}, lifetime=${enemyManager.bulletLifetime[index].toFixed(2)}s, damage=${enemyManager.damage[index]}`);

      return;
    }
    
    // Multiple projectiles case
    const startAngle = baseAngle - (this.spread * (this.projectileCount - 1) / 2);

    for (let i = 0; i < this.projectileCount; i++) {
      // Base angle plus spread offset plus inaccuracy
      const angle = startAngle + (this.spread * i) + (Math.random() * 2 - 1) * this.inaccuracy;

      const spawnX = enemyManager.x[index] + Math.cos(angle) * spawnOffset;
      const spawnY = enemyManager.y[index] + Math.sin(angle) * spawnOffset;

      const bulletId = bulletManager.addBullet({
        x: spawnX,
        y: spawnY,
        vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
        vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: enemyManager.bulletLifetime[index],
        width: 0.3,
        height: 0.3,
        isEnemy: true,
        spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
        worldId: enemyManager.worldId[index]
      });

      console.log(`  ↳ Created bullet ${i + 1}/${this.projectileCount}: ${bulletId} from ${enemyManager.id[index]}, angle=${(angle * 180 / Math.PI).toFixed(1)}°`);
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
 * MoveLine behavior - Move in a straight line for a specified distance
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
    
    // TODO: implement swirl motion.  Currently state.angle/phase updated elsewhere.
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
export class Grenade2 extends Behavior {
  /**
   * @param {number} cooldownMultiplier - Multiplier for grenade cooldown
   * @param {number} count - Number of grenades to throw
   * @param {number} range - Maximum range for grenade throwing
   * @param {number} effect - Effect radius of explosion
   * @param {string} effectSprite - Sprite for explosion effect
   */
  constructor(cooldownMultiplier = 1.0, count = 1, range = 120, effect = 30, effectSprite = 'explosion') {
    super();
    this.cooldownMultiplier = cooldownMultiplier;
    this.count = count;
    this.range = range;
    this.effect = effect;
    this.effectSprite = effectSprite;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target || !bulletManager) return;
    
    // Skip if on cooldown
    if (enemyManager.currentCooldown[index] > 0) return;
    
    // Calculate distance to target
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Skip if out of range
    if (distance > this.range) return;
    
    // Reset cooldown
    enemyManager.currentCooldown[index] = enemyManager.cooldown[index] * this.cooldownMultiplier;
    
    // Throw grenades
    for (let i = 0; i < this.count; i++) {
      // Add some spread for multiple grenades
      const spread = this.count > 1 ? (i - this.count/2) * 0.3 : 0;
      const targetX = target.x + Math.cos(spread) * 20;
      const targetY = target.y + Math.sin(spread) * 20;
      
      // Create grenade projectile with special properties
      bulletManager.addBullet({
        x: enemyManager.x[index],
        y: enemyManager.y[index],
        vx: (targetX - enemyManager.x[index]) / 1.5, // Slower travel time
        vy: (targetY - enemyManager.y[index]) / 1.5,
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: 1.5, // Time to explosion
        width: 0.8,
        height: 0.8,
        isGrenade: true,
        explosionRadius: this.effect,
        explosionSprite: this.effectSprite,
        spriteName: 'grenade',
        worldId: enemyManager.worldId[index]
      });
    }
  }
}

/**
 * HealSelf behavior - Self-healing with cooldown and condition effects
 * Based on the C# RotMG HealSelf behavior
 */
export class HealSelf extends Behavior {
  /**
   * @param {number} cooldownMultiplier - Multiplier for heal cooldown
   * @param {number} amount - Amount to heal (0-1 for percentage, >1 for fixed amount)
   * @param {number} range - Range to check for threats before healing
   */
  constructor(cooldownMultiplier = 5.0, amount = 0.25, range = 80) {
    super();
    this.cooldownMultiplier = cooldownMultiplier;
    this.amount = amount;
    this.range = range;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize heal state
    if (!stateData.healState) {
      stateData.healState = {
        cooldown: 0
      };
    }
    
    // Update cooldown
    if (stateData.healState.cooldown > 0) {
      stateData.healState.cooldown -= deltaTime;
      return;
    }
    
    // Check if healing is needed
    const healthPercent = enemyManager.health[index] / enemyManager.maxHealth[index];
    if (healthPercent >= 0.8) return; // Don't heal if above 80% health
    
    // Check if safe to heal (no players nearby)
    if (target) {
      const dx = target.x - enemyManager.x[index];
      const dy = target.y - enemyManager.y[index];
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < this.range) return; // Too dangerous to heal
    }
    
    // Perform heal
    let healAmount;
    if (this.amount <= 1.0) {
      // Percentage heal
      healAmount = enemyManager.maxHealth[index] * this.amount;
    } else {
      // Fixed amount heal
      healAmount = this.amount;
    }
    
    enemyManager.health[index] = Math.min(
      enemyManager.health[index] + healAmount,
      enemyManager.maxHealth[index]
    );
    
    // Set cooldown
    stateData.healState.cooldown = enemyManager.cooldown[index] * this.cooldownMultiplier;
    
    // Visual effect (could be implemented with particle system)
    // TODO: Add healing visual effect
  }
}

/**
 * Spawn behavior - Spawn child entities with complex parameters
 * Based on the C# RotMG Spawn behavior
 */
export class Spawn extends Behavior {
  /**
   * @param {string} children - Type of child entities to spawn
   * @param {number} maxChildren - Maximum number of children
   * @param {number} cooldownMultiplier - Spawn cooldown multiplier
   * @param {number} range - Range around spawner to place children
   */
  constructor(children = 'goblin', maxChildren = 3, cooldownMultiplier = 3.0, range = 50) {
    super();
    this.children = children;
    this.maxChildren = maxChildren;
    this.cooldownMultiplier = cooldownMultiplier;
    this.range = range;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize spawn state
    if (!stateData.spawnState) {
      stateData.spawnState = {
        cooldown: 0,
        spawnedChildren: []
      };
    }
    
    // Update cooldown
    if (stateData.spawnState.cooldown > 0) {
      stateData.spawnState.cooldown -= deltaTime;
      return;
    }
    
    // Clean up dead children from tracking
    stateData.spawnState.spawnedChildren = stateData.spawnState.spawnedChildren.filter(childId => {
      // Check if child still exists and is alive
      const childIndex = enemyManager.findIndexById(childId);
      return childIndex !== -1 && enemyManager.health[childIndex] > 0;
    });
    
    // Check if we can spawn more children
    if (stateData.spawnState.spawnedChildren.length >= this.maxChildren) return;
    
    // Spawn new child
    const angle = Math.random() * Math.PI * 2;
    const distance = Math.random() * this.range;
    const spawnX = enemyManager.x[index] + Math.cos(angle) * distance;
    const spawnY = enemyManager.y[index] + Math.sin(angle) * distance;
    
    // Determine child type (would need enemy type mapping)
    let childType = 0; // Default to goblin
    if (this.children === 'orc') childType = 1;
    else if (this.children === 'skeleton') childType = 2;
    
    const childId = enemyManager.spawnEnemy(
      childType, 
      spawnX, 
      spawnY, 
      enemyManager.worldId[index]
    );
    
    if (childId) {
      stateData.spawnState.spawnedChildren.push(childId);
    }
    
    // Set cooldown
    stateData.spawnState.cooldown = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
}

/**
 * MoveLine behavior - Move in a straight line for a specified distance
 * Based on the C# RotMG MoveLine behavior
 */
export class MoveLineDistance extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for movement
   * @param {number} distance - Distance to travel
   * @param {number} direction - Direction in radians (optional, uses random if not specified)
   */
  constructor(speed = 1.0, distance = 100, direction = null) {
    super();
    this.speed = speed;
    this.distance = distance;
    this.direction = direction;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize state
    if (!stateData.moveLineState) {
      stateData.moveLineState = {
        startX: enemyManager.x[index],
        startY: enemyManager.y[index],
        direction: this.direction !== null ? this.direction : Math.random() * Math.PI * 2,
        traveled: 0
      };
    }
    
    const state = stateData.moveLineState;
    
    // Check if we've traveled the full distance
    if (state.traveled >= this.distance) {
      return; // Movement complete
    }
    
    // Calculate movement
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const moveAmount = moveSpeed * deltaTime;
    
    const dx = Math.cos(state.direction) * moveAmount;
    const dy = Math.sin(state.direction) * moveAmount;
    
    // Apply movement
    enemyManager.x[index] += dx;
    enemyManager.y[index] += dy;
    
    // Update traveled distance
    state.traveled += moveAmount;
  }
}

/**
 * MoveTo behavior - Move to specific coordinates
 * Based on the C# RotMG MoveTo behavior
 */
export class MoveToExact extends Behavior {
  /**
   * @param {number} x - Target X coordinate
   * @param {number} y - Target Y coordinate
   * @param {number} speed - Speed multiplier for movement
   */
  constructor(x, y, speed = 1.0) {
    super();
    this.targetX = x;
    this.targetY = y;
    this.speed = speed;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    const dx = this.targetX - enemyManager.x[index];
    const dy = this.targetY - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // Check if we've reached the destination
    if (distance < 1.0) {
      return; // Reached destination
    }
    
    // Normalize direction
    const dirX = dx / distance;
    const dirY = dy / distance;
    
    // Calculate movement
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const moveAmount = moveSpeed * deltaTime;
    
    // Apply movement
    enemyManager.x[index] += dirX * moveAmount;
    enemyManager.y[index] += dirY * moveAmount;
  }
}

/**
 * StayAbove behavior - Maintain position above a certain altitude/distance
 * Based on the C# RotMG StayAbove behavior
 */
export class StayAboveAltitude extends Behavior {
  /**
   * @param {number} altitude - Minimum distance to maintain from target
   * @param {number} speed - Speed multiplier for movement
   */
  constructor(altitude = 50, speed = 1.0) {
    super();
    this.altitude = altitude;
    this.speed = speed;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;
    
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // If too close, move away
    if (distance < this.altitude) {
      const dirX = -dx / distance; // Reverse direction
      const dirY = -dy / distance;
      
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
      const moveAmount = moveSpeed * deltaTime;
      
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
    }
  }
}

/**
 * StayBack behavior - Maintain distance from target
 * Based on the C# RotMG StayBack behavior
 */
export class StayBack extends Behavior {
  /**
   * @param {number} distance - Distance to maintain from target
   * @param {number} speed - Speed multiplier for movement
   */
  constructor(distance = 80, speed = 1.0) {
    super();
    this.distance = distance;
    this.speed = speed;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;
    
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const currentDistance = Math.sqrt(dx * dx + dy * dy);
    
    if (currentDistance < this.distance) {
      // Too close, move away
      const dirX = -dx / currentDistance;
      const dirY = -dy / currentDistance;
      
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
      const moveAmount = moveSpeed * deltaTime;
      
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
    } else if (currentDistance > this.distance * 1.5) {
      // Too far, move closer
      const dirX = dx / currentDistance;
      const dirY = dy / currentDistance;
      
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed * 0.5; // Slower approach
      const moveAmount = moveSpeed * deltaTime;
      
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
    }
  }
}

/**
 * StayCloseToSpawn behavior - Stay within range of spawn point
 * Based on the C# RotMG StayCloseToSpawn behavior
 */
export class StayCloseToSpawn extends Behavior {
  /**
   * @param {number} range - Maximum distance from spawn point
   * @param {number} speed - Speed multiplier for return movement
   */
  constructor(range = 100, speed = 1.0) {
    super();
    this.range = range;
    this.speed = speed;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize spawn position if needed
    if (!stateData.spawnPos) {
      stateData.spawnPos = {
        x: enemyManager.x[index],
        y: enemyManager.y[index]
      };
    }
    
    const dx = stateData.spawnPos.x - enemyManager.x[index];
    const dy = stateData.spawnPos.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // If outside range, return to spawn
    if (distance > this.range) {
      const dirX = dx / distance;
      const dirY = dy / distance;
      
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
      const moveAmount = moveSpeed * deltaTime;
      
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
    }
  }
}

/**
 * ReturnToSpawn behavior - Return to original spawn location
 * Based on the C# RotMG ReturnToSpawn behavior
 */
export class ReturnToSpawn extends Behavior {
  /**
   * @param {number} speed - Speed multiplier for return movement
   * @param {boolean} once - Whether to return only once or continuously
   */
  constructor(speed = 1.0, once = false) {
    super();
    this.speed = speed;
    this.once = once;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize spawn position if needed
    if (!stateData.spawnReturn) {
      stateData.spawnReturn = {
        spawnX: enemyManager.x[index],
        spawnY: enemyManager.y[index],
        hasReturned: false
      };
    }
    
    const state = stateData.spawnReturn;
    
    // If 'once' and already returned, do nothing
    if (this.once && state.hasReturned) return;
    
    const dx = state.spawnX - enemyManager.x[index];
    const dy = state.spawnY - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    // If not at spawn, move toward it
    if (distance > 1.0) {
      const dirX = dx / distance;
      const dirY = dy / distance;
      
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
      const moveAmount = moveSpeed * deltaTime;
      
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
    } else if (this.once) {
      state.hasReturned = true;
    }
  }
}

/**
 * Buzz behavior - Buzzing movement pattern
 * Based on the C# RotMG Buzz behavior
 */
export class Buzz extends Behavior {
  /**
   * @param {number} dist - Distance of buzz movement
   * @param {number} speed - Speed multiplier for buzzing
   */
  constructor(dist = 20, speed = 1.0) {
    super();
    this.dist = dist;
    this.speed = speed;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize buzz state
    if (!stateData.buzzState) {
      stateData.buzzState = {
        centerX: enemyManager.x[index],
        centerY: enemyManager.y[index],
        angle: Math.random() * Math.PI * 2,
        radius: Math.random() * this.dist
      };
    }
    
    const state = stateData.buzzState;
    
    // Update angle for buzzing motion
    const angleSpeed = this.speed * 3.0; // Faster angular movement for buzzing
    state.angle += angleSpeed * deltaTime;
    
    // Vary radius slightly for more organic movement
    state.radius += (Math.random() - 0.5) * this.dist * 0.1 * deltaTime;
    state.radius = Math.max(0, Math.min(this.dist, state.radius));
    
    // Calculate target position
    const targetX = state.centerX + Math.cos(state.angle) * state.radius;
    const targetY = state.centerY + Math.sin(state.angle) * state.radius;
    
    // Move toward target position
    const dx = targetX - enemyManager.x[index];
    const dy = targetY - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > 0) {
      const dirX = dx / distance;
      const dirY = dy / distance;
      
      const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
      const moveAmount = moveSpeed * deltaTime;
      
      enemyManager.x[index] += dirX * moveAmount;
      enemyManager.y[index] += dirY * moveAmount;
    }
  }
}

/**
 * Aoe behavior - Area of Effect attacks
 * Based on the C# RotMG Aoe behavior
 */
export class Aoe extends Behavior {
  /**
   * @param {number} radius - Effect radius
   * @param {number} damage - Damage amount
   * @param {string} effect - Visual effect name
   * @param {number} cooldownMultiplier - Cooldown multiplier
   */
  constructor(radius = 50, damage = 20, effect = 'explosion', cooldownMultiplier = 2.0) {
    super();
    this.radius = radius;
    this.damage = damage;
    this.effect = effect;
    this.cooldownMultiplier = cooldownMultiplier;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target || !bulletManager) return;
    
    // Skip if on cooldown
    if (enemyManager.currentCooldown[index] > 0) return;
    
    // Check if target is in range
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance <= this.radius) {
      // Create AOE effect
      bulletManager.addBullet({
        x: enemyManager.x[index],
        y: enemyManager.y[index],
        vx: 0, // Static AOE
        vy: 0,
        ownerId: enemyManager.id[index],
        damage: this.damage,
        lifetime: 0.1, // Very short lifetime for instant effect
        width: this.radius * 2,
        height: this.radius * 2,
        isAOE: true,
        effectName: this.effect,
        worldId: enemyManager.worldId[index]
      });
      
      // Set cooldown
      enemyManager.currentCooldown[index] = enemyManager.cooldown[index] * this.cooldownMultiplier;
    }
  }
}

/**
 * TalismanAttack behavior - Special talisman-based attack
 * Based on the C# RotMG TalismanAttack behavior
 */
export class TalismanAttack extends Behavior {
  /**
   * @param {number} range - Attack range
   * @param {number} cooldownMultiplier - Cooldown multiplier
   * @param {string} talismanType - Type of talisman effect
   */
  constructor(range = 100, cooldownMultiplier = 1.5, talismanType = 'magic') {
    super();
    this.range = range;
    this.cooldownMultiplier = cooldownMultiplier;
    this.talismanType = talismanType;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target || !bulletManager) return;
    
    // Skip if on cooldown
    if (enemyManager.currentCooldown[index] > 0) return;
    
    // Check range
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > this.range) return;
    
    // Create talisman projectile with special properties
    const angle = Math.atan2(dy, dx);
    
    bulletManager.addBullet({
      x: enemyManager.x[index],
      y: enemyManager.y[index],
      vx: Math.cos(angle) * enemyManager.bulletSpeed[index] * 1.5, // Faster
      vy: Math.sin(angle) * enemyManager.bulletSpeed[index] * 1.5,
      ownerId: enemyManager.id[index],
      damage: enemyManager.damage[index] * 1.5, // Higher damage
      lifetime: 4.0,
      width: 0.6,
      height: 0.6,
      isTalisman: true,
      talismanType: this.talismanType,
      piercing: true, // Can hit multiple targets
      spriteName: `talisman_${this.talismanType}`,
      worldId: enemyManager.worldId[index]
    });
    
    // Set cooldown
    enemyManager.currentCooldown[index] = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
}

/**
 * InvisiToss behavior - Throw projectiles while invisible
 * Based on the C# RotMG InvisiToss behavior
 */
export class InvisiToss extends Behavior {
  /**
   * @param {number} radius - Toss radius
   * @param {number} cooldownMultiplier - Cooldown multiplier
   * @param {number} projectileCount - Number of projectiles to toss
   */
  constructor(radius = 80, cooldownMultiplier = 1.0, projectileCount = 3) {
    super();
    this.radius = radius;
    this.cooldownMultiplier = cooldownMultiplier;
    this.projectileCount = projectileCount;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target || !bulletManager) return;
    
    // Skip if on cooldown
    if (enemyManager.currentCooldown[index] > 0) return;
    
    // Check range
    const dx = target.x - enemyManager.x[index];
    const dy = target.y - enemyManager.y[index];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > this.radius) return;
    
    // Become invisible (set flag for rendering)
    if (!stateData.invisiState) {
      stateData.invisiState = {
        invisible: true,
        invisTimer: 2.0 // Invisible for 2 seconds
      };
    }
    
    // Toss projectiles in random directions around target
    for (let i = 0; i < this.projectileCount; i++) {
      const angle = Math.random() * Math.PI * 2;
      const tossDistance = Math.random() * this.radius;
      
      // Calculate toss target position
      const tossX = target.x + Math.cos(angle) * tossDistance;
      const tossY = target.y + Math.sin(angle) * tossDistance;
      
      // Calculate velocity to reach toss target
      const tossTime = 1.5; // Time to reach target
      const vx = (tossX - enemyManager.x[index]) / tossTime;
      const vy = (tossY - enemyManager.y[index]) / tossTime;
      
      bulletManager.addBullet({
        x: enemyManager.x[index],
        y: enemyManager.y[index],
        vx: vx,
        vy: vy,
        ownerId: enemyManager.id[index],
        damage: enemyManager.damage[index],
        lifetime: tossTime,
        width: 0.5,
        height: 0.5,
        isInvisiToss: true,
        spriteName: 'invisi_projectile',
        worldId: enemyManager.worldId[index]
      });
    }
    
    // Set cooldown
    enemyManager.currentCooldown[index] = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
}

/**
 * SpawnGroup behavior - Spawn groups of entities
 * Based on the C# RotMG SpawnGroup behavior
 */
export class SpawnGroup extends Behavior {
  /**
   * @param {string} groupType - Type of group to spawn
   * @param {number} maxGroups - Maximum number of groups
   * @param {number} cooldownMultiplier - Spawn cooldown multiplier
   * @param {number} range - Range around spawner
   */
  constructor(groupType = 'goblins', maxGroups = 2, cooldownMultiplier = 5.0, range = 80) {
    super();
    this.groupType = groupType;
    this.maxGroups = maxGroups;
    this.cooldownMultiplier = cooldownMultiplier;
    this.range = range;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize spawn state
    if (!stateData.groupSpawnState) {
      stateData.groupSpawnState = {
        cooldown: 0,
        spawnedGroups: [],
        groupCounter: 0
      };
    }
    
    // Update cooldown
    if (stateData.groupSpawnState.cooldown > 0) {
      stateData.groupSpawnState.cooldown -= deltaTime;
      return;
    }
    
    // Clean up dead groups
    stateData.groupSpawnState.spawnedGroups = stateData.groupSpawnState.spawnedGroups.filter(group => {
      // Check if any group member is still alive
      return group.members.some(memberId => {
        const memberIndex = enemyManager.findIndexById(memberId);
        return memberIndex !== -1 && enemyManager.health[memberIndex] > 0;
      });
    });
    
    // Check if we can spawn more groups
    if (stateData.groupSpawnState.spawnedGroups.length >= this.maxGroups) return;
    
    // Spawn new group
    const group = this.spawnGroup(index, enemyManager);
    if (group.members.length > 0) {
      stateData.groupSpawnState.spawnedGroups.push(group);
    }
    
    // Set cooldown
    stateData.groupSpawnState.cooldown = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
  
  spawnGroup(spawnerIndex, enemyManager) {
    const group = {
      id: `group_${Date.now()}_${Math.random()}`,
      members: [],
      formation: 'circle'
    };
    
    // Determine group composition
    let groupSize = 3;
    let memberType = 0; // Default to goblins
    
    switch (this.groupType) {
      case 'goblins':
        groupSize = 4;
        memberType = 0;
        break;
      case 'orcs':
        groupSize = 3;
        memberType = 1;
        break;
      case 'skeletons':
        groupSize = 5;
        memberType = 2;
        break;
    }
    
    // Spawn group members in formation
    const centerX = enemyManager.x[spawnerIndex];
    const centerY = enemyManager.y[spawnerIndex];
    
    for (let i = 0; i < groupSize; i++) {
      const angle = (i / groupSize) * Math.PI * 2;
      const formationRadius = 20;
      const spawnX = centerX + Math.cos(angle) * formationRadius;
      const spawnY = centerY + Math.sin(angle) * formationRadius;
      
      const memberId = enemyManager.spawnEnemy(
        memberType,
        spawnX,
        spawnY,
        enemyManager.worldId[spawnerIndex]
      );
      
      if (memberId) {
        group.members.push(memberId);
      }
    }
    
    return group;
  }
}

/**
 * RelativeSpawn behavior - Spawn entities at relative positions
 * Based on the C# RotMG RelativeSpawn behavior
 */
export class RelativeSpawn extends Behavior {
  /**
   * @param {string} children - Type of children to spawn
   * @param {Array} positions - Array of relative positions {x, y}
   * @param {number} cooldownMultiplier - Spawn cooldown multiplier
   */
  constructor(children = 'goblin', positions = [{x: 20, y: 0}, {x: -20, y: 0}], cooldownMultiplier = 4.0) {
    super();
    this.children = children;
    this.positions = positions;
    this.cooldownMultiplier = cooldownMultiplier;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize spawn state
    if (!stateData.relativeSpawnState) {
      stateData.relativeSpawnState = {
        cooldown: 0,
        spawnedChildren: []
      };
    }
    
    // Update cooldown
    if (stateData.relativeSpawnState.cooldown > 0) {
      stateData.relativeSpawnState.cooldown -= deltaTime;
      return;
    }
    
    // Clean up dead children
    stateData.relativeSpawnState.spawnedChildren = stateData.relativeSpawnState.spawnedChildren.filter(childId => {
      const childIndex = enemyManager.findIndexById(childId);
      return childIndex !== -1 && enemyManager.health[childIndex] > 0;
    });
    
    // Check if all positions are filled
    if (stateData.relativeSpawnState.spawnedChildren.length >= this.positions.length) return;
    
    // Spawn children at relative positions
    const spawnerX = enemyManager.x[index];
    const spawnerY = enemyManager.y[index];
    
    for (let i = stateData.relativeSpawnState.spawnedChildren.length; i < this.positions.length; i++) {
      const pos = this.positions[i];
      const spawnX = spawnerX + pos.x;
      const spawnY = spawnerY + pos.y;
      
      // Determine child type
      let childType = 0;
      if (this.children === 'orc') childType = 1;
      else if (this.children === 'skeleton') childType = 2;
      
      const childId = enemyManager.spawnEnemy(
        childType,
        spawnX,
        spawnY,
        enemyManager.worldId[index]
      );
      
      if (childId) {
        stateData.relativeSpawnState.spawnedChildren.push(childId);
      }
    }
    
    // Set cooldown
    stateData.relativeSpawnState.cooldown = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
}

/**
 * HealGroup behavior - Heal multiple entities in range
 * Based on the C# RotMG HealGroup behavior
 */
export class HealGroup extends Behavior {
  /**
   * @param {number} range - Healing range
   * @param {number} amount - Amount to heal
   * @param {number} cooldownMultiplier - Healing cooldown multiplier
   * @param {string} healType - Type of healing ('ally' or 'all')
   */
  constructor(range = 60, amount = 15, cooldownMultiplier = 3.0, healType = 'ally') {
    super();
    this.range = range;
    this.amount = amount;
    this.cooldownMultiplier = cooldownMultiplier;
    this.healType = healType;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize heal state
    if (!stateData.groupHealState) {
      stateData.groupHealState = {
        cooldown: 0
      };
    }
    
    // Update cooldown
    if (stateData.groupHealState.cooldown > 0) {
      stateData.groupHealState.cooldown -= deltaTime;
      return;
    }
    
    // Find entities in range to heal
    const healTargets = this.findHealTargets(index, enemyManager);
    
    if (healTargets.length === 0) return;
    
    // Heal all targets
    for (const targetIndex of healTargets) {
      const currentHealth = enemyManager.health[targetIndex];
      const maxHealth = enemyManager.maxHealth[targetIndex];
      
      if (currentHealth < maxHealth) {
        enemyManager.health[targetIndex] = Math.min(
          currentHealth + this.amount,
          maxHealth
        );
        
        // Visual effect could be added here
      }
    }
    
    // Set cooldown
    stateData.groupHealState.cooldown = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
  
  findHealTargets(healerIndex, enemyManager) {
    const targets = [];
    const healerX = enemyManager.x[healerIndex];
    const healerY = enemyManager.y[healerIndex];
    const healerType = enemyManager.type[healerIndex];
    
    for (let i = 0; i < enemyManager.enemyCount; i++) {
      if (i === healerIndex) continue; // Don't heal self
      if (enemyManager.health[i] <= 0) continue; // Skip dead
      
      // Check heal type
      if (this.healType === 'ally' && enemyManager.type[i] !== healerType) {
        continue; // Only heal same type
      }
      
      // Check range
      const dx = enemyManager.x[i] - healerX;
      const dy = enemyManager.y[i] - healerY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= this.range) {
        targets.push(i);
      }
    }
    
    return targets;
  }
}

/**
 * HealEntity behavior - Heal specific entities
 * Based on the C# RotMG HealEntity behavior
 */
export class HealEntity extends Behavior {
  /**
   * @param {string} entityType - Type of entity to heal
   * @param {number} range - Healing range
   * @param {number} amount - Amount to heal
   * @param {number} cooldownMultiplier - Healing cooldown multiplier
   */
  constructor(entityType = 'any', range = 50, amount = 20, cooldownMultiplier = 2.0) {
    super();
    this.entityType = entityType;
    this.range = range;
    this.amount = amount;
    this.cooldownMultiplier = cooldownMultiplier;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    // Initialize heal state
    if (!stateData.entityHealState) {
      stateData.entityHealState = {
        cooldown: 0
      };
    }
    
    // Update cooldown
    if (stateData.entityHealState.cooldown > 0) {
      stateData.entityHealState.cooldown -= deltaTime;
      return;
    }
    
    // Find the most wounded entity of specified type in range
    const healTarget = this.findMostWoundedEntity(index, enemyManager);
    
    if (healTarget === -1) return;
    
    // Heal the target
    const currentHealth = enemyManager.health[healTarget];
    const maxHealth = enemyManager.maxHealth[healTarget];
    
    enemyManager.health[healTarget] = Math.min(
      currentHealth + this.amount,
      maxHealth
    );
    
    // Set cooldown
    stateData.entityHealState.cooldown = enemyManager.cooldown[index] * this.cooldownMultiplier;
  }
  
  findMostWoundedEntity(healerIndex, enemyManager) {
    let mostWounded = -1;
    let lowestHealthRatio = 1.0;
    
    const healerX = enemyManager.x[healerIndex];
    const healerY = enemyManager.y[healerIndex];
    
    for (let i = 0; i < enemyManager.enemyCount; i++) {
      if (i === healerIndex) continue;
      if (enemyManager.health[i] <= 0) continue;
      
      // Check entity type filter
      if (this.entityType !== 'any') {
        const targetTypeName = enemyManager.enemyTypes[enemyManager.type[i]]?.name || '';
        if (targetTypeName.toLowerCase() !== this.entityType.toLowerCase()) {
          continue;
        }
      }
      
      // Check range
      const dx = enemyManager.x[i] - healerX;
      const dy = enemyManager.y[i] - healerY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance <= this.range) {
        const healthRatio = enemyManager.health[i] / enemyManager.maxHealth[i];
        if (healthRatio < lowestHealthRatio) {
          lowestHealthRatio = healthRatio;
          mostWounded = i;
        }
      }
    }
    
    return mostWounded;
  }
} 