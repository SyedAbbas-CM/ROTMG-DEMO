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
   * @param {boolean} clockwise - Whether to orbit clockwise (true) or counter-clockwise (false)
   */
  constructor(speed = 1.0, radius = 100, clockwise = true) {
    super();
    this.speed = speed;
    this.radius = radius;
    this.direction = clockwise ? 1 : -1;
  }
  
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData) {
    if (!target) return;
    
    // Calculate current position relative to target
    const dx = enemyManager.x[index] - target.x;
    const dy = enemyManager.y[index] - target.y;
    const currentDistance = Math.sqrt(dx * dx + dy * dy);
    
    // Calculate current angle
    let angle = Math.atan2(dy, dx);
    
    // Adjust orbit distance if needed
    const distanceAdjustment = (this.radius - currentDistance) * 0.1;
    
    // Calculate angular speed based on enemy's move speed
    const baseSpeed = enemyManager.moveSpeed[index] * this.speed;
    const angularSpeed = (baseSpeed / this.radius) * this.direction;
    
    // Update angle
    angle += angularSpeed * deltaTime;
    
    // Calculate new position
    const newDist = currentDistance + distanceAdjustment;
    const newX = target.x + Math.cos(angle) * newDist;
    const newY = target.y + Math.sin(angle) * newDist;
    
    // Move to new position
    const moveSpeed = enemyManager.moveSpeed[index] * this.speed;
    const maxMove = moveSpeed * deltaTime;
    
    // Limit movement per frame
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