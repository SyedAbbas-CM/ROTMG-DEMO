    // File: /src/Behaviours/EnemyBehaviors.js

/**
 * A modular behavior system for enemies that works with Structure of Arrays (SoA) architecture.
 * Instead of creating behavior objects, we use behavior functions that directly manipulate
 * the arrays in the EnemyManager.
 */

// Behavior Types Enum
export const BehaviorType = {
    IDLE: 0,
    CHASE: 1,
    SHOOT: 2,
    PATROL: 3,
    WANDER: 4,
    FLEE: 5,
    SUICIDE: 6
  };
  
  /**
   * Behavior state arrays - these store parameters for each behavior type
   * Each enemy can have multiple behaviors active at once
   */
  export class BehaviorSystem {
    constructor(maxEnemies) {
      this.maxEnemies = maxEnemies;
      
      // Core behavior tracking
      this.behaviorTypes = new Array(maxEnemies).fill(null).map(() => []);
      this.behaviorParams = new Array(maxEnemies).fill(null).map(() => []);
      this.behaviorTimers = new Array(maxEnemies).fill(null).map(() => []);
      
      // Common behavior parameters that are shared between behaviors
      this.targetX = new Float32Array(maxEnemies);
      this.targetY = new Float32Array(maxEnemies);
      this.moveSpeed = new Float32Array(maxEnemies);
      this.shootCooldown = new Float32Array(maxEnemies);
      this.shootRange = new Float32Array(maxEnemies);
      this.currentCooldowns = new Float32Array(maxEnemies);
      this.patrolPoints = new Array(maxEnemies).fill(null).map(() => []);
      this.patrolIndex = new Uint16Array(maxEnemies);
      this.wanderRadius = new Float32Array(maxEnemies);
      this.wanderTime = new Float32Array(maxEnemies);
      this.homeX = new Float32Array(maxEnemies);
      this.homeY = new Float32Array(maxEnemies);
      this.projectileSpeed = new Float32Array(maxEnemies);
      this.projectileDamage = new Float32Array(maxEnemies);
      this.projectileLifetime = new Float32Array(maxEnemies);
      this.projectileCount = new Uint8Array(maxEnemies);
      this.projectileSpread = new Float32Array(maxEnemies);
      this.proximityRadius = new Float32Array(maxEnemies);
    }
  
    /**
     * Add a behavior to an enemy
     * @param {number} index - Enemy index in the arrays
     * @param {number} behaviorType - Type of behavior from BehaviorType enum
     * @param {Object} params - Behavior-specific parameters
     */
    addBehavior(index, behaviorType, params = {}) {
      if (index >= this.maxEnemies) return;
      
      // Add behavior to tracking arrays
      this.behaviorTypes[index].push(behaviorType);
      this.behaviorParams[index].push(params);
      this.behaviorTimers[index].push(0);
      
      // Set common parameters based on the behavior type
      switch (behaviorType) {
        case BehaviorType.CHASE:
          this.moveSpeed[index] = params.speed || 50;
          this.proximityRadius[index] = params.radius || 200;
          break;
          
        case BehaviorType.SHOOT:
          this.shootRange[index] = params.range || 150;
          this.shootCooldown[index] = params.cooldown || 2;
          this.currentCooldowns[index] = params.initialCooldown || 0;
          this.projectileSpeed[index] = params.projectileSpeed || 100;
          this.projectileDamage[index] = params.damage || 10;
          this.projectileLifetime[index] = params.lifetime || 2;
          this.projectileCount[index] = params.count || 1;
          this.projectileSpread[index] = params.spread || (params.count > 1 ? Math.PI/6 : 0);
          break;
          
        case BehaviorType.PATROL:
          this.moveSpeed[index] = params.speed || 30;
          this.patrolPoints[index] = params.points || [];
          this.patrolIndex[index] = 0;
          if (this.patrolPoints[index].length > 0) {
            const firstPoint = this.patrolPoints[index][0];
            this.targetX[index] = firstPoint.x;
            this.targetY[index] = firstPoint.y;
          }
          break;
          
        case BehaviorType.WANDER:
          this.moveSpeed[index] = params.speed || 20;
          this.wanderRadius[index] = params.radius || 100;
          this.wanderTime[index] = params.time || 3;
          this.homeX[index] = params.homeX || 0;
          this.homeY[index] = params.homeY || 0;
          break;
          
        case BehaviorType.FLEE:
          this.moveSpeed[index] = params.speed || 70;
          this.proximityRadius[index] = params.radius || 150;
          break;
          
        case BehaviorType.SUICIDE:
          this.proximityRadius[index] = params.radius || 50;
          break;
      }
    }
    
    /**
     * Update all behaviors for an enemy
     * @param {number} index - Enemy index
     * @param {Object} enemyManager - Reference to the EnemyManager
     * @param {Object} targetEntity - Target entity (player, etc.)
     * @param {Object} bulletManager - Reference to the BulletManager
     * @param {number} deltaTime - Time since last update in seconds
     * @returns {boolean} - True if the enemy is still active
     */
    updateBehaviors(index, enemyManager, targetEntity, bulletManager, deltaTime) {
      // Skip if no behaviors
      if (!this.behaviorTypes[index] || this.behaviorTypes[index].length === 0) return true;
      
      // Update cooldowns
      this.currentCooldowns[index] = Math.max(0, this.currentCooldowns[index] - deltaTime);
      
      // Track if enemy should still be active
      let isActive = true;
      
      // Process each behavior
      for (let i = 0; i < this.behaviorTypes[index].length; i++) {
        const behaviorType = this.behaviorTypes[index][i];
        const params = this.behaviorParams[index][i];
        this.behaviorTimers[index][i] += deltaTime;
        
        switch (behaviorType) {
          case BehaviorType.CHASE:
            this._processChase(index, enemyManager, targetEntity, deltaTime);
            break;
            
          case BehaviorType.SHOOT:
            this._processShoot(index, enemyManager, targetEntity, bulletManager, deltaTime);
            break;
            
          case BehaviorType.PATROL:
            this._processPatrol(index, enemyManager, deltaTime);
            break;
            
          case BehaviorType.WANDER:
            this._processWander(index, enemyManager, deltaTime);
            break;
            
          case BehaviorType.FLEE:
            this._processFlee(index, enemyManager, targetEntity, deltaTime);
            break;
            
          case BehaviorType.SUICIDE:
            if (this._processSuicide(index, enemyManager, targetEntity, deltaTime)) {
              isActive = false; // Enemy should be removed
            }
            break;
        }
      }
      
      return isActive;
    }
    
    /**
     * Process a CHASE behavior
     * @private
     */
    _processChase(index, enemyManager, targetEntity, deltaTime) {
      if (!targetEntity) return;
      
      // Calculate distance to target
      const dx = targetEntity.x - enemyManager.x[index];
      const dy = targetEntity.y - enemyManager.y[index];
      const distSq = dx * dx + dy * dy;
      
      // Only chase if target is within range
      if (distSq <= this.proximityRadius[index] * this.proximityRadius[index]) {
        const dist = Math.sqrt(distSq);
        
        // Move toward target
        if (dist > 0) {
          const moveAmount = this.moveSpeed[index] * deltaTime;
          
          // Normalize direction
          const normalizedDx = dx / dist;
          const normalizedDy = dy / dist;
          
          // Apply movement
          enemyManager.x[index] += normalizedDx * moveAmount;
          enemyManager.y[index] += normalizedDy * moveAmount;
        }
      }
    }
    
    /**
     * Process a SHOOT behavior
     * @private
     */
    _processShoot(index, enemyManager, targetEntity, bulletManager, deltaTime) {
      if (!targetEntity || !bulletManager) return;
      
      // Calculate distance to target
      const dx = targetEntity.x - enemyManager.x[index];
      const dy = targetEntity.y - enemyManager.y[index];
      const distSq = dx * dx + dy * dy;
      
      // Only shoot if target is within range
      if (distSq <= this.shootRange[index] * this.shootRange[index]) {
        // Check cooldown
        if (this.currentCooldowns[index] <= 0) {
          // Calculate angle to target
          const angle = Math.atan2(dy, dx);
          
          // Create bullets based on projectileCount and spread
          for (let i = 0; i < this.projectileCount[index]; i++) {
            // Calculate spread angle
            let bulletAngle = angle;
            if (this.projectileCount[index] > 1) {
              bulletAngle += this.projectileSpread[index] * (i - (this.projectileCount[index]-1)/2);
            }
            
            // Create bullet
            if (bulletManager.addBullet) {
              bulletManager.addBullet({
                x: enemyManager.x[index],
                y: enemyManager.y[index],
                vx: Math.cos(bulletAngle) * this.projectileSpeed[index],
                vy: Math.sin(bulletAngle) * this.projectileSpeed[index],
                damage: this.projectileDamage[index],
                lifetime: this.projectileLifetime[index],
                ownerId: enemyManager.id[index],
                worldId: enemyManager.worldId[index]
              });
            }
          }
          
          // Reset cooldown
          this.currentCooldowns[index] = this.shootCooldown[index];
        }
      }
    }
    
    /**
     * Process a PATROL behavior
     * @private
     */
    _processPatrol(index, enemyManager, deltaTime) {
      // Skip if no patrol points
      if (!this.patrolPoints[index] || this.patrolPoints[index].length === 0) return;
      
      // Get current target point
      const currentPoint = this.patrolPoints[index][this.patrolIndex[index]];
      if (!currentPoint) return;
      
      // Calculate distance to current target
      const dx = currentPoint.x - enemyManager.x[index];
      const dy = currentPoint.y - enemyManager.y[index];
      const distSq = dx * dx + dy * dy;
      
      // Move toward target
      const dist = Math.sqrt(distSq);
      if (dist > 5) { // Threshold to avoid jittering
        const moveAmount = this.moveSpeed[index] * deltaTime;
        
        // Normalize direction
        const normalizedDx = dx / dist;
        const normalizedDy = dy / dist;
        
        // Apply movement
        enemyManager.x[index] += normalizedDx * moveAmount;
        enemyManager.y[index] += normalizedDy * moveAmount;
      } else {
        // Reached target point, move to next point
        this.patrolIndex[index] = (this.patrolIndex[index] + 1) % this.patrolPoints[index].length;
      }
    }
    
    /**
     * Process a WANDER behavior
     * @private
     */
    _processWander(index, enemyManager, deltaTime) {
      const timer = this.behaviorTimers[index][
        this.behaviorTypes[index].indexOf(BehaviorType.WANDER)
      ];
      
      // Check if we need a new wander target
      if (timer >= this.wanderTime[index]) {
        // Reset timer
        this.behaviorTimers[index][
          this.behaviorTypes[index].indexOf(BehaviorType.WANDER)
        ] = 0;
        
        // Create a new random target within wander radius
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * this.wanderRadius[index];
        
        this.targetX[index] = this.homeX[index] + Math.cos(angle) * radius;
        this.targetY[index] = this.homeY[index] + Math.sin(angle) * radius;
      }
      
      // Move toward current wander target
      const dx = this.targetX[index] - enemyManager.x[index];
      const dy = this.targetY[index] - enemyManager.y[index];
      const distSq = dx * dx + dy * dy;
      
      // Move toward target
      const dist = Math.sqrt(distSq);
      if (dist > 5) { // Threshold to avoid jittering
        const moveAmount = this.moveSpeed[index] * deltaTime;
        
        // Normalize direction
        const normalizedDx = dx / dist;
        const normalizedDy = dy / dist;
        
        // Apply movement
        enemyManager.x[index] += normalizedDx * moveAmount;
        enemyManager.y[index] += normalizedDy * moveAmount;
      }
    }
    
    /**
     * Process a FLEE behavior
     * @private
     */
    _processFlee(index, enemyManager, targetEntity, deltaTime) {
      if (!targetEntity) return;
      
      // Calculate distance to target
      const dx = targetEntity.x - enemyManager.x[index];
      const dy = targetEntity.y - enemyManager.y[index];
      const distSq = dx * dx + dy * dy;
      
      // Only flee if target is within range
      if (distSq <= this.proximityRadius[index] * this.proximityRadius[index]) {
        const dist = Math.sqrt(distSq);
        
        // Move away from target
        if (dist > 0) {
          const moveAmount = this.moveSpeed[index] * deltaTime;
          
          // Normalize direction and invert to flee
          const normalizedDx = -dx / dist;
          const normalizedDy = -dy / dist;
          
          // Apply movement
          enemyManager.x[index] += normalizedDx * moveAmount;
          enemyManager.y[index] += normalizedDy * moveAmount;
        }
      }
    }
    
    /**
     * Process a SUICIDE behavior
     * @private
     * @returns {boolean} True if the enemy should be removed
     */
    _processSuicide(index, enemyManager, targetEntity, deltaTime) {
      if (!targetEntity) return false;
      
      // Calculate distance to target
      const dx = targetEntity.x - enemyManager.x[index];
      const dy = targetEntity.y - enemyManager.y[index];
      const distSq = dx * dx + dy * dy;
      
      // If close enough, trigger suicide (explode, die, etc.)
      if (distSq <= this.proximityRadius[index] * this.proximityRadius[index]) {
        // Signal that this enemy should be removed
        return true;
      }
      
      return false;
    }
    
    /**
     * Clear all behaviors for an enemy
     * @param {number} index - Enemy index
     */
    clearBehaviors(index) {
      if (index >= this.maxEnemies) return;
      
      this.behaviorTypes[index] = [];
      this.behaviorParams[index] = [];
      this.behaviorTimers[index] = [];
    }
  }