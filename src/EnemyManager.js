// File: /src/Managers/EnemyManager.js

import BehaviorSystem from './BehaviorSystem.js';

/**
 * EnemyManager handles enemy creation, updating, and removal.
 * Uses Structure of Arrays (SoA) for data layout optimization.
 */
export default class EnemyManager {
  /**
   * Creates an enemy manager
   * @param {number} maxEnemies - Maximum number of enemies to allow
   */
  constructor(maxEnemies = 1000) {
    this.maxEnemies = maxEnemies;
    this.enemyCount = 0;
    this.nextEnemyId = 1; // For assigning unique IDs

    // SoA data layout for position and basic properties
    this.id = new Array(maxEnemies);         // Unique enemy IDs
    this.x = new Float32Array(maxEnemies);   // X position
    this.y = new Float32Array(maxEnemies);   // Y position
    this.width = new Float32Array(maxEnemies);  // Collision width
    this.height = new Float32Array(maxEnemies); // Collision height
    this.type = new Uint8Array(maxEnemies);  // Enemy type (0-4)
    this.health = new Float32Array(maxEnemies); // Current health
    this.maxHealth = new Float32Array(maxEnemies); // Maximum health
    
    // Behavior properties
    this.moveSpeed = new Float32Array(maxEnemies); // Movement speed
    this.chaseRadius = new Float32Array(maxEnemies); // Chase detection radius
    this.shootRange = new Float32Array(maxEnemies); // Shooting range
    this.cooldown = new Float32Array(maxEnemies); // Shoot cooldown time
    this.currentCooldown = new Float32Array(maxEnemies); // Current cooldown timer
    this.damage = new Float32Array(maxEnemies); // Bullet damage
    this.bulletSpeed = new Float32Array(maxEnemies); // Bullet speed
    this.projectileCount = new Uint8Array(maxEnemies); // Number of projectiles to fire
    this.projectileSpread = new Float32Array(maxEnemies); // Angular spread for multiple projectiles
    this.canChase = new Uint8Array(maxEnemies); // Whether enemy can chase (1 or 0)
    this.canShoot = new Uint8Array(maxEnemies); // Whether enemy can shoot (1 or 0)
    
    // New fields for visual effects
    this.flashTimer = new Float32Array(maxEnemies); // Timer for flash effect
    this.isFlashing = new Uint8Array(maxEnemies); // Whether enemy is flashing
    this.deathTimer = new Float32Array(maxEnemies); // Timer for death animation
    this.isDying = new Uint8Array(maxEnemies); // Whether enemy is dying

    // Mapping from ID to index for fast lookups
    this.idToIndex = new Map();
    
    // Define enemy types with default values
    this.enemyDefaults = [
      // Type 0: Basic enemy - moderate speed, moderate health, single shot
      {
        width: 25,
        height: 25,
        health: 100,
        maxHealth: 100,
        moveSpeed: 40,
        chaseRadius: 250,
        shootRange: 200,
        cooldown: 2.0,
        damage: 10,
        bulletSpeed: 100,
        projectileCount: 1,
        projectileSpread: 0,
        canChase: 1,
        canShoot: 1
      },
      // Type 1: Fast enemy - high speed, low health, quick shots
      {
        width: 20,
        height: 20,
        health: 60,
        maxHealth: 60,
        moveSpeed: 70,
        chaseRadius: 300,
        shootRange: 150,
        cooldown: 1.0,
        damage: 8,
        bulletSpeed: 120,
        projectileCount: 1,
        projectileSpread: 0,
        canChase: 1,
        canShoot: 1
      },
      // Type 2: Heavy enemy - slow speed, high health, multiple shots
      {
        width: 35,
        height: 35,
        health: 200,
        maxHealth: 200,
        moveSpeed: 25,
        chaseRadius: 350,
        shootRange: 250,
        cooldown: 3.0,
        damage: 15,
        bulletSpeed: 80,
        projectileCount: 3,
        projectileSpread: Math.PI/8,
        canChase: 1,
        canShoot: 1
      },
      // Type 3: Stationary turret - no movement, medium health, fast shots
      {
        width: 28,
        height: 28,
        health: 120,
        maxHealth: 120,
        moveSpeed: 0,
        chaseRadius: 0,
        shootRange: 300,
        cooldown: 1.5,
        damage: 12,
        bulletSpeed: 150,
        projectileCount: 1,
        projectileSpread: 0,
        canChase: 0,
        canShoot: 1
      },
      // Type 4: Melee enemy - fast speed, medium health, no shooting
      {
        width: 30,
        height: 30,
        health: 150,
        maxHealth: 150,
        moveSpeed: 60,
        chaseRadius: 200,
        shootRange: 0,
        cooldown: 0,
        damage: 0,
        bulletSpeed: 0,
        projectileCount: 0,
        projectileSpread: 0,
        canChase: 1,
        canShoot: 0
      }
    ];
    
    // Initialize behavior system
    this.behaviorSystem = new BehaviorSystem();
  }

  /**
   * Spawn a new enemy
   * @param {number} type - Enemy type (0-4)
   * @param {number} x - X position to spawn
   * @param {number} y - Y position to spawn
   * @returns {string} The ID of the new enemy
   */
  spawnEnemy(type, x, y) {
    if (this.enemyCount >= this.maxEnemies) {
      console.warn('EnemyManager: Maximum enemy capacity reached');
      return null;
    }
    
    // Validate enemy type
    if (type < 0 || type >= this.enemyDefaults.length) {
      type = 0; // Default to type 0 if invalid
    }
    
    // Get default values for this enemy type
    const defaults = this.enemyDefaults[type];
    
    // Assign unique ID and store in manager
    const enemyId = `enemy_${this.nextEnemyId++}`;
    const index = this.enemyCount++;
    
    // Store basic properties
    this.id[index] = enemyId;
    this.x[index] = x;
    this.y[index] = y;
    this.width[index] = defaults.width;
    this.height[index] = defaults.height;
    this.type[index] = type;
    this.health[index] = defaults.health;
    this.maxHealth[index] = defaults.maxHealth;
    
    // Store behavior properties
    this.moveSpeed[index] = defaults.moveSpeed;
    this.chaseRadius[index] = defaults.chaseRadius;
    this.shootRange[index] = defaults.shootRange;
    this.cooldown[index] = defaults.cooldown;
    this.currentCooldown[index] = 0; // Start with no cooldown
    this.damage[index] = defaults.damage;
    this.bulletSpeed[index] = defaults.bulletSpeed;
    this.projectileCount[index] = defaults.projectileCount;
    this.projectileSpread[index] = defaults.projectileSpread;
    this.canChase[index] = defaults.canChase;
    this.canShoot[index] = defaults.canShoot;
    
    // Initialize visual effect timers
    this.flashTimer[index] = 0;
    this.isFlashing[index] = 0;
    this.deathTimer[index] = 0;
    this.isDying[index] = 0;
    
    // Store ID to index mapping
    this.idToIndex.set(enemyId, index);
    
    // Initialize behavior for this enemy
    this.behaviorSystem.initBehavior(index, type);
    
    console.log(`Spawned enemy ${enemyId} of type ${type} at position (${x.toFixed(2)}, ${y.toFixed(2)}), health: ${defaults.health}`);
    
    return enemyId;
  }

  /**
   * Update all enemies
   * @param {number} deltaTime - Time elapsed since last update in seconds
   * @param {Object} bulletManager - Reference to the bullet manager for shooting
   * @param {Object} target - Optional target entity (e.g., player)
   * @returns {number} The number of active enemies
   */
  update(deltaTime, bulletManager, target = null) {
    // Skip update if no target or bullet manager
    if (!target) return this.getActiveEnemyCount();
    
    // Count of active enemies
    let activeCount = 0;
    
    for (let i = 0; i < this.enemyCount; i++) {
      // Skip dead enemies
      if (this.health[i] <= 0) {
        if (this.isDying[i]) {
          // Update death animation
          this.updateDeathAnimation(i, deltaTime);
          activeCount++; // Still count dying enemies as active
        }
        continue;
      }
      
      activeCount++;
      
      // Update cooldowns
      if (this.currentCooldown[i] > 0) {
        this.currentCooldown[i] -= deltaTime;
      }
      
      // Update flash effect
      if (this.isFlashing[i]) {
        this.updateFlashEffect(i, deltaTime);
      }
      
      // Update enemy behavior using the behavior system
      this.behaviorSystem.updateBehavior(i, this, bulletManager, target, deltaTime);
    }
    
    return activeCount;
  }
  
  /**
   * Apply a hit effect (flash) to an enemy
   * @param {number} index - Enemy index
   */
  applyHitEffect(index) {
    this.isFlashing[index] = 1;
    this.flashTimer[index] = 0.1; // Flash for 100ms
  }
  
  /**
   * Update flash effect
   * @param {number} index - Enemy index
   * @param {number} deltaTime - Time elapsed since last update
   */
  updateFlashEffect(index, deltaTime) {
    if (this.flashTimer[index] > 0) {
      this.flashTimer[index] -= deltaTime;
    } else {
      this.isFlashing[index] = 0;
    }
  }
  
  /**
   * Start the death animation for an enemy
   * @param {number} index - Enemy index
   */
  startDeathAnimation(index) {
    this.isDying[index] = 1;
    this.deathTimer[index] = 0.5; // Death animation duration
  }
  
  /**
   * Update death animation
   * @param {number} index - Enemy index
   * @param {number} deltaTime - Time elapsed since last update
   */
  updateDeathAnimation(index, deltaTime) {
    if (this.deathTimer[index] > 0) {
      this.deathTimer[index] -= deltaTime;
    } else {
      // Animation finished, remove the enemy
      this.removeEnemy(index);
    }
  }
  
  /**
   * Apply damage to an enemy
   * @param {number} index - Enemy index
   * @param {number} damage - Amount of damage to apply
   * @returns {Object} Result with new health and whether the enemy was killed
   */
  applyDamage(index, damage) {
    if (index < 0 || index >= this.enemyCount) {
      return { valid: false, reason: 'Invalid enemy index' };
    }
    
    // Apply damage
    this.health[index] -= damage;
    
    // Apply hit effect
    this.applyHitEffect(index);
    
    // Check if killed
    const killed = this.health[index] <= 0;
    if (killed) {
      // Start death animation
      this.startDeathAnimation(index);
      
      // Call onDeath to handle death effects
      this.onDeath(index);
    }
    
    return {
      valid: true,
      health: this.health[index],
      killed: killed
    };
  }

  /**
   * Remove an enemy using the swap-and-pop technique
   * @param {number} index - Index of enemy to remove
   */
  removeEnemy(index) {
    const last = this.enemyCount - 1;
    
    // Remove from ID mapping
    const id = this.id[index];
    this.idToIndex.delete(id);
    
    if (index !== last) {
      // Swap with the last enemy - basic properties
      this.id[index] = this.id[last];
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
      this.type[index] = this.type[last];
      this.health[index] = this.health[last];
      this.maxHealth[index] = this.maxHealth[last];
      
      // Swap behavior properties
      this.moveSpeed[index] = this.moveSpeed[last];
      this.chaseRadius[index] = this.chaseRadius[last];
      this.shootRange[index] = this.shootRange[last];
      this.cooldown[index] = this.cooldown[last];
      this.currentCooldown[index] = this.currentCooldown[last];
      this.damage[index] = this.damage[last];
      this.bulletSpeed[index] = this.bulletSpeed[last];
      this.projectileCount[index] = this.projectileCount[last];
      this.projectileSpread[index] = this.projectileSpread[last];
      this.canChase[index] = this.canChase[last];
      this.canShoot[index] = this.canShoot[last];
      
      // Swap visual effect properties
      this.flashTimer[index] = this.flashTimer[last];
      this.isFlashing[index] = this.isFlashing[last];
      this.deathTimer[index] = this.deathTimer[last];
      this.isDying[index] = this.isDying[last];
      
      // Update ID mapping for the swapped enemy
      this.idToIndex.set(this.id[index], index);
    }
    
    this.enemyCount--;
  }
  
  /**
   * Find enemy index by ID
   * @param {string} enemyId - Enemy ID to find
   * @returns {number} Index of enemy or -1 if not found
   */
  findIndexById(enemyId) {
    // Use the ID to index mapping
    const index = this.idToIndex.get(enemyId);
    return index !== undefined ? index : -1;
  }
  
  /**
   * Handle enemy death
   * @param {number} index - Index of enemy that died
   * @param {string} killedBy - ID of player who killed the enemy
   */
  onDeath(index, killedBy) {
    // The actual removal is now handled by the death animation
    // Additional death effects or drops can be added here
  }

  /**
   * Get number of active enemies
   * @returns {number} Count of active enemies
   */
  getActiveEnemyCount() {
    return this.enemyCount;
  }

  /**
   * Clean up resources
   */
  cleanup() {
    this.enemyCount = 0;
    this.idToIndex.clear();
  }

  /**
   * Get enemy data array for network transmission
   * @returns {Array} Array of enemy data objects
   */
  getEnemiesData() {
    const enemies = [];
    
    for (let i = 0; i < this.enemyCount; i++) {
      // Skip completely dead enemies (those done with death animation)
      if (this.health[i] <= 0 && !this.isDying[i]) continue;
      
      enemies.push({
        id: this.id[i],
        x: this.x[i],
        y: this.y[i],
        width: this.width[i],
        height: this.height[i],
        type: this.type[i],
        health: this.health[i],
        maxHealth: this.maxHealth[i],
        isFlashing: this.isFlashing[i],
        isDying: this.isDying[i],
        deathStage: this.isDying[i] ? Math.floor((1 - this.deathTimer[i] / 0.5) * 4) : 0
      });
    }
    
    return enemies;
  }

  /**
   * Add a new enemy - alias for spawnEnemy for backward compatibility
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {number} type - Enemy type (0-4)
   * @returns {string} The ID of the new enemy
   */
  addEnemy(x, y, type = 0) {
    return this.spawnEnemy(type, x, y);
  }
}