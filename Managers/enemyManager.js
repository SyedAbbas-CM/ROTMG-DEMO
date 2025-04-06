// File: /src/Managers/EnemyManager.js

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
    
    // Store ID to index mapping
    this.idToIndex.set(enemyId, index);
    
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
    
    // Count enemies that moved or shot
    let movedCount = 0;
    let shotCount = 0;
    
    for (let i = 0; i < this.enemyCount; i++) {
      // Skip inactive enemies
      if (this.health[i] <= 0) continue;
      
      // Update cooldowns
      if (this.currentCooldown[i] > 0) {
        this.currentCooldown[i] -= deltaTime;
      }
      
      // Store original position for movement tracking
      const originalX = this.x[i];
      const originalY = this.y[i];
      
      // Update chase behavior
      if (this.canChase[i]) {
        this.updateChase(i, target, deltaTime);
      }
      
      // Check if enemy moved
      if (Math.abs(originalX - this.x[i]) > 0.01 || Math.abs(originalY - this.y[i]) > 0.01) {
        movedCount++;
      }
      
      // Update shooting behavior
      if (this.canShoot[i] && bulletManager) {
        const didShoot = this.updateShoot(i, target, bulletManager);
        if (didShoot) {
          shotCount++;
        }
      }
    }
    
    const activeCount = this.getActiveEnemyCount();
    // Only log if there's actual activity
    if (movedCount > 0 || shotCount > 0) {
      console.log(`Enemies updated: ${activeCount} active, ${movedCount} moved, ${shotCount} shot at target`);
    }
    
    return activeCount;
  }
  
  /**
   * Update chase behavior for an enemy
   * @param {number} index - Enemy index
   * @param {Object} target - Target entity
   * @param {number} deltaTime - Time elapsed since last update
   */
  updateChase(index, target, deltaTime) {
    const dx = target.x - this.x[index];
    const dy = target.y - this.y[index];
    const distanceSquared = dx * dx + dy * dy;
    
    // Only chase if within chase radius
    if (distanceSquared > this.chaseRadius[index] * this.chaseRadius[index]) {
      return false;
    }
    
    // Normalize direction
    const distance = Math.sqrt(distanceSquared);
    const dirX = dx / distance;
    const dirY = dy / distance;
    
    // Move toward target
    const moveAmount = this.moveSpeed[index] * deltaTime;
    this.x[index] += dirX * moveAmount;
    this.y[index] += dirY * moveAmount;
    
    return true;
  }
  
  /**
   * Update shooting behavior for an enemy
   * @param {number} index - Enemy index
   * @param {Object} target - Target entity
   * @param {Object} bulletManager - Reference to the bullet manager
   * @returns {boolean} True if the enemy shot, false otherwise
   */
  updateShoot(index, target, bulletManager) {
    // Skip if on cooldown
    if (this.currentCooldown[index] > 0) {
      return false;
    }
    
    const dx = target.x - this.x[index];
    const dy = target.y - this.y[index];
    const distanceSquared = dx * dx + dy * dy;
    
    // Only shoot if within range
    if (distanceSquared > this.shootRange[index] * this.shootRange[index]) {
      return false;
    }
    
    // Calculate angle to target
    const angle = Math.atan2(dy, dx);
    
    // Reset cooldown
    this.currentCooldown[index] = this.cooldown[index];
    
    // Shoot multiple projectiles if needed
    const count = this.projectileCount[index];
    const spread = this.projectileSpread[index];
    
    // Single projectile case
    if (count <= 1) {
      bulletManager.addBullet({
        x: this.x[index],
        y: this.y[index],
        vx: Math.cos(angle) * this.bulletSpeed[index],
        vy: Math.sin(angle) * this.bulletSpeed[index],
        ownerId: this.id[index],
        damage: this.damage[index],
        lifetime: 3.0,
        isEnemy: true
      });
      
      console.log(`Enemy ${this.id[index]} fired at target, angle: ${angle.toFixed(2)}`);
      return true;
    }
    
    // Multiple projectiles case
    const startAngle = angle - (spread * (count - 1) / 2);
    for (let i = 0; i < count; i++) {
      const bulletAngle = startAngle + (spread * i);
      bulletManager.addBullet({
        x: this.x[index],
        y: this.y[index],
        vx: Math.cos(bulletAngle) * this.bulletSpeed[index],
        vy: Math.sin(bulletAngle) * this.bulletSpeed[index],
        ownerId: this.id[index],
        damage: this.damage[index],
        lifetime: 3.0,
        isEnemy: true
      });
    }
    
    console.log(`Enemy ${this.id[index]} fired ${count} bullets at target in a spread pattern`);
    return true;
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
    
    // Check if killed
    const killed = this.health[index] <= 0;
    if (killed) {
      // Call onDeath to handle death effects
      this.onDeath(index);
    }
    
    console.log(`Enemy ${this.id[index]} took ${damage} damage, health: ${this.health[index].toFixed(0)}/${this.maxHealth[index]}, killed: ${killed}`);
    
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
    // Remove the enemy
    this.removeEnemy(index);
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
      enemies.push({
        id: this.id[i],
        x: this.x[i],
        y: this.y[i],
        width: this.width[i],
        height: this.height[i],
        type: this.type[i],
        health: this.health[i],
        maxHealth: this.maxHealth[i]
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