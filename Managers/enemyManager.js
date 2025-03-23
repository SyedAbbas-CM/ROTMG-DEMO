// File: /src/managers/EnemyManager.js

import {
  createBlackOverlord,
  createRedBerserker,
  createPurpleIllusionist,
  createEmeraldRegenerator,
  createNavyTurtle
} from '../Behaviours/Experimental/ColorCodedIndex.js';

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
    
    // Store more complex behavior objects
    this.behaviors = new Array(maxEnemies);  // Behavior objects (from ColorCodedIndex)
    
    // Mapping from ID to index for fast lookups
    this.idToIndex = new Map();
    
    // Enemy factory functions for different types
    this.enemyFactories = [
      createBlackOverlord,    // Type 0
      createRedBerserker,     // Type 1
      createPurpleIllusionist, // Type 2
      createEmeraldRegenerator, // Type 3
      createNavyTurtle        // Type 4
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
    
    // Get the correct factory function for this enemy type
    const createEnemy = this.enemyFactories[type];
    if (!createEnemy) {
      console.warn(`EnemyManager: Unknown enemy type: ${type}`);
      return null;
    }
    
    // Create the enemy with the behavior system
    const enemy = createEnemy();
    
    // Assign unique ID and store in manager
    const enemyId = `enemy_${this.nextEnemyId++}`;
    const index = this.enemyCount++;
    
    // Store basic properties in SoA
    this.id[index] = enemyId;
    this.x[index] = x;
    this.y[index] = y;
    this.width[index] = enemy.width || 20;
    this.height[index] = enemy.height || 20;
    this.type[index] = type;
    this.health[index] = enemy.hp;
    this.maxHealth[index] = enemy.maxHp;
    
    // Store the behavior object
    this.behaviors[index] = enemy;
    
    // Update x/y in the behavior object to match our SoA
    enemy.x = x;
    enemy.y = y;
    
    // Store ID to index mapping
    this.idToIndex.set(enemyId, index);
    
    return enemyId;
  }

  /**
   * Update all enemies
   * @param {number} deltaTime - Time elapsed since last update in seconds
   * @param {Object} bulletManager - Reference to the bullet manager for shooting
   */
  update(deltaTime, bulletManager) {
    for (let i = 0; i < this.enemyCount; i++) {
      const enemy = this.behaviors[i];
      
      // Skip inactive enemies
      if (!enemy || this.health[i] <= 0) continue;
      
      // Update enemy behavior (using the ColorCodedIndex behavior system)
      const target = { x: 0, y: 0 }; // TODO: Get player/target position
      enemy.update(target, deltaTime * 1000); // Convert to ms for behavior system
      
      // Sync position from behavior object to SoA
      this.x[i] = enemy.x;
      this.y[i] = enemy.y;
    }
  }
  
  /**
   * Apply damage to an enemy
   * @param {number} index - Enemy index
   * @param {number} damage - Amount of damage to apply
   * @returns {number} Remaining health
   */
  applyDamage(index, damage) {
    if (index < 0 || index >= this.enemyCount) return 0;
    
    this.health[index] -= damage;
    
    // Ensure health doesn't go below 0
    if (this.health[index] < 0) {
      this.health[index] = 0;
    }
    
    return this.health[index];
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
      // Swap with the last enemy
      this.id[index] = this.id[last];
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
      this.type[index] = this.type[last];
      this.health[index] = this.health[last];
      this.maxHealth[index] = this.maxHealth[last];
      this.behaviors[index] = this.behaviors[last];
      
      // Update ID mapping for the swapped enemy
      this.idToIndex.set(this.id[index], index);
    }
    
    // Clear the behavior reference to allow garbage collection
    this.behaviors[last] = null;
    
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
    // TODO: Handle drops, score, etc.
    
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
    // Clear behavior references to allow garbage collection
    for (let i = 0; i < this.enemyCount; i++) {
      this.behaviors[i] = null;
    }
    
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
}