// File: /src/Managers/CollisionManager.js

/**
 * CollisionManager.js
 * Handles collision detection and processing between entities
 */

export default class CollisionManager {
  /**
   * Creates a collision manager
   * @param {Object} bulletManager - The bullet manager instance
   * @param {Object} enemyManager - The enemy manager instance
   * @param {Object} mapManager - The map manager (optional)
   */
  constructor(bulletManager, enemyManager, mapManager = null) {
    this.bulletManager = bulletManager;
    this.enemyManager = enemyManager;
    this.mapManager = mapManager;
    
    // Tracking processed collisions to avoid duplicates
    this.processedCollisions = new Map(); // collisionId -> timestamp
    this.cleanupInterval = setInterval(() => this.cleanupProcessedCollisions(), 10000);
  }
  
  /**
   * Check for all collisions in the current state
   * Called on each update cycle
   */
  checkCollisions() {
    // Skip if managers aren't properly initialized
    if (!this.bulletManager || !this.enemyManager) return;
    
    // For each bullet, check collision with enemies
    for (let bi = 0; bi < this.bulletManager.bulletCount; bi++) {
      // Skip expired bullets
      if (this.bulletManager.life[bi] <= 0) continue;
      
      const bulletX = this.bulletManager.x[bi];
      const bulletY = this.bulletManager.y[bi];
      const bulletWidth = this.bulletManager.width[bi];
      const bulletHeight = this.bulletManager.height[bi];
      const bulletId = this.bulletManager.id[bi];
      const bulletOwnerId = this.bulletManager.ownerId[bi];
      
      // Check for collisions with walls/obstacles if map manager exists
      if (this.mapManager && this.mapManager.isWallOrOutOfBounds) {
        if (this.mapManager.isWallOrOutOfBounds(bulletX, bulletY)) {
          // Bullet hit a wall, mark for removal
          this.bulletManager.markForRemoval(bi);
          continue;
        }
      }
      
      // Check for enemy collisions
      for (let ei = 0; ei < this.enemyManager.enemyCount; ei++) {
        // Skip dead enemies
        if (this.enemyManager.health[ei] <= 0) continue;
        
        const enemyX = this.enemyManager.x[ei];
        const enemyY = this.enemyManager.y[ei];
        const enemyWidth = this.enemyManager.width[ei];
        const enemyHeight = this.enemyManager.height[ei];
        const enemyId = this.enemyManager.id[ei];
        
        // Check if bullet and enemy collide (AABB)
        if (this.checkAABBCollision(
          bulletX, bulletY, bulletWidth, bulletHeight,
          enemyX, enemyY, enemyWidth, enemyHeight
        )) {
          // Create collision ID to track this collision
          const collisionId = `${bulletId}_${enemyId}`;
          
          // Skip if already processed
          if (this.processedCollisions.has(collisionId)) continue;
          
          // Process this collision
          this.processCollision(bi, ei, bulletOwnerId);
          
          // Mark as processed
          this.processedCollisions.set(collisionId, Date.now());
          
          // Break the enemy loop since bullet hit something
          break;
        }
      }
    }
  }
  
  /**
   * Validate client-reported collision (server-side)
   * @param {Object} data - Collision data from client
   * @returns {Object} Validation result
   */
  validateCollision(data) {
    const { bulletId, enemyId, timestamp, clientId } = data;
    
    // Find bullet and enemy indices using IDs
    const bulletIndex = this.findBulletIndex(bulletId);
    const enemyIndex = this.findEnemyIndex(enemyId);
    
    // Check if both entities exist
    if (bulletIndex === -1 || enemyIndex === -1) {
      return { 
        valid: false, 
        reason: 'Entity not found',
        bulletId,
        enemyId
      };
    }
    
    // Check if this collision was already processed recently
    const collisionId = `${bulletId}_${enemyId}`;
    if (this.processedCollisions.has(collisionId)) {
      return { 
        valid: false, 
        reason: 'Already processed',
        bulletId,
        enemyId
      };
    }
    
    // Check if timestamp is reasonable (within 500ms from now)
    const now = Date.now();
    if (Math.abs(now - timestamp) > 500) {
      return { 
        valid: false, 
        reason: 'Timestamp too old',
        bulletId,
        enemyId 
      };
    }
    
    // Check for line of sight obstruction
    if (this.mapManager && this.mapManager.hasLineOfSight) {
      if (!this.mapManager.hasLineOfSight(
        this.bulletManager.x[bulletIndex],
        this.bulletManager.y[bulletIndex],
        this.enemyManager.x[enemyIndex],
        this.enemyManager.y[enemyIndex]
      )) {
        return { 
          valid: false, 
          reason: 'No line of sight',
          bulletId,
          enemyId 
        };
      }
    }
    
    // Check for actual collision (AABB)
    if (!this.checkAABBCollision(
      this.bulletManager.x[bulletIndex],
      this.bulletManager.y[bulletIndex],
      this.bulletManager.width[bulletIndex],
      this.bulletManager.height[bulletIndex],
      this.enemyManager.x[enemyIndex],
      this.enemyManager.y[enemyIndex],
      this.enemyManager.width[enemyIndex],
      this.enemyManager.height[enemyIndex]
    )) {
      return { 
        valid: false, 
        reason: 'No collision detected',
        bulletId,
        enemyId 
      };
    }
    
    // Collision is valid - process it and store result
    const result = this.processCollision(bulletIndex, enemyIndex, clientId);
    
    // Mark as processed to avoid duplicates
    this.processedCollisions.set(collisionId, now);
    
    return {
      valid: true,
      ...result
    };
  }
  
  /**
   * Process a valid collision
   * @param {number} bulletIndex - Index of bullet in bulletManager
   * @param {number} enemyIndex - Index of enemy in enemyManager
   * @param {string|number} clientId - ID of the client that reported the collision
   * @returns {Object} Collision results
   */
  processCollision(bulletIndex, enemyIndex, clientId) {
    // Get bullet and enemy details
    const bulletId = this.bulletManager.id[bulletIndex];
    const enemyId = this.enemyManager.id[enemyIndex];
    
    // Calculate damage
    const damage = this.bulletManager.damage ? 
      this.bulletManager.damage[bulletIndex] : 10; // Default damage
    
    // Apply damage to enemy
    const remainingHealth = this.enemyManager.applyDamage(enemyIndex, damage);
    
    // Remove bullet
    if (this.bulletManager.markForRemoval) {
      this.bulletManager.markForRemoval(bulletIndex);
    } else if (this.bulletManager.life) {
      // Alternative removal method
      this.bulletManager.life[bulletIndex] = 0;
    }
    
    // Handle enemy death if needed
    let enemyKilled = false;
    if (remainingHealth <= 0) {
      enemyKilled = true;
      
      // Call enemy manager's death handler
      if (this.enemyManager.onDeath) {
        this.enemyManager.onDeath(enemyIndex, clientId);
      }
    }
    
    // Return collision result
    return {
      bulletId,
      enemyId,
      damage,
      enemyHealth: remainingHealth,
      enemyKilled,
      clientId
    };
  }
  
  /**
   * Find bullet index by ID
   * @param {string|number} bulletId - ID of the bullet
   * @returns {number} Bullet index or -1 if not found
   */
  findBulletIndex(bulletId) {
    // If bulletManager has a lookup method, use it
    if (this.bulletManager.findIndexById) {
      return this.bulletManager.findIndexById(bulletId);
    }
    
    // Otherwise search by ID array
    for (let i = 0; i < this.bulletManager.bulletCount; i++) {
      if (this.bulletManager.id[i] === bulletId) {
        return i;
      }
    }
    
    return -1;
  }
  
  /**
   * Find enemy index by ID
   * @param {string|number} enemyId - ID of the enemy
   * @returns {number} Enemy index or -1 if not found
   */
  findEnemyIndex(enemyId) {
    // If enemyManager has a lookup method, use it
    if (this.enemyManager.findIndexById) {
      return this.enemyManager.findIndexById(enemyId);
    }
    
    // Otherwise search by ID array
    for (let i = 0; i < this.enemyManager.enemyCount; i++) {
      if (this.enemyManager.id[i] === enemyId) {
        return i;
      }
    }
    
    return -1;
  }
  
  /**
   * AABB collision check
   */
  checkAABBCollision(ax, ay, awidth, aheight, bx, by, bwidth, bheight) {
    return (
      ax < bx + bwidth &&
      ax + awidth > bx &&
      ay < by + bheight &&
      ay + aheight > by
    );
  }
  
  /**
   * Clean up old processed collisions to prevent memory leaks
   */
  cleanupProcessedCollisions() {
    const now = Date.now();
    const expiryTime = 5000; // 5 seconds
    
    for (const [id, timestamp] of this.processedCollisions.entries()) {
      if (now - timestamp > expiryTime) {
        this.processedCollisions.delete(id);
      }
    }
  }
  
  /**
   * Clean up when shutting down
   */
  cleanup() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
  }
}