/**
 * ServerCollisionManager.js
 * Handles server-side collision validation and authoritative collision processing.
 */

class ServerCollisionManager {
  /**
   * Creates a server-side collision manager
   * @param {BulletManager} bulletManager - The server's bullet manager instance
   * @param {EnemyManager} enemyManager - The server's enemy manager instance
   * @param {MapManager} mapManager - The server's map manager
   */
  constructor(bulletManager, enemyManager, mapManager) {
    this.bulletManager = bulletManager;
    this.enemyManager = enemyManager;
    this.mapManager = mapManager;
    
    // Tracking processed collisions to avoid duplicates
    this.processedCollisions = new Map(); // collisionId -> timestamp
    this.cleanupInterval = setInterval(() => this.cleanupProcessedCollisions(), 10000);
  }
  
  /**
   * Validate client-reported collision
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
    if (!this.hasLineOfSight(
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
    const bulletId = this.bulletManager.id ? 
      this.bulletManager.id[bulletIndex] : bulletIndex;
      
    const enemyId = this.enemyManager.id ? 
      this.enemyManager.id[enemyIndex] : enemyIndex;
    
    // Calculate damage
    const damage = this.bulletManager.damage ? 
      this.bulletManager.damage[bulletIndex] : 10; // Default damage
    
    // Apply damage to enemy
    if (this.enemyManager.health) {
      this.enemyManager.health[enemyIndex] -= damage;
    }
    
    // Get current enemy health
    const enemyHealth = this.enemyManager.health ? 
      this.enemyManager.health[enemyIndex] : 0;
    
    // Remove bullet
    if (this.bulletManager.markForRemoval) {
      this.bulletManager.markForRemoval(bulletIndex);
    } else {
      // Default removal method
      this.bulletManager.life[bulletIndex] = 0;
    }
    
    // Handle enemy death if needed
    let enemyKilled = false;
    if (enemyHealth <= 0) {
      enemyKilled = true;
      
      // Call enemy manager's death handler if available
      if (this.enemyManager.onDeath) {
        this.enemyManager.onDeath(enemyIndex, clientId);
      }
    }
    
    // Return collision result
    return {
      bulletId,
      enemyId,
      damage,
      enemyHealth,
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
    // If using numerical indices as IDs
    if (typeof bulletId === 'number' && 
        bulletId >= 0 && 
        bulletId < this.bulletManager.bulletCount) {
      return bulletId;
    }
    
    // If bulletManager has a lookup method, use it
    if (this.bulletManager.findIndexById) {
      return this.bulletManager.findIndexById(bulletId);
    }
    
    // Otherwise search by ID array if it exists
    if (this.bulletManager.id) {
      for (let i = 0; i < this.bulletManager.bulletCount; i++) {
        if (this.bulletManager.id[i] === bulletId) {
          return i;
        }
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
    // If using numerical indices as IDs
    if (typeof enemyId === 'number' && 
        enemyId >= 0 && 
        enemyId < this.enemyManager.enemyCount) {
      return enemyId;
    }
    
    // If enemyManager has a lookup method, use it
    if (this.enemyManager.findIndexById) {
      return this.enemyManager.findIndexById(enemyId);
    }
    
    // Otherwise search by ID array if it exists
    if (this.enemyManager.id) {
      for (let i = 0; i < this.enemyManager.enemyCount; i++) {
        if (this.enemyManager.id[i] === enemyId) {
          return i;
        }
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
   * Check line of sight between two points (tile-based)
   * @param {number} x1 - Starting X position
   * @param {number} y1 - Starting Y position
   * @param {number} x2 - Ending X position
   * @param {number} y2 - Ending Y position
   * @returns {boolean} True if line of sight exists
   */
  hasLineOfSight(x1, y1, x2, y2) {
    // If no map manager or it doesn't have getTile, always return true
    if (!this.mapManager || !this.mapManager.getTile) {
      return true;
    }
    
    // Bresenham's line algorithm to check tiles between points
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    const sx = x1 < x2 ? 1 : -1;
    const sy = y1 < y2 ? 1 : -1;
    let err = dx - dy;
    
    let tileSize = this.mapManager.TILE_SIZE || 12; // Default if not specified
    let currentX = x1;
    let currentY = y1;
    
    while (currentX !== x2 || currentY !== y2) {
      // Convert to tile coordinates
      const tileX = Math.floor(currentX / tileSize);
      const tileY = Math.floor(currentY / tileSize);
      
      // Check if this tile blocks line of sight
      const tile = this.mapManager.getTile(tileX, tileY);
      
      // Check if tile is a wall or other solid obstacle
      // Adjust tile types based on your game's tile system
      if (tile && (
          tile.type === 1 || // Wall
          tile.type === 4    // Mountain
      )) {
        return false;
      }
      
      // Move to next position
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        currentX += sx;
      }
      if (e2 < dx) {
        err += dx;
        currentY += sy;
      }
    }
    
    return true;
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
    clearInterval(this.cleanupInterval);
  }
}

module.exports = ServerCollisionManager;