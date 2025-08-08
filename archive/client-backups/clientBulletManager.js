/**
 * ClientBulletManager.js
 * Client-side manager for bullets with client prediction and reconciliation
 */
export class ClientBulletManager {
    /**
     * Creates a client-side bullet manager
     * @param {number} maxBullets - Maximum number of bullets to support
     */
    constructor(maxBullets = 10000) {
      this.maxBullets = maxBullets;
      this.bulletCount = 0;
      
      // Structure of Arrays for performance
      this.id = new Array(maxBullets);        // Unique bullet IDs
      this.x = new Float32Array(maxBullets);  // X position
      this.y = new Float32Array(maxBullets);  // Y position
      this.vx = new Float32Array(maxBullets); // X velocity
      this.vy = new Float32Array(maxBullets); // Y velocity
      this.life = new Float32Array(maxBullets); // Remaining life in seconds
      this.width = new Float32Array(maxBullets);  // Width for collision
      this.height = new Float32Array(maxBullets); // Height for collision
      this.ownerId = new Array(maxBullets);   // Who fired this bullet
      
      // Visual properties
      this.sprite = new Array(maxBullets);    // Sprite or null
      
      // Mapping from ID to index for fast lookups
      this.idToIndex = new Map();
      
      // Local prediction bullets (client-created, not yet confirmed by server)
      this.localBullets = new Set();
    }
    
    /**
     * Add a new bullet
     * @param {Object} bulletData - Bullet properties
     * @returns {string} Bullet ID
     */
    addBullet(bulletData) {
      if (this.bulletCount >= this.maxBullets) {
        console.warn('ClientBulletManager: Maximum bullet capacity reached');
        return null;
      }
      
      const index = this.bulletCount++;
      const bulletId = bulletData.id || `local_${Date.now()}_${index}`;
      
      // Store bullet properties
      this.id[index] = bulletId;
      this.x[index] = bulletData.x;
      this.y[index] = bulletData.y;
      this.vx[index] = bulletData.vx;
      this.vy[index] = bulletData.vy;
      this.life[index] = bulletData.lifetime || 3.0; // Default 3 seconds
      this.width[index] = bulletData.width || 5;
      this.height[index] = bulletData.height || 5;
      this.ownerId[index] = bulletData.ownerId || null;
      this.sprite[index] = bulletData.sprite || null;
      
      // Store index for lookup
      this.idToIndex.set(bulletId, index);
      
      // If this is a local prediction bullet (client-created)
      if (bulletId.startsWith('local_')) {
        this.localBullets.add(bulletId);
      }
      
      return bulletId;
    }
    
    /**
     * Update all bullets
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    update(deltaTime) {
      let count = this.bulletCount;
      
      for (let i = 0; i < count; i++) {
        // Update position
        this.x[i] += this.vx[i] * deltaTime;
        this.y[i] += this.vy[i] * deltaTime;
        
        // Decrement lifetime
        this.life[i] -= deltaTime;
        
        // Remove expired bullets
        if (this.life[i] <= 0) {
          this.swapRemove(i);
          count--;
          i--;
        }
      }
      
      this.bulletCount = count;
    }
    
    /**
     * Remove a bullet using swap-and-pop technique
     * @param {number} index - Index of bullet to remove
     */
    swapRemove(index) {
      if (index < 0 || index >= this.bulletCount) return;
      
      // Remove from ID mapping and local bullets set
      const bulletId = this.id[index];
      this.idToIndex.delete(bulletId);
      this.localBullets.delete(bulletId);
      
      // Swap with the last bullet (if not already the last)
      const lastIndex = this.bulletCount - 1;
      if (index !== lastIndex) {
        // Copy properties from last bullet to this position
        this.id[index] = this.id[lastIndex];
        this.x[index] = this.x[lastIndex];
        this.y[index] = this.y[lastIndex];
        this.vx[index] = this.vx[lastIndex];
        this.vy[index] = this.vy[lastIndex];
        this.life[index] = this.life[lastIndex];
        this.width[index] = this.width[lastIndex];
        this.height[index] = this.height[lastIndex];
        this.ownerId[index] = this.ownerId[lastIndex];
        this.sprite[index] = this.sprite[lastIndex];
        
        // Update index in mapping
        this.idToIndex.set(this.id[index], index);
      }
      
      this.bulletCount--;
    }
    
    /**
     * Find bullet index by ID
     * @param {string} bulletId - ID of bullet to find
     * @returns {number} Index of bullet or -1 if not found
     */
    findIndexById(bulletId) {
      const index = this.idToIndex.get(bulletId);
      return index !== undefined ? index : -1;
    }
    
    /**
     * Remove a bullet by ID
     * @param {string} bulletId - ID of bullet to remove
     * @returns {boolean} True if bullet was found and removed
     */
    removeBulletById(bulletId) {
      const index = this.findIndexById(bulletId);
      if (index !== -1) {
        this.swapRemove(index);
        return true;
      }
      return false;
    }
    
    /**
     * Set initial bullets list from server
     * @param {Array} bullets - Array of bullet data from server
     */
    setBullets(bullets) {
      // Clear existing bullets except local predictions
      this.clearNonLocalBullets();
      
      // Add new bullets from server
      for (const bullet of bullets) {
        // Skip if we already have a local prediction for this ID
        if (this.findIndexById(bullet.id) !== -1) continue;
        
        this.addBullet(bullet);
      }
    }
    
    /**
     * Clear all non-local (server confirmed) bullets
     */
    clearNonLocalBullets() {
      // Remove bullets that aren't local predictions
      for (let i = 0; i < this.bulletCount; i++) {
        if (!this.localBullets.has(this.id[i])) {
          this.swapRemove(i);
          i--;
        }
      }
    }
    
    /**
     * Update bullets based on server data
     * @param {Array} bullets - Array of bullet data from server
     */
    updateBullets(bullets) {
      // Process server bullets
      for (const bullet of bullets) {
        const index = this.findIndexById(bullet.id);
        
        if (index !== -1) {
          // Update existing bullet
          this.x[index] = bullet.x;
          this.y[index] = bullet.y;
          this.vx[index] = bullet.vx;
          this.vy[index] = bullet.vy;
          this.life[index] = bullet.life;
        } else {
          // Add new bullet if we don't have it
          this.addBullet(bullet);
        }
      }
      
      // Remove bullets that aren't in the server update and aren't local predictions
      const serverBulletIds = new Set(bullets.map(b => b.id));
      
      for (let i = 0; i < this.bulletCount; i++) {
        const bulletId = this.id[i];
        
        // Keep local predictions and bullets from server update
        if (!this.localBullets.has(bulletId) && !serverBulletIds.has(bulletId)) {
          this.swapRemove(i);
          i--;
        }
      }
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
      this.bulletCount = 0;
      this.idToIndex.clear();
      this.localBullets.clear();
    }
  }