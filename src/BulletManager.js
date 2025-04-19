// File: /src/managers/BulletManager.js

/**
 * BulletManager handles bullet creation, updating, and removal.
 * Uses Structure of Arrays (SoA) for data layout optimization.
 */
export default class BulletManager {
  /**
   * Creates a bullet manager
   * @param {number} maxBullets - Maximum number of bullets to allow
   */
  constructor(maxBullets = 10000) {
    this.maxBullets = maxBullets;
    this.bulletCount = 0;
    this.nextBulletId = 1; // For assigning unique IDs

    // SoA data layout
    this.id = new Array(maxBullets);        // Unique bullet IDs
    this.x = new Float32Array(maxBullets);  // X position
    this.y = new Float32Array(maxBullets);  // Y position
    this.vx = new Float32Array(maxBullets); // X velocity
    this.vy = new Float32Array(maxBullets); // Y velocity
    this.life = new Float32Array(maxBullets); // Remaining life in seconds
    this.width = new Float32Array(maxBullets);  // Collision width
    this.height = new Float32Array(maxBullets); // Collision height
    this.damage = new Float32Array(maxBullets);  // Damage amount
    this.ownerId = new Array(maxBullets);   // ID of entity that created this bullet
  }

  /**
   * Add a new bullet
   * @param {Object} bulletData - Bullet properties
   * @returns {string} The ID of the new bullet
   */
  addBullet(bulletData) {
    if (this.bulletCount >= this.maxBullets) {
      console.warn('BulletManager: Maximum bullet capacity reached');
      return null;
    }
    
    const bulletId = bulletData.id || `bullet_${this.nextBulletId++}`;
    const index = this.bulletCount++;
    
    // Set bullet properties
    this.id[index] = bulletId;
    this.x[index] = bulletData.x;
    this.y[index] = bulletData.y;
    this.vx[index] = bulletData.vx;
    this.vy[index] = bulletData.vy;
    this.life[index] = bulletData.lifetime || 3.0; // Default 3 seconds
    this.width[index] = bulletData.width || 5;
    this.height[index] = bulletData.height || 5;
    this.damage[index] = bulletData.damage || 10;
    this.ownerId[index] = bulletData.ownerId || null;
    
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
   * Remove a bullet using the swap-and-pop technique
   * @param {number} index - Index of bullet to remove
   */
  swapRemove(index) {
    const last = this.bulletCount - 1;
    
    if (index !== last) {
      // Swap with the last bullet
      this.id[index] = this.id[last];
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.vx[index] = this.vx[last];
      this.vy[index] = this.vy[last];
      this.life[index] = this.life[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
      this.damage[index] = this.damage[last];
      this.ownerId[index] = this.ownerId[last];
    }
    
    this.bulletCount--;
  }
  
  /**
   * Remove a bullet by ID
   * @param {string} bulletId - ID of bullet to remove
   * @returns {boolean} True if bullet was found and removed
   */
  removeBulletById(bulletId) {
    for (let i = 0; i < this.bulletCount; i++) {
      if (this.id[i] === bulletId) {
        this.swapRemove(i);
        return true;
      }
    }
    return false;
  }
  
  /**
   * Find bullet index by ID
   * @param {string} bulletId - Bullet ID to find
   * @returns {number} Index of bullet or -1 if not found
   */
  findIndexById(bulletId) {
    for (let i = 0; i < this.bulletCount; i++) {
      if (this.id[i] === bulletId) {
        return i;
      }
    }
    return -1;
  }
  
  /**
   * Mark a bullet for removal
   * @param {number} index - Index of bullet to remove
   */
  markForRemoval(index) {
    if (index >= 0 && index < this.bulletCount) {
      this.life[index] = 0;
    }
  }

  /**
   * Get number of active bullets
   * @returns {number} Count of active bullets
   */
  getActiveBulletCount() {
    return this.bulletCount;
  }

  /**
   * Clean up resources
   */
  cleanup() {
    this.bulletCount = 0;
  }

  /**
   * Get bullet data array for network transmission
   * @returns {Array} Array of bullet data objects
   */
  getBulletsData() {
    const bullets = [];
    
    for (let i = 0; i < this.bulletCount; i++) {
      bullets.push({
        id: this.id[i],
        x: this.x[i],
        y: this.y[i],
        vx: this.vx[i],
        vy: this.vy[i],
        width: this.width[i],
        height: this.height[i],
        life: this.life[i],
        damage: this.damage[i],
        ownerId: this.ownerId[i]
      });
    }
    
    return bullets;
  }
}