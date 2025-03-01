// File: /src/managers/BulletManager.js

export default class BulletManager {
    constructor(maxBullets = 10000) {
      this.maxBullets = maxBullets;
      this.bulletCount = 0;
  
      // Structure of Arrays (SoA)
      this.x = new Float32Array(maxBullets);
      this.y = new Float32Array(maxBullets);
      this.vx = new Float32Array(maxBullets);
      this.vy = new Float32Array(maxBullets);
      this.life = new Float32Array(maxBullets);
      this.width = new Float32Array(maxBullets);
      this.height = new Float32Array(maxBullets);
    }
  
    /**
     * Adds a new bullet into the manager.
     * @param {number} x - Initial X position
     * @param {number} y - Initial Y position
     * @param {number} vx - Velocity in X
     * @param {number} vy - Velocity in Y
     * @param {number} life - Lifetime in seconds
     * @param {number} [width=5] - Bullet width
     * @param {number} [height=5] - Bullet height
     */
    addBullet(x, y, vx, vy, life, width = 5, height = 5) {
      if (this.bulletCount >= this.maxBullets) {
        console.warn('BulletManager: Max bullet capacity reached.');
        return;
      }
  
      const index = this.bulletCount++;
      this.x[index] = x;
      this.y[index] = y;
      this.vx[index] = vx;
      this.vy[index] = vy;
      this.life[index] = life;
      this.width[index] = width;
      this.height[index] = height;
    }
  
    /**
     * Updates all active bullets, removing any that have expired.
     * @param {number} deltaTime - Time elapsed since last update, in seconds
     */
    update(deltaTime) {
      let count = this.bulletCount;
      for (let i = 0; i < count; i++) {
        // Update position
        this.x[i] += this.vx[i] * deltaTime;
        this.y[i] += this.vy[i] * deltaTime;
        // Decrement life
        this.life[i] -= deltaTime;
  
        // Check if bullet expired
        if (this.life[i] <= 0) {
          this.swapRemove(i);
          count--;
          i--;
        }
      }
      this.bulletCount = count;
    }
  
    /**
     * Removes a bullet at the given index by swapping it with the last active bullet, then decrementing the count.
     * @param {number} index - The index of the bullet to remove
     */
    swapRemove(index) {
      const last = this.bulletCount - 1;
      if (index !== last) {
        this.x[index] = this.x[last];
        this.y[index] = this.y[last];
        this.vx[index] = this.vx[last];
        this.vy[index] = this.vy[last];
        this.life[index] = this.life[last];
        this.width[index] = this.width[last];
        this.height[index] = this.height[last];
      }
    }
  
    /**
     * Returns the current active number of bullets.
     * @returns {number}
     */
    getActiveBulletCount() {
      return this.bulletCount;
    }
  
    /**
     * Cleanup or reset if needed (placeholder for future).
     */
    cleanup() {
      this.bulletCount = 0;
    }
  }
  