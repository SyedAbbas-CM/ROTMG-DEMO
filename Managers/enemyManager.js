// File: /src/managers/EnemyManager.js

export default class EnemyManager {
  constructor(maxEnemies = 1000) {
    this.maxEnemies = maxEnemies;
    this.enemyCount = 0;

    // Structure of Arrays (SoA) for enemies
    this.x = new Float32Array(maxEnemies);
    this.y = new Float32Array(maxEnemies);
    this.width = new Float32Array(maxEnemies);
    this.height = new Float32Array(maxEnemies);
    this.shootCooldown = new Float32Array(maxEnemies);
    // In a future step, we can add 'state' arrays or references to behavior trees.
  }

  /**
   * Adds a new enemy into the manager.
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {number} [width=20] - Enemy width
   * @param {number} [height=20] - Enemy height
   * @param {number} [initialCooldown=1.0] - Initial shooting cooldown
   */
  addEnemy(x, y, width = 20, height = 20, initialCooldown = 1.0) {
    if (this.enemyCount >= this.maxEnemies) {
      console.warn('EnemyManager: Max enemy capacity reached.');
      return;
    }
    const index = this.enemyCount++;
    this.x[index] = x;
    this.y[index] = y;
    this.width[index] = width;
    this.height[index] = height;
    this.shootCooldown[index] = initialCooldown;
  }

  /**
   * Updates all enemies, decrementing shoot cooldowns and optionally spawning bullets.
   * @param {number} deltaTime - Time elapsed in seconds
   * @param {object} bulletManager - A reference to the BulletManager
   */
  update(deltaTime, bulletManager) {
    for (let i = 0; i < this.enemyCount; i++) {
      // Decrement cooldown
      if (this.shootCooldown[i] > 0) {
        this.shootCooldown[i] -= deltaTime;
      } else {
        // Shoot: spawn a bullet traveling to the right
        const bulletX = this.x[i] + this.width[i];
        const bulletY = this.y[i] + this.height[i] * 0.5;
        bulletManager.addBullet(bulletX, bulletY, 200, 0, 3.0, 5, 5);

        // Reset the cooldown
        this.shootCooldown[i] = 1.0;
      }
    }
  }

  /**
   * Returns the current active number of enemies.
   * @returns {number}
   */
  getActiveEnemyCount() {
    return this.enemyCount;
  }

  /**
   * Placeholder for future expansions: removing enemies, applying damage, etc.
   */
  removeEnemy(index) {
    // Similar swap-and-pop logic can go here if needed.
  }

  /**
   * Cleanup or reset if needed (placeholder for future).
   */
  cleanup() {
    this.enemyCount = 0;
  }
}
