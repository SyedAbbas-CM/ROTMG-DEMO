// File: /src/managers/EnemyManager.js

export default class EnemyManager {
  constructor(maxEnemies = 1000) {
    this.maxEnemies = maxEnemies;
    this.enemyCount = 0;

    // SoA for enemy positioning and size
    this.x = new Float32Array(maxEnemies);
    this.y = new Float32Array(maxEnemies);
    this.width = new Float32Array(maxEnemies);
    this.height = new Float32Array(maxEnemies);

    // Simple cooldown for shooting
    this.shootCooldown = new Float32Array(maxEnemies);

    // Optional state machine or behavior tree ID
    this.aiState = new Uint8Array(maxEnemies);

    // Potential partial updates or LOD
    this.updateInterval = new Uint8Array(maxEnemies);
    this.frameCounter = 0;
  }

  addEnemy(x, y, width = 20, height = 20, initialCooldown = 1.0, initialState = 0) {
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
    this.aiState[index] = initialState; // e.g. 0 = idle
    this.updateInterval[index] = 1;     // update every frame by default
  }

  update(deltaTime, bulletManager) {
    this.frameCounter++;

    for (let i = 0; i < this.enemyCount; i++) {
      // If partial updates or LOD needed:
      // if (this.updateInterval[i] > 1 && (this.frameCounter % this.updateInterval[i] !== 0)) {
      //   continue; // skip updating this enemy this frame
      // }

      // Decrement cooldown
      if (this.shootCooldown[i] > 0) {
        this.shootCooldown[i] -= deltaTime;
      } else {
        // Enemy shoots a bullet traveling to the right
        const bulletX = this.x[i] + this.width[i];
        const bulletY = this.y[i] + this.height[i] * 0.5;
        bulletManager.addBullet(bulletX, bulletY, 200, 0, 3.0, 5, 5);
        this.shootCooldown[i] = 1.0;
      }

      // Optional: A tiny, naive "AI state" approach
      switch (this.aiState[i]) {
        case 0: // idle
          // do nothing
          break;
        case 1: // patrolling, for example
          // this.x[i] += someVelocity * deltaTime
          break;
        // etc.
      }
    }
  }

  /**
   * Returns the current active number of enemies.
   */
  getActiveEnemyCount() {
    return this.enemyCount;
  }

  /**
   * Example for removing an enemy via swap-and-pop
   */
  removeEnemy(index) {
    const last = this.enemyCount - 1;
    if (index !== last) {
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
      this.shootCooldown[index] = this.shootCooldown[last];
      this.aiState[index] = this.aiState[last];
      this.updateInterval[index] = this.updateInterval[last];
    }
    this.enemyCount--;
  }

  cleanup() {
    this.enemyCount = 0;
  }
}
