// File: /src/managers/CollisionManager.js

export default class CollisionManager {
    constructor(bulletManager, enemyManager) {
      this.bulletManager = bulletManager;
      this.enemyManager = enemyManager;
    }
  
    /**
     * Performs a naive O(N*M) AABB collision check between bullets and enemies.
     * Expires bullets that collide with enemies. Future expansions: damage enemies, etc.
     */
    checkCollisions() {
      const bCount = this.bulletManager.getActiveBulletCount();
      const eCount = this.enemyManager.getActiveEnemyCount();
  
      const bx = this.bulletManager.x;
      const by = this.bulletManager.y;
      const bw = this.bulletManager.width;
      const bh = this.bulletManager.height;
      const blife = this.bulletManager.life; // used to expire bullet
  
      const ex = this.enemyManager.x;
      const ey = this.enemyManager.y;
      const ew = this.enemyManager.width;
      const eh = this.enemyManager.height;
  
      for (let i = 0; i < bCount; i++) {
        for (let j = 0; j < eCount; j++) {
          if (
            bx[i] < ex[j] + ew[j] &&
            bx[i] + bw[i] > ex[j] &&
            by[i] < ey[j] + eh[j] &&
            by[i] + bh[i] > ey[j]
          ) {
            console.log(`Collision: Bullet ${i} hit Enemy ${j}`);
            blife[i] = 0; // Mark bullet for removal
            // Potentially apply damage or remove enemy here in a future step
          }
        }
      }
    }
  }
  