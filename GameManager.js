// File: /src/GameManager.js

import BulletManager from './Managers/bulletManager.js';
import EnemyManager from './Managers/enemyManager.js';
import CollisionManager from './managers/CollisionManager.js';

export default class GameManager {
  constructor() {
    // Instantiate managers
    this.bulletManager = new BulletManager(10000);
    this.enemyManager = new EnemyManager(1000);
    this.collisionManager = new CollisionManager(
      this.bulletManager,
      this.enemyManager
    );

    // Timing
    this.lastTime = performance.now();
    this.isRunning = false;

    // For convenience, we can automatically add some test enemies:
    this.initializeTestData();
  }

  initializeTestData() {
    // Add a couple of enemies for demonstration
    this.enemyManager.addEnemy(100, 100);
    this.enemyManager.addEnemy(300, 150);
  }

  /**
   * Main update loop, orchestrating bullet/enemy updates and collision checks.
   * @param {number} deltaTime - time in seconds since last frame
   */
  update(deltaTime) {
    // 1. Update enemies (which can spawn bullets)
    this.enemyManager.update(deltaTime, this.bulletManager);

    // 2. Update all bullets
    this.bulletManager.update(deltaTime);

    // 3. Check collisions
    this.collisionManager.checkCollisions();
  }

  /**
   * Starts the main game loop using requestAnimationFrame.
   */
  start() {
    this.isRunning = true;
    this.lastTime = performance.now();

    const loop = () => {
      if (!this.isRunning) return;

      const now = performance.now();
      const deltaTime = (now - this.lastTime) / 1000; // convert ms to seconds
      this.lastTime = now;

      this.update(deltaTime);

      requestAnimationFrame(loop);
    };

    loop();
  }

  /**
   * Stops the main game loop.
   */
  stop() {
    this.isRunning = false;
  }

  /**
   * Clean up or reset if needed (placeholder).
   */
  cleanup() {
    this.bulletManager.cleanup();
    this.enemyManager.cleanup();
    // No direct cleanup for collisions for now
  }
}
