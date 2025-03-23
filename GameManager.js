// File: /src/GameManager.js

import BulletManager from './Managers/BulletManager.js';
import EnemyManager from './Managers/EnemyManager.js';
import CollisionManager from './Managers/serverCollisionManager.js';

/**
 * If you're using Node + WebSocket only, you might not use requestAnimationFrame,
 * but let's keep the structure for demonstration. 
 * On the server, you'd just call gameManager.update(dt) inside setInterval or so.
 */
export default class GameManager {
  constructor() {
    this.bulletManager = new BulletManager(10000);
    this.enemyManager = new EnemyManager(1000);
    this.collisionManager = new CollisionManager(
      this.bulletManager,
      this.enemyManager
    );

    this.lastTime = performance.now();
    this.isRunning = false;

    this.initializeTestData();
  }

  initializeTestData() {
    this.enemyManager.addEnemy(100, 100);
    this.enemyManager.addEnemy(300, 150);
  }

  /**
   * The main update routine
   */
  update(deltaTime) {
    // 1) Enemies update first
    this.enemyManager.update(deltaTime, this.bulletManager);
    // 2) Then bullets
    this.bulletManager.update(deltaTime);
    // 3) Then collisions
    this.collisionManager.checkCollisions();
  }

  start() {
    this.isRunning = true;
    this.lastTime = performance.now();

    const loop = () => {
      if (!this.isRunning) return;

      const now = performance.now();
      const dt = (now - this.lastTime) / 1000; 
      this.lastTime = now;

      this.update(dt);
      requestAnimationFrame(loop);
    };

    loop();
  }

  stop() {
    this.isRunning = false;
  }

  cleanup() {
    this.bulletManager.cleanup();
    this.enemyManager.cleanup();
  }

  /**
   * For passing data to websockets or debugging
   */
  getBulletData() {
    return this.bulletManager.getDataArray();
  }
  getEnemyData() {
    return this.enemyManager.getDataArray();
  }
}
