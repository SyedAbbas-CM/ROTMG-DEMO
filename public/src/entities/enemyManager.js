// src/entities/enemyManager.js

import { Enemy } from './enemy.js';
import { assets } from '../assets/assets.js';
import { gameState } from '../game/gamestate.js';
import { TILE_SIZE, TILE_IDS } from '../constants/constants.js';

export class EnemyManager {
  constructor(scene) {
    this.scene = scene;
    this.enemies = [];
    this.enemyTexture = new THREE.Texture(assets.enemySpriteSheet);
    this.enemyTexture.needsUpdate = true;
  }

  // Method to spawn a new enemy at a given position
  spawnEnemy(x, y) {
    // Ensure enemies spawn in valid locations (not inside walls)
    if (this.isValidSpawn(x, y)) {
      const enemy = new Enemy(x, y);
      enemy.createSprite(this.scene, this.enemyTexture);
      this.enemies.push(enemy);
      gameState.enemies.push(enemy);
    }
  }

  // Method to remove an enemy
  removeEnemy(enemy) {
    enemy.removeSprite(this.scene);
    this.enemies = this.enemies.filter(e => e !== enemy);
    gameState.enemies = gameState.enemies.filter(e => e !== enemy);
  }

  // Method to update all enemies
  update(delta) {
    this.enemies.forEach(enemy => {
      enemy.update(delta, gameState.character);
      // Additional logic (e.g., checking health) can be added here
      if (enemy.health <= 0) {
        this.removeEnemy(enemy);
      }
    });

    // Example: Spawn new enemies periodically or based on certain conditions
    // Implement spawning logic as needed
  }

  // Method to initialize enemies (e.g., spawn initial enemies)
  initializeEnemies(initialCount) {
    for (let i = 0; i < initialCount; i++) {
      const x = Math.random() * (25 * TILE_SIZE);
      const y = Math.random() * (25 * TILE_SIZE);
      this.spawnEnemy(x, y);
    }
  }

  // Check if the spawn location is valid (e.g., not inside a wall)
  isValidSpawn(x, y) {
    const tileX = Math.floor(x / TILE_SIZE);
    const tileY = Math.floor(y / TILE_SIZE);
    const tile = gameState.map.getTile(tileX, tileY); // Use gameState.map.getTile
    return tile && tile.type !== TILE_IDS.WALL;
  }
}
