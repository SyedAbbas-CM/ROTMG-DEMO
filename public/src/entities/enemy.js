// src/entities/enemy.js

import { UNIT_SIZE, ENEMY_SPRITE_POSITIONS } from '../constants/constants.js';

export class Enemy {
  constructor(x, y) {
    this.x = x; // X position in world coordinates
    this.y = y; // Y position in world coordinates
    this.z = 0; // Height for 3D rendering
    this.speed = 1.5; // Units per second
    this.health = 50; // Enemy health
    this.width = UNIT_SIZE;
    this.height = UNIT_SIZE;
    this.spriteX = ENEMY_SPRITE_POSITIONS.DEFAULT.x; // X position on the sprite sheet
    this.spriteY = ENEMY_SPRITE_POSITIONS.DEFAULT.y; // Y position on the sprite sheet
    this.rotation = { yaw: 0 }; // Rotation for directional movement
    this.sprite = null; // Linked Three.js sprite (for 3D rendering)
  }

  // Method to update enemy behavior
  update(delta, player) {
    // Simple AI: Move towards the player
    const dx = player.x - this.x;
    const dy = player.y - this.y;
    const distance = Math.hypot(dx, dy);

    if (distance > 1) { // Avoid overlapping
      const moveX = (dx / distance) * this.speed * delta;
      const moveY = (dy / distance) * this.speed * delta;

      // Update position
      this.x += moveX;
      this.y += moveY;

      // Update rotation to face the player
      this.rotation.yaw = Math.atan2(-dy, dx); // Negative dy to account for coordinate mapping

      // Update Three.js sprite position if it exists
      if (this.sprite) {
        this.sprite.position.set(
          this.x * UNIT_SIZE,
          this.z,
          -this.y * UNIT_SIZE
        );
        this.sprite.rotation.y = this.rotation.yaw;
      }

      // Debug log
      console.log(`Enemy: Moved to (${this.x.toFixed(2)}, ${this.y.toFixed(2)}).`);
    }

    // Additional behaviors (e.g., attacking) can be implemented here
  }

  // Method to create Three.js sprite for the enemy
  createSprite(scene, enemyTexture) {
    const spriteMaterial = new THREE.SpriteMaterial({ map: enemyTexture, transparent: true });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.set(
      this.x * UNIT_SIZE,
      this.z,
      -this.y * UNIT_SIZE
    );
    sprite.scale.set(this.width, this.height, 1); // Adjust scale if needed
    sprite.castShadow = true;
    scene.add(sprite);
    this.sprite = sprite;

    // Debug log
    console.log(`Enemy: Created sprite at (${this.x}, ${this.y}, ${this.z}).`);
  }

  // Method to remove the enemy's sprite from the scene
  removeSprite(scene) {
    if (this.sprite) {
      scene.remove(this.sprite);
      this.sprite = null;
    }
  }
}
