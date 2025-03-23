/**
 * Player.js
 * Represents the local player character
 */
export class Player {
    /**
     * Create a new player
     * @param {Object} options - Player options
     */
    constructor(options = {}) {
      // Core properties
      this.id = options.id || null;
      this.name = options.name || 'Player';
      this.x = options.x || 50;
      this.y = options.y || 50;
      this.width = options.width || 20;
      this.height = options.height || 20;
      this.rotation = options.rotation || 0; // Rotation in radians
      
      // Movement
      this.speed = options.speed || 150; // Pixels per second
      this.isMoving = false;
      this.moveDirection = { x: 0, y: 0 };
      
      // Combat
      this.health = options.health || 100;
      this.maxHealth = options.maxHealth || 100;
      this.damage = options.damage || 10;
      this.projectileSpeed = options.projectileSpeed || 300;
      this.shootCooldown = options.shootCooldown || 0.3; // Seconds
      this.lastShootTime = 0;
      
      // Visual
      this.sprite = options.sprite || null;
      this.spriteX = options.spriteX || 0;
      this.spriteY = options.spriteY || 0;
      
      // State
      this.isDead = false;
    }
    
    /**
     * Update player
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    update(deltaTime) {
      // Update cooldowns
      if (this.lastShootTime > 0) {
        this.lastShootTime -= deltaTime;
        if (this.lastShootTime < 0) this.lastShootTime = 0;
      }
    }
    
    /**
     * Move player
     * @param {number} dx - X direction (-1 to 1)
     * @param {number} dy - Y direction (-1 to 1)
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    move(dx, dy, deltaTime) {
      // Normalize direction if moving diagonally
      if (dx !== 0 && dy !== 0) {
        const length = Math.sqrt(dx * dx + dy * dy);
        dx /= length;
        dy /= length;
      }
      
      // Save move direction
      this.moveDirection.x = dx;
      this.moveDirection.y = dy;
      this.isMoving = dx !== 0 || dy !== 0;
      
      // Apply movement
      const distance = this.speed * deltaTime;
      this.x += dx * distance;
      this.y += dy * distance;
    }
    
    /**
     * Rotate player to face a position
     * @param {number} targetX - Target X position
     * @param {number} targetY - Target Y position
     */
    rotateTo(targetX, targetY) {
      const dx = targetX - this.x;
      const dy = targetY - this.y;
      this.rotation = Math.atan2(dy, dx);
    }
    
    /**
     * Check if player can shoot
     * @returns {boolean} True if player can shoot
     */
    canShoot() {
      return this.lastShootTime <= 0;
    }
    
    /**
     * Start shoot cooldown
     */
    startShootCooldown() {
      this.lastShootTime = this.shootCooldown;
    }
    
    /**
     * Apply damage to player
     * @param {number} amount - Amount of damage
     * @returns {number} Remaining health
     */
    takeDamage(amount) {
      this.health -= amount;
      
      if (this.health <= 0) {
        this.health = 0;
        this.isDead = true;
      }
      
      return this.health;
    }
    
    /**
     * Heal player
     * @param {number} amount - Amount to heal
     * @returns {number} New health
     */
    heal(amount) {
      this.health += amount;
      
      if (this.health > this.maxHealth) {
        this.health = this.maxHealth;
      }
      
      if (this.isDead && this.health > 0) {
        this.isDead = false;
      }
      
      return this.health;
    }
    
    /**
     * Convert to network-friendly format
     * @returns {Object} Serialized player data
     */
    serialize() {
      return {
        id: this.id,
        name: this.name,
        x: this.x,
        y: this.y,
        rotation: this.rotation,
        health: this.health,
        maxHealth: this.maxHealth,
        spriteX: this.spriteX,
        spriteY: this.spriteY
      };
    }
  }