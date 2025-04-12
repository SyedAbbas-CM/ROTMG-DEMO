/**
 * Player.js
 * Represents the local player character
 */
import { generateUUID } from '../utils/uuid.js';
import { EntityAnimator } from './EntityAnimator.js';
import { spriteManager } from '../assets/spriteManager.js';
import { SCALE, TILE_SIZE } from '../constants/constants.js';

export class Player {
    /**
     * Create a new player
     * @param {Object} options - Player options
     */
    constructor(options = {}) {
      // Generate a unique ID for this player
      // NOTE: This will be overridden by the server-assigned ID when connected
      this.id = options.id || generateUUID();
      
      // Log the generated ID to help with debugging
      console.log(`[Player] Created new player with initial ID: ${this.id}`);
      
      // Core properties
      this.name = options.name || 'Player';
      this.x = options.x || 0;
      this.y = options.y || 0;
      this.width = options.width || 10;
      this.height = options.height || 10;
      this.rotation = 0;
      
      // Movement
      this.speed = options.speed || 100; // Pixels per second
      this.vx = 0;
      this.vy = 0;
      this.isMoving = false;
      this.moveDirection = { x: 0, y: 0 };
      
      // Combat
      this.health = options.health !== undefined ? options.health : 100;
      this.maxHealth = options.maxHealth || 100;
      this.damage = options.damage || 10;
      this.projectileSpeed = options.projectileSpeed || 200;
      this.shootCooldown = options.shootCooldown || 0.5; // Seconds
      this.lastShotTime = 0;
      
      // Visual
      this.spriteX = options.spriteX || 0;
      this.spriteY = options.spriteY || 0;
      
      // State
      this.isDead = false;
      
      // Network properties
      this.isLocal = options.isLocal !== undefined ? options.isLocal : true;
      this.lastUpdate = Date.now();
      
      // Initialize the animator
      this.animator = new EntityAnimator({
        defaultState: 'idle',
        frameCount: 4,
        frameDuration: 0.15,
        attackDuration: 0.3,
        attackCooldownTime: this.shootCooldown,
        spriteWidth: TILE_SIZE,
        spriteHeight: TILE_SIZE,
        spriteSheet: 'character_sprites'
      });
    }
    
    /**
     * Update player
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    update(deltaTime) {
      // Update cooldowns
      if (this.lastShotTime > 0) {
        this.lastShotTime -= deltaTime;
        if (this.lastShotTime < 0) this.lastShotTime = 0;
      }
      
      // Make sure movement properties are set correctly for animation
      const isMoving = this.isMoving || (Math.abs(this.moveDirection.x) > 0.01 || Math.abs(this.moveDirection.y) > 0.01);
      
      // Log animation state for debugging
      if (isMoving) {
        console.log(`[Player.update] Moving: ${isMoving}, Direction: ${this.animator.direction}, MoveDir: ${JSON.stringify(this.moveDirection)}`);
      }
      
      // Update animator with clear movement state
      this.animator.update(
        deltaTime, 
        isMoving, 
        this.moveDirection // Pass move direction vector directly 
      );
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
      
      // Set movement state
      this.isMoving = (dx !== 0 || dy !== 0);
      
      // Save move direction
      this.moveDirection.x = dx;
      this.moveDirection.y = dy;
      
      // Update animator direction based on movement
      if (this.isMoving && this.animator) {
        // Determine dominant direction
        if (Math.abs(dx) > Math.abs(dy)) {
          // Horizontal movement is dominant
          this.animator.direction = dx < 0 ? 1 : 3; // 1 = left, 3 = right
        } else {
          // Vertical movement is dominant
          this.animator.direction = dy < 0 ? 2 : 0; // 2 = up, 0 = down
        }
      }
      
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
      
      // Update animator direction based on rotation
      if (Math.abs(dx) > Math.abs(dy)) {
        this.animator.direction = dx < 0 ? 1 : 3; // Left or right
      } else {
        this.animator.direction = dy < 0 ? 2 : 0; // Up or down
      }
    }
    
    /**
     * Check if player can shoot
     * @returns {boolean} True if player can shoot
     */
    canShoot() {
      return this.lastShotTime <= 0;
    }
    
    /**
     * Start shoot cooldown
     */
    startShootCooldown() {
      this.lastShotTime = this.shootCooldown;
    }
    
    /**
     * Set the last shot time directly
     * @param {number} time - The timestamp of the last shot
     */
    setLastShotTime(time) {
      this.lastShotTime = this.shootCooldown;
      
      // Also trigger attack animation
      this.animator.attack();
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
        
        // Set death animation
        this.animator.setAnimationState(this.animator.states.DEATH, this.animator.direction);
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
        // Reset animation to idle
        this.animator.setAnimationState(this.animator.states.IDLE, this.animator.direction);
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
    
    /**
     * Draw the player on the canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
     * @param {Object} cameraPosition - Camera position {x, y}
     */
    draw(ctx, cameraPosition) {
      if (!ctx) return;
      
      // Get sprite sheet
      const spriteSheetObj = spriteManager.getSpriteSheet(this.animator.spriteSheet);
      if (!spriteSheetObj) {
        this.drawDebugRect(ctx, cameraPosition);
        return;
      }
      
      const spriteSheet = spriteSheetObj.image;
      
      // Get screen dimensions
      const screenWidth = ctx.canvas.width;
      const screenHeight = ctx.canvas.height;
      
      // Check for direct view scaling flags set by renderCharacter
      let viewScaleFactor;
      if (this._viewScaleFactor !== undefined) {
        // Use the flag directly set on the character object
        viewScaleFactor = this._viewScaleFactor;
        console.log(`[Player.draw] Using direct scale factor: ${viewScaleFactor}`);
      } else {
        // Fallback to checking the view type
        const isStrategicView = window.gameState?.camera?.viewType === 'strategic';
        viewScaleFactor = isStrategicView ? 0.5 : 1.0; // 50% smaller in strategic view
        console.log(`[Player.draw] Using fallback scale factor: ${viewScaleFactor}, view: ${window.gameState?.camera?.viewType}`);
      }
      
      // Define scale based on view type
      const scaleFactor = viewScaleFactor;
      
      // Calculate screen position
      const screenX = (this.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
      const screenY = (this.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
      
      // Apply the appropriate scale factor for rendering
      const width = this.width * SCALE * viewScaleFactor;
      const height = this.height * SCALE * viewScaleFactor;
      
      // Save context for rotation
      ctx.save();
      
      // Translate to player position
      ctx.translate(screenX, screenY);
      
      // If player has a rotation, use it
      if (typeof this.rotation === 'number') {
        ctx.rotate(this.rotation);
      }
      
      // Get animation source rect
      const sourceRect = this.animator.getSourceRect();
      
      // Draw player sprite
      ctx.drawImage(
        spriteSheet,
        sourceRect.x, sourceRect.y,
        sourceRect.width, sourceRect.height,
        -width/2, -height/2,
        width, height
      );
      
      // Draw player name
      if (this.name) {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.textAlign = 'center';
        ctx.font = '12px Arial';
        ctx.strokeText(this.name, 0, -height/2 - 5);
        ctx.fillText(this.name, 0, -height/2 - 5);
      }
      
      // Draw health bar if health is defined
      if (this.health !== undefined && this.maxHealth !== undefined) {
        const healthPercent = this.health / this.maxHealth;
        const barWidth = width;
        const barHeight = 3;
        const barY = height/2 + 5;
        
        // Background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(-barWidth/2, barY, barWidth, barHeight);
        
        // Health
        ctx.fillStyle = healthPercent > 0.6 ? 'green' : healthPercent > 0.3 ? 'yellow' : 'red';
        ctx.fillRect(-barWidth/2, barY, barWidth * healthPercent, barHeight);
      }
      
      // Restore context
      ctx.restore();
    }
    
    /**
     * Draw a simple rectangle for debugging
     * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
     * @param {Object} cameraPosition - Camera position {x, y}
     */
    drawDebugRect(ctx, cameraPosition) {
      // Get screen dimensions
      const screenWidth = ctx.canvas.width;
      const screenHeight = ctx.canvas.height;
      
      // Determine view scaling factor based on view type
      const isStrategicView = window.gameState?.camera?.viewType === 'strategic';
      const viewScaleFactor = isStrategicView ? 0.5 : 1.0; // 50% smaller in strategic view
      
      // Define scale based on view type
      const scaleFactor = viewScaleFactor;
      
      // Calculate screen position
      const screenX = (this.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
      const screenY = (this.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
      
      // Apply the appropriate scale factor for rendering
      const width = this.width * SCALE * viewScaleFactor;
      const height = this.height * SCALE * viewScaleFactor;
      
      // Draw debug rectangle
      ctx.fillStyle = 'red';
      ctx.fillRect(
        screenX - width/2,
        screenY - height/2,
        width, height
      );
      
      // Draw player name
      if (this.name) {
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.textAlign = 'center';
        ctx.font = '12px Arial';
        ctx.strokeText(this.name, screenX, screenY - height/2 - 5);
        ctx.fillText(this.name, screenX, screenY - height/2 - 5);
      }
    }
}