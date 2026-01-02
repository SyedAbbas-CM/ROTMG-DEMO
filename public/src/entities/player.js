/**
 * Player.js
 * Represents the local player character
 */
import { generateUUID } from '../utils/uuid.js';
import { EntityAnimator } from './EntityAnimator.js';
import { spriteManager } from '../assets/spriteManager.js';
import { SCALE, TILE_SIZE, TILE_PROPERTIES } from '../constants/constants.js';
import { createLogger, LOG_LEVELS } from '../utils/logger.js';

// Create a logger for this module
const logger = createLogger('player');

// CRITICAL DEBUG: This log proves player.js is being loaded
console.log('====== player.js FILE LOADED! TIMESTAMP:', new Date().toISOString(), '======');
console.log('[player.js] Module loaded, Player class will be exported');

// ROTMG-like speed calculation constants
const BASE_SPEED = 3; // Base speed stat (extremely slow)
const SPEED_MULTIPLIER = 2.0; // Further reduced multiplier
const MIN_SPEED = 2; // Minimum speed
const MAX_SPEED = 10; // Maximum speed

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
      logger.info(`Created new player with initial ID: ${this.id}`);
      
      // Core properties
      this.name = options.name || 'Player';
      this.x = options.x || 0;
      this.y = options.y || 0;
      // Physical hit-box (tile units)
      this.collisionWidth  = options.collisionWidth  ?? 1;
      this.collisionHeight = options.collisionHeight ?? 1;

      // Increase default sprite scale for better visibility (2.5x larger than previous)
      this.renderScale = options.renderScale ?? 10;

      // Keep legacy width/height fields for code that still expects them, but
      // deprecate their use in favour of collisionWidth / renderScale.
      this.width  = this.collisionWidth;
      this.height = this.collisionHeight;
      this.rotation = 0;
      
      // Movement - ROTMG-style stats
      this.speedStat = options.speedStat || BASE_SPEED; // Base speed stat (0-75 in ROTMG)
      this.speed = this.calculateSpeed(this.speedStat); // Calculated actual speed
      logger.info(`Initial speed set to ${this.speed} pixels/sec (from stat: ${this.speedStat})`);
      
      this.vx = 0;
      this.vy = 0;
      this.isMoving = false;
      this.moveDirection = { x: 0, y: 0 };
      
      // Combat (defaults match class HP, usually 200 - server sends actual value)
      this.health = options.health !== undefined ? options.health : 200;
      this.maxHealth = options.maxHealth || 200;
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
     * Calculate actual movement speed from speed stat
     * @param {number} speedStat - The speed stat value (ROTMG-like)
     * @returns {number} - The actual movement speed in pixels per second
     */
    calculateSpeed(speedStat) {
      // Clamp speed stat between MIN_SPEED and MAX_SPEED
      const clampedStat = Math.max(MIN_SPEED, Math.min(MAX_SPEED, speedStat));
      
      // ROTMG-like formula: Convert stat to actual speed
      const calculatedSpeed = clampedStat * SPEED_MULTIPLIER;
      logger.debug(`Speed stat: ${speedStat}, Clamped: ${clampedStat}, Final Speed: ${calculatedSpeed}`);
      return calculatedSpeed;
    }
    
    /**
     * Set the player's speed stat
     * @param {number} newSpeedStat - New speed stat value
     */
    setSpeedStat(newSpeedStat) {
      this.speedStat = newSpeedStat;
      this.speed = this.calculateSpeed(this.speedStat);
      logger.info(`Speed stat set to ${this.speedStat}, actual speed: ${this.speed.toFixed(1)} px/s`);
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
      
      // CRITICAL FIX: Double-check if we're really moving based on moveDirection
      const actuallyMoving = Math.abs(this.moveDirection.x) > 0.01 || Math.abs(this.moveDirection.y) > 0.01;
      
      // Handle any discrepancy between isMoving flag and actual movement
      if (this.isMoving !== actuallyMoving) {
        // Force consistency between the flag and actual state
        this.isMoving = actuallyMoving;
        
        // If we're truly stopped but animation says otherwise, force a reset
        if (!actuallyMoving && this.animator && 
            this.animator.currentState === this.animator.states.WALK) {
          // Double-check safety: directly force back to idle state
          this.animator.resetToIdle();
        }
      }
      
      // Make absolutely sure animation state exactly matches movement state
      if (!this.isMoving && this.animator && 
          this.animator.currentState === this.animator.states.WALK) {
        // Force back to idle state if somehow we still have a walk animation
        this.animator.resetToIdle();
      }
      
      // Update animator with all protection layers applied
      this.animator.update(
        deltaTime, 
        this.isMoving,  // Corrected movement state
        this.moveDirection
      );
    }
    
    /**
     * Move player
     * @param {number} dx - X direction (-1 to 1)
     * @param {number} dy - Y direction (-1 to 1)
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    move(dx, dy, deltaTime) {
      // CRITICAL FIX: Detect if we're trying to stop
      const tryingToStop = Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001;
      
      // Normalize direction if moving diagonally
      if (!tryingToStop && dx !== 0 && dy !== 0) {
        const length = Math.sqrt(dx * dx + dy * dy);
        dx /= length;
        dy /= length;
      }
      
      // CRITICAL FIX: More aggressive handling of the stopped state
      if (tryingToStop) {
        // If we're stopping, explicitly reset everything
        this.isMoving = false;
        this.moveDirection.x = 0;
        this.moveDirection.y = 0;
        
        // Explicitly reset to idle animation if we were walking
        if (this.animator && this.animator.currentState === this.animator.states.WALK) {
          this.animator.resetToIdle();
        }
      } else {
        // Set movement state for normal movement
        this.isMoving = true;
        
        // Save move direction
        this.moveDirection.x = dx;
        this.moveDirection.y = dy;
        
        // Update animator direction based on movement
        if (this.animator) {
          // Determine dominant direction
          if (Math.abs(dx) > Math.abs(dy)) {
            // Horizontal movement is dominant
            this.animator.direction = dx < 0 ? 1 : 3; // 1 = left, 3 = right
          } else {
            // Vertical movement is dominant
            this.animator.direction = dy < 0 ? 2 : 0; // 2 = up, 0 = down
          }
        }
        
        // Apply movement with tile-based speed modifier
        let distance = this.speed * deltaTime;

        // Check tile at player position and apply movement cost (water slow, etc.)
        if (window.gameState?.map) {
          const tileX = Math.floor(this.x);
          const tileY = Math.floor(this.y);
          const tile = window.gameState.map.getTile(tileX, tileY);

          if (tile && TILE_PROPERTIES[tile.type]) {
            const movementCost = TILE_PROPERTIES[tile.type].movementCost || 1.0;
            // Higher cost = slower movement (divide speed by cost)
            distance = distance / movementCost;

            // Log when moving through non-normal tiles
            if (movementCost !== 1.0) {
              logger.occasional(0.05, LOG_LEVELS.DEBUG,
                `Moving through tile type ${tile.type} with cost ${movementCost.toFixed(1)}x (${movementCost === 2.0 ? 'WATER' : movementCost === 1.5 ? 'SAND' : 'OTHER'})`);
            }
          }
        }

        // Debug log movement occasionally
        logger.occasional(0.01, LOG_LEVELS.DEBUG,
          `Speed: ${this.speed.toFixed(2)}, DeltaTime: ${deltaTime.toFixed(4)}, Distance this frame: ${distance.toFixed(4)}`);
        logger.occasional(0.01, LOG_LEVELS.VERBOSE,
          `Actual move: dx=${dx.toFixed(2)}, dy=${dy.toFixed(2)}, resulting in +${(dx * distance).toFixed(4)}, +${(dy * distance).toFixed(4)}`);

        this.x += dx * distance;
        this.y += dy * distance;
      }
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
     * @param {boolean} skipAnimation - Whether to skip triggering the attack animation
     */
    setLastShotTime(time, skipAnimation = false) {
      logger.debug(`Setting cooldown: ${this.shootCooldown.toFixed(2)}, skipAnimation: ${skipAnimation}`);
      this.lastShotTime = this.shootCooldown;
      
      // Only trigger attack animation if not skipped
      // This allows handleShoot to control the attack animation with the correct direction
      if (!skipAnimation && this.animator && typeof this.animator.attack === 'function') {
        logger.debug(`Triggering attack animation`);
        this.animator.attack();
      }
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
      console.log('[DEBUG player.draw()] CALLED! ctx=', !!ctx, 'cameraPosition=', cameraPosition);
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
        logger.verbose(`Using direct scale factor: ${viewScaleFactor}`);
      } else {
        // Fallback to checking the view type
        const isStrategicView = window.gameState?.camera?.viewType === 'strategic';
        viewScaleFactor = isStrategicView ? 0.5 : 1.0; // 50% smaller in strategic view
        logger.debug(`Using fallback scale factor: ${viewScaleFactor}, view: ${window.gameState?.camera?.viewType}`);
      }
      
      // Define scale based on view type
      const scaleFactor = viewScaleFactor;
      
      // Calculate screen position
      const screenX = (this.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
      const screenY = (this.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
      
      // Apply the appropriate scale factor for rendering
      const width = this.width * this.renderScale * SCALE * viewScaleFactor;
      const height = this.height * this.renderScale * SCALE * viewScaleFactor;
      
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

      // Check if player is on water for submersion effect
      const tileX = Math.floor(this.x);
      const tileY = Math.floor(this.y);
      const tile = window.gameState?.map?.getTile(tileX, tileY) || window.gameState?.mapManager?.getTile(tileX, tileY);

      // DEBUG: Log submersion check ALWAYS initially
      if (!window._submersionLogged) {
        window._submersionLogged = true;
        console.log('[PLAYER SUBMERSION] INITIAL CHECK:', {
          pos: `(${tileX}, ${tileY})`,
          hasTile: !!tile,
          spriteName: tile?.spriteName,
          tileType: tile?.type,
          hasMap: !!window.gameState?.map,
          hasMapManager: !!window.gameState?.mapManager,
          tile: tile
        });
      }

      // Check if water tile by sprite name or type ID
      let isOnWater = false;
      if (tile && tile.spriteName) {
        const spriteName = tile.spriteName.toLowerCase();
        isOnWater = spriteName.includes('water') || spriteName.includes('deep') || spriteName.includes('lava');
        if (isOnWater) {
          console.log('[PLAYER SUBMERSION] ✓ Detected water/lava tile:', spriteName);
        }
      } else if (tile && (tile.type === 3 || tile.type === 6)) { // TILE_IDS.WATER or LAVA
        isOnWater = true;
        console.log('[PLAYER SUBMERSION] ✓ Detected water/lava by type:', tile.type);
      }

      // Log if we're applying submersion
      if (isOnWater && !window._submersionDetected) {
        window._submersionDetected = true;
        console.log('[PLAYER SUBMERSION] 🌊 APPLYING SUBMERSION EFFECT!');
      }

      // Water submersion: Show only top 50% of sprite when on water
      const submersionRatio = isOnWater ? 0.5 : 1.0;
      const spriteSourceHeight = sourceRect.height * submersionRatio;
      const renderHeight = height * submersionRatio;
      const submersionYOffset = isOnWater ? (height * 0.25) : 0;

      // Draw black outline for visibility (8-directional shadow)
      const outlineSize = 1.5;
      ctx.shadowColor = 'black';
      ctx.shadowBlur = 0;
      for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 4) {
        ctx.shadowOffsetX = Math.cos(angle) * outlineSize;
        ctx.shadowOffsetY = Math.sin(angle) * outlineSize;
        ctx.drawImage(
          spriteSheet,
          sourceRect.x, sourceRect.y,
          sourceRect.width, spriteSourceHeight,
          -width/2, -renderHeight/2 + submersionYOffset,
          width, renderHeight
        );
      }

      // Reset shadow and draw main sprite
      ctx.shadowColor = 'transparent';
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      ctx.drawImage(
        spriteSheet,
        sourceRect.x, sourceRect.y,
        sourceRect.width, spriteSourceHeight,
        -width/2, -renderHeight/2 + submersionYOffset,
        width, renderHeight
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
        // Fixed width health bar (not based on sprite size)
        const barWidth = 40; // Fixed 40 pixels wide
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
      const width = this.width * this.renderScale * SCALE * viewScaleFactor;
      const height = this.height * this.renderScale * SCALE * viewScaleFactor;
      
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