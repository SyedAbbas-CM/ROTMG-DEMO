import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { gameState } from '../game/gamestate.js';
import { spriteManager } from '../assets/spriteManager.js';

/**
 * AnimatedPlayer class to manage player animations and state
 * This class handles player movement, collision, and animations
 */
export class AnimatedPlayer {
    constructor(options = {}) {
        // Core properties
        this.id = options.id || `player_${Date.now()}`;
        this.x = options.x || 0;
        this.y = options.y || 0;
        this.width = options.width || TILE_SIZE;
        this.height = options.height || TILE_SIZE;
        this.speed = options.speed || 5;
        this.rotation = options.rotation || 0;
        this.name = options.name || "Player";
        
        // Animation states
        this.STATES = {
            IDLE: 'idle',
            WALK: 'walk',
            ATTACK: 'attack'
        };
        
        // Movement and direction
        this.velocity = { x: 0, y: 0 };
        this.isMoving = false;
        this.direction = 0; // 0: down, 1: left, 2: up, 3: right
        
        // Animation properties
        this.currentState = this.STATES.IDLE;
        this.frameIndex = 0;
        this.frameCount = options.frameCount || 4;
        this.frameDuration = options.frameDuration || 0.15; // seconds per frame
        this.frameTimer = 0;
        
        // Attack properties
        this.isAttacking = false;
        this.attackCooldown = 0;
        this.attackDuration = options.attackDuration || 0.3;
        this.attackCooldownTime = options.attackCooldownTime || 0.5;
        
        // Collision properties
        this.collisionEnabled = options.collisionEnabled !== undefined ? options.collisionEnabled : true;
        this.collisionRadius = options.collisionRadius || Math.min(this.width, this.height) / 2;
        
        // Debug
        this.visualDebug = options.visualDebug || false;
    }
    
    /**
     * Update player position and animation state
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    update(deltaTime) {
        // Update position based on velocity
        if (this.isMoving && (this.velocity.x !== 0 || this.velocity.y !== 0)) {
            this.x += this.velocity.x * this.speed * deltaTime;
            this.y += this.velocity.y * this.speed * deltaTime;
        }
        
        // Update attack state
        if (this.isAttacking) {
            this.attackCooldown -= deltaTime;
            if (this.attackCooldown <= 0) {
                this.isAttacking = false;
            }
        }
        
        // Update animation state
        this.updateAnimationState();
        
        // Update animation frame
        this.frameTimer += deltaTime;
        if (this.frameTimer >= this.frameDuration) {
            this.frameTimer = 0;
            this.frameIndex = (this.frameIndex + 1) % this.frameCount;
        }
    }
    
    /**
     * Update the animation state based on movement and actions
     */
    updateAnimationState() {
        // Animation state priority: attack > walk > idle
        if (this.isAttacking) {
            this.currentState = this.STATES.ATTACK;
        } else if (this.isMoving) {
            this.currentState = this.STATES.WALK;
            
            // Update direction based on velocity
            if (Math.abs(this.velocity.x) > Math.abs(this.velocity.y)) {
                // Horizontal movement is dominant
                this.direction = this.velocity.x < 0 ? 1 : 3; // Left or right
            } else if (this.velocity.y !== 0) {
                // Vertical movement is dominant
                this.direction = this.velocity.y < 0 ? 2 : 0; // Up or down
            }
        } else {
            this.currentState = this.STATES.IDLE;
        }
    }
    
    /**
     * Start attack animation if not already attacking
     * @returns {boolean} Whether attack was started
     */
    attack() {
        if (!this.isAttacking) {
            this.isAttacking = true;
            this.attackCooldown = this.attackDuration;
            this.frameIndex = 0; // Reset animation frame
            return true;
        }
        return false;
    }
    
    /**
     * Set player velocity
     * @param {number} x - X velocity component
     * @param {number} y - Y velocity component
     */
    setVelocity(x, y) {
        this.velocity.x = x;
        this.velocity.y = y;
        
        // Normalize velocity vector for diagonal movement
        if (x !== 0 && y !== 0) {
            const length = Math.sqrt(x * x + y * y);
            this.velocity.x = x / length;
            this.velocity.y = y / length;
        }
        
        // Update moving state
        this.isMoving = x !== 0 || y !== 0;
    }
    
    /**
     * Calculate the source rectangle for the current animation frame
     * @returns {Object} Source rectangle {x, y, width, height}
     */
    getSourceRect() {
        // Calculate state index
        let stateIndex = 0;
        switch (this.currentState) {
            case this.STATES.IDLE:
                stateIndex = 0;
                break;
            case this.STATES.WALK:
                stateIndex = 1;
                break;
            case this.STATES.ATTACK:
                stateIndex = 2;
                break;
        }
        
        // Calculate source position in spritesheet
        const sx = this.frameIndex * this.width;
        const sy = (stateIndex * 4 + this.direction) * this.height;
        
        return {
            x: sx,
            y: sy,
            width: this.width,
            height: this.height
        };
    }
    
    /**
     * Draw the player on canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
     * @param {Object} cameraPosition - Camera position {x, y}
     */
    draw(ctx, cameraPosition) {
        if (!ctx) return;
        
        // Get sprite sheet
        const spriteSheetObj = spriteManager.getSpriteSheet('character_sprites');
        if (!spriteSheetObj) {
            this.drawDebugRect(ctx, cameraPosition);
            return;
        }
        
        const spriteSheet = spriteSheetObj.image;
        
        // Get screen dimensions
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        
        // Define scale based on view type
        const scaleFactor = gameState.camera?.viewType === 'strategic' ? 0.5 : 1;
        
        // Calculate screen position
        const screenX = (this.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
        const screenY = (this.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
        
        // Apply scale
        const width = this.width * SCALE;
        const height = this.height * SCALE;
        
        // Save context for rotation
        ctx.save();
        
        // Translate to player position
        ctx.translate(screenX, screenY);
        
        // Apply rotation if any
        if (typeof this.rotation === 'number') {
            ctx.rotate(this.rotation);
        }
        
        // Get animation source rectangle
        const sourceRect = this.getSourceRect();
        
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
        
        // Draw debug visuals if enabled
        if (this.visualDebug) {
            // Draw collision circle
            ctx.beginPath();
            ctx.arc(0, 0, this.collisionRadius * TILE_SIZE * scaleFactor, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw direction indicator
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(
                Math.cos(this.rotation) * this.width/2 * SCALE,
                Math.sin(this.rotation) * this.height/2 * SCALE
            );
            ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        // Restore context
        ctx.restore();
    }
    
    /**
     * Draw a debug rectangle if sprite sheet not available
     * @param {CanvasRenderingContext2D} ctx - Canvas rendering context
     * @param {Object} cameraPosition - Camera position {x, y}
     */
    drawDebugRect(ctx, cameraPosition) {
        // Get screen dimensions
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        
        // Define scale
        const scaleFactor = gameState.camera?.viewType === 'strategic' ? 0.5 : 1;
        
        // Calculate screen position
        const screenX = (this.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
        const screenY = (this.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
        
        // Draw a colored rectangle as fallback
        ctx.save();
        ctx.translate(screenX, screenY);
        
        if (typeof this.rotation === 'number') {
            ctx.rotate(this.rotation);
        }
        
        // Draw pink debug rectangle
        ctx.fillStyle = 'rgba(255, 105, 180, 0.7)';
        const width = this.width * SCALE;
        const height = this.height * SCALE;
        ctx.fillRect(-width/2, -height/2, width, height);
        
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
        
        ctx.restore();
    }
} 