/**
 * EntityAnimator.js
 * A reusable animation system for game entities like players and enemies
 */

export class EntityAnimator {
  /**
   * Create a new animator for an entity
   * @param {Object} config - Animation configuration 
   */
  constructor(config = {}) {
    // Animation states
    this.states = {
      IDLE: 'idle',
      WALK: 'walk',
      ATTACK: 'attack',
      DEATH: 'death'
    };
    
    // Current state
    this.currentState = config.defaultState || this.states.IDLE;
    
    // Direction (0 = down, 1 = left, 2 = up, 3 = right)
    this.direction = 0;
    
    // Animation properties
    this.frameIndex = 0;
    this.frameCount = config.frameCount || 4; // Default frame count per animation
    this.frameDuration = config.frameDuration || 0.1; // Time per frame in seconds
    this.frameTimer = 0;
    
    // Action flags
    this.isAttacking = false;
    this.attackCooldown = 0;
    this.attackDuration = config.attackDuration || 0.3; // Time to complete attack animation
    this.attackCooldownTime = config.attackCooldownTime || 0.5; // Time between attacks
    
    // Sprite sheet configuration
    this.spriteWidth = config.spriteWidth || 64;  // Width of each frame in the spritesheet
    this.spriteHeight = config.spriteHeight || 64; // Height of each frame in the spritesheet
    this.spriteSheet = config.spriteSheet || 'character_sprites';
  }
  
  /**
   * Update animation state and frame
   * @param {number} deltaTime - Time since last frame in seconds
   * @param {boolean} isMoving - Whether entity is moving
   * @param {Object} velocity - Current velocity vector {x, y}
   */
  update(deltaTime, isMoving, velocity = { x: 0, y: 0 }) {
    // Update animation timers
    this.frameTimer += deltaTime;
    
    // Update attack state
    if (this.isAttacking) {
      this.attackCooldown -= deltaTime;
      if (this.attackCooldown <= 0) {
        this.isAttacking = false;
      }
    }
    
    // Determine animation state
    this.updateAnimationState(isMoving, velocity);
    
    // Update animation frame
    if (this.frameTimer >= this.frameDuration) {
      this.frameTimer = 0;
      this.frameIndex = (this.frameIndex + 1) % this.frameCount;
    }
  }
  
  /**
   * Update the current animation state based on entity movement and actions
   * @param {boolean} isMoving - Whether entity is moving
   * @param {Object} velocity - Entity velocity {x, y}
   */
  updateAnimationState(isMoving, velocity) {
    // Priority: attack -> walk -> idle
    if (this.isAttacking) {
      this.currentState = this.states.ATTACK;
    } else if (isMoving) {
      this.currentState = this.states.WALK;
      
      // Update direction based on movement
      if (Math.abs(velocity.x) > Math.abs(velocity.y)) {
        // Horizontal movement is dominant
        this.direction = velocity.x < 0 ? 1 : 3; // Left or right
      } else {
        // Vertical movement is dominant
        this.direction = velocity.y < 0 ? 2 : 0; // Up or down
      }
    } else {
      this.currentState = this.states.IDLE;
    }
  }
  
  /**
   * Start attack animation
   * @returns {boolean} Whether attack was started
   */
  attack() {
    if (!this.isAttacking && this.attackCooldown <= 0) {
      this.isAttacking = true;
      this.attackCooldown = this.attackDuration;
      this.frameIndex = 0; // Reset animation frame for attack
      return true;
    }
    return false;
  }
  
  /**
   * Force a specific animation state
   * @param {string} state - State name (use this.states constants)
   * @param {number} direction - Direction index (0-3)
   */
  setAnimationState(state, direction) {
    // Only change if the state is valid
    if (Object.values(this.states).includes(state)) {
      this.currentState = state;
    }
    
    // Validate direction
    if (direction >= 0 && direction <= 3) {
      this.direction = direction;
    }
    
    // Reset frame index when changing state
    this.frameIndex = 0;
  }
  
  /**
   * Get source rectangle for current animation frame
   * @returns {Object} Source rectangle {x, y, width, height}
   */
  getSourceRect() {
    // Calculate state index
    let stateIndex = 0;
    switch (this.currentState) {
      case this.states.IDLE:
        stateIndex = 0;
        break;
      case this.states.WALK:
        stateIndex = 1;
        break;
      case this.states.ATTACK:
        stateIndex = 2;
        break;
      case this.states.DEATH:
        stateIndex = 3;
        break;
    }
    
    // Calculate source position in spritesheet
    const sx = this.frameIndex * this.spriteWidth;
    const sy = (stateIndex * 4 + this.direction) * this.spriteHeight;
    
    return {
      x: sx,
      y: sy,
      width: this.spriteWidth,
      height: this.spriteHeight
    };
  }
} 