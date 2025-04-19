/**
 * EntityAnimator.js
 * A reusable animation system for game entities like players and enemies.
 *
 * This updated version is specifically tailored to the provided sprite sheet:
 * 
 * Sprite sheet arrangement (0-indexed columns):
 * 0: wizard_idle_r_1    8: wizard_walk_r_1    16: SKIP_FRAME
 * 1: wizard_idle_d_1    9: wizard_walk_d_1    17: wizard_atk_r_1
 * 2: wizard_idle_u_1    10: wizard_walk_u_1   18: wizard_atk_d_1
 * 3: wizard_idle_l_1    11: wizard_walk_l_2   19: wizard_atk_d_2
 * 4: wizard_idle_r_2    12: wizard_walk_r_2   20: wizard_atk_u_1
 * 5: wizard_idle_d_2    13: wizard_walk_d_2   21: wizard_atk_u_2
 * 6: wizard_idle_u_2    14: wizard_walk_u_2   22: UNUSED
 * 7: wizard_idle_l_2    15: wizard_walk_l_2   23: SKIP_FRAME
 *                                             24: wizard_atk_l_1
 */
export class EntityAnimator {
  /**
   * Create a new animator for an entity
   * @param {Object} config - Animation configuration 
   */
  constructor(config = {}) {
    // Animation states – note: DEATH is not mapped here (it will fall back to idle)
    this.states = {
      IDLE: 'idle',
      WALK: 'walk',
      ATTACK: 'attack',
      DEATH: 'death'
    };
    
    // Current state
    this.currentState = config.defaultState || this.states.IDLE;
    this.previousState = this.currentState; // Track previous state for transitions
    
    // Direction (original mapping: 0 = down, 1 = left, 2 = up, 3 = right)
    this.direction = 0;
    
    // Animation properties
    this.frameIndex = 0;
    this.frameCount = config.frameCount || 4; // (only used for cycling in walk and attack)
    this.frameDuration = config.frameDuration || 0.1; // Time per frame in seconds
    this.frameTimer = 0;
    
    // Stop animation when idle
    this.shouldAnimateIdle = false; // Set to false to freeze idle animation
    
    // Action flags
    this.isAttacking = false;
    this.attackCooldown = 0;
    this.attackDuration = config.attackDuration || 0.3; // Time to complete attack animation
    this.attackCooldownTime = config.attackCooldownTime || 0.5; // Time between attacks
    
    // Sprite sheet configuration
    this.spriteWidth = config.spriteWidth || 12;
    this.spriteHeight = config.spriteHeight || 12;
    this.spriteSheet = config.spriteSheet || 'character_sprites';
    
    // For this arrangement, we expect each character's frames are in one row
    this.characterIndex = config.characterIndex || 0;
  }
  
  /**
   * Update animation state and frame
   * @param {number} deltaTime - Time since last frame in seconds
   * @param {boolean} isMoving - Whether entity is moving
   * @param {Object} velocity - Current velocity vector {x, y}
   */
  update(deltaTime, isMoving, velocity = { x: 0, y: 0 }) {
    // Save previous state for comparison
    this.previousState = this.currentState;
    
    // Update animation timers
    this.frameTimer += deltaTime;
    
    // Update attack state
    if (this.isAttacking) {
      this.attackCooldown -= deltaTime;
      
      if (this.attackCooldown <= 0) {
        this.isAttacking = false;
        // Reset to idle when attack animation completes
        this.resetToIdle();
      }
    }
    
    // Only determine animation state if not attacking
    if (!this.isAttacking) {
      // Handle movement state
      if (isMoving) {
        // Set to walk state
        this.currentState = this.states.WALK;
        
        // Update direction based on movement
        if (Math.abs(velocity.x) > Math.abs(velocity.y)) {
          this.direction = velocity.x < 0 ? 1 : 3; // 1=left, 3=right
        } else {
          this.direction = velocity.y < 0 ? 2 : 0; // 2=up, 0=down
        }
      } else {
        // Not moving, set to idle
        this.currentState = this.states.IDLE;
        // Note: we don't change direction when stopping - we keep facing last direction
      }
    }
    
    // Only cycle animation frames if we're in a state that has animation
    if (this.currentState === this.states.WALK || this.currentState === this.states.ATTACK) {
      // Use different frame durations based on animation state
      const frameDurationToUse = (this.currentState === this.states.WALK) ? 
                                 this.frameDuration * 0.5 : // Walk faster for smoother animation
                                 this.frameDuration;       // Normal speed for attack
      
      // Update frame when timer exceeds duration
      if (this.frameTimer >= frameDurationToUse) {
        // Reset frame timer
        this.frameTimer = 0;
        
        // Cycle frames based on state
        if (this.currentState === this.states.WALK) {
          // Cycle between frames for walk animation
          this.frameIndex = (this.frameIndex + 1) % 2;
        } 
        else if (this.currentState === this.states.ATTACK) {
          // For attack animation
          if (this.direction === 1) { // Left attack
            // Keep the frameIndex at 0 for left attack
            this.frameIndex = 0;
          } else {
            // Cycle between frames for other directions
            this.frameIndex = (this.frameIndex + 1) % 2;
          }
        }
      }
    } else {
      // For idle state, we don't animate - just use frame 0
      this.frameIndex = 0;
    }
  }
  
  /**
   * Reset to idle animation
   * This is a critical helper method for forcing the character back to idle state
   */
  resetToIdle() {
    this.currentState = this.states.IDLE;
    this.frameIndex = 0;
    this.frameTimer = 0;
    this.isAttacking = false;
    // Note: we don't reset direction here, so character keeps facing last direction
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
      // Only change to walk if we were previously idle or already walking
      if (this.currentState === this.states.IDLE || this.currentState === this.states.WALK) {
        this.currentState = this.states.WALK;
        
        // Update direction based on movement (using original mapping: 0=down, 1=left, 2=up, 3=right)
        if (Math.abs(velocity.x) > Math.abs(velocity.y)) {
          this.direction = velocity.x < 0 ? 1 : 3;
        } else {
          this.direction = velocity.y < 0 ? 2 : 0;
        }
      }
    } else {
      // Not attacking and not moving - reset to idle
      // Only reset to idle if we're not attacking and currently walking
      if (this.currentState === this.states.WALK) {
        this.resetToIdle();
      }
    }
    
    // If we changed states, reset animation frame
    if (this.previousState !== this.currentState) {
      this.frameIndex = 0;
      this.frameTimer = 0;
    }
  }
  
  /**
   * Start attack animation
   * @param {number} direction - Optional direction override (0-3)
   * @returns {boolean} Whether attack was started
   */
  attack(direction) {
    // Check if we can attack
    const canAttack = !this.isAttacking && this.attackCooldown <= 0;
    console.log(`[Animator.attack] Can attack: ${canAttack}, isAttacking: ${this.isAttacking}, cooldown: ${this.attackCooldown.toFixed(2)}`);
    
    if (canAttack) {
      // Set attack flags
      this.isAttacking = true;
      this.attackCooldown = this.attackDuration;
      
      // If a direction was provided, override the current direction
      if (direction !== undefined && direction >= 0 && direction <= 3) {
        console.log(`[Animator.attack] Overriding direction: ${this.direction} -> ${direction}`);
        this.direction = direction;
      }
      
      console.log(`[Animator.attack] Starting attack in direction: ${this.direction} (${['down', 'left', 'up', 'right'][this.direction]})`);
      
      // Always set the animation state to ATTACK
      this.currentState = this.states.ATTACK;
      
      // Reset animation frame and timer to start the sequence
      this.frameIndex = 0;
      this.frameTimer = 0;
      
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
    if (Object.values(this.states).includes(state)) {
      this.currentState = state;
    }
    if (direction >= 0 && direction <= 3) {
      this.direction = direction;
    }
    this.frameIndex = 0;
    this.frameTimer = 0;
  }
  
  /**
   * Set the current state
   * @param {string} state - State name (use this.states constants)
   */
  setCurrentState(state) {
    if (Object.values(this.states).includes(state)) {
      this.currentState = state;
      this.frameIndex = 0;
      this.frameTimer = 0;
    }
  }
  
  /**
   * Set the entity's direction
   * @param {number} direction - Direction index (0=down, 1=left, 2=up, 3=right)
   */
  setDirection(direction) {
    if (direction >= 0 && direction <= 3) {
      this.direction = direction;
    }
  }
  
  /**
   * Set direction based on an angle (in radians)
   * @param {number} angle - Direction angle in radians
   */
  setDirectionFromAngle(angle) {
    // DEBUG: Log the raw angle for diagnostic purposes
    console.log(`[setDirectionFromAngle] Raw angle: ${angle.toFixed(2)} radians`);
    
    // Convert angle to direction: 0=down, 1=left, 2=up, 3=right
    // Normalize angle to 0-2π range
    const normalizedAngle = (angle + 2 * Math.PI) % (2 * Math.PI);
    console.log(`[setDirectionFromAngle] Normalized angle: ${normalizedAngle.toFixed(2)} radians`);
    
    // CORRECTION: FIX THE ANGLE MAPPING
    // 
    // The angle is in standard mathematical orientation where:
    // - 0 or 2π = right (positive X axis)
    // - π/2 = down (positive Y axis in canvas)
    // - π = left (negative X axis)
    // - 3π/2 = up (negative Y axis in canvas)
    //
    // Map these angles to our direction indices:
    // - Right (3): -π/4 to π/4 (around 0)
    // - Down (0): π/4 to 3π/4 (around π/2)
    // - Left (1): 3π/4 to 5π/4 (around π)
    // - Up (2): 5π/4 to 7π/4 (around 3π/2)
    
    let direction;
    if (normalizedAngle >= 7 * Math.PI / 4 || normalizedAngle < Math.PI / 4) {
      direction = 3; // Right (around 0 radians)
    } else if (normalizedAngle >= Math.PI / 4 && normalizedAngle < 3 * Math.PI / 4) {
      direction = 0; // Down (around π/2 radians)
    } else if (normalizedAngle >= 3 * Math.PI / 4 && normalizedAngle < 5 * Math.PI / 4) {
      direction = 1; // Left (around π radians)
    } else {
      direction = 2; // Up (around 3π/2 radians)
    }
    
    console.log(`[setDirectionFromAngle] Setting direction to: ${direction} (${['down', 'left', 'up', 'right'][direction]})`);
    this.direction = direction;
  }
  
  /**
   * Get source rectangle for current animation frame.
   * This method calculates the correct cell based on the specific sprite sheet arrangement.
   * @returns {Object} Source rectangle {x, y, width, height}
   */
  getSourceRect() {
    let col;
    
    switch (this.currentState) {
      case this.states.IDLE:
        // Idle frames are in the first 4 columns (one per direction)
        // Map: 0 (down) -> 1, 1 (left) -> 3, 2 (up) -> 2, 3 (right) -> 0
        {
          const idleMap = { 0: 1, 1: 3, 2: 2, 3: 0 };
          col = idleMap[this.direction];
        }
        break;
      
      case this.states.WALK:
        // Walk frames are split across columns based on direction and frame
        {
          const walkBaseMap = { 
            0: [9, 13],  // down: frames in columns 9 & 13
            1: [11, 15], // left: frames in columns 11 & 15
            2: [10, 14], // up: frames in columns 10 & 14
            3: [8, 12]   // right: frames in columns 8 & 12
          };
          col = walkBaseMap[this.direction][this.frameIndex];
        }
        break;
      
      case this.states.ATTACK:
        // Fixed mapping for attack animations
        {
          if (this.direction === 0) { // down - has 2 frames
            // Use columns 18 & 19 for down attack
            col = this.frameIndex === 0 ? 18 : 19;
          } else if (this.direction === 1) { // left - only 1 attack frame
            // For left attack, ONLY use the attack frame (col 24)
            col = 24; 
          } else if (this.direction === 2) { // up - has 2 frames
            // Use columns 20 & 21
            col = this.frameIndex === 0 ? 20 : 21;
          } else { // right - only 1 attack frame
            // For right attack, use column 17 and alternate with idle right (column 0)
            col = this.frameIndex === 0 ? 17 : 0;
          }
        }
        break;
      
      case this.states.DEATH:
      default:
        // Fallback to idle frames if death or unknown state
        {
          const idleMap = { 0: 1, 1: 3, 2: 2, 3: 0 };
          col = idleMap[this.direction];
        }
        break;
    }
    
    // Since all frames for a character are in one row on the sprite sheet,
    // the row is determined by the characterIndex.
    const row = this.characterIndex;
    
    // Calculate sprite position
    const spriteX = col * this.spriteWidth;
    const spriteY = row * this.spriteHeight;
    
    // Final source rect
    return {
      x: spriteX,
      y: spriteY,
      width: this.spriteWidth,
      height: this.spriteHeight
    };
  }
}
