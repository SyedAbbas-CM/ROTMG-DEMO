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
 * 4: wizard_idle_r_2    12: wizard_walk_r_2   20: wizard_atk_d_2
 * 5: wizard_idle_d_2    13: wizard_walk_d_2   21: wizard_atk_u_1
 * 6: wizard_idle_u_2    14: wizard_walk_u_2   22: wizard_atk_u_2
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
    
    // Determine animation state based on current conditions
    this.updateAnimationState(isMoving, velocity);
    
    // Only update animation frames for animated states (WALK and ATTACK)
    // or if we've just changed states (to reset the frame)
    if ((this.currentState === this.states.WALK || 
         this.currentState === this.states.ATTACK) && 
        this.frameTimer >= this.frameDuration) {
      
      this.frameTimer = 0;
      
      // For walk, cycle between frames 0 and 1
      if (this.currentState === this.states.WALK) {
        this.frameIndex = (this.frameIndex + 1) % 2;
      }
      // For attack, behavior depends on direction
      else if (this.currentState === this.states.ATTACK) {
        // Left and right attacks only have one frame in this sprite sheet
        if (this.direction === 1 || this.direction === 3) {
          // Keep frameIndex at 0 for left/right attacks
          this.frameIndex = 0;
        } else {
          // Down and up attacks have 2 frames
          this.frameIndex = (this.frameIndex + 1) % 2;
        }
      }
    }
  }
  
  /**
   * Reset to idle state
   */
  resetToIdle() {
    this.currentState = this.states.IDLE;
    this.frameIndex = 0;
    this.frameTimer = 0;
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
      // Only reset to idle if we're not attacking
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
   * @returns {boolean} Whether attack was started
   */
  attack() {
    if (!this.isAttacking && this.attackCooldown <= 0) {
      this.isAttacking = true;
      this.attackCooldown = this.attackDuration;
      this.currentState = this.states.ATTACK;
      this.frameIndex = 0; // Reset animation frame for attack
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
   * Get source rectangle for current animation frame.
   * This method calculates the correct cell based on the specific sprite sheet arrangement:
   * 
   * Sprite mapping for directions (0=down, 1=left, 2=up, 3=right):
   * 
   * IDLE frames (first frame only):
   * - Down (0): column 1
   * - Left (1): column 3
   * - Up (2): column 2
   * - Right (3): column 0
   * 
   * WALK frames (2 frames per direction):
   * - Down (0): columns 9 & 13
   * - Left (1): columns 11 & 15
   * - Up (2): columns 10 & 14
   * - Right (3): columns 8 & 12
   * 
   * ATTACK frames:
   * - Down (0): columns 18 & 19 (2 frames)
   * - Left (1): column 24 (1 frame)
   * - Up (2): columns 21 & 22 (2 frames)
   * - Right (3): column 17 (1 frame)
   *
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
        // Attack frames - note some directions only have 1 frame
        {
          if (this.direction === 0) { // down
            col = this.frameIndex === 0 ? 18 : 19;
          } else if (this.direction === 1) { // left
            col = 24; // Only one frame for left attack
          } else if (this.direction === 2) { // up
            col = this.frameIndex === 0 ? 21 : 22;
          } else { // right
            col = 17; // Only one frame for right attack
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
    
    // Final source rect
    return {
      x: col * this.spriteWidth,
      y: row * this.spriteHeight,
      width: this.spriteWidth,
      height: this.spriteHeight
    };
  }
}
