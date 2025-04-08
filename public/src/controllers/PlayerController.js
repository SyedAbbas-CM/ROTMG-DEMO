/**
 * PlayerController.js
 * Handles input for controlling the AnimatedPlayer
 */

export class PlayerController {
  /**
   * Create a new player controller
   * @param {AnimatedPlayer} player - The player to control
   */
  constructor(player) {
    this.player = player;
    this.keys = {
      up: false,
      down: false,
      left: false,
      right: false,
      attack: false
    };
    
    // Movement vector
    this.movement = { x: 0, y: 0 };
    
    // Bind event listeners
    this.bindEvents();
  }
  
  /**
   * Bind keyboard event listeners
   */
  bindEvents() {
    // Keyboard down event
    window.addEventListener('keydown', (e) => {
      this.handleKeyDown(e);
    });
    
    // Keyboard up event
    window.addEventListener('keyup', (e) => {
      this.handleKeyUp(e);
    });
  }
  
  /**
   * Handle key down events
   * @param {KeyboardEvent} e - Keyboard event
   */
  handleKeyDown(e) {
    switch (e.key.toLowerCase()) {
      case 'w':
      case 'arrowup':
        this.keys.up = true;
        break;
      case 's':
      case 'arrowdown':
        this.keys.down = true;
        break;
      case 'a':
      case 'arrowleft':
        this.keys.left = true;
        break;
      case 'd':
      case 'arrowright':
        this.keys.right = true;
        break;
      case ' ':
      case 'shift':
        if (!this.keys.attack) {
          this.keys.attack = true;
          this.player.attack();
        }
        break;
    }
    
    // Update movement vector
    this.updateMovement();
  }
  
  /**
   * Handle key up events
   * @param {KeyboardEvent} e - Keyboard event
   */
  handleKeyUp(e) {
    switch (e.key.toLowerCase()) {
      case 'w':
      case 'arrowup':
        this.keys.up = false;
        break;
      case 's':
      case 'arrowdown':
        this.keys.down = false;
        break;
      case 'a':
      case 'arrowleft':
        this.keys.left = false;
        break;
      case 'd':
      case 'arrowright':
        this.keys.right = false;
        break;
      case ' ':
      case 'shift':
        this.keys.attack = false;
        break;
    }
    
    // Update movement vector
    this.updateMovement();
  }
  
  /**
   * Update the movement vector based on key states
   */
  updateMovement() {
    this.movement.x = 0;
    this.movement.y = 0;
    
    if (this.keys.right) this.movement.x += 1;
    if (this.keys.left) this.movement.x -= 1;
    if (this.keys.down) this.movement.y += 1;
    if (this.keys.up) this.movement.y -= 1;
    
    // Normalize diagonal movement
    if (this.movement.x !== 0 && this.movement.y !== 0) {
      const length = Math.sqrt(this.movement.x * this.movement.x + this.movement.y * this.movement.y);
      this.movement.x /= length;
      this.movement.y /= length;
    }
  }
  
  /**
   * Update the controller state
   * @param {number} deltaTime - Time elapsed since last frame in seconds
   */
  update(deltaTime) {
    // Update the player with the current movement vector
    this.player.update(deltaTime, this.movement);
  }
  
  /**
   * Clean up event listeners
   */
  destroy() {
    window.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('keyup', this.handleKeyUp);
  }
} 