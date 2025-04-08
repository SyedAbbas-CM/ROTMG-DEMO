/**
 * Animator.js
 * A class to manage sprite animations
 */

import { spriteManager } from '../assets/spriteManager.js';

/**
 * Animation defines a sequence of sprite frames
 */
class Animation {
  /**
   * Create an animation sequence
   * @param {Object} config - Animation configuration
   * @param {string} config.name - Animation name (e.g. "idle", "walk", "attack")
   * @param {Array<Object>} config.frames - Array of frame objects with {x, y} coordinates in the spritesheet
   * @param {number} config.frameDuration - Duration of each frame in seconds
   * @param {boolean} config.loop - Whether the animation should loop
   */
  constructor(config) {
    this.name = config.name;
    this.frames = config.frames || [];
    this.frameDuration = config.frameDuration || 0.1; // Default 10 FPS
    this.loop = config.loop !== undefined ? config.loop : true;
    this.frameCount = this.frames.length;
  }
}

/**
 * Animator handles sprite animations for an entity
 */
export class Animator {
  /**
   * Create a new animator
   * @param {Object} options - Configuration options
   * @param {string} options.spriteSheet - ID of the sprite sheet to use
   * @param {Object} options.animations - Animation definitions
   * @param {string} options.defaultAnimation - The default animation to play
   */
  constructor({ spriteSheet, animations, defaultAnimation }) {
    this.spriteSheet = spriteSheet;
    this.animations = animations || {};
    this.currentAnimation = null;
    this.currentFrame = 0;
    this.frameTimer = 0;
    this.isPlaying = false;
    this.isLooping = true;
    this.onAnimationEnd = null;
    
    // Set default animation if provided
    if (defaultAnimation && this.animations[defaultAnimation]) {
      this.play(defaultAnimation);
    }
  }
  
  /**
   * Add a new animation
   * @param {string} name - The animation name
   * @param {Object} config - Animation configuration
   * @param {Array} config.frames - Array of frame indices
   * @param {number} config.frameDuration - Duration of each frame in seconds
   * @param {boolean} config.loop - Whether the animation should loop
   */
  addAnimation(name, { frames, frameDuration, loop = true }) {
    this.animations[name] = {
      frames: frames || [],
      frameDuration: frameDuration || 0.1,
      loop: loop
    };
  }
  
  /**
   * Play an animation
   * @param {string} name - The animation name
   * @param {Object} options - Playback options
   * @param {boolean} options.loop - Whether to loop the animation
   * @param {Function} options.onEnd - Callback when animation ends
   */
  play(name, { loop = true, onEnd = null } = {}) {
    // Don't restart if already playing this animation
    if (this.currentAnimation === name && this.isPlaying) {
      return;
    }
    
    const animation = this.animations[name];
    if (!animation) {
      console.warn(`Animation "${name}" not found.`);
      return;
    }
    
    this.currentAnimation = name;
    this.currentFrame = 0;
    this.frameTimer = 0;
    this.isPlaying = true;
    this.isLooping = loop !== undefined ? loop : animation.loop;
    this.onAnimationEnd = onEnd;
  }
  
  /**
   * Stop the current animation
   */
  stop() {
    this.isPlaying = false;
  }
  
  /**
   * Update the animation state
   * @param {number} deltaTime - Time elapsed since last frame in seconds
   */
  update(deltaTime) {
    if (!this.isPlaying || !this.currentAnimation) {
      return;
    }
    
    const animation = this.animations[this.currentAnimation];
    if (!animation) {
      return;
    }
    
    this.frameTimer += deltaTime;
    
    // Time to advance to the next frame
    if (this.frameTimer >= animation.frameDuration) {
      this.frameTimer = 0;
      this.currentFrame++;
      
      // Check if we've reached the end of the animation
      if (this.currentFrame >= animation.frames.length) {
        if (this.isLooping) {
          // Loop back to the beginning
          this.currentFrame = 0;
        } else {
          // Stop at the last frame
          this.currentFrame = animation.frames.length - 1;
          this.isPlaying = false;
          
          // Call the end callback if provided
          if (typeof this.onAnimationEnd === 'function') {
            this.onAnimationEnd();
          }
        }
      }
    }
  }
  
  /**
   * Get the current frame index from the sprite sheet
   * @returns {number} The current frame index
   */
  getCurrentFrameIndex() {
    if (!this.currentAnimation) {
      return 0;
    }
    
    const animation = this.animations[this.currentAnimation];
    if (!animation || animation.frames.length === 0) {
      return 0;
    }
    
    return animation.frames[this.currentFrame];
  }
  
  /**
   * Get the name of the current animation
   * @returns {string} Current animation name
   */
  getCurrentAnimation() {
    return this.currentAnimation;
  }
  
  /**
   * Check if an animation is currently playing
   * @returns {boolean} True if animation is playing
   */
  isAnimationPlaying() {
    return this.isPlaying;
  }
  
  /**
   * Get the sprite sheet ID
   * @returns {string} Sprite sheet ID
   */
  getSpriteSheet() {
    return this.spriteSheet;
  }
  
  /**
   * Draw the current animation frame
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {number} x - X coordinate (center of entity)
   * @param {number} y - Y coordinate (center of entity)
   * @param {number} width - Width to draw
   * @param {number} height - Height to draw
   * @param {number} rotation - Rotation in radians
   */
  draw(ctx, x, y, width, height, rotation = 0) {
    // Get current frame
    const frame = this.getCurrentFrame();
    
    // Draw sprite
    ctx.save();
    ctx.translate(x, y);
    if (rotation) {
      ctx.rotate(rotation);
    }
    
    spriteManager.drawSprite(
      ctx,
      this.spriteSheet,
      frame.x * this.frameWidth, 
      frame.y * this.frameHeight,
      -width / 2,
      -height / 2,
      width,
      height
    );
    
    ctx.restore();
  }
}

/**
 * Common animation definitions
 * These can be used as templates for most characters
 */
export const CommonAnimations = {
  // Simple 2-frame walking animation
  WALK: {
    frames: [
      { x: 0, y: 0 }, // Frame 1
      { x: 1, y: 0 }  // Frame 2
    ],
    frameDuration: 0.2,
    loop: true
  },
  
  // Idle animation (single frame)
  IDLE: {
    frames: [
      { x: 0, y: 0 } // Just the base frame
    ],
    frameDuration: 0.1,
    loop: true
  },
  
  // Attack animation (3 frames)
  ATTACK: {
    frames: [
      { x: 2, y: 0 }, // Wind up
      { x: 3, y: 0 }, // Attack
      { x: 2, y: 0 }  // Return
    ],
    frameDuration: 0.1,
    loop: false
  },
  
  // Death animation
  DEATH: {
    frames: [
      { x: 4, y: 0 }, // Start dying
      { x: 5, y: 0 }  // Dead
    ],
    frameDuration: 0.3,
    loop: false
  }
};

// Export Animation class for extending
export { Animation }; 