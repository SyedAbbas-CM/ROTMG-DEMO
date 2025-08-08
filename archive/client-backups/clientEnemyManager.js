/**
 * ClientEnemyManager.js
 * Client-side manager for enemies with visual effects and interpolation
 */
export class ClientEnemyManager {
    /**
     * Creates a client-side enemy manager
     * @param {number} maxEnemies - Maximum number of enemies to support
     */
    constructor(maxEnemies = 1000) {
      this.maxEnemies = maxEnemies;
      this.enemyCount = 0;
      
      // Structure of Arrays for performance
      this.id = new Array(maxEnemies);         // Unique enemy IDs
      this.x = new Float32Array(maxEnemies);   // X position
      this.y = new Float32Array(maxEnemies);   // Y position
      this.width = new Float32Array(maxEnemies);  // Width for collision
      this.height = new Float32Array(maxEnemies); // Height for collision
      this.type = new Uint8Array(maxEnemies);  // Enemy type (0-4)
      this.health = new Float32Array(maxEnemies); // Current health
      this.maxHealth = new Float32Array(maxEnemies); // Maximum health
      
      // Visual and animation properties
      this.sprite = new Array(maxEnemies);     // Sprite reference
      this.spriteX = new Uint16Array(maxEnemies); // Sprite X position in sheet
      this.spriteY = new Uint16Array(maxEnemies); // Sprite Y position in sheet
      this.animFrame = new Uint8Array(maxEnemies); // Current animation frame
      this.animTime = new Float32Array(maxEnemies); // Animation timer
      
      // Visual effects
      this.flashTime = new Float32Array(maxEnemies); // Hit flash effect timer
      this.deathTime = new Float32Array(maxEnemies); // Death animation timer
      
      // Movement interpolation
      this.prevX = new Float32Array(maxEnemies); // Previous X position
      this.prevY = new Float32Array(maxEnemies); // Previous Y position
      this.targetX = new Float32Array(maxEnemies); // Target X position
      this.targetY = new Float32Array(maxEnemies); // Target Y position
      this.interpTime = new Float32Array(maxEnemies); // Interpolation timer
      
      // ID to index mapping
      this.idToIndex = new Map();
      
      // Enemy type to sprite mapping
      this.typeToSprite = [
        { x: 0, y: 0 },   // Type 0: Black Overlord
        { x: 8, y: 0 },   // Type 1: Red Berserker
        { x: 16, y: 0 },  // Type 2: Purple Illusionist
        { x: 24, y: 0 },  // Type 3: Emerald Regenerator
        { x: 32, y: 0 }   // Type 4: Navy Turtle
      ];
    }
    
    /**
     * Add a new enemy
     * @param {Object} enemyData - Enemy properties
     * @returns {string} Enemy ID
     */
    addEnemy(enemyData) {
      if (this.enemyCount >= this.maxEnemies) {
        console.warn('ClientEnemyManager: Maximum enemy capacity reached');
        return null;
      }
      
      const index = this.enemyCount++;
      const enemyId = enemyData.id || `enemy_${index}`;
      
      // Store enemy properties
      this.id[index] = enemyId;
      this.x[index] = enemyData.x;
      this.y[index] = enemyData.y;
      this.width[index] = enemyData.width || 20;
      this.height[index] = enemyData.height || 20;
      this.type[index] = enemyData.type || 0;
      this.health[index] = enemyData.health || 100;
      this.maxHealth[index] = enemyData.maxHealth || 100;
      
      // Initialize visual properties
      const spriteData = this.typeToSprite[this.type[index]] || this.typeToSprite[0];
      this.spriteX[index] = spriteData.x;
      this.spriteY[index] = spriteData.y;
      this.animFrame[index] = 0;
      this.animTime[index] = 0;
      this.flashTime[index] = 0;
      this.deathTime[index] = 0;
      
      // Initialize interpolation
      this.prevX[index] = this.x[index];
      this.prevY[index] = this.y[index];
      this.targetX[index] = this.x[index];
      this.targetY[index] = this.y[index];
      this.interpTime[index] = 0;
      
      // Store index for lookup
      this.idToIndex.set(enemyId, index);
      
      return enemyId;
    }
    
    /**
     * Update all enemies
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    update(deltaTime) {
      for (let i = 0; i < this.enemyCount; i++) {
        // Update position interpolation
        this.updateInterpolation(i, deltaTime);
        
        // Update animations
        this.updateAnimation(i, deltaTime);
        
        // Update visual effects
        this.updateEffects(i, deltaTime);
      }
    }
    
    /**
     * Update position interpolation
     * @param {number} index - Enemy index
     * @param {number} deltaTime - Time elapsed since last update
     */
    updateInterpolation(index, deltaTime) {
      // Interpolation factor (0 to 1)
      const INTERP_SPEED = 5; // Adjust for smoother/faster interpolation
      this.interpTime[index] += deltaTime * INTERP_SPEED;
      
      if (this.interpTime[index] >= 1) {
        // Reached target, prepare for next interpolation
        this.prevX[index] = this.targetX[index];
        this.prevY[index] = this.targetY[index];
        this.x[index] = this.targetX[index];
        this.y[index] = this.targetY[index];
        this.interpTime[index] = 1;
      } else {
        // Interpolate between previous and target positions
        const t = this.interpTime[index];
        this.x[index] = this.prevX[index] + (this.targetX[index] - this.prevX[index]) * t;
        this.y[index] = this.prevY[index] + (this.targetY[index] - this.prevY[index]) * t;
      }
    }
    
    /**
     * Update animations
     * @param {number} index - Enemy index
     * @param {number} deltaTime - Time elapsed since last update
     */
    updateAnimation(index, deltaTime) {
      // Animation timing
      this.animTime[index] += deltaTime;
      
      // Change animation frame every 0.2 seconds (adjust as needed)
      if (this.animTime[index] >= 0.2) {
        this.animTime[index] = 0;
        this.animFrame[index] = (this.animFrame[index] + 1) % 4; // Assuming 4 frames
        
        // Update sprite sheet position for animation
        // This assumes your sprite sheet has animation frames horizontally
        const baseX = this.typeToSprite[this.type[index]].x;
        this.spriteX[index] = baseX + this.animFrame[index] * 8; // Assuming 8px width sprites
      }
    }
    
    /**
     * Update visual effects
     * @param {number} index - Enemy index
     * @param {number} deltaTime - Time elapsed since last update
     */
    updateEffects(index, deltaTime) {
      // Handle hit flash effect
      if (this.flashTime[index] > 0) {
        this.flashTime[index] -= deltaTime;
      }
      
      // Handle death animation
      if (this.deathTime[index] > 0) {
        this.deathTime[index] -= deltaTime;
        
        // Remove enemy when death animation completes
        if (this.deathTime[index] <= 0) {
          this.swapRemove(index);
        }
      }
    }
    
    /**
     * Apply hit effect to an enemy
     * @param {number} index - Enemy index
     */
    applyHitEffect(index) {
      if (index < 0 || index >= this.enemyCount) return;
      
      // Set flash timer (duration in seconds)
      this.flashTime[index] = 0.1;
    }
    
    /**
     * Start death animation for an enemy
     * @param {number} index - Enemy index
     */
    startDeathAnimation(index) {
      if (index < 0 || index >= this.enemyCount) return;
      
      // Set death animation timer (duration in seconds)
      this.deathTime[index] = 0.5;
    }
    
    /**
     * Set enemy health
     * @param {string} enemyId - Enemy ID
     * @param {number} health - New health value
     * @returns {boolean} True if enemy was found
     */
    setEnemyHealth(enemyId, health) {
      const index = this.findIndexById(enemyId);
      if (index === -1) return false;
      
      // Apply hit effect if health decreased
      if (health < this.health[index]) {
        this.applyHitEffect(index);
      }
      
      // Update health
      this.health[index] = health;
      
      // Start death animation if health is 0
      if (health <= 0 && this.deathTime[index] <= 0) {
        this.startDeathAnimation(index);
      }
      
      return true;
    }
    
    /**
     * Remove an enemy using swap-and-pop technique
     * @param {number} index - Index of enemy to remove
     */
    swapRemove(index) {
      if (index < 0 || index >= this.enemyCount) return;
      
      // Remove from ID mapping
      const enemyId = this.id[index];
      this.idToIndex.delete(enemyId);
      
      // Swap with the last enemy (if not already the last)
      const lastIndex = this.enemyCount - 1;
      if (index !== lastIndex) {
        // Copy all properties from last enemy to this position
        this.id[index] = this.id[lastIndex];
        this.x[index] = this.x[lastIndex];
        this.y[index] = this.y[lastIndex];
        this.width[index] = this.width[lastIndex];
        this.height[index] = this.height[lastIndex];
        this.type[index] = this.type[lastIndex];
        this.health[index] = this.health[lastIndex];
        this.maxHealth[index] = this.maxHealth[lastIndex];
        this.sprite[index] = this.sprite[lastIndex];
        this.spriteX[index] = this.spriteX[lastIndex];
        this.spriteY[index] = this.spriteY[lastIndex];
        this.animFrame[index] = this.animFrame[lastIndex];
        this.animTime[index] = this.animTime[lastIndex];
        this.flashTime[index] = this.flashTime[lastIndex];
        this.deathTime[index] = this.deathTime[lastIndex];
        this.prevX[index] = this.prevX[lastIndex];
        this.prevY[index] = this.prevY[lastIndex];
        this.targetX[index] = this.targetX[lastIndex];
        this.targetY[index] = this.targetY[lastIndex];
        this.interpTime[index] = this.interpTime[lastIndex];
        
        // Update index in mapping
        this.idToIndex.set(this.id[index], index);
      }
      
      this.enemyCount--;
    }
    
    /**
     * Find enemy index by ID
     * @param {string} enemyId - ID of enemy to find
     * @returns {number} Index of enemy or -1 if not found
     */
    findIndexById(enemyId) {
      const index = this.idToIndex.get(enemyId);
      return index !== undefined ? index : -1;
    }
    
    /**
     * Remove an enemy by ID
     * @param {string} enemyId - ID of enemy to remove
     * @returns {boolean} True if enemy was found and removed
     */
    removeEnemyById(enemyId) {
      const index = this.findIndexById(enemyId);
      if (index !== -1) {
        this.swapRemove(index);
        return true;
      }
      return false;
    }
    
    /**
     * Set initial enemies list from server
     * @param {Array} enemies - Array of enemy data from server
     */
    setEnemies(enemies) {
      // Clear existing enemies
      this.enemyCount = 0;
      this.idToIndex.clear();
      
      // Add new enemies from server
      for (const enemy of enemies) {
        this.addEnemy(enemy);
      }
    }
    
    /**
     * Update enemies based on server data
     * @param {Array} enemies - Array of enemy data from server
     */
    updateEnemies(enemies) {
      // Track which enemies we've seen in this update
      const seenEnemies = new Set();
      
      // Update existing enemies and add new ones
      for (const enemy of enemies) {
        const index = this.findIndexById(enemy.id);
        
        if (index !== -1) {
          // Update existing enemy
          // Store previous position for interpolation
          this.prevX[index] = this.x[index];
          this.prevY[index] = this.y[index];
          
          // Set target position (for interpolation)
          this.targetX[index] = enemy.x;
          this.targetY[index] = enemy.y;
          
          // Reset interpolation timer
          this.interpTime[index] = 0;
          
          // Update health and other properties
          if (this.health[index] !== enemy.health) {
            this.setEnemyHealth(enemy.id, enemy.health);
          }
          
          this.maxHealth[index] = enemy.maxHealth;
        } else {
          // Add new enemy
          this.addEnemy(enemy);
        }
        
        seenEnemies.add(enemy.id);
      }
      
      // Remove enemies that aren't in the server update (except those in death animation)
      for (let i = 0; i < this.enemyCount; i++) {
        if (!seenEnemies.has(this.id[i]) && this.deathTime[i] <= 0) {
          this.swapRemove(i);
          i--; // Adjust index after removal
        }
      }
    }
    
    /**
     * Get enemy data for rendering
     * @returns {Array} Array of enemy render data
     */
    getEnemiesForRender() {
      const enemies = [];
      
      for (let i = 0; i < this.enemyCount; i++) {
        // Skip enemies in death animation
        if (this.health[i] <= 0 && this.deathTime[i] <= 0) continue;
        
        enemies.push({
          id: this.id[i],
          x: this.x[i],
          y: this.y[i],
          spriteX: this.spriteX[i],
          spriteY: this.spriteY[i],
          width: this.width[i],
          height: this.height[i],
          health: this.health[i],
          maxHealth: this.maxHealth[i],
          isFlashing: this.flashTime[i] > 0,
          isDying: this.deathTime[i] > 0
        });
      }
      
      return enemies;
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
      this.enemyCount = 0;
      this.idToIndex.clear();
    }
  }