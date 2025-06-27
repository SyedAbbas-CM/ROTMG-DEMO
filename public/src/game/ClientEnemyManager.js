/**
 * ClientEnemyManager.js
 * Client-side manager for enemies with visual effects and interpolation
 */
import { spriteDatabase } from '../assets/SpriteDatabase.js';

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
      this.spriteX = new Float32Array(maxEnemies);
      this.spriteY = new Float32Array(maxEnemies);
      this.spriteName = new Array(maxEnemies);
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
      
      // Enemy type definitions with sprite names for the new system
      this.enemyTypes = [
        { 
          name: "Goblin",      
          spriteName: 'goblin',  // Use sprite database name
          frames: 4,
          behaviors: ['chase', 'shoot'],
          moveSpeed: 20,
          attackRange: 5,
          attackCooldown: 2,
          projectileSpeed: 40,
          damagePerHit: 10
        },  
        { 
          name: "Orc",       
          spriteName: 'orc',  // Use sprite database name
          frames: 4,
          behaviors: ['chase'],
          moveSpeed: 40,
          attackRange: 1,
          attackCooldown: 1,
          damagePerHit: 5
        },  
        { 
          name: "Skeleton",      
          spriteName: 'skeleton', // Use sprite database name
          frames: 4,
          behaviors: ['chase', 'shoot'],
          moveSpeed: 10,
          attackRange: 8,
          attackCooldown: 3,
          projectileSpeed: 30,
          projectileCount: 3,
          projectileSpread: Math.PI/6,
          damagePerHit: 15
        },  
        { 
          name: "Troll",     
          spriteName: 'troll', // Use sprite database name
          frames: 4,
          behaviors: ['shoot'],
          moveSpeed: 0, // Doesn't move
          attackRange: 10,
          attackCooldown: 2.5,
          projectileSpeed: 60,
          damagePerHit: 12
        },  
        { 
          name: "Wizard",      
          spriteName: 'wizard', // Use sprite database name
          frames: 4,
          behaviors: ['chase'],
          moveSpeed: 25,
          attackRange: 1.5,
          attackCooldown: 1,
          damagePerHit: 20
        }   
      ];
      
      /* Merge external enemy templates that the server may use (loaded via entityDatabase). */
      if (typeof window !== 'undefined' && window.entityDatabase?.getAll) {
        try {
          const extraDefs = window.entityDatabase.getAll('enemies');
          extraDefs.forEach(def => {
            if (this.enemyTypes.find(e => e.name === def.name)) return; // skip duplicates
            this.enemyTypes.push({
              name: def.name || `enemy_${this.enemyTypes.length}`,
              spriteName: (def.sprite || '').replace(/^chars:/, ''),
              frames: 1,
              behaviors: ['chase'],
              moveSpeed: def.speed || 15,
              attackRange: def.attack?.range || 6,
              attackCooldown: (def.attack?.cooldown || 2000) / 1000,
              projectileSpeed: def.attack?.speed || 30,
              damagePerHit: def.attack?.damage || 10
            });
          });
        } catch (mergeErr) {
          console.warn('[ClientEnemyManager] Could not merge external enemy definitions:', mergeErr);
        }
      }
      
      // Initialize sprite coordinates from the sprite database
      this.initializeSpriteCoordinates();
      
      // Render scale per enemy (in tile units)
      this.renderScale = new Float32Array(maxEnemies);
      
      // After base enemyTypes array is initialised, merge external defs if available.
      try {
        const extDefs = (typeof window !== 'undefined' && window.entityDatabase?.getAll) ? window.entityDatabase.getAll('enemies') : [];
        if (extDefs && extDefs.length) {
          extDefs.forEach(ent => {
            // Use ent.id as index by order if provided; otherwise push sequentially
            const idx = this.enemyTypes.findIndex(e => e.name === ent.name);
            if (idx === -1) {
              this.enemyTypes.push({
                name: ent.name || ent.id || `enemy_${this.enemyTypes.length}`,
                spriteName: ent.sprite || ent.spriteName || '',
                spriteX: ent.spriteX || 0,
                spriteY: ent.spriteY || 0,
                frames: 1
              });
            }
          });
        }
      } catch (err) {
        console.warn('[ClientEnemyManager] Failed to merge external enemy defs', err);
      }
    }
    
    /**
     * Initialize sprite coordinates from the sprite database
     */
    initializeSpriteCoordinates() {
      // Check if sprite database is loaded
      if (!spriteDatabase || spriteDatabase.getStats().atlasesLoaded === 0) {
        console.warn('⚠️ Sprite database not loaded yet, using fallback coordinates');
        // Use fallback coordinates
        this.enemyTypes.forEach((type, index) => {
          type.spriteX = index * 8;
          type.spriteY = 0;
        });
        this.typeToSprite = this.enemyTypes.map(type => ({ x: type.spriteX, y: type.spriteY }));
        return;
      }
      
      this.enemyTypes.forEach((type, index) => {
        const sprite = spriteDatabase.getSprite(type.spriteName);
        if (sprite) {
          type.spriteX = sprite.x;
          type.spriteY = sprite.y;
          console.log(`Enemy type ${type.name} mapped to sprite (${sprite.x}, ${sprite.y})`);
        } else {
          // Try to grab sprite from default chars2 atlas grid if available
          const gridSpr = spriteDatabase.getSpriteByGrid('chars2', 0, index);
          if (gridSpr) {
            type.spriteX = gridSpr.x;
            type.spriteY = gridSpr.y;
            console.log(`Enemy type ${type.name} assigned grid sprite (${gridSpr.x}, ${gridSpr.y}) from chars2 atlas`);
          } else {
            console.warn(`Sprite '${type.spriteName}' not found for enemy type ${type.name}, using fallback coordinates`);
            type.spriteX = index * 8; // Fallback grid position
            type.spriteY = 0;
          }
        }
      });
      
      // Update the backwards-compatible mapping
      this.typeToSprite = this.enemyTypes.map(type => ({ x: type.spriteX, y: type.spriteY }));
      
      // Log sprite mapping for debugging
      console.log("ClientEnemyManager sprite mapping:", 
        this.enemyTypes.map(t => `${t.name}(${t.spriteX},${t.spriteY})`).join(', '));
    }
    
    /**
     * Reinitialize sprite coordinates (call this after sprite database is loaded)
     */
    reinitializeSprites() {
      console.log('🔄 Reinitializing enemy sprite coordinates...');
      this.initializeSpriteCoordinates();
      
      // Update existing enemies with new sprite coordinates
      for (let i = 0; i < this.enemyCount; i++) {
        const typeData = this.enemyTypes[this.type[i]];
        if (typeData) {
          this.spriteX[i] = typeData.spriteX;
          this.spriteY[i] = typeData.spriteY;
        }
      }
      
      console.log('✅ Enemy sprite coordinates updated');
    }
    
    /**
     * Add a new enemy
     * @param {Object} enemyData - Enemy properties
     * @returns {string} Enemy ID
     */
    addEnemy(enemyData) {
      const playerWorld = window.gameState?.character?.worldId;
      if (enemyData.worldId && playerWorld && enemyData.worldId !== playerWorld) {
        return null; // Skip off-world enemy
      }
      
      if (this.enemyCount >= this.maxEnemies) {
        console.warn('ClientEnemyManager: Maximum enemy capacity reached');
        return null;
      }
      
      const index = this.enemyCount++;
      const enemyId = enemyData.id || `enemy_${index}`;
      
      // Ensure spriteDatabase has a fetchGridSprite stub to avoid TypeErrors in older call sites
      if (typeof window !== 'undefined' && window.spriteDatabase && typeof window.spriteDatabase.fetchGridSprite !== 'function') {
        window.spriteDatabase.fetchGridSprite = () => {};
      }
      
      // Store enemy properties
      this.id[index] = enemyId;
      this.x[index] = enemyData.x;
      this.y[index] = enemyData.y;
      this.width[index] = enemyData.width || 20;
      this.height[index] = enemyData.height || 20;
      this.type[index] = enemyData.type || 0;
      this.health[index] = enemyData.health || 100;
      this.maxHealth[index] = enemyData.maxHealth || 100;
      
      // Ensure we have a definition for this enemy type index.
      const typeIdx = this.type[index];
      if (typeIdx >= this.enemyTypes.length) {
        console.warn(`Unknown enemy type index ${typeIdx} – creating placeholder entries up to that index`);
        // Add placeholders until the array is long enough to include typeIdx.
        while (this.enemyTypes.length <= typeIdx) {
          const placeholderIdx = this.enemyTypes.length;
          // Spread placeholders across a simple 8x8 grid (chars2-style) so they don't all overlap.
          const gridX = (placeholderIdx % 8) * 8;
          const gridY = Math.floor(placeholderIdx / 8) * 8;
          // Register a placeholder alias via the classic spriteManager so we can render something.
          try {
            const sm = window.spriteManager;
            if (sm && typeof sm.fetchGridSprite === 'function') {
              const aliasName = `dynamic_${placeholderIdx}`;
              if (!sm.aliases || !sm.aliases[aliasName]) {
                sm.fetchGridSprite('enemy_sprites', Math.floor(placeholderIdx / 8), placeholderIdx % 8, aliasName, 8, 8);
              }
            }
          } catch (e) {
            console.warn('Failed to register placeholder sprite alias via spriteManager:', e);
          }
          this.enemyTypes.push({
            name: `dynamic_${placeholderIdx}`,
            spriteX: gridX,
            spriteY: gridY,
            frames: 1
          });
          // Register a matching sprite alias in the SpriteDatabase so that future renders
          // can draw via spriteName if we decide to assign it.
          const sm2 = window.spriteManager;
          if (sm2 && typeof sm2.fetchGridSprite === 'function') {
            const alias = `dynamic_${placeholderIdx}`;
            if (!sm2.aliases?.[alias] && typeof spriteDatabase.fetchGridSprite === 'function') {
              // Guard against environments where fetchGridSprite is not present
              if (typeof spriteDatabase.fetchGridSprite === 'function') {
                spriteDatabase.fetchGridSprite('enemy_sprites', Math.floor(placeholderIdx / 8), placeholderIdx % 8, alias, 8, 8);
              }
            }
          }
        }
        // Keep legacy mapping array (if already generated) in sync
        if (this.typeToSprite) {
          this.typeToSprite[typeIdx] = {
            x: this.enemyTypes[typeIdx].spriteX,
            y: this.enemyTypes[typeIdx].spriteY
          };
        }
      }
      
      // If server provided spriteName use database; else fall back to type spriteX/Y
      if (enemyData.spriteName) {
        this.spriteName[index] = enemyData.spriteName;
        const s = window.spriteDatabase?.getSprite(enemyData.spriteName);
        this.spriteX[index] = s ? s.x : 0;
        this.spriteY[index] = s ? s.y : 0;
      } else {
        const typeData = this.enemyTypes[this.type[index]];
        this.spriteX[index] = typeData.spriteX;
        this.spriteY[index] = typeData.spriteY;
        this.spriteName[index] = null;
        if (!this.spriteName[index]) {
          this.spriteName[index] = typeData.name;
        }
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
      }
      
      // Store index for lookup
      this.idToIndex.set(enemyId, index);
      
      // Add render scale
      this.renderScale[index] = enemyData.renderScale || 2;
      
      console.log(`Added enemy ${enemyId} of type ${this.enemyTypes[this.type[index]].name} at (${enemyData.x}, ${enemyData.y})`);
      
      return enemyId;
    }
    
    /**
     * Update all enemies
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    update(deltaTime) {
      for (let i = 0; i < this.enemyCount; i++) {
        // Interpolate position towards target smoothly
        this.updateInterpolation(i, deltaTime);
        
        // Skip animations - enemies don't animate
        
        // Only update visual effects (flashing and death)
        this.updateEffects(i, deltaTime);
        
        // UPDATE BEHAVIORS
        // Note: By default enemies don't move on the client, but we support behaviors
        // for custom gameplay modes where the server might not be involved
        if (window.ALLOW_CLIENT_ENEMY_BEHAVIOR) {
          this.updateBehaviors(i, deltaTime);
        }
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
      // DISABLED ANIMATIONS - Always use frame 0
      this.animFrame[index] = 0;
      
      // Set sprite position based on enemy type with no animation
      const typeData = this.enemyTypes[this.type[index]];
      this.spriteX[index] = typeData.spriteX;
      this.spriteY[index] = typeData.spriteY;
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
        this.spriteName[index] = this.spriteName[lastIndex];
        this.animFrame[index] = this.animFrame[lastIndex];
        this.animTime[index] = this.animTime[lastIndex];
        this.flashTime[index] = this.flashTime[lastIndex];
        this.deathTime[index] = this.deathTime[lastIndex];
        this.prevX[index] = this.prevX[lastIndex];
        this.prevY[index] = this.prevY[lastIndex];
        this.targetX[index] = this.targetX[lastIndex];
        this.targetY[index] = this.targetY[lastIndex];
        this.interpTime[index] = this.interpTime[lastIndex];
        this.renderScale[index] = this.renderScale[lastIndex];
        
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
      // Filter out enemies from other worlds
      const playerWorld = window.gameState?.character?.worldId;
      if (playerWorld) {
        enemies = enemies.filter(e => e.worldId === playerWorld);
      }
      
      // Clear existing enemies
      this.enemyCount = 0;
      this.idToIndex.clear();
      
      // Add new enemies from server (filtered)
      for (const enemy of enemies) {
        this.addEnemy(enemy);
      }
    }
    
    /**
     * Update enemies based on server data
     * @param {Array} enemies - Array of enemy data from server
     */
    updateEnemies(enemies) {
      const playerWorld = window.gameState?.character?.worldId;
      if (playerWorld) {
        enemies = enemies.filter(e => e.worldId === playerWorld);
      }
      
      // Track which enemies we've seen in this update
      const seenEnemies = new Set();
      
      // Update existing enemies and add new ones
      for (const enemy of enemies) {
        if (playerWorld && enemy.worldId && enemy.worldId !== playerWorld) {
          // Ensure we purge if we already had that id cached
          this.removeEnemyById(enemy.id);
          continue;
        }
        
        const index = this.findIndexById(enemy.id);
        
        if (index !== -1) {
          // Update existing enemy (interpolated movement)

          // Update target position for interpolation
          this.prevX[index] = this.x[index];
          this.prevY[index] = this.y[index];
          this.targetX[index] = enemy.x;
          this.targetY[index] = enemy.y;
          this.interpTime[index] = 0; // reset interpolation timer

          // Update health (with hit flash)
          if (this.health[index] !== enemy.health) {
            this.setEnemyHealth(enemy.id, enemy.health);
          }

          // Update other dynamic properties if needed (e.g., type switch)
          this.type[index] = enemy.type;
          this.maxHealth[index] = enemy.maxHealth;
          this.renderScale[index] = enemy.renderScale || this.renderScale[index];
          if (enemy.spriteName) this.spriteName[index] = enemy.spriteName;
        } else {
          // Add new enemy (only first time)
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
          spriteName: this.spriteName[i],
          width: this.width[i],
          height: this.height[i],
          health: this.health[i],
          maxHealth: this.maxHealth[i],
          isFlashing: this.flashTime[i] > 0,
          isDying: this.deathTime[i] > 0,
          renderScale: this.renderScale[i] || 2
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
    
    /**
     * Verify that sprite sheet and type mapping is correct
     * Call this method after sprite sheets are loaded to validate configuration
     */
    verifySpriteData() {
      // Try to get sprite sheet from global spriteManager
      const spriteManager = window.spriteManager;
      if (!spriteManager) {
        console.error("Cannot verify enemy sprites: spriteManager not available");
        return false;
      }
      
      const enemySheet = spriteManager.getSpriteSheet('enemy_sprites');
      if (!enemySheet) {
        console.error("Cannot verify enemy sprites: enemy_sprites sheet not found");
        return false;
      }
      
      console.log("Enemy sprite sheet loaded:", enemySheet.config);
      
      // Verify each type's sprite coordinates are within bounds
      const sheetWidth = enemySheet.image.width;
      const sheetHeight = enemySheet.image.height;
      
      let allValid = true;
      this.enemyTypes.forEach((type, index) => {
        const maxX = type.spriteX + (type.frames * 8);
        if (maxX > sheetWidth || type.spriteY + 8 > sheetHeight) {
          console.error(`Enemy type ${index} (${type.name}) has sprite coordinates out of bounds:`,
            `X range: ${type.spriteX}-${maxX}, Y: ${type.spriteY}`,
            `Sheet size: ${sheetWidth}x${sheetHeight}`);
          allValid = false;
        }
      });
      
      if (allValid) {
        console.log("All enemy sprite mappings verified successfully");
      } else {
        console.error("Some enemy sprite mappings are invalid - check sprite sheet configuration");
      }
      
      return allValid;
    }
    
    /**
     * Update enemy behaviors
     * @param {number} index - Enemy index
     * @param {number} deltaTime - Time elapsed since last update
     */
    updateBehaviors(index, deltaTime) {
      // Skip if enemy is dead
      if (this.health[index] <= 0) return;
      
      // Get enemy's type
      const type = this.type[index];
      
      // Get type data
      const typeData = this.enemyTypes[type];
      if (!typeData || !typeData.behaviors) return;
      
      // Initialize behavior data if needed
      if (!this.behaviorData) {
        this.behaviorData = new Array(this.maxEnemies);
      }
      
      // Create behavior data for this enemy if it doesn't exist
      if (!this.behaviorData[index]) {
        this.behaviorData[index] = {
          lastAttackTime: 0,
          targetId: null,
          patrolPoint: null,
          state: 'idle'
        };
      }
      
      const behaviorData = this.behaviorData[index];
      
      // Find target (player) if needed
      const target = this.findClosestTarget(index);
      
      // Update behaviors based on type
      for (const behavior of typeData.behaviors) {
        switch (behavior) {
          case 'chase':
            this.updateChase(index, target, typeData, deltaTime, behaviorData);
            break;
          case 'shoot':
            this.updateShoot(index, target, typeData, deltaTime, behaviorData);
            break;
          case 'patrol':
            this.updatePatrol(index, typeData, deltaTime, behaviorData);
            break;
          case 'teleport':
            this.updateTeleport(index, target, typeData, deltaTime, behaviorData);
            break;
          default:
            // Unknown behavior
            break;
        }
      }
    }
    
    /**
     * Find closest target for enemy
     * @param {number} index - Enemy index
     * @returns {Object|null} Target object or null
     */
    findClosestTarget(index) {
      // In a real implementation, this would find the closest player
      // For now, just return the local player if available
      if (window.gameState && window.gameState.character) {
        return window.gameState.character;
      }
      return null;
    }
    
    /**
     * Update chase behavior
     * @param {number} index - Enemy index
     * @param {Object} target - Target object
     * @param {Object} typeData - Enemy type data
     * @param {number} deltaTime - Time elapsed since last update
     * @param {Object} behaviorData - Enemy behavior state
     */
    updateChase(index, target, typeData, deltaTime, behaviorData) {
      // Skip if no target or enemy can't move
      if (!target || typeData.moveSpeed <= 0) return;
      
      // Get current position
      const x = this.x[index];
      const y = this.y[index];
      
      // Calculate distance to target
      const dx = target.x - x;
      const dy = target.y - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // Check if within attack range
      if (distance <= typeData.attackRange) {
        // Close enough to attack, stop moving
        behaviorData.state = 'attacking';
        return;
      }
      
      // Calculate movement direction
      const dirX = dx / distance;
      const dirY = dy / distance;
      
      // Move towards target
      const moveSpeed = typeData.moveSpeed * deltaTime;
      this.x[index] = x + dirX * moveSpeed;
      this.y[index] = y + dirY * moveSpeed;
      
      // Update behavior state
      behaviorData.state = 'chasing';
    }
    
    /**
     * Update shoot behavior
     * @param {number} index - Enemy index
     * @param {Object} target - Target object
     * @param {Object} typeData - Enemy type data
     * @param {number} deltaTime - Time elapsed since last update
     * @param {Object} behaviorData - Enemy behavior state
     */
    updateShoot(index, target, typeData, deltaTime, behaviorData) {
      // Skip if no target
      if (!target) return;
      
      // Get current position
      const x = this.x[index];
      const y = this.y[index];
      
      // Calculate distance to target
      const dx = target.x - x;
      const dy = target.y - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // Check if within attack range
      if (distance > typeData.attackRange) {
        return; // Too far to attack
      }
      
      // Check cooldown
      const now = Date.now();
      if (now - behaviorData.lastAttackTime < typeData.attackCooldown * 1000) {
        return; // Still on cooldown
      }
      
      // Fire projectile
      this.fireProjectile(index, target, typeData);
      
      // Update last attack time
      behaviorData.lastAttackTime = now;
    }
    
    /**
     * Fire a projectile
     * @param {number} index - Enemy index
     * @param {Object} target - Target object
     * @param {Object} typeData - Enemy type data
     */
    fireProjectile(index, target, typeData) {
      // Skip if bullet manager not available
      if (!window.gameState || !window.gameState.bulletManager) return;
      
      const bulletManager = window.gameState.bulletManager;
      
      // Calculate angle to target
      const x = this.x[index];
      const y = this.y[index];
      const dx = target.x - x;
      const dy = target.y - y;
      const angle = Math.atan2(dy, dx);
      
      // Determine number of projectiles
      const projectileCount = typeData.projectileCount || 1;
      const spread = typeData.projectileSpread || 0;
      
      for (let i = 0; i < projectileCount; i++) {
        // Calculate angle with spread
        let projectileAngle = angle;
        if (projectileCount > 1) {
          projectileAngle = angle - (spread / 2) + (spread * i / (projectileCount - 1));
        }
        
        // Create bullet
        const bulletData = {
          x: x,
          y: y,
          vx: Math.cos(projectileAngle) * typeData.projectileSpeed,
          vy: Math.sin(projectileAngle) * typeData.projectileSpeed,
          ownerId: this.id[index],
          damage: typeData.damagePerHit || 10,
          lifetime: 3.0,
          width: 8,
          height: 8,
          // Add sprite info if available
          spriteSheet: 'bullet_sprites',
          spriteX: 0, // Default sprite position
          spriteY: 0, // Default sprite position
          spriteWidth: 8,
          spriteHeight: 8
        };
        
        bulletManager.addBullet(bulletData);
      }
    }
    
    /**
     * Update patrol behavior
     * @param {number} index - Enemy index
     * @param {Object} typeData - Enemy type data
     * @param {number} deltaTime - Time elapsed since last update
     * @param {Object} behaviorData - Enemy behavior state
     */
    updatePatrol(index, typeData, deltaTime, behaviorData) {
      // Not implemented yet - for server-side behavior
    }
    
    /**
     * Update teleport behavior
     * @param {number} index - Enemy index
     * @param {Object} target - Target object
     * @param {Object} typeData - Enemy type data
     * @param {number} deltaTime - Time elapsed since last update
     * @param {Object} behaviorData - Enemy behavior state
     */
    updateTeleport(index, target, typeData, deltaTime, behaviorData) {
      // Not implemented yet - for server-side behavior
    }
  }
