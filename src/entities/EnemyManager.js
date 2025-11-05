// File: /src/Managers/EnemyManager.js

import BehaviorSystem from '../Behaviours/BehaviorSystem.js';
import { entityDatabase } from '../assets/EntityDatabase.js';
import fs from 'fs';
import path from 'path';
import { parseBehaviourTree, BehaviourTreeRunner } from '../Behaviours/BehaviorTree.js';
import { rollDropTable } from './DropSystem.js';
import { loadEnemyDefinitions, compileEnemy } from '../EnemyDefinitionLoader.js';

/**
 * EnemyManager handles enemy creation, updating, and removal.
 * Uses Structure of Arrays (SoA) for data layout optimization.
 */
export default class EnemyManager {
  /**
   * Creates an enemy manager
   * @param {number} maxEnemies - Maximum number of enemies to allow
   */
  constructor(maxEnemies = 1000, itemManager = null) {
    this.maxEnemies = maxEnemies;
    this.enemyCount = 0;
    this.nextEnemyId = 1; // For assigning unique IDs

    // Injected dependencies (avoids global state)
    this.itemManager = itemManager;

    // SoA data layout for position and basic properties
    this.id = new Array(maxEnemies);         // Unique enemy IDs
    this.x = new Float32Array(maxEnemies);   // X position
    this.y = new Float32Array(maxEnemies);   // Y position
    this.width = new Float32Array(maxEnemies);  // Collision width
    this.height = new Float32Array(maxEnemies); // Collision height
    this.type = new Uint8Array(maxEnemies);  // Enemy type (0-4)
    this.health = new Float32Array(maxEnemies); // Current health
    this.maxHealth = new Float32Array(maxEnemies); // Maximum health
    this.renderScale = new Float32Array(maxEnemies); // Visual render scale
    
    // Behavior properties
    this.moveSpeed = new Float32Array(maxEnemies); // Movement speed
    this.chaseRadius = new Float32Array(maxEnemies); // Chase detection radius
    this.shootRange = new Float32Array(maxEnemies); // Shooting range
    this.cooldown = new Float32Array(maxEnemies); // Shoot cooldown time
    this.currentCooldown = new Float32Array(maxEnemies); // Current cooldown timer
    this.damage = new Float32Array(maxEnemies); // Bullet damage
    this.bulletSpeed = new Float32Array(maxEnemies); // Bullet speed
    this.bulletLifetime = new Float32Array(maxEnemies); // Bullet lifetime in seconds
    this.projectileCount = new Uint8Array(maxEnemies); // Number of projectiles to fire
    this.projectileSpread = new Float32Array(maxEnemies); // Angular spread for multiple projectiles
    this.canChase = new Uint8Array(maxEnemies); // Whether enemy can chase (1 or 0)
    this.canShoot = new Uint8Array(maxEnemies); // Whether enemy can shoot (1 or 0)
    
    // Sprite for projectiles fired by this enemy
    this.bulletSpriteName = new Array(maxEnemies);

    // Map external entity ID -> internal type index
    this.enemyIdToTypeIndex = new Map();
    
    // New fields for visual effects
    this.flashTimer = new Float32Array(maxEnemies); // Timer for flash effect
    this.isFlashing = new Uint8Array(maxEnemies); // Whether enemy is flashing
    this.deathTimer = new Float32Array(maxEnemies); // Timer for death animation
    this.isDying = new Uint8Array(maxEnemies); // Whether enemy is dying

    // Mapping from ID to index for fast lookups
    this.idToIndex = new Map();

    // Track which world / map this enemy belongs to so the server can filter
    // them per-world when broadcasting.  We keep string identifiers (mapId).
    this.worldId = new Array(maxEnemies);

    // Debug: Timer for periodic position logging
    this.debugPositionTimer = 0;
    this.debugPositionInterval = 3.0; // Log every 3 seconds
    
    // Load enemy type definitions from entityDatabase, fallback to built-ins
    const dbEnemies = entityDatabase.getAll('enemies');
    if (dbEnemies.length > 0) {
      // Filter out bullet definitions (only load actual enemies with hp)
      const actualEnemies = dbEnemies.filter(e => e.hp);

      // Normalize definitions coming from JSON so core fields exist
      this.enemyTypes = actualEnemies.map((e, idx) => {
        const attack = e.attack || {};
        return {
          id: idx,
          name: e.name || e.id,
          spriteName: (e.sprite || '').replace(/^chars:/,'') || 'unknown',
          maxHealth: e.hp || e.health || 50,
          speed: e.speed || 10,
          damage: attack.damage || 10,
          width: e.width || 1,
          height: e.height || 1,
          renderScale: e.renderScale || 2,
          shootRange: attack.range || 120,
          shootCooldown: (attack.cooldown||1000) / 1000,
          bulletSpeed: attack.speed || 20,
          bulletLifetime: (attack.lifetime||2000) / 1000,
          projectileCount: attack.count || attack.projectileCount || 1,
          spread: (attack.spread||0) * Math.PI / 180,
          inaccuracy: (attack.inaccuracy||0) * Math.PI / 180,
          behavior: e.ai?.behavior || 'aggressive',
          bulletSpriteName: attack.sprite || null
        };
      });

      // IMPORTANT: Build ID→index mapping for EntityDB enemies (use filtered array indices)
      actualEnemies.forEach((e, idx) => {
        if (e.id) {
          this.enemyIdToTypeIndex.set(e.id, idx);
        }
      });

      console.log(`[EnemyManager] Loaded ${this.enemyTypes.length} enemy templates from EntityDB`);
      console.log(`[EnemyManager] Mapped ${this.enemyIdToTypeIndex.size} enemy IDs: ${Array.from(this.enemyIdToTypeIndex.keys()).join(', ')}`);
    } else {
      console.warn('[EnemyManager] EntityDB returned zero enemies – falling back to hard-coded defaults');
      this.enemyTypes = [
        {
          id: 0,
          name: 'Goblin',
          spriteName: 'goblin',
          maxHealth: 30,
          speed: 15,
          damage: 5,
          width: 1,
          height: 1,
          renderScale: 2,
          shootRange: 50,
          shootCooldown: 2.5,
          bulletSpeed: 25,
          bulletLifetime: 2.0,
          projectileCount: 1,
          spread: 0,
          inaccuracy: 0.1,
          behavior: 'aggressive',
          dropTable:[
            { id:1001, prob:0.3, bagType:0 },  // Ironveil Sword in white bag
            { id:1004, prob:0.1, bagType:1 },  // Greenwatch Sword in brown bag
            { id:1007, prob:0.02, bagType:2 } // Skysteel Sword in purple bag
          ]
        },
        {
          id: 1,
          name: 'Orc',
          spriteName: 'orc',
          maxHealth: 50,
          speed: 12,
          damage: 8,
          width: 1,
          height: 1,
          renderScale: 2,
          shootRange: 60,
          shootCooldown: 2.0,
          bulletSpeed: 30,
          bulletLifetime: 2.5,
          projectileCount: 1,
          spread: 0,
          inaccuracy: 0.05,
          behavior: 'defensive'
        },
        {
          id: 2,
          name: 'Skeleton',
          spriteName: 'skeleton',
          maxHealth: 25,
          speed: 20,
          damage: 6,
          width: 1,
          height: 1,
          renderScale: 2,
          shootRange: 80,
          shootCooldown: 1.8,
          bulletSpeed: 35,
          bulletLifetime: 3.0,
          projectileCount: 1,
          spread: 0,
          inaccuracy: 0.08,
          behavior: 'patrol'
        },
        {
          id: 3,
          name: 'Troll',
          spriteName: 'troll',
          maxHealth: 100,
          speed: 8,
          damage: 15,
          width: 1,
          height: 1,
          renderScale: 2,
          shootRange: 40,
          shootCooldown: 3.0,
          bulletSpeed: 20,
          bulletLifetime: 2.0,
          projectileCount: 3,
          spread: 0.5,
          inaccuracy: 0.2,
          behavior: 'guard'
        },
        {
          id: 4,
          name: 'Wizard',
          spriteName: 'wizard',
          maxHealth: 40,
          speed: 10,
          damage: 12,
          width: 1,
          height: 1,
          renderScale: 2,
          shootRange: 100,
          shootCooldown: 2.2,
          bulletSpeed: 40,
          bulletLifetime: 4.0,
          projectileCount: 1,
          spread: 0,
          inaccuracy: 0.02,
          behavior: 'ranged'
        }
      ];
    }
    
    // Initialize behavior system
    this.behaviorSystem = new BehaviorSystem();

    // Storage for parsed behaviour trees (per type index)
    this.behaviourTreeRunners = [];

    // Load additional enemy definitions from the entity database (e.g. Red Demon)
    this._loadExternalEnemyDefs();

    this.rootStateCache = new Map(); // type -> rootState for lazy registration
  }

  /**
   * Load enemy definitions from public/assets/entities/enemies.json and append to enemyTypes
   * Allows designers to author enemies in JSON without touching server code.
   * Populates enemyIdToTypeIndex for quick lookup.
   * Currently uses a basic mapping; all external enemies get the generic BasicEnemy behavior template.
   */
  _loadExternalEnemyDefs() {
    try {
      const customDir = path.join(process.cwd(), 'public','assets','enemies-custom');
      const customDefs = loadEnemyDefinitions(customDir);
      customDefs.forEach(def=>{
        const {template, rootState} = compileEnemy(def);
        const idx = this.enemyTypes.length;
        this.enemyTypes.push(template);
        this.enemyIdToTypeIndex.set(def.id, idx);
        if(rootState){
          this.rootStateCache.set(idx, rootState); // defer registration until first spawn
        }
      });
      if(customDefs.length) console.log(`[EnemyManager] Loaded ${customDefs.length} custom enemy JSON definitions`);
    }catch(err){console.error('[EnemyManager] custom enemy load failed',err);}
  }

  /**
   * Convenience helper to spawn by entity ID defined in JSON (e.g. 101 → Red Demon)
   */
  spawnEnemyById(entityId, x, y, worldId='default') {
    const typeIdx = this.enemyIdToTypeIndex.get(Number(entityId)) ?? this.enemyIdToTypeIndex.get(entityId);
    if (typeIdx === undefined) {
      console.warn(`spawnEnemyById: unknown entity ID ${entityId} – defaulting to type 0`);
      return this.spawnEnemy(0, x, y, worldId);
    }
    return this.spawnEnemy(typeIdx, x, y, worldId);
  }

  /**
   * Spawn a new enemy
   * @param {number} type - Enemy type (0-4)
   * @param {number} x - X position to spawn
   * @param {number} y - Y position to spawn
   * @param {string} worldId - World ID to associate with the enemy
   * @returns {string} The ID of the new enemy
   */
  spawnEnemy(type, x, y, worldId='default') {
    if (this.enemyCount >= this.maxEnemies) {
      console.warn('EnemyManager: Maximum enemy capacity reached');
      return null;
    }
    
    // Validate enemy type
    if (type < 0 || type >= this.enemyTypes.length) {
      type = 0; // Default to type 0 if invalid
    }
    
    // Get default values for this enemy type
    const defaults = this.enemyTypes[type];
    
    // Assign unique ID and store in manager
    const enemyId = `enemy_${this.nextEnemyId++}`;
    const index = this.enemyCount++;
    
    // Store basic properties
    this.id[index] = enemyId;
    this.x[index] = x;
    this.y[index] = y;
    this.width[index] = defaults.width;
    this.height[index] = defaults.height;
    this.type[index] = type;
    this.health[index] = defaults.maxHealth;
    this.maxHealth[index] = defaults.maxHealth;
    this.renderScale[index] = defaults.renderScale;
    
    // Store behavior properties
    this.moveSpeed[index] = defaults.speed;
    this.chaseRadius[index] = defaults.shootRange;
    this.shootRange[index] = defaults.shootRange;
    this.cooldown[index] = defaults.shootCooldown;
    this.currentCooldown[index] = 0; // Start with no cooldown
    this.damage[index] = defaults.damage;
    this.bulletSpeed[index] = defaults.bulletSpeed;
    this.bulletLifetime[index] = defaults.bulletLifetime;
    this.projectileCount[index] = defaults.projectileCount;
    this.projectileSpread[index] = defaults.spread;
    this.canChase[index] = 1;
    this.canShoot[index] = 1;
    // Store bullet sprite for rendering on clients
    this.bulletSpriteName[index] = defaults.bulletSpriteName || null;
    
    // Initialize visual effect timers
    this.flashTimer[index] = 0;
    this.isFlashing[index] = 0;
    this.deathTimer[index] = 0;
    this.isDying[index] = 0;
    
    // Store ID to index mapping
    this.idToIndex.set(enemyId, index);
    
    // Lazy-register behavior template if not yet registered
    if(!this.behaviorSystem.behaviorTemplates.has(type)){
      const rs = this.rootStateCache.get(type);
      if(rs){
        this.behaviorSystem.registerBehaviorTemplate(type, rs);
      }
    }
    
    // Initialize behavior for this enemy
    this.behaviorSystem.initBehavior(index, type);
    
    // Ensure BT runner array sized
    if (!this.behaviourTreeRunners[type]) {
        this.behaviourTreeRunners[type] = null;
    }
    
    // Record world ownership
    this.worldId[index] = worldId;
    
    console.log(`Spawned enemy ${enemyId} of type ${type} at position (${x.toFixed(2)}, ${y.toFixed(2)}), health: ${defaults.maxHealth}`);
    
    return enemyId;
  }

  /**
   * Update all enemies
   * @param {number} deltaTime - Time elapsed since last update in seconds
   * @param {Object} bulletManager - Reference to the bullet manager for shooting
   * @param {Object} target - Optional target entity (e.g., player)
   * @param {Object} mapManager - Optional map manager for collision checks
   * @returns {number} The number of active enemies
   */
  update(deltaTime, bulletManager, target = null, mapManager = null) {
    // Skip update if no target or bullet manager
    if (!target) return this.getActiveEnemyCount();
    
    // Count of active enemies
    let activeCount = 0;
    
    for (let i = 0; i < this.enemyCount; i++) {
      // Skip enemies that belong to a different world – but only if *both* sides
      // have a defined worldId.  This prevents the AI from idling when the player
      // hasn't been assigned one yet (proc-gen default map).
      if (target && this.worldId[i] && target.worldId && this.worldId[i] !== target.worldId) {
        continue;
      }
      
      // Skip dead enemies
      if (this.health[i] <= 0) {
        if (this.isDying[i]) {
          // Update death animation
          this.updateDeathAnimation(i, deltaTime);
          activeCount++; // Still count dying enemies as active
        }
        continue;
      }
      
      activeCount++;
      
      // Update cooldowns
      if (this.currentCooldown[i] > 0) {
        this.currentCooldown[i] -= deltaTime;
      }
      
      // Update flash effect
      if (this.isFlashing[i]) {
        this.updateFlashEffect(i, deltaTime);
      }
      
      // Store previous position for potential rollback if collision occurs
      const prevX = this.x[i];
      const prevY = this.y[i];

      // Update enemy behavior using the behavior system (expects bulletManager before target)
      const runner = this.behaviourTreeRunners[this.type[i]];
      if (runner) {
        runner.tick(i, this, bulletManager, target, deltaTime);
      } else {
        this.behaviorSystem.updateBehavior(i, this, bulletManager, target, deltaTime);
      }

      // Prevent enemies from moving into walls / outside of map bounds
      if (mapManager && mapManager.isWallOrOutOfBounds) {
        if (mapManager.isWallOrOutOfBounds(this.x[i], this.y[i])) {
          // Simple resolution: revert to previous valid position
          this.x[i] = prevX;
          this.y[i] = prevY;

          // Optional: nudge enemy in random safe direction
        }
      }

      // Check for collision with other enemies
      this.checkEnemyCollision(i);
    }

    // Debug: Periodic position logging
    this.debugPositionTimer += deltaTime;
    if (this.debugPositionTimer >= this.debugPositionInterval) {
      this.debugPositionTimer = 0;

      if (activeCount > 0) {
        console.log(`\n📍 [ENEMY POSITIONS] Active enemies: ${activeCount}`);
        for (let i = 0; i < this.enemyCount; i++) {
          if (this.health[i] > 0 && !this.isDying[i]) {
            const typeName = this.enemyTypes[this.type[i]]?.name || `Type${this.type[i]}`;
            const stateName = this.behaviorSystem?.currentState[i]?.name || 'unknown';
            console.log(`  Enemy ${i} (${typeName}): pos=(${this.x[i].toFixed(2)}, ${this.y[i].toFixed(2)}), hp=${this.health[i].toFixed(0)}/${this.maxHealth[i]}, state=${stateName}`);
          }
        }
      }
    }

    return activeCount;
  }
  
  /**
   * Apply a hit effect (flash) to an enemy
   * @param {number} index - Enemy index
   */
  applyHitEffect(index) {
    this.isFlashing[index] = 1;
    this.flashTimer[index] = 0.1; // Flash for 100ms
  }

  /**
   * Check for collisions with other enemies and apply separation
   * @param {number} index - Enemy index to check
   */
  checkEnemyCollision(index) {
    // Check against all other enemies
    for (let j = 0; j < this.enemyCount; j++) {
      // Skip self and dead/dying enemies
      if (j === index || this.health[j] <= 0 || this.isDying[j]) continue;

      // Calculate distance between enemies
      const dx = this.x[index] - this.x[j];
      const dy = this.y[index] - this.y[j];
      const distSquared = dx * dx + dy * dy;
      const dist = Math.sqrt(distSquared);

      // Calculate minimum distance (sum of radii)
      const minDist = (this.width[index] + this.width[j]) / 2;

      // If colliding (distance is less than minimum)
      if (dist < minDist && dist > 0.01) {
        // Calculate separation force (push enemies apart)
        const overlap = minDist - dist;
        const pushForce = overlap * 0.5; // Each enemy gets pushed half the overlap

        // Normalize direction vector
        const angle = Math.atan2(dy, dx);

        // Apply separation force to current enemy (push away from other enemy)
        this.x[index] += Math.cos(angle) * pushForce;
        this.y[index] += Math.sin(angle) * pushForce;
      }
    }
  }

  /**
   * Update flash effect
   * @param {number} index - Enemy index
   * @param {number} deltaTime - Time elapsed since last update
   */
  updateFlashEffect(index, deltaTime) {
    if (this.flashTimer[index] > 0) {
      this.flashTimer[index] -= deltaTime;
    } else {
      this.isFlashing[index] = 0;
    }
  }
  
  /**
   * Start the death animation for an enemy
   * @param {number} index - Enemy index
   */
  startDeathAnimation(index) {
    this.isDying[index] = 1;
    this.deathTimer[index] = 0.5; // Death animation duration
  }
  
  /**
   * Update death animation
   * @param {number} index - Enemy index
   * @param {number} deltaTime - Time elapsed since last update
   */
  updateDeathAnimation(index, deltaTime) {
    if (this.deathTimer[index] > 0) {
      this.deathTimer[index] -= deltaTime;
    } else {
      // Animation finished, remove the enemy
      this.removeEnemy(index);
    }
  }
  
  /**
   * Apply damage to an enemy
   * @param {number} index - Enemy index
   * @param {number} damage - Amount of damage to apply
   * @returns {Object} Result with new health and whether the enemy was killed
   */
  applyDamage(index, damage) {
    if (index < 0 || index >= this.enemyCount) {
      return { valid: false, reason: 'Invalid enemy index' };
    }
    
    // Apply damage
    this.health[index] -= damage;
    
    // Apply hit effect
    this.applyHitEffect(index);
    
    // Check if killed
    const killed = this.health[index] <= 0;
    if (killed) {
      // Start death animation
      this.startDeathAnimation(index);
      
      // Call onDeath to handle death effects
      this.onDeath(index);
    }
    
    return {
      valid: true,
      health: this.health[index],
      killed: killed
    };
  }

  /**
   * Remove an enemy using the swap-and-pop technique
   * @param {number} index - Index of enemy to remove
   */
  removeEnemy(index) {
    const last = this.enemyCount - 1;

    // Remove from ID mapping
    const id = this.id[index];
    this.idToIndex.delete(id);

    if (index !== last) {
      // Swap with the last enemy - basic properties
      this.id[index] = this.id[last];
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
      this.type[index] = this.type[last];
      this.health[index] = this.health[last];
      this.maxHealth[index] = this.maxHealth[last];
      this.renderScale[index] = this.renderScale[last];

      // Swap behavior properties
      this.moveSpeed[index] = this.moveSpeed[last];
      this.chaseRadius[index] = this.chaseRadius[last];
      this.shootRange[index] = this.shootRange[last];
      this.cooldown[index] = this.cooldown[last];
      this.currentCooldown[index] = this.currentCooldown[last];
      this.damage[index] = this.damage[last];
      this.bulletSpeed[index] = this.bulletSpeed[last];
      this.bulletLifetime[index] = this.bulletLifetime[last];
      this.projectileCount[index] = this.projectileCount[last];
      this.projectileSpread[index] = this.projectileSpread[last];
      this.canChase[index] = this.canChase[last];
      this.canShoot[index] = this.canShoot[last];

      // Swap visual effect properties
      this.flashTimer[index] = this.flashTimer[last];
      this.isFlashing[index] = this.isFlashing[last];
      this.deathTimer[index] = this.deathTimer[last];
      this.isDying[index] = this.isDying[last];

      // CRITICAL: Swap behavior system data to maintain correct behavior state
      // Without this, the swapped enemy would inherit the dead enemy's behavior!
      this.behaviorSystem.swapBehaviorData(index, last);

      // Update ID mapping for the swapped enemy
      this.idToIndex.set(this.id[index], index);
    }

    this.enemyCount--;
  }
  
  /**
   * Find enemy index by ID
   * @param {string} enemyId - Enemy ID to find
   * @returns {number} Index of enemy or -1 if not found
   */
  findIndexById(enemyId) {
    // Use the ID to index mapping
    const index = this.idToIndex.get(enemyId);
    return index !== undefined ? index : -1;
  }
  
  /**
   * Handle enemy death
   * @param {number} index - Index of enemy that died
   * @param {string} killedBy - ID of player who killed the enemy
   */
  onDeath(index, killedBy) {
    if (!this._bagManager || !this.itemManager) return;
    const enemyType = this.type[index];
    const template = this.enemyTypes[enemyType] || {};
    const dropTable = template.dropTable || [];
    const {items, bagType} = rollDropTable(dropTable);
    if(items.length===0) return;
    const itemInstanceIds = items.map(defId=>{
      const inst = this.itemManager.createItem(defId,{x:this.x[index],y:this.y[index]});
      return inst?.id;
    }).filter(Boolean);
    if(itemInstanceIds.length===0) return;
    this._bagManager.spawnBag(this.x[index], this.y[index], itemInstanceIds, this.worldId[index], 300, bagType);
  }

  /**
   * Get number of active enemies
   * @returns {number} Count of active enemies
   */
  getActiveEnemyCount() {
    return this.enemyCount;
  }

  /**
   * Clean up resources
   */
  cleanup() {
    this.enemyCount = 0;
    this.idToIndex.clear();
  }

  /**
   * Get enemy data array for network transmission
   * @param {string} filterWorldId - World ID to filter enemies by
   * @returns {Array} Array of enemy data objects
   */
  getEnemiesData(filterWorldId=null) {
    const enemies = [];
    
    for (let i = 0; i < this.enemyCount; i++) {
      if (filterWorldId && this.worldId[i] !== filterWorldId) continue;
      enemies.push({
        id: this.id[i],
        x: this.x[i],
        y: this.y[i],
        width: this.width[i],
        height: this.height[i],
        type: this.type[i],
        spriteName: this.enemyTypes[this.type[i]]?.spriteName || null,
        health: this.health[i],
        maxHealth: this.maxHealth[i],
        renderScale: this.renderScale[i],
        isFlashing: this.isFlashing[i],
        isDying: this.isDying[i],
        deathStage: this.isDying[i] ? Math.floor((1 - this.deathTimer[i] / 0.5) * 4) : 0,
        worldId: this.worldId[i]
      });
    }
    
    return enemies;
  }

  /**
   * Add a new enemy - alias for spawnEnemy for backward compatibility
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {number} type - Enemy type (0-4)
   * @returns {string} The ID of the new enemy
   */
  addEnemy(x, y, type = 0) {
    return this.spawnEnemy(type, x, y);
  }
}