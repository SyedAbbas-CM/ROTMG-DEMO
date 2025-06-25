// File: /src/Managers/EnemyManager.js

import BehaviorSystem from './BehaviorSystem.js';
import { entityDatabase } from './assets/EntityDatabase.js';
import fs from 'fs';
import path from 'path';
import { parseBehaviourTree, BehaviourTreeRunner } from './BehaviorTree.js';

/**
 * EnemyManager handles enemy creation, updating, and removal.
 * Uses Structure of Arrays (SoA) for data layout optimization.
 */
export default class EnemyManager {
  /**
   * Creates an enemy manager
   * @param {number} maxEnemies - Maximum number of enemies to allow
   */
  constructor(maxEnemies = 1000) {
    this.maxEnemies = maxEnemies;
    this.enemyCount = 0;
    this.nextEnemyId = 1; // For assigning unique IDs

    // SoA data layout for position and basic properties
    this.id = new Array(maxEnemies);         // Unique enemy IDs
    this.x = new Float32Array(maxEnemies);   // X position
    this.y = new Float32Array(maxEnemies);   // Y position
    this.width = new Float32Array(maxEnemies);  // Collision width
    this.height = new Float32Array(maxEnemies); // Collision height
    this.type = new Uint8Array(maxEnemies);  // Enemy type (0-4)
    this.health = new Float32Array(maxEnemies); // Current health
    this.maxHealth = new Float32Array(maxEnemies); // Maximum health
    
    // Behavior properties
    this.moveSpeed = new Float32Array(maxEnemies); // Movement speed
    this.chaseRadius = new Float32Array(maxEnemies); // Chase detection radius
    this.shootRange = new Float32Array(maxEnemies); // Shooting range
    this.cooldown = new Float32Array(maxEnemies); // Shoot cooldown time
    this.currentCooldown = new Float32Array(maxEnemies); // Current cooldown timer
    this.damage = new Float32Array(maxEnemies); // Bullet damage
    this.bulletSpeed = new Float32Array(maxEnemies); // Bullet speed
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
    
    // Load enemy type definitions from entityDatabase, fallback to built-ins
    const dbEnemies = entityDatabase.getAll('enemies');
    if (dbEnemies.length > 0) {
      // Normalize definitions coming from JSON so core fields exist
      this.enemyTypes = dbEnemies.map((e, idx) => {
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
      console.log(`[EnemyManager] Loaded ${this.enemyTypes.length} enemy templates from EntityDB`);
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
          behavior: 'aggressive'
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
  }

  /**
   * Load enemy definitions from public/assets/entities/enemies.json and append to enemyTypes
   * Allows designers to author enemies in JSON without touching server code.
   * Populates enemyIdToTypeIndex for quick lookup.
   * Currently uses a basic mapping; all external enemies get the generic BasicEnemy behavior template.
   */
  _loadExternalEnemyDefs() {
    try {
      const entitiesDir = path.join(process.cwd(), 'public', 'assets', 'entities');
      const enemyPath = path.join(entitiesDir, 'enemies.json');
      if (!fs.existsSync(enemyPath)) return;

      const raw = JSON.parse(fs.readFileSync(enemyPath, 'utf8'));
      // Build a quick lookup for bullet templates in same file (ids that appear in other enemies' attack.bulletId)
      const bulletLookup = new Map();
      raw.forEach(e => {
        if (!e.attack && !e.hp) {
          // Treat as projectile / auxiliary record (e.g. Red Demon Bullet)
          bulletLookup.set(e.id, e);
        }
      });

      raw.forEach(ent => {
        if (!ent.attack) return; // Skip non-enemy rows (likely projectiles)

        // Skip duplicates
        if (this.enemyIdToTypeIndex.has(ent.id)) return;

        const bulletEnt = bulletLookup.get(ent.attack.bulletId);

        const bulletSpeed = bulletEnt?.speed || 8;
        const bulletLifetime = (bulletEnt?.lifetime || 1500) / 1000;

        // Determine projectile pattern
        const projCount = ent.attack.projectileCount || ent.attack.count || (ent.id === 101 ? 5 : 1);
        const spreadDeg = ent.attack.spread !== undefined ? ent.attack.spread : (projCount > 1 ? 45 : 0);
        const spreadRad = spreadDeg * Math.PI / 180;

        const template = {
          id: ent.id,
          name: ent.name,
          spriteName: (ent.sprite || '').trim(),
          maxHealth: ent.hp || 100,
          speed: ent.speed || 10,
          damage: 10,
          width: 1,
          height: 1,
          renderScale: 2,
          shootRange: 120,
          shootCooldown: (ent.attack.cooldown || 800) / 1000,
          bulletSpeed,
          bulletLifetime,
          projectileCount: projCount,
          spread: spreadRad,
          inaccuracy: 0,
          behavior: 'aggressive',
          bulletSpriteName: bulletEnt?.sprite || null
        };

        const idx = this.enemyTypes.length;
        this.enemyTypes.push(template);
        this.enemyIdToTypeIndex.set(ent.id, idx);

        // Register a basic behavior for this new enemy type so AI functions correctly
        const shootBehavior = this.behaviorSystem.createCustomShootBehavior({
          projectileCount: template.projectileCount,
          spread: template.spread,
          shootCooldown: template.shootCooldown
        });
        this.behaviorSystem.registerBehaviorTemplate(idx, shootBehavior);

        // Parse behaviour tree if provided
        if (ent.behaviourTree) {
          const root = parseBehaviourTree(ent.behaviourTree);
          this.behaviourTreeRunners[idx] = new BehaviourTreeRunner(root);
        }
      });

      if (this.enemyIdToTypeIndex.size > 0) {
        console.log(`[EnemyManager] Loaded ${this.enemyIdToTypeIndex.size} external enemy definitions from entity DB`);
      }
    } catch (err) {
      console.error('[EnemyManager] Failed to load external enemy definitions', err);
    }
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
    
    // Store behavior properties
    this.moveSpeed[index] = defaults.speed;
    this.chaseRadius[index] = defaults.shootRange;
    this.shootRange[index] = defaults.shootRange;
    this.cooldown[index] = defaults.shootCooldown;
    this.currentCooldown[index] = 0; // Start with no cooldown
    this.damage[index] = defaults.damage;
    this.bulletSpeed[index] = defaults.bulletSpeed;
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
      
      // Swap behavior properties
      this.moveSpeed[index] = this.moveSpeed[last];
      this.chaseRadius[index] = this.chaseRadius[last];
      this.shootRange[index] = this.shootRange[last];
      this.cooldown[index] = this.cooldown[last];
      this.currentCooldown[index] = this.currentCooldown[last];
      this.damage[index] = this.damage[last];
      this.bulletSpeed[index] = this.bulletSpeed[last];
      this.projectileCount[index] = this.projectileCount[last];
      this.projectileSpread[index] = this.projectileSpread[last];
      this.canChase[index] = this.canChase[last];
      this.canShoot[index] = this.canShoot[last];
      
      // Swap visual effect properties
      this.flashTimer[index] = this.flashTimer[last];
      this.isFlashing[index] = this.isFlashing[last];
      this.deathTimer[index] = this.deathTimer[last];
      this.isDying[index] = this.isDying[last];
      
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
    // The actual removal is now handled by the death animation
    // Additional death effects or drops can be added here
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