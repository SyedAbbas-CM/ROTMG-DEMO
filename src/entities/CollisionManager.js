// File: /src/Managers/CollisionManager.js

/**
 * CollisionManager.js
 * Handles collision detection and processing between entities
 */

export default class CollisionManager {
  /**
   * Creates a collision manager
   * @param {Object} bulletManager - The bullet manager instance
   * @param {Object} enemyManager - The enemy manager instance
   * @param {Object} mapManager - The map manager (optional)
   */
  constructor(bulletManager, enemyManager, mapManager = null) {
    this.bulletManager = bulletManager;
    this.enemyManager = enemyManager;
    this.mapManager = mapManager;
    
    // Tracking processed collisions to avoid duplicates
    this.processedCollisions = new Map(); // collisionId -> timestamp
    this.cleanupInterval = setInterval(() => this.cleanupProcessedCollisions(), 10000);
  }
  
  /**
   * Check for all collisions in the current state
   * Called on each update cycle
   * @param {number} deltaTime - Time elapsed since last update in seconds
   * @param {Array} players - Array of player objects in this world
   */
  checkCollisions(deltaTime = 0.033, players = []) {
    // Skip if managers aren't properly initialized
    if (!this.bulletManager || !this.enemyManager) return;
    
    // For each bullet, check collision with enemies
    for (let bi = 0; bi < this.bulletManager.bulletCount; bi++) {
      // Skip expired bullets
      if (this.bulletManager.life[bi] <= 0) continue;
      
      const bulletX = this.bulletManager.x[bi];
      const bulletY = this.bulletManager.y[bi];
      const bulletWidth = this.bulletManager.width[bi];
      const bulletHeight = this.bulletManager.height[bi];
      const bulletId = this.bulletManager.id[bi];
      const bulletOwnerId = this.bulletManager.ownerId[bi];
      
      /* ------- Wall / obstacle collision (sub-stepped) ------- */
      // RE-ENABLED WITH DETAILED LOGGING TO FIND ROOT CAUSE
      if (this.mapManager && this.mapManager.isWallOrOutOfBounds) {
        const vx = this.bulletManager.vx[bi];
        const vy = this.bulletManager.vy[bi];

        // CRITICAL FIX: Calculate actual movement for THIS FRAME using deltaTime
        // vx/vy are in tiles/SECOND, so multiply by deltaTime to get tiles moved this frame
        const dx = vx * deltaTime;  // e.g., -10 tiles/sec * 0.033s = -0.33 tiles
        const dy = vy * deltaTime;

        // Maximum distance bullet will move this tick (tile-units)
        const maxDelta = Math.max(Math.abs(dx), Math.abs(dy));
        // Break the motion into ≤0.5-tile chunks – prevents tunnelling
        const steps = Math.max(1, Math.ceil(maxDelta / 0.5));

        let bxStep = bulletX;
        let byStep = bulletY;
        let collided = false;

        for (let s = 0; s < steps; s++) {
          bxStep += dx / steps;
          byStep += dy / steps;

          if (this.mapManager.isWallOrOutOfBounds(bxStep, byStep)) {
            collided = true;

            // Enhanced debug logging for bullet collision with coordinates
            const tile = this.mapManager.getTile ? this.mapManager.getTile(Math.floor(bxStep), Math.floor(byStep)) : null;
            const tileType = tile ? tile.type : 'UNKNOWN';
            const isOutOfBounds = bxStep < 0 || byStep < 0 ||
                                  (this.mapManager.width && bxStep >= this.mapManager.width) ||
                                  (this.mapManager.height && byStep >= this.mapManager.height);
            const reason = isOutOfBounds ? 'OUT_OF_BOUNDS' : (tile ? 'WALL' : 'MISSING_CHUNK');

            const tileX = Math.floor(bxStep);
            const tileY = Math.floor(byStep);

            // DIAGNOSTIC: Always log collisions at X=8-11 range
            if (bxStep >= 8 && bxStep <= 11) {
              console.error(`❌ [COLLISION AT X=9!] Bullet ${bulletId} at (${bxStep.toFixed(4)}, ${byStep.toFixed(4)}), Tile: (${tileX}, ${tileY}), Reason: ${reason}, TileType: ${tileType}`);
              if (tile) {
                console.error(`  Tile properties:`, JSON.stringify(tile));
              }
            } else {
              console.log(`[SERVER BULLET] ID: ${bulletId}, Pos: (${bxStep.toFixed(4)}, ${byStep.toFixed(4)}), Tile: (${tileX}, ${tileY}), Reason: ${reason}`);
            }
            console.log(`  Start Pos: (${bulletX.toFixed(4)}, ${bulletY.toFixed(4)})`);
            console.log(`  Velocity: (${vx.toFixed(4)}, ${vy.toFixed(4)}) tiles/sec`);
            console.log(`  Movement this frame: (${dx.toFixed(4)}, ${dy.toFixed(4)}) tiles (deltaTime=${deltaTime.toFixed(4)}s)`);
            console.log(`  Map Bounds: (0, 0) to (${this.mapManager.width}, ${this.mapManager.height})`);
            console.log(`  Tile Type: ${tileType}`);
            console.log(`  Out of bounds: bxStep < 0: ${bxStep < 0}, byStep < 0: ${byStep < 0}, bxStep >= width: ${bxStep >= this.mapManager.width}, byStep >= height: ${byStep >= this.mapManager.height}`);

            break;
          }
        }

        if (collided) {
          // CRITICAL FIX: Update bullet position to MAP BOUNDARY before marking for removal
          // Clamp to [0, mapWidth] and [0, mapHeight] so the Server.js clamp() function
          // doesn't filter it out (clamp rejects x < 0 or y < 0)
          // This allows clients to see bullets reach the boundary instead of disappearing at X≈9.5
          const mapWidth = this.mapManager.width || 512;
          const mapHeight = this.mapManager.height || 512;

          this.bulletManager.x[bi] = Math.max(0, Math.min(bxStep, mapWidth - 0.01));
          this.bulletManager.y[bi] = Math.max(0, Math.min(byStep, mapHeight - 0.01));

          this.bulletManager.markForRemoval(bi);
          if (this.bulletManager.registerRemoval) {
            this.bulletManager.registerRemoval('wallHit');
          }
          continue; // Skip enemy checks for this bullet
        }
      }

      /* ------- Object collision (trees, boulders, etc.) ------- */
      // RE-ENABLED: Object collision for trees, boulders, etc.
      if (this.mapManager && this.mapManager.getObjects) {
        const worldId = this.bulletManager.worldId[bi];
        const worldObjects = this.mapManager.getObjects(worldId) || [];

        let objectCollision = false;
        for (const obj of worldObjects) {
          // Only check non-walkable objects (trees, boulders, etc.)
          // walkable: false means it blocks movement/bullets
          if (obj.walkable !== false) continue;

          // Check AABB collision between bullet and object
          // FIX: Default to 0.4 tile size (40%) to match typical sprite sizes
          // Objects don't have width/height defined, so we need accurate defaults
          const objX = obj.x || 0;
          const objY = obj.y || 0;
          const objWidth = obj.width || 0.4;   // 40% of tile (0.4 tiles) - matches sprite size
          const objHeight = obj.height || 0.4;  // 40% of tile (0.4 tiles) - matches sprite size

          // CRITICAL: checkAABBCollision expects CENTER coordinates
          // Objects are positioned at tile coordinates (top-left), so add 0.5 to get tile center
          const objCenterX = objX + 0.5;  // Tile center X
          const objCenterY = objY + 0.5;  // Tile center Y

          if (this.checkAABBCollision(
            bulletX, bulletY, bulletWidth, bulletHeight,
            objCenterX, objCenterY, objWidth, objHeight
          )) {
            // DEBUG: Log object collision details
            console.log(`[SERVER] 🎯 BULLET-OBJECT COLLISION:
  Bullet: pos=(${bulletX.toFixed(2)}, ${bulletY.toFixed(2)}), size=${bulletWidth.toFixed(2)}x${bulletHeight.toFixed(2)}
  Object: tile=(${objX}, ${objY}), center=(${objCenterX.toFixed(2)}, ${objCenterY.toFixed(2)}), size=${objWidth.toFixed(2)}x${objHeight.toFixed(2)}
  Object sprite: ${obj.sprite || 'none'}
  Distance: ${Math.sqrt(Math.pow(bulletX - objCenterX, 2) + Math.pow(bulletY - objCenterY, 2)).toFixed(2)} tiles`);

            objectCollision = true;
            break;
          }
        }

        if (objectCollision) {
          this.bulletManager.markForRemoval(bi);
          if (this.bulletManager.registerRemoval) {
            this.bulletManager.registerRemoval('objectHit');
          }
          continue; // Skip enemy checks for this bullet
        }
      }

      // Check for enemy collisions
      for (let ei = 0; ei < this.enemyManager.enemyCount; ei++) {
        // Skip enemies from a different world
        if (this.bulletManager.worldId[bi] !== this.enemyManager.worldId[ei]) {
          continue;
        }
        // Skip dead enemies
        if (this.enemyManager.health[ei] <= 0) continue;
        
        const enemyX = this.enemyManager.x[ei];
        const enemyY = this.enemyManager.y[ei];
        const enemyWidth = this.enemyManager.width[ei];
        const enemyHeight = this.enemyManager.height[ei];
        const enemyId = this.enemyManager.id[ei];
        
        // Skip self-collision (enemy bullets colliding with their owner)
        if (bulletOwnerId === enemyId) {
          // DIAGNOSTIC: Log self-collision skip occasionally
          if (Math.random() < 0.01) {
            console.log(`[COLLISION] Skipping self-collision: Enemy ${enemyId} won't collide with own bullet ${bulletId}`);
          }
          continue; // Bullet belongs to this enemy – ignore
        }

        // Skip enemy bullets hitting other enemies (they should only hit players)
        const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
        if (isEnemyBullet) {
          continue; // Check players instead (handled separately below)
        }
        
        // Check if bullet and enemy collide (AABB)
        if (this.checkAABBCollision(
          bulletX, bulletY, bulletWidth, bulletHeight,
          enemyX, enemyY, enemyWidth, enemyHeight
        )) {
          // Create collision ID to track this collision
          const collisionId = `${bulletId}_${enemyId}`;
          
          // Skip if already processed
          if (this.processedCollisions.has(collisionId)) continue;
          
          // Process this collision
          this.processCollision(bi, ei, bulletOwnerId);
          
          // Mark as processed
          this.processedCollisions.set(collisionId, Date.now());
          
          // Break the enemy loop since bullet hit something
          break;
        }
      }

      // NEW: Check enemy bullets against players
      const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
      if (isEnemyBullet && players && players.length > 0) {
        for (const player of players) {
          // Skip if player is dead or in different world
          if (!player || player.health <= 0) continue;
          if (this.bulletManager.worldId[bi] !== player.worldId) continue;

          const playerX = player.x;
          const playerY = player.y;
          const playerWidth = player.collisionWidth || 1;
          const playerHeight = player.collisionHeight || 1;
          const playerId = player.id;

          // Check if bullet and player collide (AABB)
          if (this.checkAABBCollision(
            bulletX, bulletY, bulletWidth, bulletHeight,
            playerX, playerY, playerWidth, playerHeight
          )) {
            // Create collision ID to track this collision
            const collisionId = `${bulletId}_${playerId}`;

            // Skip if already processed
            if (this.processedCollisions.has(collisionId)) continue;

            // Apply damage to player
            const damage = this.bulletManager.damage ? this.bulletManager.damage[bi] : 10;

            console.log(`[SERVER] ⚔️ ENEMY BULLET HIT PLAYER: Bullet ${bulletId} from ${bulletOwnerId} hit player ${playerId} for ${damage} damage`);
            console.log(`[SERVER]   Player health: ${player.health} → ${Math.max(0, player.health - damage)}`);

            // Apply damage
            if (typeof player.takeDamage === 'function') {
              player.takeDamage(damage);
            } else {
              player.health = Math.max(0, player.health - damage);
            }

            // Mark bullet for removal
            this.bulletManager.markForRemoval(bi);

            // Mark as processed
            this.processedCollisions.set(collisionId, Date.now());

            // Break the player loop since bullet hit something
            break;
          }
        }
      }
    }
  }

  /**
   * Validate client-reported collision (server-side)
   * @param {Object} data - Collision data from client
   * @returns {Object} Validation result
   */
  validateCollision(data) {
    const { bulletId, enemyId, timestamp, clientId } = data;

    console.log(`[SERVER] 📨 COLLISION REPORT received from client ${clientId}: Bullet ${bulletId} vs Enemy ${enemyId}`);

    // Find bullet and enemy indices using IDs
    const bulletIndex = this.findBulletIndex(bulletId);
    const enemyIndex = this.findEnemyIndex(enemyId);

    // Check if both entities exist
    if (bulletIndex === -1 || enemyIndex === -1) {
      console.log(`[SERVER] ❌ VALIDATION FAILED: Entity not found | Bullet index: ${bulletIndex}, Enemy index: ${enemyIndex}`);
      return {
        valid: false,
        reason: 'Entity not found',
        bulletId,
        enemyId
      };
    }
    
    // Ensure bullet and enemy are in the same world to prevent cross-realm hits
    if (this.bulletManager.worldId[bulletIndex] !== this.enemyManager.worldId[enemyIndex]) {
      console.log(`[SERVER] ❌ VALIDATION FAILED: Different world | Bullet world: ${this.bulletManager.worldId[bulletIndex]}, Enemy world: ${this.enemyManager.worldId[enemyIndex]}`);
      return {
        valid: false,
        reason: 'Different world',
        bulletId,
        enemyId
      };
    }

    // Check if this collision was already processed recently
    const collisionId = `${bulletId}_${enemyId}`;
    if (this.processedCollisions.has(collisionId)) {
      console.log(`[SERVER] ❌ VALIDATION FAILED: Already processed | CollisionId: ${collisionId}`);
      return {
        valid: false,
        reason: 'Already processed',
        bulletId,
        enemyId
      };
    }

    // Check if timestamp is reasonable (within 500ms from now)
    const now = Date.now();
    if (Math.abs(now - timestamp) > 500) {
      console.log(`[SERVER] ❌ VALIDATION FAILED: Timestamp too old | Age: ${Math.abs(now - timestamp)}ms`);
      return {
        valid: false,
        reason: 'Timestamp too old',
        bulletId,
        enemyId
      };
    }
    
    // Check for line of sight obstruction
    if (this.mapManager && this.mapManager.hasLineOfSight) {
      if (!this.mapManager.hasLineOfSight(
        this.bulletManager.x[bulletIndex],
        this.bulletManager.y[bulletIndex],
        this.enemyManager.x[enemyIndex],
        this.enemyManager.y[enemyIndex]
      )) {
        console.log(`[SERVER] ❌ VALIDATION FAILED: No line of sight`);
        return {
          valid: false,
          reason: 'No line of sight',
          bulletId,
          enemyId
        };
      }
    }

    // Check for actual collision (AABB)
    const bulletPos = {
      x: this.bulletManager.x[bulletIndex],
      y: this.bulletManager.y[bulletIndex]
    };
    const enemyPos = {
      x: this.enemyManager.x[enemyIndex],
      y: this.enemyManager.y[enemyIndex]
    };

    if (!this.checkAABBCollision(
      bulletPos.x,
      bulletPos.y,
      this.bulletManager.width[bulletIndex],
      this.bulletManager.height[bulletIndex],
      enemyPos.x,
      enemyPos.y,
      this.enemyManager.width[enemyIndex],
      this.enemyManager.height[enemyIndex]
    )) {
      console.log(`[SERVER] ❌ VALIDATION FAILED: No collision detected | ` +
                 `Bullet pos: (${bulletPos.x.toFixed(2)}, ${bulletPos.y.toFixed(2)}), ` +
                 `Enemy pos: (${enemyPos.x.toFixed(2)}, ${enemyPos.y.toFixed(2)})`);
      return {
        valid: false,
        reason: 'No collision detected',
        bulletId,
        enemyId
      };
    }

    console.log(`[SERVER] ✅ VALIDATION SUCCESS: Processing collision between Bullet ${bulletId} and Enemy ${enemyId}`);

    // Collision is valid - process it and store result
    const result = this.processCollision(bulletIndex, enemyIndex, clientId);

    // Mark as processed to avoid duplicates
    this.processedCollisions.set(collisionId, now);

    return {
      valid: true,
      ...result
    };
  }
  
  /**
   * Process a valid collision
   * @param {number} bulletIndex - Index of bullet in bulletManager
   * @param {number} enemyIndex - Index of enemy in enemyManager
   * @param {string|number} clientId - ID of the client that reported the collision
   * @returns {Object} Collision results
   */
  processCollision(bulletIndex, enemyIndex, clientId) {
    // Get bullet and enemy details
    const bulletId = this.bulletManager.id[bulletIndex];
    const enemyId = this.enemyManager.id[enemyIndex];

    const enemyHealthBefore = this.enemyManager.health[enemyIndex];

    console.log(`[SERVER] ⚔️ PROCESSING COLLISION: Bullet ${bulletId} hitting Enemy ${enemyId} | ` +
               `Enemy health before: ${enemyHealthBefore}`);

    // Calculate damage
    const damage = this.bulletManager.damage ?
      this.bulletManager.damage[bulletIndex] : 10; // Default damage

    // Apply damage to enemy (may return an object)
    const dmgResult = this.enemyManager.applyDamage(enemyIndex, damage);
    const remainingHealth = typeof dmgResult === 'object' && dmgResult !== null && 'health' in dmgResult
      ? dmgResult.health
      : (typeof dmgResult === 'number' ? dmgResult : this.enemyManager.health[enemyIndex]);

    console.log(`[SERVER] 💥 DAMAGE APPLIED: ${damage} damage | ` +
               `Enemy ${enemyId} health: ${enemyHealthBefore} → ${remainingHealth}`);

    // Remove bullet and register
    if (this.bulletManager.markForRemoval) {
      this.bulletManager.markForRemoval(bulletIndex);
    } else if (this.bulletManager.life) {
      // Alternative removal method
      this.bulletManager.life[bulletIndex] = 0;
    }

    if (this.bulletManager.registerRemoval) {
      this.bulletManager.registerRemoval('entityHit');
    }

    console.log(`[SERVER] 🔫 BULLET REMOVED: Bullet ${bulletId} marked for removal`);

    // Handle enemy death if needed
    let enemyKilled = false;
    if (remainingHealth <= 0) {
      enemyKilled = true;

      console.log(`[SERVER] 💀 ENEMY KILLED: Enemy ${enemyId} has been defeated by client ${clientId}!`);

      // Call enemy manager's death handler
      if (this.enemyManager.onDeath) {
        this.enemyManager.onDeath(enemyIndex, clientId);
      }
    }

    // Return collision result
    return {
      bulletId,
      enemyId,
      damage,
      enemyHealth: remainingHealth,
      enemyKilled,
      clientId
    };
  }
  
  /**
   * Find bullet index by ID
   * @param {string|number} bulletId - ID of the bullet
   * @returns {number} Bullet index or -1 if not found
   */
  findBulletIndex(bulletId) {
    // If bulletManager has a lookup method, use it
    if (this.bulletManager.findIndexById) {
      return this.bulletManager.findIndexById(bulletId);
    }
    
    // Otherwise search by ID array
    for (let i = 0; i < this.bulletManager.bulletCount; i++) {
      if (this.bulletManager.id[i] === bulletId) {
        return i;
      }
    }
    
    return -1;
  }
  
  /**
   * Find enemy index by ID
   * @param {string|number} enemyId - ID of the enemy
   * @returns {number} Enemy index or -1 if not found
   */
  findEnemyIndex(enemyId) {
    // If enemyManager has a lookup method, use it
    if (this.enemyManager.findIndexById) {
      return this.enemyManager.findIndexById(enemyId);
    }
    
    // Otherwise search by ID array
    for (let i = 0; i < this.enemyManager.enemyCount; i++) {
      if (this.enemyManager.id[i] === enemyId) {
        return i;
      }
    }
    
    return -1;
  }
  
  /**
   * AABB collision check (treats x,y as entity centres)
   */
  checkAABBCollision(ax, ay, awidth, aheight, bx, by, bwidth, bheight) {
    // Convert centre positions to min/max extents for each axis
    const aMinX = ax - awidth / 2;
    const aMaxX = ax + awidth / 2;
    const aMinY = ay - aheight / 2;
    const aMaxY = ay + aheight / 2;

    const bMinX = bx - bwidth / 2;
    const bMaxX = bx + bwidth / 2;
    const bMinY = by - bheight / 2;
    const bMaxY = by + bheight / 2;

    return (
      aMinX < bMaxX &&
      aMaxX > bMinX &&
      aMinY < bMaxY &&
      aMaxY > bMinY
    );
  }
  
  /**
   * Clean up old processed collisions to prevent memory leaks
   */
  cleanupProcessedCollisions() {
    const now = Date.now();
    const expiryTime = 5000; // 5 seconds
    
    for (const [id, timestamp] of this.processedCollisions.entries()) {
      if (now - timestamp > expiryTime) {
        this.processedCollisions.delete(id);
      }
    }
  }
  
  /**
   * Clean up when shutting down
   */
  cleanup() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
  }
}