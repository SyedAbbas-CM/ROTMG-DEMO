// File: /src/Managers/CollisionManager.js

/**
 * CollisionManager.js
 * Handles collision detection and processing between entities
 */

import CollisionValidator from './CollisionValidator.js';

export default class CollisionManager {
  /**
   * Creates a collision manager
   * @param {Object} bulletManager - The bullet manager instance
   * @param {Object} enemyManager - The enemy manager instance
   * @param {Object} mapManager - The map manager (optional)
   * @param {Object} lagCompensation - Lag compensation system (optional)
   * @param {Object} fileLogger - File logger for collision events (optional)
   */
  constructor(bulletManager, enemyManager, mapManager = null, lagCompensation = null, fileLogger = null, options = {}) {
    this.bulletManager = bulletManager;
    this.enemyManager = enemyManager;
    this.mapManager = mapManager;
    this.lagCompensation = lagCompensation;
    this.fileLogger = fileLogger;

    // PVP settings
    this.pvpEnabled = options.pvpEnabled || false;

    // Initialize collision validator
    this.collisionValidator = new CollisionValidator({ fileLogger });

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

    // Lag Compensation: Rewind player positions before collision detection
    let originalPositions = new Map();
    let rewindAmount = 0;
    let lagCompensationActive = false;

    if (this.lagCompensation && this.lagCompensation.enabled && players.length > 0) {
      // Calculate average RTT across all players for rewind amount
      let totalRTT = 0;
      let rttCount = 0;

      for (const player of players) {
        if (player.rtt && player.rtt > 0) {
          totalRTT += player.rtt;
          rttCount++;
        }
      }

      if (rttCount > 0) {
        const avgRTT = totalRTT / rttCount;
        rewindAmount = this.lagCompensation.calculateRewindAmount(avgRTT);

        if (rewindAmount > 0) {
          const currentTime = Date.now();
          originalPositions = this.lagCompensation.rewindAllPlayers(players, rewindAmount, currentTime);
          lagCompensationActive = true;

          if (this.lagCompensation.debug && players.length > 0) {
            console.log(`[LAG_COMP] Rewound ${players.length} players by ${rewindAmount.toFixed(2)}ms (avgRTT: ${avgRTT.toFixed(2)}ms)`);
          }
        }
      }
    }

    // Use try-finally to guarantee position restoration
    try {

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

            // Log wall collision to file instead of console
            if (this.fileLogger) {
              this.fileLogger.logCollision('WALL_HIT', {
                bulletId,
                position: { x: bxStep, y: byStep },
                tile: { x: tileX, y: tileY },
                reason,
                tileType,
                startPos: { x: bulletX, y: bulletY },
                velocity: { x: vx, y: vy },
                deltaTime
              });
            }

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
            if (this.fileLogger) {
              this.fileLogger.logCollision('OBJECT_HIT', {
                bulletId,
                bulletPos: { x: bulletX, y: bulletY },
                bulletSize: { w: bulletWidth, h: bulletHeight },
                object: { tile: { x: objX, y: objY }, center: { x: objCenterX, y: objCenterY }, sprite: obj.sprite }
              });
            }
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

          // DEBUG: Log positions for first few bullets (sample every 30 frames)
          if (bi === 0 && !this._posLogCount) this._posLogCount = 0;
          if (bi === 0 && this._posLogCount++ % 30 === 0) {
            const dist = Math.sqrt((bulletX - playerX) ** 2 + (bulletY - playerY) ** 2);
            console.log(`[COLLISION POS] Bullet(${bulletX.toFixed(1)},${bulletY.toFixed(1)}) sz=${bulletWidth.toFixed(2)} Player(${playerX.toFixed(1)},${playerY.toFixed(1)}) sz=${playerWidth} dist=${dist.toFixed(1)}`);
          }

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

            // DEBUG: Log collision hit
            console.log(`[COLLISION HIT] Bullet ${bulletId} hit player ${playerId}! Damage: ${damage}, Health: ${player.health} -> ${player.health - damage}`);

            if (this.fileLogger) {
              this.fileLogger.bulletHit(bulletId, 'player', playerId, damage, { x: playerX, y: playerY });
            }

            // Apply damage
            if (typeof player.takeDamage === 'function') {
              player.takeDamage(damage);
            } else {
              player.health = Math.max(0, player.health - damage);
            }

            // Check for player death
            if (player.health <= 0 && !player.isDead) {
              player.isDead = true;
              player.deathX = player.x;
              player.deathY = player.y;
              player.deathTimestamp = Date.now();

              if (this.fileLogger) {
                this.fileLogger.logCollision('PLAYER_DEATH', { playerId, position: { x: player.x, y: player.y } });
              }

              // Spawn grave object at death location
              if (this.mapManager && this.mapManager.addObject) {
                const graveId = `grave_${playerId}_${Date.now()}`;
                const graveObject = {
                  id: graveId,
                  type: 'grave',
                  x: player.deathX,
                  y: player.deathY,
                  playerId: playerId,
                  timestamp: player.deathTimestamp,
                  sprite: 'grave',
                  width: 2,
                  height: 2
                };

                const worldId = player.worldId || 'overworld';
                this.mapManager.addObject(worldId, graveObject);
              }
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

      // PVP: Check player bullets against other players
      if (this.pvpEnabled) {
        const isPlayerBullet = typeof bulletOwnerId === 'number' ||
                               (typeof bulletOwnerId === 'string' && !bulletOwnerId.startsWith('enemy_'));

        if (isPlayerBullet && players && players.length > 0) {
          for (const player of players) {
            // Skip self-damage
            if (player.id === bulletOwnerId) continue;

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
              const collisionId = `pvp_${bulletId}_${playerId}`;
              if (this.processedCollisions.has(collisionId)) continue;

              const damage = this.bulletManager.damage ? this.bulletManager.damage[bi] : 10;

              if (this.fileLogger) {
                this.fileLogger.bulletHit(bulletId, 'player_pvp', playerId, damage, { x: playerX, y: playerY });
              }

              // Apply damage
              if (typeof player.takeDamage === 'function') {
                player.takeDamage(damage);
              } else {
                player.health = Math.max(0, player.health - damage);
              }

              // Check for player death
              if (player.health <= 0 && !player.isDead) {
                player.isDead = true;
                player.deathX = player.x;
                player.deathY = player.y;
                player.deathTimestamp = Date.now();
                player.killedBy = bulletOwnerId;

                if (this.fileLogger) {
                  this.fileLogger.logCollision('PVP_KILL', {
                    killerId: bulletOwnerId,
                    victimId: playerId,
                    position: { x: player.x, y: player.y }
                  });
                }
              }

              this.bulletManager.markForRemoval(bi);
              this.processedCollisions.set(collisionId, Date.now());
              break;
            }
          }
        }
      }
    }

    // ============================================================================
    // ENEMY-PLAYER CONTACT COLLISIONS (New System)
    // ============================================================================
    // Check for enemy body collisions with players
    const enemyContactCollisions = [];

    if (players.length > 0 && this.enemyManager.enemyCount > 0 && this.fileLogger) {
      this.fileLogger.collisionCheck('enemy-player', this.enemyManager.enemyCount, players.length);
    }

    for (let ei = 0; ei < this.enemyManager.enemyCount; ei++) {
      // Skip dead enemies
      if (this.enemyManager.health[ei] <= 0) continue;

      const enemyX = this.enemyManager.x[ei];
      const enemyY = this.enemyManager.y[ei];
      const enemyWidth = this.enemyManager.width[ei];
      const enemyHeight = this.enemyManager.height[ei];
      const enemyWorldId = this.enemyManager.worldId[ei];
      const enemyContactDamage = this.enemyManager.contactDamage[ei] || 0;
      const enemyKnockback = this.enemyManager.knockbackForce[ei] || 0;

      // Only check if enemy has contact damage
      if (enemyContactDamage <= 0) continue;

      // Check collision with each player
      for (const player of players) {
        if (!player || player.health <= 0) continue;
        if (enemyWorldId !== player.worldId) continue;

        // Check AABB collision between enemy and player
        const playerWidth = 1; // Default player collision size
        const playerHeight = 1;

        if (this.checkAABBCollision(
          enemyX, enemyY, enemyWidth, enemyHeight,
          player.x, player.y, playerWidth, playerHeight
        )) {
          // Collision detected! Calculate knockback direction
          const dx = player.x - enemyX;
          const dy = player.y - enemyY;
          const distance = Math.sqrt(dx * dx + dy * dy);

          let knockbackX = 0;
          let knockbackY = 0;

          if (distance > 0) {
            // Knockback away from enemy
            knockbackX = (dx / distance) * enemyKnockback;
            knockbackY = (dy / distance) * enemyKnockback;
          }

          enemyContactCollisions.push({
            enemyIndex: ei,
            player: player,
            playerId: player.id,
            contactDamage: enemyContactDamage,
            knockbackX,
            knockbackY,
            enemyX,
            enemyY
          });

          if (this.fileLogger) {
            this.fileLogger.contactDamage(ei, player.id, enemyContactDamage, { x: knockbackX, y: knockbackY }, { enemy: { x: enemyX, y: enemyY }, player: { x: player.x, y: player.y } });
          }
        }
      }
    }

    // Return enemy contact collisions for server to process
    return {
      enemyContactCollisions
    };

    } finally {
      // Lag Compensation: Always restore player positions
      if (this.lagCompensation && originalPositions.size > 0) {
        this.lagCompensation.restoreAllPlayers(players, originalPositions);
      }
    }
  }

  /**
   * Validate client-reported collision (server-side)
   * @param {Object} data - Collision data from client
   * @returns {Object} Validation result
   */
  validateCollision(data) {
    const { bulletId, enemyId, timestamp, clientId, playerPosition, playerRTT } = data;

    // Find bullet and enemy indices using IDs
    const bulletIndex = this.findBulletIndex(bulletId);
    const enemyIndex = this.findEnemyIndex(enemyId);

    // Check if both entities exist
    if (bulletIndex === -1 || enemyIndex === -1) {
      if (this.fileLogger) this.fileLogger.collisionValidation('rejected', { reason: 'entity_not_found', bulletId, enemyId, clientId });
      return { valid: false, reason: 'Entity not found', bulletId, enemyId };
    }
    
    // Ensure bullet and enemy are in the same world to prevent cross-realm hits
    if (this.bulletManager.worldId[bulletIndex] !== this.enemyManager.worldId[enemyIndex]) {
      if (this.fileLogger) this.fileLogger.collisionValidation('rejected', { reason: 'different_world', bulletId, enemyId, clientId });
      return { valid: false, reason: 'Different world', bulletId, enemyId };
    }

    // Check if this collision was already processed recently
    const collisionId = `${bulletId}_${enemyId}`;
    if (this.processedCollisions.has(collisionId)) {
      if (this.fileLogger) this.fileLogger.collisionValidation('rejected', { reason: 'already_processed', bulletId, enemyId, clientId });
      return { valid: false, reason: 'Already processed', bulletId, enemyId };
    }

    // Check if timestamp is reasonable (within 500ms from now)
    const now = Date.now();
    if (Math.abs(now - timestamp) > 500) {
      if (this.fileLogger) this.fileLogger.collisionValidation('rejected', { reason: 'timestamp_old', age: Math.abs(now - timestamp), bulletId, enemyId, clientId });
      return { valid: false, reason: 'Timestamp too old', bulletId, enemyId };
    }
    
    // Check for line of sight obstruction
    if (this.mapManager && this.mapManager.hasLineOfSight) {
      if (!this.mapManager.hasLineOfSight(
        this.bulletManager.x[bulletIndex],
        this.bulletManager.y[bulletIndex],
        this.enemyManager.x[enemyIndex],
        this.enemyManager.y[enemyIndex]
      )) {
        if (this.fileLogger) this.fileLogger.collisionValidation('rejected', { reason: 'no_line_of_sight', bulletId, enemyId, clientId });
        return { valid: false, reason: 'No line of sight', bulletId, enemyId };
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
      if (this.fileLogger) this.fileLogger.collisionValidation('rejected', { reason: 'no_collision', bulletPos, enemyPos, bulletId, enemyId, clientId });
      return { valid: false, reason: 'No collision detected', bulletId, enemyId };
    }

    // Validate collision position if player data provided
    if (playerPosition && this.collisionValidator && this.collisionValidator.enabled) {
      const positionValidation = this.collisionValidator.validatePosition({
        serverPosition: bulletPos,
        clientPosition: playerPosition,
        rtt: playerRTT || 0,
        playerId: clientId,
        collisionType: 'bullet_hit'
      });

      // In strict mode, reject invalid collisions
      if (!positionValidation.valid && this.collisionValidator.mode === 'strict') {
        return {
          valid: false,
          reason: 'Position validation failed',
          bulletId,
          enemyId,
          details: positionValidation
        };
      }
    }

    if (this.fileLogger) this.fileLogger.collisionValidation('valid', { bulletId, enemyId, clientId });

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

    // Calculate damage
    const damage = this.bulletManager.damage ?
      this.bulletManager.damage[bulletIndex] : 10;

    // Apply damage to enemy (may return an object)
    const dmgResult = this.enemyManager.applyDamage(enemyIndex, damage);
    const remainingHealth = typeof dmgResult === 'object' && dmgResult !== null && 'health' in dmgResult
      ? dmgResult.health
      : (typeof dmgResult === 'number' ? dmgResult : this.enemyManager.health[enemyIndex]);

    if (this.fileLogger) {
      this.fileLogger.bulletHit(bulletId, 'enemy', enemyId, damage, { healthBefore: enemyHealthBefore, healthAfter: remainingHealth });
    }

    // Remove bullet
    if (this.bulletManager.markForRemoval) {
      this.bulletManager.markForRemoval(bulletIndex);
    } else if (this.bulletManager.life) {
      this.bulletManager.life[bulletIndex] = 0;
    }

    if (this.bulletManager.registerRemoval) {
      this.bulletManager.registerRemoval('entityHit');
    }

    // Handle enemy death
    let enemyKilled = false;
    if (remainingHealth <= 0) {
      enemyKilled = true;
      if (this.fileLogger) {
        this.fileLogger.logCollision('ENEMY_DEATH', { enemyId, killedBy: clientId });
      }
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