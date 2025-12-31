/**
 * ClientBulletManager.js
 * Client-side manager for bullets with client prediction and reconciliation
 */
export class ClientBulletManager {
  /**
   * Creates a client-side bullet manager
   * @param {number} maxBullets - Maximum number of bullets to support
   */
  constructor(maxBullets = 10000) {
    this.maxBullets = maxBullets;
    this.bulletCount = 0;
    
    // Structure of Arrays for performance
    this.id = new Array(maxBullets);        // Unique bullet IDs
    this.x = new Float32Array(maxBullets);  // X position (rendered)
    this.y = new Float32Array(maxBullets);  // Y position (rendered)
    this.targetX = new Float32Array(maxBullets);  // Target X from server (for interpolation)
    this.targetY = new Float32Array(maxBullets);  // Target Y from server (for interpolation)
    this.vx = new Float32Array(maxBullets); // X velocity
    this.vy = new Float32Array(maxBullets); // Y velocity
    this.life = new Float32Array(maxBullets); // Remaining life in seconds
    this.width = new Float32Array(maxBullets);  // Width for collision
    this.height = new Float32Array(maxBullets); // Height for collision
    this.ownerId = new Array(maxBullets);   // Who fired this bullet
    this.damage = new Float32Array(maxBullets); // Damage value
    this.faction = new Uint8Array(maxBullets); // Faction layer (0=enemy, 1-12=player factions)

    // Interpolation settings
    this.interpolationSpeed = 15.0; // How fast to blend towards server position (higher = faster snap)

    // World association so client can filter/collide correctly
    this.worldId = new Array(maxBullets);
    
    // Visual properties
    this.sprite = new Array(maxBullets);    // Legacy sprite sheet info
    this.spriteName = new Array(maxBullets); // SpriteDatabase name (new system)
    
    // Mapping from ID to index for fast lookups
    this.idToIndex = new Map();
    
    // Local prediction bullets (client-created, not yet confirmed by server)
    this.localBullets = new Set();
    
    // Scale factor for bullet rendering
    this.bulletScale = 1.0;
    
    console.log("ClientBulletManager initialized with capacity for", maxBullets, "bullets");
  }
  
  /**
   * Add a new bullet
   * @param {Object} bulletData - Bullet properties
   * @returns {string} Bullet ID
   */
  addBullet(bulletData) {
    if (this.bulletCount >= this.maxBullets) {
      console.warn('ClientBulletManager: Maximum bullet capacity reached');
      return null;
    }

    const index = this.bulletCount++;
    const bulletId = bulletData.id || `local_${Date.now()}_${index}`;

    // DIAGNOSTIC: Log enemy bullets being added
    if (typeof bulletData.ownerId === 'string' && bulletData.ownerId.startsWith('enemy_')) {
      console.log(`✅ [ENEMY BULLET ADD] ID: ${bulletId}, Owner: ${bulletData.ownerId}, Pos: (${bulletData.x.toFixed(2)}, ${bulletData.y.toFixed(2)})`);
    }

    // Store bullet properties
    this.id[index] = bulletId;
    this.x[index] = bulletData.x || 0;
    this.y[index] = bulletData.y || 0;
    this.targetX[index] = bulletData.x || 0;  // Initialize target to current position
    this.targetY[index] = bulletData.y || 0;
    // CRITICAL: Default to 0 if velocity not provided (binary protocol may omit)
    this.vx[index] = bulletData.vx || 0;
    this.vy[index] = bulletData.vy || 0;
    this.life[index] = bulletData.life || bulletData.lifetime || 2.0; // Default 2 seconds
    // CRITICAL FIX: Match server bullet size (0.6 tiles, not 5 pixels)
    // Server uses 0.6 tiles (60% of tile), client must match for consistent hitboxes
    this.width[index] = bulletData.width || 0.6;
    this.height[index] = bulletData.height || 0.6;
    this.ownerId[index] = bulletData.ownerId || null;
    this.damage[index] = bulletData.damage || 10;
    this.worldId[index] = bulletData.worldId;

    // Determine faction layer
    if (bulletData.faction !== undefined) {
      this.faction[index] = bulletData.faction;
    } else {
      // Auto-detect from ownerId
      const isEnemyBullet = typeof bulletData.ownerId === 'string' && bulletData.ownerId.startsWith('enemy_');
      this.faction[index] = isEnemyBullet ? 0 : 1; // 0=enemy, 1=default player faction
    }
    
    // Store sprite info
    this.sprite[index] = bulletData.spriteSheet ? {
      spriteSheet: bulletData.spriteSheet,
      spriteX: bulletData.spriteX || 0,
      spriteY: bulletData.spriteY || 0,
      spriteWidth: bulletData.spriteWidth || 8,
      spriteHeight: bulletData.spriteHeight || 8
    } : null;
    
    // Store spriteName for new rendering path
    this.spriteName[index] = bulletData.spriteName || null;
    
    // Store index for lookup
    this.idToIndex.set(bulletId, index);
    
    // If this is a local prediction bullet (client-created)
    if (bulletId.startsWith('local_')) {
      this.localBullets.add(bulletId);
    }
    
    return bulletId;
  }
  
  /**
   * Update all bullets
   * @param {number} deltaTime - Time elapsed since last update in seconds
   */
  update(deltaTime) {
    if (deltaTime <= 0) {
      return; // Skip if delta time is zero or negative
    }

    // DIAGNOSTIC: Log deltaTime to check if it's in correct units
    if (this.bulletCount > 0 && Math.random() < 0.05) {
      // console.error(`⏱️ [BULLET UPDATE] deltaTime=${deltaTime.toFixed(4)}s, bulletCount=${this.bulletCount}`);
    }

    let count = this.bulletCount;

    for (let i = 0; i < count; i++) {
      const lifeBefore = this.life[i];

      // DEFENSIVE: Skip if velocity is NaN (would corrupt position)
      const vx = this.vx[i];
      const vy = this.vy[i];
      if (!isFinite(vx) || !isFinite(vy)) {
        // Log once and remove the corrupted bullet
        console.error(`[BULLET] Removing bullet with NaN velocity: id=${this.id[i]}, vx=${vx}, vy=${vy}`);
        this.swapRemove(i);
        count--;
        i--;
        continue;
      }

      // Update target position based on velocity (server prediction)
      this.targetX[i] += vx * deltaTime;
      this.targetY[i] += vy * deltaTime;

      // DEFENSIVE: Check if target became NaN somehow
      if (!isFinite(this.targetX[i]) || !isFinite(this.targetY[i])) {
        console.error(`[BULLET] Target became NaN: id=${this.id[i]}, targetX=${this.targetX[i]}, targetY=${this.targetY[i]}`);
        this.swapRemove(i);
        count--;
        i--;
        continue;
      }

      // Smoothly interpolate actual position towards target (reduces jitter)
      const interpFactor = Math.min(1.0, this.interpolationSpeed * deltaTime);
      this.x[i] += (this.targetX[i] - this.x[i]) * interpFactor;
      this.y[i] += (this.targetY[i] - this.y[i]) * interpFactor;

      // Decrement lifetime
      this.life[i] -= deltaTime;

      // DIAGNOSTIC: Log lifetime changes
      if (Math.random() < 0.05 && lifeBefore > 0) {
        // console.error(`⏱️ [LIFE] Bullet ${this.id[i]}: life ${lifeBefore.toFixed(3)} → ${this.life[i].toFixed(3)} (delta: ${deltaTime.toFixed(4)})`);
      }

      // Remove expired bullets
      if (this.life[i] <= 0) {
        // DIAGNOSTIC: Log bullet removal
        // console.error(`❌ [BULLET EXPIRED] ID: ${this.id[i]}, Pos: (${this.x[i].toFixed(2)}, ${this.y[i].toFixed(2)}), Life: ${this.life[i].toFixed(3)}, DeltaTime: ${deltaTime.toFixed(4)}`);
        this.swapRemove(i);
        count--;
        i--;
      }
    }
    
    this.bulletCount = count;
  }
  
  /**
   * Remove a bullet using swap-and-pop technique
   * @param {number} index - Index of bullet to remove
   */
  swapRemove(index) {
    if (index < 0 || index >= this.bulletCount) return;
    
    // Remove from ID mapping and local bullets set
    const bulletId = this.id[index];
    this.idToIndex.delete(bulletId);
    this.localBullets.delete(bulletId);
    
    // Swap with the last bullet (if not already the last)
    const lastIndex = this.bulletCount - 1;
    if (index !== lastIndex) {
      // Copy properties from last bullet to this position
      this.id[index] = this.id[lastIndex];
      this.x[index] = this.x[lastIndex];
      this.y[index] = this.y[lastIndex];
      this.targetX[index] = this.targetX[lastIndex];  // FIX: Copy interpolation targets
      this.targetY[index] = this.targetY[lastIndex];  // FIX: Copy interpolation targets
      this.vx[index] = this.vx[lastIndex];
      this.vy[index] = this.vy[lastIndex];
      this.life[index] = this.life[lastIndex];
      this.width[index] = this.width[lastIndex];
      this.height[index] = this.height[lastIndex];
      this.ownerId[index] = this.ownerId[lastIndex];
      this.damage[index] = this.damage[lastIndex];
      this.faction[index] = this.faction[lastIndex];
      this.worldId[index] = this.worldId[lastIndex];
      this.sprite[index] = this.sprite[lastIndex];
      this.spriteName[index] = this.spriteName[lastIndex];

      // Update index in mapping
      this.idToIndex.set(this.id[index], index);
    }
    
    this.bulletCount--;
  }
  
  /**
   * Find bullet index by ID
   * @param {string} bulletId - ID of bullet to find
   * @returns {number} Index of bullet or -1 if not found
   */
  findIndexById(bulletId) {
    const index = this.idToIndex.get(bulletId);
    return index !== undefined ? index : -1;
  }
  
  /**
   * Remove a bullet by ID
   * @param {string} bulletId - ID of bullet to remove
   * @returns {boolean} True if bullet was found and removed
   */
  removeBulletById(bulletId) {
    const index = this.findIndexById(bulletId);
    if (index !== -1) {
      this.swapRemove(index);
      return true;
    }
    return false;
  }
  
  /**
   * Mark a bullet for removal
   * @param {number} index - Index of bullet to remove
   */
  markForRemoval(index) {
    if (index >= 0 && index < this.bulletCount) {
      this.life[index] = 0;
    }
  }
  
  /**
   * Set initial bullets list from server
   * @param {Array} bullets - Array of bullet data from server
   */
  setBullets(bullets) {
    const playerWorld = window.gameState?.character?.worldId;
    if (playerWorld) bullets = bullets.filter(b => b.worldId === playerWorld);
    
    // Clear existing bullets except local predictions
    this.clearNonLocalBullets();
    
    // Add new bullets from server
    for (const bullet of bullets) {
      // Skip if we already have a local prediction for this ID
      if (this.findIndexById(bullet.id) !== -1) continue;
      
      // Make sure all required properties are set
      const bulletData = {
        ...bullet,
        width: bullet.width || 5,
        height: bullet.height || 5,
        damage: bullet.damage || 10
      };
      
      this.addBullet(bulletData);
    }

    // Bullet debug logs disabled - collision fix verified ✓
    // console.log(`Set ${bullets.length} bullets from server, total bullets: ${this.bulletCount}`);
  }
  
  /**
   * Clear all non-local (server confirmed) bullets
   */
  clearNonLocalBullets() {
    // Remove bullets that aren't local predictions
    for (let i = 0; i < this.bulletCount; i++) {
      if (!this.localBullets.has(this.id[i])) {
        this.swapRemove(i);
        i--;
      }
    }
  }
  
  /**
   * Update bullets based on server data
   * @param {Array} bullets - Array of bullet data from server
   */
  updateBullets(bullets) {
    if (!bullets || !Array.isArray(bullets)) {
      console.warn("Invalid bullets data in updateBullets");
      return;
    }

    const playerWorld = window.gameState?.character?.worldId;
    const beforeFilter = bullets.length;
    if (playerWorld) bullets = bullets.filter(b => b.worldId === playerWorld);
    const afterFilter = bullets.length;

    // DIAGNOSTIC: Always log bullet updates
    if (bullets.length > 0) {
      // console.error(`🔵 [BULLET UPDATE] Received ${beforeFilter} bullets, ${afterFilter} after worldId filter. First: ID=${bullets[0].id}, Pos=(${bullets[0].x?.toFixed(2)}, ${bullets[0].y?.toFixed(2)}), WorldID=${bullets[0].worldId}`);
    } else if (beforeFilter > 0) {
      console.error(`❌ [BULLET UPDATE] ${beforeFilter} bullets FILTERED OUT by worldId! Player world: ${playerWorld}`);
    }

    // Process server bullets
    for (const bullet of bullets) {
      const index = this.findIndexById(bullet.id);

      if (index !== -1) {
        // Update existing bullet - use interpolation to prevent jitter
        // Set TARGET position from server, actual position will interpolate towards it
        this.targetX[index] = bullet.x;
        this.targetY[index] = bullet.y;
        // Only update velocity if provided (binary protocol may only send position)
        if (bullet.vx !== undefined) this.vx[index] = bullet.vx;
        if (bullet.vy !== undefined) this.vy[index] = bullet.vy;

        // If this is the first update or bullet jumped too far (>10 tiles), snap immediately
        const deltaX = bullet.x - this.x[index];
        const deltaY = bullet.y - this.y[index];
        const distSq = deltaX * deltaX + deltaY * deltaY;
        if (distSq > 100) { // > 10 tiles, snap immediately
          this.x[index] = bullet.x;
          this.y[index] = bullet.y;
        }
        if (bullet.life !== undefined || bullet.lifetime !== undefined) {
          this.life[index] = bullet.life || bullet.lifetime || 3.0;
        }
        if (bullet.width !== undefined) this.width[index] = bullet.width;
        if (bullet.height !== undefined) this.height[index] = bullet.height;
        if (bullet.damage !== undefined) this.damage[index] = bullet.damage;
        if (bullet.ownerId !== undefined) this.ownerId[index] = bullet.ownerId;
        if (bullet.worldId !== undefined) this.worldId[index] = bullet.worldId;
        if (bullet.spriteName) {
          this.spriteName[index] = bullet.spriteName;
        }

        // DIAGNOSTIC: Log when enemy bullet ownerId is set
        if (typeof bullet.ownerId === 'string' && bullet.ownerId.startsWith('enemy_') && Math.random() < 0.1) {
          console.log(`🔄 [ENEMY BULLET UPDATE] ID: ${bullet.id}, Owner: ${bullet.ownerId} updated`);
        }
      } else {
        // Try to reconcile with a locally predicted bullet (same owner & close position)
        if (bullet.ownerId === window.gameState?.character?.id) {
          let bestIdx = -1;
          let bestDistSq = Infinity;
          for (const localId of this.localBullets) {
            const idx = this.findIndexById(localId);
            if (idx === -1) continue;
            // Compare positions
            const dx = this.x[idx] - bullet.x;
            const dy = this.y[idx] - bullet.y;
            const distSq = dx*dx + dy*dy;
            if (distSq < 1.0 && distSq < bestDistSq) { // ε² = 1 tile²
              bestDistSq = distSq;
              bestIdx = idx;
            }
          }
          if (bestIdx !== -1) {
            // Overwrite local bullet slot with authoritative data
            this.idToIndex.delete(this.id[bestIdx]);
            this.localBullets.delete(this.id[bestIdx]);

            this.id[bestIdx] = bullet.id;
            this.x[bestIdx] = bullet.x;
            this.y[bestIdx] = bullet.y;
            this.vx[bestIdx] = bullet.vx;
            this.vy[bestIdx] = bullet.vy;
            this.life[bestIdx] = bullet.life || bullet.lifetime || 3.0;
            this.width[bestIdx] = bullet.width || 5;
            this.height[bestIdx] = bullet.height || 5;
            this.damage[bestIdx] = bullet.damage || 10;
            this.ownerId[bestIdx] = bullet.ownerId;
            this.worldId[bestIdx] = bullet.worldId;
            this.spriteName[bestIdx] = bullet.spriteName || this.spriteName[bestIdx];

            this.idToIndex.set(bullet.id, bestIdx);
            continue; // reconciled, skip normal add
          }
        }
        // Add new bullet if no reconciliation happened
        this.addBullet(bullet);
      }
    }
    
    // Remove bullets that aren't in the server update and aren't local predictions
    const serverBulletIds = new Set(bullets.map(b => b.id));
    
    for (let i = 0; i < this.bulletCount; i++) {
      const bulletId = this.id[i];
      
      // Keep local predictions and bullets from server update
      if (!this.localBullets.has(bulletId) && !serverBulletIds.has(bulletId)) {
        this.swapRemove(i);
        i--;
      }
    }
  }
  
  /**
   * Get bullet rendering data
   * @returns {Array} Array of bullet data for rendering
   */
  getBulletsForRender() {
    const bullets = [];
    
    for (let i = 0; i < this.bulletCount; i++) {
      bullets.push({
        id: this.id[i],
        x: this.x[i],
        y: this.y[i],
        width: this.width[i],
        height: this.height[i],
        isLocal: this.localBullets.has(this.id[i])
      });
    }
    
    return bullets;
  }
  
  /**
   * Clean up resources
   */
  cleanup() {
    this.bulletCount = 0;
    this.idToIndex.clear();
    this.localBullets.clear();
  }
}
