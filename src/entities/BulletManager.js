// File: /src/managers/BulletManager.js

/**
 * BulletManager handles bullet creation, updating, and removal.
 * Uses Structure of Arrays (SoA) for data layout optimization.
 */
export default class BulletManager {
  /**
   * Creates a bullet manager
   * @param {number} maxBullets - Maximum number of bullets to allow
   *
   * COORDINATE SYSTEM:
   * - All positions (x, y) are in TILE UNITS (not pixels)
   * - All sizes (width, height) are in TILE UNITS (not pixels)
   * - All velocities (vx, vy) are in TILES PER SECOND
   * - This matches the server MapManager coordinate system
   * - TILE_SIZE = 12 pixels per tile
   */
  constructor(maxBullets = 10000) {
    this.maxBullets = maxBullets;
    this.bulletCount = 0;
    this.nextBulletId = 1; // For assigning unique IDs

    // SoA data layout
    this.id = new Array(maxBullets);        // Unique bullet IDs
    this.x = new Float32Array(maxBullets);  // X position (TILE UNITS)
    this.y = new Float32Array(maxBullets);  // Y position (TILE UNITS)
    this.vx = new Float32Array(maxBullets); // X velocity (TILES/SEC)
    this.vy = new Float32Array(maxBullets); // Y velocity (TILES/SEC)
    this.life = new Float32Array(maxBullets); // Remaining life in seconds
    this.width = new Float32Array(maxBullets);  // Collision width (TILE UNITS)
    this.height = new Float32Array(maxBullets); // Collision height (TILE UNITS)
    this.damage = new Float32Array(maxBullets);  // Damage amount
    this.ownerId = new Array(maxBullets);   // ID of entity that created this bullet
    this.spriteName = new Array(maxBullets); // For client rendering
    this.worldId = new Array(maxBullets);
    this.faction = new Uint8Array(maxBullets); // Faction layer (0=enemy, 1-12=player factions)

    // Per-bullet speed multiplier (used for acceleration / slow effects)
    this.speedScale = new Float32Array(maxBullets);
    this.speedScale.fill(1);

    // VAE v2 properties: acceleration, curve, wave motion
    this.ax = new Float32Array(maxBullets);           // X acceleration (tiles/sec^2)
    this.ay = new Float32Array(maxBullets);           // Y acceleration (tiles/sec^2)
    this.angularVel = new Float32Array(maxBullets);   // Angular velocity for curved paths (rad/sec)
    this.waveAmp = new Float32Array(maxBullets);      // Wave amplitude (tiles)
    this.waveFreq = new Float32Array(maxBullets);     // Wave frequency (Hz)
    this.wavePhase = new Float32Array(maxBullets);    // Wave phase offset (radians)
    this.age = new Float32Array(maxBullets);          // Time since spawn (seconds)

    // Track removed bullets for network sync
    this.removedBullets = new Set();

    // Debug / analytics counters (reset each update)
    this.stats = {
      created: 0,
      expired: 0,
      wallHit: 0,
      entityHit: 0
    };
  }

  /**
   * Add a new bullet
   * @param {Object} bulletData - Bullet properties
   * @returns {string} The ID of the new bullet
   */
  addBullet(bulletData) {
    if (this.bulletCount >= this.maxBullets) {
      console.warn('BulletManager: Maximum bullet capacity reached');
      return null;
    }
    
    // ------------------------------------------------------------------
    // Sanity: every bullet **must** belong to a world.  Hard-reject if not.
    // ------------------------------------------------------------------
    if (!bulletData.worldId) {
      console.warn('[BulletManager] REJECTED bullet without worldId', bulletData);
      return null;
    }

    const bulletId = bulletData.id || `bullet_${this.nextBulletId++}`;
    const index = this.bulletCount++;
    
    // Set bullet properties
    this.id[index] = bulletId;
    this.x[index] = bulletData.x;
    this.y[index] = bulletData.y;
    this.vx[index] = bulletData.vx;
    this.vy[index] = bulletData.vy;
    this.life[index] = bulletData.lifetime || 3.0; // Default 3 seconds
    this.width[index] = bulletData.width || 0.4;   // TILE UNITS: 40% of a tile (was 5 pixels - WRONG!)
    this.height[index] = bulletData.height || 0.4; // TILE UNITS: 40% of a tile (was 5 pixels - WRONG!)
    this.damage[index] = bulletData.damage || 10;
    this.ownerId[index] = bulletData.ownerId || null;
    this.spriteName[index] = bulletData.spriteName || null;
    this.worldId[index] = bulletData.worldId;
    this.speedScale[index] = 1;

    // VAE v2 properties
    this.ax[index] = bulletData.ax || 0;
    this.ay[index] = bulletData.ay || 0;
    this.angularVel[index] = bulletData.angularVel || 0;
    this.waveAmp[index] = bulletData.waveAmp || 0;
    this.waveFreq[index] = bulletData.waveFreq || 0;
    this.wavePhase[index] = bulletData.wavePhase || 0;
    this.age[index] = 0;

    // Determine faction layer
    if (bulletData.faction !== undefined) {
      this.faction[index] = bulletData.faction;
    } else {
      // Auto-detect from ownerId
      const isEnemyBullet = typeof bulletData.ownerId === 'string' && bulletData.ownerId.startsWith('enemy_');
      this.faction[index] = isEnemyBullet ? 0 : (bulletData.playerFaction || 1);
    }

    if (this.stats) this.stats.created++;
    return bulletId;
  }

  /**
   * Update all bullets
   * @param {number} deltaTime - Time elapsed since last update in seconds
   */
  update(deltaTime) {
    let count = this.bulletCount;
    
    // Reset per-frame stats (except created which should persist until next flush)
    if (this.stats) {
      this.stats.expired = 0;
      // wallHit / entityHit updated externally; keep their values for overlay then reset later if desired
    }
    
    for (let i = 0; i < count; i++) {
      // Update age
      this.age[i] += deltaTime;

      // Apply acceleration
      this.vx[i] += this.ax[i] * deltaTime;
      this.vy[i] += this.ay[i] * deltaTime;

      // Apply angular velocity (curved paths)
      if (this.angularVel[i] !== 0) {
        const angle = Math.atan2(this.vy[i], this.vx[i]);
        const speed = Math.sqrt(this.vx[i] * this.vx[i] + this.vy[i] * this.vy[i]);
        const newAngle = angle + this.angularVel[i] * deltaTime;
        this.vx[i] = Math.cos(newAngle) * speed;
        this.vy[i] = Math.sin(newAngle) * speed;
      }

      // Calculate wave offset (perpendicular wobble)
      let waveOffsetX = 0, waveOffsetY = 0;
      if (this.waveAmp[i] > 0.001) {
        const moveAngle = Math.atan2(this.vy[i], this.vx[i]);
        const perpAngle = moveAngle + Math.PI / 2;
        const waveValue = Math.sin(this.age[i] * this.waveFreq[i] * 2 * Math.PI + this.wavePhase[i]);
        waveOffsetX = Math.cos(perpAngle) * waveValue * this.waveAmp[i] * deltaTime * 10;
        waveOffsetY = Math.sin(perpAngle) * waveValue * this.waveAmp[i] * deltaTime * 10;
      }

      // Update position
      const scale = this.speedScale[i] || 1;
      this.x[i] += this.vx[i] * scale * deltaTime + waveOffsetX;
      this.y[i] += this.vy[i] * scale * deltaTime + waveOffsetY;

      // Decrement lifetime
      this.life[i] -= deltaTime;

      // Remove expired bullets
      if (this.life[i] <= 0) {
        if (this.stats) this.stats.expired++;

        // Debug cavalry bullets
        const ownerIdStr = String(this.ownerId[i] || '');
        if (ownerIdStr.includes('enemy_9') || ownerIdStr.includes('enemy_10')) {
          console.log(`⚠️  CAVALRY BULLET EXPIRED: ${this.id[i]} from ${this.ownerId[i]} at (${this.x[i].toFixed(2)}, ${this.y[i].toFixed(2)})`);
        }

        this.swapRemove(i);
        count--;
        i--;
      }
    }
    
    this.bulletCount = count;

    return this.bulletCount;
  }

  /**
   * Remove a bullet using the swap-and-pop technique
   * @param {number} index - Index of bullet to remove
   */
  swapRemove(index) {
    // Track the removed bullet ID for network sync
    const removedId = this.id[index];
    if (removedId) {
      this.removedBullets.add(removedId);
    }

    const last = this.bulletCount - 1;

    if (index !== last) {
      // Swap with the last bullet
      this.id[index] = this.id[last];
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.vx[index] = this.vx[last];
      this.vy[index] = this.vy[last];
      this.life[index] = this.life[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
      this.damage[index] = this.damage[last];
      this.ownerId[index] = this.ownerId[last];
      this.spriteName[index] = this.spriteName[last];
      this.worldId[index] = this.worldId[last];
      this.speedScale[index] = this.speedScale[last];
      this.faction[index] = this.faction[last];
      // VAE v2 properties
      this.ax[index] = this.ax[last];
      this.ay[index] = this.ay[last];
      this.angularVel[index] = this.angularVel[last];
      this.waveAmp[index] = this.waveAmp[last];
      this.waveFreq[index] = this.waveFreq[last];
      this.wavePhase[index] = this.wavePhase[last];
      this.age[index] = this.age[last];
    }

    this.bulletCount--;
  }

  /**
   * Get and clear removed bullet IDs for network sync
   * @returns {Array} Array of removed bullet IDs
   */
  getAndClearRemovedBullets() {
    const removed = Array.from(this.removedBullets);
    this.removedBullets.clear();
    return removed;
  }
  
  /**
   * Remove a bullet by ID
   * @param {string} bulletId - ID of bullet to remove
   * @returns {boolean} True if bullet was found and removed
   */
  removeBulletById(bulletId) {
    for (let i = 0; i < this.bulletCount; i++) {
      if (this.id[i] === bulletId) {
        this.swapRemove(i);
        return true;
      }
    }
    return false;
  }

  /**
   * Simple AoE explosion – iterate bullets or future entity grid.
   * For now this just marks the projectile for removal.
   * @param {string|number} bulletIdOrIdx – bullet id or index
   * @param {number} radius – blast radius (tiles)
   * @param {number} damage – damage to inflict to entities (todo)
   */
  explode(bulletIdOrIdx, radius = 1, damage = 10) {
    let idx = -1;
    if (typeof bulletIdOrIdx === 'number') idx = bulletIdOrIdx;
    else idx = this.findIndexById(bulletIdOrIdx);
    if (idx === -1) return false;
    // TODO: apply damage to entities in radius
    this.markForRemoval(idx);
    return true;
  }
  
  /**
   * Find bullet index by ID
   * @param {string} bulletId - Bullet ID to find
   * @returns {number} Index of bullet or -1 if not found
   */
  findIndexById(bulletId) {
    for (let i = 0; i < this.bulletCount; i++) {
      if (this.id[i] === bulletId) {
        return i;
      }
    }
    return -1;
  }
  
  /**
   * Mark a bullet for removal
   * @param {number} index - Index of bullet to remove
   *
   * CRITICAL: Sets life to 0.001 instead of 0 to allow ONE MORE broadcast
   * with the updated collision position before natural expiration
   */
  markForRemoval(index) {
    if (index >= 0 && index < this.bulletCount) {
      // Set to tiny positive value instead of 0 to allow final broadcast
      // The next update() will decrement this below 0 and remove it
      this.life[index] = 0.001;
    }
  }

  /**
   * Get number of active bullets
   * @returns {number} Count of active bullets
   */
  getActiveBulletCount() {
    return this.bulletCount;
  }

  /**
   * Clean up resources
   */
  cleanup() {
    this.bulletCount = 0;
  }

  /**
   * Get bullet data array for network transmission
   * @param {string} filterWorldId - Filter bullets by world ID
   * @returns {Array} Array of bullet data objects
   */
  getBulletsData(filterWorldId = null) {
    const bullets = [];
    // Truncate helper - reduces JSON size by ~30%
    const t = (v) => Math.round(v * 100) / 100;

    for (let i = 0; i < this.bulletCount; i++) {
      // Skip bullets marked for removal (life <= 0)
      // These bullets have already collided server-side. Don't broadcast them.
      // The client will naturally stop receiving updates and clean up.
      if (this.life[i] <= 0) continue;

      if (filterWorldId && this.worldId[i] !== filterWorldId) continue;

      bullets.push({
        id: this.id[i],
        x: t(this.x[i]),
        y: t(this.y[i]),
        vx: t(this.vx[i]),
        vy: t(this.vy[i]),
        width: t(this.width[i]),
        height: t(this.height[i]),
        life: t(this.life[i]),
        damage: this.damage[i],
        ownerId: this.ownerId[i],
        spriteName: this.spriteName[i],
        worldId: this.worldId[i],
        faction: this.faction[i]
      });
    }

    return bullets;
  }

  /**
   * External systems (collision manager) can record why a bullet was removed.
   * @param {string} reason - 'wallHit' | 'entityHit'
   */
  registerRemoval(reason) {
    if (!this.stats) return;
    if (reason === 'wallHit') this.stats.wallHit++;
    if (reason === 'entityHit') this.stats.entityHit++;
  }
}