/**
 * ClientCollisionManager.js
 * Handles optimized collision detection on the client using WebAssembly and spatial partitioning.
 */
import { loadCollisionWasm, getWasmMemory } from '../wasm/clientWasmLoader.js';
import SpatialGrid from '../../shared/SpatialGrid.js';

export class ClientCollisionManager {
  /**
   * Creates a client-side collision manager
   * @param {BulletManager} bulletManager - The client's bullet manager instance
   * @param {EnemyManager} enemyManager - The client's enemy manager instance
   * @param {WebSocket} socket - WebSocket connection to the server
   */
  constructor(bulletManager, enemyManager, socket) {
    this.bulletManager = bulletManager;
    this.enemyManager = enemyManager;
    this.socket = socket;
    
    // Player's position (center of visible area)
    this.playerX = 0;
    this.playerY = 0;
    
    // Visible area dimensions
    this.visibleWidth = 2000;  // Adjust based on your view distance
    this.visibleHeight = 1500;
    
    // Grid configuration
    this.cellSize = 64; // Should match server's cell size for consistency
    
    // Create spatial grid for the visible area
    this.spatialGrid = new SpatialGrid(
      this.cellSize, 
      this.visibleWidth, 
      this.visibleHeight
    );
    
    // WASM related properties
    this.wasm = null;
    this.isWasmReady = false;
    this.maxCollisions = 200; // Maximum collisions to detect in one frame
    
    // Collision result buffer
    this.collisionPairs = new Int32Array(this.maxCollisions * 2);
    
    // Tracker for processed collisions to avoid sending duplicates
    this.processedCollisions = new Set();
    this.processedTimeout = 1000; // Time in ms to consider a collision "processed"
    
    // Initialize WASM
    this.initWasm();
  }
  
  /**
   * Initialize WebAssembly module
   */
  async initWasm() {
    try {
      this.wasm = await loadCollisionWasm();
      this.isWasmReady = !!this.wasm;
      console.log('Client collision WASM initialized:', this.isWasmReady);
    } catch (error) {
      console.error('Failed to initialize client collision WASM:', error);
      this.isWasmReady = false;
    }
  }
  
  /**
   * Update the player's position (center of visibility)
   * @param {number} x - Player's X coordinate
   * @param {number} y - Player's Y coordinate
   */
  updatePlayerPosition(x, y) {
    this.playerX = x;
    this.playerY = y;
  }
  
  /**
   * Check if an entity is within the visible area
   * @param {number} x - Entity X position
   * @param {number} y - Entity Y position
   * @returns {boolean} True if entity is visible
   */
  isInVisibleArea(x, y) {
    const halfWidth = this.visibleWidth / 2;
    const halfHeight = this.visibleHeight / 2;
    
    return (
      x >= this.playerX - halfWidth &&
      x <= this.playerX + halfWidth &&
      y >= this.playerY - halfHeight &&
      y <= this.playerY + halfHeight
    );
  }
  
  /**
   * Convert from world to local grid coordinates
   * @param {number} worldX - World X coordinate
   * @param {number} worldY - World Y coordinate
   * @returns {Object} Local coordinates {x, y}
   */
  worldToLocalPosition(worldX, worldY) {
    return {
      x: worldX - (this.playerX - this.visibleWidth / 2),
      y: worldY - (this.playerY - this.visibleHeight / 2)
    };
  }
  
  /**
   * Update the spatial grid with current entity positions
   */
  updateSpatialGrid() {
    // Clear previous data
    this.spatialGrid.clear();
    
    // Add visible bullets to grid
    for (let i = 0; i < this.bulletManager.bulletCount; i++) {
      if (this.isInVisibleArea(this.bulletManager.x[i], this.bulletManager.y[i])) {
        const localPos = this.worldToLocalPosition(
          this.bulletManager.x[i], 
          this.bulletManager.y[i]
        );
        
        this.spatialGrid.insertBullet(
          i,
          localPos.x,
          localPos.y,
          this.bulletManager.width[i],
          this.bulletManager.height[i]
        );
      }
    }
    
    // Add visible enemies to grid
    for (let i = 0; i < this.enemyManager.enemyCount; i++) {
      if (this.isInVisibleArea(this.enemyManager.x[i], this.enemyManager.y[i])) {
        const localPos = this.worldToLocalPosition(
          this.enemyManager.x[i], 
          this.enemyManager.y[i]
        );
        
        this.spatialGrid.insertEnemy(
          i,
          localPos.x,
          localPos.y,
          this.enemyManager.width[i],
          this.enemyManager.height[i]
        );
      }
    }
  }
  
  /**
   * Main collision detection and processing
   */
  update() {
    if (!this.bulletManager || !this.enemyManager) return;
    
    // Get collision pairs
    const collisions = this.isWasmReady ? 
      this.detectCollisionsWasm() : 
      this.detectCollisionsJS();
    
    // Process each collision
    for (const collision of collisions) {
      const { bulletIndex, enemyIndex } = collision;
      
      // Skip if indices are invalid
      if (bulletIndex >= this.bulletManager.bulletCount || 
          enemyIndex >= this.enemyManager.enemyCount) {
        continue;
      }
      
      // Generate a unique collision ID
      // If you have persistent IDs, use those instead of indices
      const bulletId = this.bulletManager.id ? 
        this.bulletManager.id[bulletIndex] : 
        `bullet_${bulletIndex}`;
        
      const enemyId = this.enemyManager.id ? 
        this.enemyManager.id[enemyIndex] : 
        `enemy_${enemyIndex}`;
      
      const collisionId = `${bulletId}_${enemyId}`;
      
      // Check if we've already processed this collision recently
      if (this.processedCollisions.has(collisionId)) {
        continue;
      }
      
      // Apply immediate client-side effects
      this.applyCollisionEffects(bulletIndex, enemyIndex);
      
      // Mark as processed
      this.processedCollisions.add(collisionId);
      setTimeout(() => {
        this.processedCollisions.delete(collisionId);
      }, this.processedTimeout);
      
      // Send to server for validation
      this.sendCollisionToServer(bulletIndex, enemyIndex);
    }
  }
  
  /**
   * Apply immediate visual effects for client feedback
   * @param {number} bulletIndex - Index of bullet in bulletManager
   * @param {number} enemyIndex - Index of enemy in enemyManager
   */
  applyCollisionEffects(bulletIndex, enemyIndex) {
    // Mark bullet for removal (client prediction)
    this.bulletManager.life[bulletIndex] = 0;
    
    // Apply visual hit effect to enemy if method exists
    if (this.enemyManager.applyHitEffect) {
      this.enemyManager.applyHitEffect(enemyIndex);
    }
    
    // You could add particle effects or other visual feedback here
  }
  
  /**
   * Send collision data to server for validation
   * @param {number} bulletIndex - Index of bullet in bulletManager
   * @param {number} enemyIndex - Index of enemy in enemyManager
   */
  sendCollisionToServer(bulletIndex, enemyIndex) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    
    // Get persistent IDs if available, otherwise use indices
    const bulletId = this.bulletManager.id ? 
      this.bulletManager.id[bulletIndex] : bulletIndex;
      
    const enemyId = this.enemyManager.id ? 
      this.enemyManager.id[enemyIndex] : enemyIndex;
    
    // Send collision event to server
    this.socket.send(JSON.stringify({
      type: 'COLLISION',
      bulletId: bulletId,
      enemyId: enemyId,
      timestamp: Date.now()
    }));
  }
  
  /**
   * Handle server's response to a collision report
   * @param {Object} response - Server's validated collision result
   */
  handleServerResponse(response) {
    const { valid, bulletId, enemyId, damage, enemyHealth } = response;
    
    if (valid) {
      // Find indices based on IDs (if using persistent IDs)
      let enemyIndex = -1;
      
      if (this.enemyManager.findIndexById) {
        enemyIndex = this.enemyManager.findIndexById(enemyId);
      } else if (typeof enemyId === 'number') {
        // Assume enemyId is the index if no mapping function exists
        enemyIndex = enemyId;
      }
      
      // Update enemy state with authoritative server data
      if (enemyIndex !== -1 && enemyIndex < this.enemyManager.enemyCount) {
        // Update health
        if (this.enemyManager.health) {
          this.enemyManager.health[enemyIndex] = enemyHealth;
        }
        
        // Handle enemy death if needed
        if (enemyHealth <= 0 && this.enemyManager.onDeath) {
          this.enemyManager.onDeath(enemyIndex);
        }
      }
    } else {
      // Server rejected the collision, could implement reconciliation here
      console.log('Server rejected collision:', response);
    }
  }
  
  /**
   * Detect collisions using JavaScript (fallback if WASM fails)
   * @returns {Array} Array of collision objects {bulletIndex, enemyIndex}
   */
  detectCollisionsJS() {
    // Update the spatial grid
    this.updateSpatialGrid();
    
    // Get potential bullet-enemy pairs
    const potentialPairs = this.spatialGrid.getPotentialCollisionPairs();
    
    // Perform precise collision checks
    const collisions = [];
    
    for (const [bulletIndex, enemyIndex] of potentialPairs) {
      // Verify indices are still valid
      if (bulletIndex >= this.bulletManager.bulletCount || 
          enemyIndex >= this.enemyManager.enemyCount) {
        continue;
      }
      
      // AABB collision check
      if (this.checkAABBCollision(
        this.bulletManager.x[bulletIndex],
        this.bulletManager.y[bulletIndex],
        this.bulletManager.width[bulletIndex],
        this.bulletManager.height[bulletIndex],
        this.enemyManager.x[enemyIndex],
        this.enemyManager.y[enemyIndex],
        this.enemyManager.width[enemyIndex],
        this.enemyManager.height[enemyIndex]
      )) {
        collisions.push({
          bulletIndex,
          enemyIndex
        });
      }
    }
    
    return collisions;
  }
  
  /**
   * Basic AABB collision check
   */
  checkAABBCollision(ax, ay, awidth, aheight, bx, by, bwidth, bheight) {
    return (
      ax < bx + bwidth &&
      ax + awidth > bx &&
      ay < by + bheight &&
      ay + aheight > by
    );
  }
  
  /**
   * Detect collisions using WebAssembly for better performance
   * @returns {Array} Array of collision objects {bulletIndex, enemyIndex}
   */
  detectCollisionsWasm() {
    if (!this.isWasmReady || !this.wasm) {
      return this.detectCollisionsJS();
    }
    
    const memory = getWasmMemory();
    if (!memory) {
      return this.detectCollisionsJS();
    }
    
    const bulletCount = this.bulletManager.bulletCount;
    const enemyCount = this.enemyManager.enemyCount;
    
    if (bulletCount === 0 || enemyCount === 0) {
      return [];
    }
    
    // Only allocate memory for entities in visible area
    const visibleBullets = [];
    const visibleEnemies = [];
    
    // Gather visible bullets
    for (let i = 0; i < bulletCount; i++) {
      if (this.isInVisibleArea(this.bulletManager.x[i], this.bulletManager.y[i])) {
        visibleBullets.push({
          index: i,
          pos: this.worldToLocalPosition(
            this.bulletManager.x[i], 
            this.bulletManager.y[i]
          ),
          width: this.bulletManager.width[i],
          height: this.bulletManager.height[i]
        });
      }
    }
    
    // Gather visible enemies
    for (let i = 0; i < enemyCount; i++) {
      if (this.isInVisibleArea(this.enemyManager.x[i], this.enemyManager.y[i])) {
        visibleEnemies.push({
          index: i,
          pos: this.worldToLocalPosition(
            this.enemyManager.x[i], 
            this.enemyManager.y[i]
          ),
          width: this.enemyManager.width[i],
          height: this.enemyManager.height[i]
        });
      }
    }
    
    const visibleBulletCount = visibleBullets.length;
    const visibleEnemyCount = visibleEnemies.length;
    
    if (visibleBulletCount === 0 || visibleEnemyCount === 0) {
      return [];
    }
    
    // Setup typed arrays in JavaScript that map to WASM memory
    const FLOAT_SIZE = 4;
    const INT_SIZE = 4;
    
    // Calculate memory needs and offsets
    const bulletsSize = visibleBulletCount * 4 * FLOAT_SIZE; // x, y, width, height
    const enemiesSize = visibleEnemyCount * 4 * FLOAT_SIZE; // x, y, width, height
    const resultsSize = this.maxCollisions * 2 * INT_SIZE; // bulletIdx, enemyIdx pairs
    const totalSize = bulletsSize + enemiesSize + resultsSize;
    
    // Ensure WASM memory is large enough
    const currentPages = memory.buffer.byteLength / 65536; // 64KB per page
    const requiredPages = Math.ceil(totalSize / 65536) + 1;
    
    if (currentPages < requiredPages) {
      memory.grow(requiredPages - currentPages);
      console.log(`Grew WASM memory to ${requiredPages} pages`);
    }
    
    // Get the memory view
    const memoryBuffer = new ArrayBuffer(memory.buffer.byteLength);
    new Uint8Array(memoryBuffer).set(new Uint8Array(memory.buffer));
    
    const dataView = new DataView(memoryBuffer);
    
    // Setup offsets in memory
    let offset = 0;
    
    // Bullet arrays
    const bulletXPtr = offset;
    offset += visibleBulletCount * FLOAT_SIZE;
    
    const bulletYPtr = offset;
    offset += visibleBulletCount * FLOAT_SIZE;
    
    const bulletWidthPtr = offset;
    offset += visibleBulletCount * FLOAT_SIZE;
    
    const bulletHeightPtr = offset;
    offset += visibleBulletCount * FLOAT_SIZE;
    
    // Enemy arrays
    const enemyXPtr = offset;
    offset += visibleEnemyCount * FLOAT_SIZE;
    
    const enemyYPtr = offset;
    offset += visibleEnemyCount * FLOAT_SIZE;
    
    const enemyWidthPtr = offset;
    offset += visibleEnemyCount * FLOAT_SIZE;
    
    const enemyHeightPtr = offset;
    offset += visibleEnemyCount * FLOAT_SIZE;
    
    // Result array
    const collisionPairsPtr = offset;
    
    // Copy bullet data to WASM memory
    for (let i = 0; i < visibleBulletCount; i++) {
      dataView.setFloat32(bulletXPtr + i * FLOAT_SIZE, visibleBullets[i].pos.x, true);
      dataView.setFloat32(bulletYPtr + i * FLOAT_SIZE, visibleBullets[i].pos.y, true);
      dataView.setFloat32(bulletWidthPtr + i * FLOAT_SIZE, visibleBullets[i].width, true);
      dataView.setFloat32(bulletHeightPtr + i * FLOAT_SIZE, visibleBullets[i].height, true);
    }
    
    // Copy enemy data to WASM memory
    for (let i = 0; i < visibleEnemyCount; i++) {
      dataView.setFloat32(enemyXPtr + i * FLOAT_SIZE, visibleEnemies[i].pos.x, true);
      dataView.setFloat32(enemyYPtr + i * FLOAT_SIZE, visibleEnemies[i].pos.y, true);
      dataView.setFloat32(enemyWidthPtr + i * FLOAT_SIZE, visibleEnemies[i].width, true);
      dataView.setFloat32(enemyHeightPtr + i * FLOAT_SIZE, visibleEnemies[i].height, true);
    }
    
    // Call WASM collision detection
    const collisionCount = this.wasm.detectCollisions(
      bulletXPtr / FLOAT_SIZE,
      bulletYPtr / FLOAT_SIZE,
      bulletWidthPtr / FLOAT_SIZE,
      bulletHeightPtr / FLOAT_SIZE,
      visibleBulletCount,
      
      enemyXPtr / FLOAT_SIZE,
      enemyYPtr / FLOAT_SIZE,
      enemyWidthPtr / FLOAT_SIZE,
      enemyHeightPtr / FLOAT_SIZE,
      visibleEnemyCount,
      
      this.cellSize,
      
      collisionPairsPtr / INT_SIZE,
      this.maxCollisions
    );
    
    // Read collision results and map back to original indices
    const collisions = [];
    for (let i = 0; i < collisionCount; i++) {
      const bulletLocalIndex = dataView.getInt32(collisionPairsPtr + i * 2 * INT_SIZE, true);
      const enemyLocalIndex = dataView.getInt32(collisionPairsPtr + i * 2 * INT_SIZE + INT_SIZE, true);
      
      // Map back to global indices
      const bulletIndex = visibleBullets[bulletLocalIndex].index;
      const enemyIndex = visibleEnemies[enemyLocalIndex].index;
      
      collisions.push({
        bulletIndex,
        enemyIndex
      });
    }
    
    return collisions;
  }
}