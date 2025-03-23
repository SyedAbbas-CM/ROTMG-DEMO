/**
 * CollisionSystem.js
 * Integrates WebAssembly-based collision detection for better performance
 */

/**
 * Base class for collision systems
 */
class CollisionSystem {
    /**
     * Create a collision system
     * @param {Object} options - System options
     * @param {BulletManager} options.bulletManager - Bullet manager
     * @param {EnemyManager} options.enemyManager - Enemy manager
     * @param {MapManager} options.mapManager - Map manager
     */
    constructor(options = {}) {
      this.bulletManager = options.bulletManager;
      this.enemyManager = options.enemyManager;
      this.mapManager = options.mapManager;
      this.enabled = true;
      this.lastUpdateTime = 0;
      this.updateInterval = 1000 / 60; // 60Hz default update rate
      this.callbacks = {
        onBulletEnemyCollision: null,
        onBulletWallCollision: null
      };
    }
    
    /**
     * Update collision detection
     * @param {number} currentTime - Current timestamp
     */
    update(currentTime) {
      if (!this.enabled) return;
      
      // Check if we should update based on interval
      if (currentTime - this.lastUpdateTime < this.updateInterval) {
        return;
      }
      
      this.lastUpdateTime = currentTime;
      
      // Check bullet-wall collisions
      this.checkBulletWallCollisions();
      
      // Check bullet-enemy collisions
      this.checkBulletEnemyCollisions();
    }
    
    /**
     * Set callback for bullet-enemy collisions
     * @param {Function} callback - Callback function(bulletIndex, enemyIndex)
     */
    setOnBulletEnemyCollision(callback) {
      this.callbacks.onBulletEnemyCollision = callback;
    }
    
    /**
     * Set callback for bullet-wall collisions
     * @param {Function} callback - Callback function(bulletIndex)
     */
    setOnBulletWallCollision(callback) {
      this.callbacks.onBulletWallCollision = callback;
    }
    
    /**
     * Check for bullet-enemy collisions
     * Implemented by subclasses
     */
    checkBulletEnemyCollisions() {
      throw new Error('Method not implemented');
    }
    
    /**
     * Check for bullet-wall collisions
     * Implemented by subclasses
     */
    checkBulletWallCollisions() {
      throw new Error('Method not implemented');
    }
  }
  
  /**
   * JavaScript-based collision system (fallback)
   */
  class JSCollisionSystem extends CollisionSystem {
    /**
     * Create a JS-based collision system
     * @param {Object} options - System options
     */
    constructor(options = {}) {
      super(options);
      this.gridCellSize = options.gridCellSize || 64;
      this.grid = new SpatialGrid(this.gridCellSize, 2000, 2000); // Adjust size as needed
    }
    
    /**
     * Check for bullet-enemy collisions
     */
    checkBulletEnemyCollisions() {
      if (!this.bulletManager || !this.enemyManager) return;
      
      // Clear grid
      this.grid.clear();
      
      // Insert bullets and enemies into grid
      for (let i = 0; i < this.bulletManager.bulletCount; i++) {
        this.grid.insertBullet(
          i,
          this.bulletManager.x[i],
          this.bulletManager.y[i],
          this.bulletManager.width[i] || 5,
          this.bulletManager.height[i] || 5
        );
      }
      
      for (let i = 0; i < this.enemyManager.enemyCount; i++) {
        this.grid.insertEnemy(
          i,
          this.enemyManager.x[i],
          this.enemyManager.y[i],
          this.enemyManager.width[i] || 20,
          this.enemyManager.height[i] || 20
        );
      }
      
      // Get potential collision pairs
      const pairs = this.grid.getPotentialCollisionPairs();
      
      // Check each pair
      for (const [bulletIndex, enemyIndex] of pairs) {
        // Skip invalid indices
        if (bulletIndex >= this.bulletManager.bulletCount || 
            enemyIndex >= this.enemyManager.enemyCount) {
          continue;
        }
        
        // Full AABB collision test
        if (this.checkAABBCollision(
          this.bulletManager.x[bulletIndex],
          this.bulletManager.y[bulletIndex],
          this.bulletManager.width[bulletIndex] || 5,
          this.bulletManager.height[bulletIndex] || 5,
          this.enemyManager.x[enemyIndex],
          this.enemyManager.y[enemyIndex],
          this.enemyManager.width[enemyIndex] || 20,
          this.enemyManager.height[enemyIndex] || 20
        )) {
          // Invoke collision callback
          if (this.callbacks.onBulletEnemyCollision) {
            this.callbacks.onBulletEnemyCollision(bulletIndex, enemyIndex);
          }
        }
      }
    }
    
    /**
     * Check for bullet-wall collisions
     */
    checkBulletWallCollisions() {
      if (!this.bulletManager || !this.mapManager) return;
      
      // Check each bullet
      for (let i = 0; i < this.bulletManager.bulletCount; i++) {
        const x = this.bulletManager.x[i];
        const y = this.bulletManager.y[i];
        
        // Check if bullet collides with wall
        if (this.mapManager.isWallOrObstacle(x, y)) {
          // Invoke collision callback
          if (this.callbacks.onBulletWallCollision) {
            this.callbacks.onBulletWallCollision(i);
          }
        }
      }
    }
    
    /**
     * Check AABB collision between two rectangles
     * @param {number} ax - First rect X
     * @param {number} ay - First rect Y
     * @param {number} awidth - First rect width
     * @param {number} aheight - First rect height
     * @param {number} bx - Second rect X
     * @param {number} by - Second rect Y
     * @param {number} bwidth - Second rect width
     * @param {number} bheight - Second rect height
     * @returns {boolean} True if colliding
     */
    checkAABBCollision(ax, ay, awidth, aheight, bx, by, bwidth, bheight) {
      return (
        ax < bx + bwidth &&
        ax + awidth > bx &&
        ay < by + bheight &&
        ay + aheight > by
      );
    }
  }
  
  /**
   * WebAssembly-based collision system
   */
  class WASMCollisionSystem extends CollisionSystem {
    /**
     * Create a WASM-based collision system
     * @param {Object} options - System options
     */
    constructor(options = {}) {
      super(options);
      this.wasm = null;
      this.memory = null;
      this.isLoaded = false;
      this.gridCellSize = options.gridCellSize || 64;
      this.maxCollisions = options.maxCollisions || 1000;
      
      // Arrays to store collision results
      this.collisionPairs = new Int32Array(this.maxCollisions * 2);
      
      // Load WASM module
      this.loadWASM();
    }
    
    /**
     * Load WASM module
     */
    async loadWASM() {
      try {
        const response = await fetch('/wasm/collision.wasm');
        const buffer = await response.arrayBuffer();
        
        // Create memory
        this.memory = new WebAssembly.Memory({
          initial: 10, // 10 pages = 640 KB
          maximum: 100 // 100 pages = 6.4 MB
        });
        
        // Instantiate module
        const result = await WebAssembly.instantiate(buffer, {
          env: {
            memory: this.memory
          }
        });
        
        // Get exports
        this.wasm = result.instance.exports;
        this.isLoaded = true;
        console.log('WASM collision module loaded');
      } catch (error) {
        console.error('Failed to load WASM collision module:', error);
      }
    }
    
    /**
     * Check for bullet-enemy collisions using WASM
     */
    checkBulletEnemyCollisions() {
      if (!this.isLoaded || !this.bulletManager || !this.enemyManager) {
        // Fall back to JS implementation if WASM not ready
        return super.checkBulletEnemyCollisions();
      }
      
      // Get entity counts
      const bulletCount = this.bulletManager.bulletCount;
      const enemyCount = this.enemyManager.enemyCount;
      
      if (bulletCount === 0 || enemyCount === 0) {
        return;
      }
      
      // Prepare memory
      const FLOAT_SIZE = 4;
      const INT_SIZE = 4;
      
      // Calculate memory needs
      const bulletsSize = bulletCount * 4 * FLOAT_SIZE; // x, y, width, height
      const enemiesSize = enemyCount * 4 * FLOAT_SIZE; // x, y, width, height
      const resultsSize = this.maxCollisions * 2 * INT_SIZE;
      const totalSize = bulletsSize + enemiesSize + resultsSize;
      
      // Ensure memory is large enough
      const currentPages = this.memory.buffer.byteLength / 65536;
      const requiredPages = Math.ceil(totalSize / 65536) + 1;
      
      if (currentPages < requiredPages) {
        this.memory.grow(requiredPages - currentPages);
      }
      
      // Get direct buffer view
      const buffer = new ArrayBuffer(this.memory.buffer.byteLength);
      new Uint8Array(buffer).set(new Uint8Array(this.memory.buffer));
      
      const dataView = new DataView(buffer);
      
      // Calculate offsets
      let offset = 0;
      
      const bulletXPtr = offset;
      offset += bulletCount * FLOAT_SIZE;
      
      const bulletYPtr = offset;
      offset += bulletCount * FLOAT_SIZE;
      
      const bulletWidthPtr = offset;
      offset += bulletCount * FLOAT_SIZE;
      
      const bulletHeightPtr = offset;
      offset += bulletCount * FLOAT_SIZE;
      
      const enemyXPtr = offset;
      offset += enemyCount * FLOAT_SIZE;
      
      const enemyYPtr = offset;
      offset += enemyCount * FLOAT_SIZE;
      
      const enemyWidthPtr = offset;
      offset += enemyCount * FLOAT_SIZE;
      
      const enemyHeightPtr = offset;
      offset += enemyCount * FLOAT_SIZE;
      
      const collisionResultsPtr = offset;
      
      // Copy bullet data to WASM memory
      for (let i = 0; i < bulletCount; i++) {
        dataView.setFloat32(bulletXPtr + i * FLOAT_SIZE, this.bulletManager.x[i], true);
        dataView.setFloat32(bulletYPtr + i * FLOAT_SIZE, this.bulletManager.y[i], true);
        dataView.setFloat32(bulletWidthPtr + i * FLOAT_SIZE, this.bulletManager.width[i] || 5, true);
        dataView.setFloat32(bulletHeightPtr + i * FLOAT_SIZE, this.bulletManager.height[i] || 5, true);
      }
      
      // Copy enemy data to WASM memory
      for (let i = 0; i < enemyCount; i++) {
        dataView.setFloat32(enemyXPtr + i * FLOAT_SIZE, this.enemyManager.x[i], true);
        dataView.setFloat32(enemyYPtr + i * FLOAT_SIZE, this.enemyManager.y[i], true);
        dataView.setFloat32(enemyWidthPtr + i * FLOAT_SIZE, this.enemyManager.width[i] || 20, true);
        dataView.setFloat32(enemyHeightPtr + i * FLOAT_SIZE, this.enemyManager.height[i] || 20, true);
      }
      
      // Call WASM function to detect collisions
      const collisionCount = this.wasm.detectCollisions(
        bulletXPtr / FLOAT_SIZE,
        bulletYPtr / FLOAT_SIZE,
        bulletWidthPtr / FLOAT_SIZE,
        bulletHeightPtr / FLOAT_SIZE,
        bulletCount,
        
        enemyXPtr / FLOAT_SIZE,
        enemyYPtr / FLOAT_SIZE,
        enemyWidthPtr / FLOAT_SIZE,
        enemyHeightPtr / FLOAT_SIZE,
        enemyCount,
        
        this.gridCellSize,
        
        collisionResultsPtr / INT_SIZE,
        this.maxCollisions
      );
      
      // Process collision results
      for (let i = 0; i < collisionCount; i++) {
        const bulletIndex = dataView.getInt32(collisionResultsPtr + i * 2 * INT_SIZE, true);
        const enemyIndex = dataView.getInt32(collisionResultsPtr + i * 2 * INT_SIZE + INT_SIZE, true);
        
        // Invoke collision callback
        if (this.callbacks.onBulletEnemyCollision) {
          this.callbacks.onBulletEnemyCollision(bulletIndex, enemyIndex);
        }
      }
    }
    
    /**
     * Check for bullet-wall collisions with batch processing
     */
    checkBulletWallCollisions() {
      if (!this.bulletManager || !this.mapManager) return;
      
      // We can use WASM for this too, but for simplicity, just use JS
      // In a future iteration, we could optimize this further with WASM
      for (let i = 0; i < this.bulletManager.bulletCount; i++) {
        const x = this.bulletManager.x[i];
        const y = this.bulletManager.y[i];
        
        if (this.mapManager.isWallOrObstacle(x, y)) {
          if (this.callbacks.onBulletWallCollision) {
            this.callbacks.onBulletWallCollision(i);
          }
        }
      }
    }
  }
  
  /**
   * Collision system factory - creates the best available system
   * @param {Object} options - System options
   * @returns {CollisionSystem} Collision system instance
   */
  function createCollisionSystem(options = {}) {
    if (typeof WebAssembly !== 'undefined') {
      return new WASMCollisionSystem(options);
    } else {
      console.log('WebAssembly not supported, using JS collision system');
      return new JSCollisionSystem(options);
    }
  }
  
  // Export modules
  export { 
    CollisionSystem, 
    JSCollisionSystem,
    WASMCollisionSystem,
    createCollisionSystem
  };
  
  // For browser
  if (typeof window !== 'undefined') {
    window.CollisionSystem = CollisionSystem;
    window.JSCollisionSystem = JSCollisionSystem;
    window.WASMCollisionSystem = WASMCollisionSystem;
    window.createCollisionSystem = createCollisionSystem;
  }