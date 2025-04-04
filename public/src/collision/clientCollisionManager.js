// public/src/collision/ClientCollisionManager.js

import SpatialGrid from '../shared/spatialGrid.js';

/**
 * ClientCollisionManager
 * Handles collision detection on the client side and reports to server
 */
export class ClientCollisionManager {
    /**
     * Create a new client collision manager
     * @param {Object} options - Collision manager options
     */
    constructor(options = {}) {
        this.bulletManager = options.bulletManager;
        this.enemyManager = options.enemyManager;
        this.mapManager = options.mapManager;
        this.networkManager = options.networkManager;
        this.localPlayerId = options.localPlayerId;
        
        // Spatial partitioning for efficient collision detection
        this.gridCellSize = 64; // Size of each grid cell
        this.grid = new SpatialGrid(this.gridCellSize, 2000, 2000);
        
        // Collision tracking to prevent duplicates
        this.processedCollisions = new Map(); // collisionId -> timestamp
        this.collisionTimeout = 500; // ms until a collision can be processed again
        
        // Setup cleanup interval
        this.cleanupInterval = setInterval(() => this.cleanupProcessedCollisions(), 5000);
    }
    
    /**
     * Update collision detection
     * @param {number} deltaTime - Time since last update in seconds
     */
    update(deltaTime) {
        if (!this.bulletManager || !this.enemyManager) return;
        
        // Clear the spatial grid
        this.grid.clear();
        
        // Insert bullets into grid
        for (let i = 0; i < this.bulletManager.bulletCount; i++) {
            // Skip bullets fired by enemies
            if (this.bulletManager.ownerId && 
                this.bulletManager.ownerId[i] && 
                typeof this.bulletManager.ownerId[i] === 'string' &&
                this.bulletManager.ownerId[i].startsWith('enemy_')) {
                continue;
            }
            
            this.grid.insertBullet(
                i,
                this.bulletManager.x[i],
                this.bulletManager.y[i],
                this.bulletManager.width[i],
                this.bulletManager.height[i]
            );
        }
        
        // Insert enemies into grid
        for (let i = 0; i < this.enemyManager.enemyCount; i++) {
            // Skip dead enemies
            if (this.enemyManager.health[i] <= 0) continue;
            
            this.grid.insertEnemy(
                i,
                this.enemyManager.x[i],
                this.enemyManager.y[i],
                this.enemyManager.width[i],
                this.enemyManager.height[i]
            );
        }
        
        // Get potential collision pairs
        const potentialPairs = this.grid.getPotentialCollisionPairs();
        
        // Check each potential collision
        for (const [bulletIndex, enemyIndex] of potentialPairs) {
            // Verify bullet and enemy still exist
            if (bulletIndex >= this.bulletManager.bulletCount || 
                enemyIndex >= this.enemyManager.enemyCount) {
                continue;
            }
            
            // Skip bullets fired by enemies
            if (this.bulletManager.ownerId &&
                this.bulletManager.ownerId[bulletIndex] &&
                typeof this.bulletManager.ownerId[bulletIndex] === 'string' &&
                this.bulletManager.ownerId[bulletIndex].startsWith('enemy_')) {
                continue;
            }
            
            // Skip dead enemies
            if (this.enemyManager.health[enemyIndex] <= 0) continue;
            
            // Full AABB collision check
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
                // Process this collision
                this.handleCollision(bulletIndex, enemyIndex);
            }
        }
        
        // Check for bullet-wall collisions if map manager exists
        this.checkBulletWallCollisions();
    }
    
    /**
     * AABB collision test
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
    
    /**
     * Check for bullet-wall collisions
     */
    checkBulletWallCollisions() {
        if (!this.mapManager || !this.bulletManager) return;
        
        for (let i = 0; i < this.bulletManager.bulletCount; i++) {
            const x = this.bulletManager.x[i];
            const y = this.bulletManager.y[i];
            
            // Check if position is wall or out of bounds
            if (this.mapManager.isWallOrObstacle(x, y)) {
                // Mark bullet for removal - handle case where markForRemoval doesn't exist
                if (this.bulletManager.markForRemoval) {
                    this.bulletManager.markForRemoval(i);
                } else if (this.bulletManager.removeBulletById && this.bulletManager.id) {
                    // Alternative method: remove by ID
                    this.bulletManager.removeBulletById(this.bulletManager.id[i]);
                } else if (this.bulletManager.life) {
                    // Last resort: set lifetime to 0
                    this.bulletManager.life[i] = 0;
                }
            }
        }
    }
    
    /**
     * Handle a bullet-enemy collision
     * @param {number} bulletIndex - Bullet index
     * @param {number} enemyIndex - Enemy index
     */
    handleCollision(bulletIndex, enemyIndex) {
        const bulletId = this.bulletManager.id[bulletIndex];
        const enemyId = this.enemyManager.id[enemyIndex];
        
        // Generate a unique collision ID
        const collisionId = `${bulletId}_${enemyId}`;
        
        // Check if this collision was already processed recently
        if (this.processedCollisions.has(collisionId)) {
            return;
        }
        
        // Mark as processed
        this.processedCollisions.set(collisionId, Date.now());
        
        // Apply client-side prediction
        this.applyClientPrediction(bulletIndex, enemyIndex);
        
        // Report collision to server if we have network manager
        if (this.networkManager && this.networkManager.isConnected()) {
            this.networkManager.sendCollision({
                bulletId,
                enemyId,
                clientId: this.localPlayerId,
                timestamp: Date.now()
            });
        }
    }
    
    /**
     * Apply client-side prediction for immediate feedback
     * @param {number} bulletIndex - Bullet index
     * @param {number} enemyIndex - Enemy index
     */
    applyClientPrediction(bulletIndex, enemyIndex) {
        // Mark bullet for removal
        this.bulletManager.markForRemoval(bulletIndex);
        
        // Apply hit effect to enemy
        if (this.enemyManager.applyHitEffect) {
            this.enemyManager.applyHitEffect(enemyIndex);
        }
    }
    
    /**
     * Clean up old processed collisions
     */
    cleanupProcessedCollisions() {
        const now = Date.now();
        
        for (const [id, timestamp] of this.processedCollisions.entries()) {
            if (now - timestamp > this.collisionTimeout) {
                this.processedCollisions.delete(id);
            }
        }
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }
    }
}