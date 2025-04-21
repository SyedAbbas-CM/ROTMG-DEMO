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
        
        // Check for network manager
        if (!this.networkManager) {
            console.warn("No networkManager provided to CollisionManager. Will attempt to get from gameState when needed.");
        }
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
            const ownerId = this.bulletManager.ownerId[i];
            const isEnemyBullet = ownerId && typeof ownerId === 'string' && ownerId.startsWith('enemy_');
            
            if (isEnemyBullet) {
                continue;
            }
            
            // Add player bullets to the grid
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
            
            // Skip bullets fired by enemies (double-check)
            const ownerId = this.bulletManager.ownerId[bulletIndex];
            const isEnemyBullet = ownerId && typeof ownerId === 'string' && ownerId.startsWith('enemy_');
            
            if (isEnemyBullet) {
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
     * AABB collision test with precise square hitboxes
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
        // Calculate center points
        const acx = ax + awidth / 2;
        const acy = ay + aheight / 2;
        const bcx = bx + bwidth / 2;
        const bcy = by + bheight / 2;
        
        // Use precise square hitboxes - reduce size by fixed percentage
        // Small bullets should have smaller hitboxes
        const bulletSizeFactor = awidth < 10 ? 0.4 : 0.5;
        const enemySizeFactor = 0.6; // Keep enemy hitboxes a bit larger
        
        // Adjust hitbox size
        const awidthAdjusted = awidth * bulletSizeFactor;
        const aheightAdjusted = aheight * bulletSizeFactor;
        const bwidthAdjusted = bwidth * enemySizeFactor;
        const bheightAdjusted = bheight * enemySizeFactor;
        
        // Calculate exact square bounds with adjusted sizes
        const a_left = acx - awidthAdjusted / 2;
        const a_right = acx + awidthAdjusted / 2;
        const a_top = acy - aheightAdjusted / 2;
        const a_bottom = acy + aheightAdjusted / 2;
        
        const b_left = bcx - bwidthAdjusted / 2;
        const b_right = bcx + bwidthAdjusted / 2;
        const b_top = bcy - bheightAdjusted / 2;
        const b_bottom = bcy + bheightAdjusted / 2;
        
        // Perfect square collision test
        const colliding = 
            a_right >= b_left && 
            a_left <= b_right &&
            a_bottom >= b_top && 
            a_top <= b_bottom;
        
        // For debugging, occassionally log collision details
        if (colliding && Math.random() < 0.1) {
            console.log(`Collision details:
- Bullet center: (${acx.toFixed(2)}, ${acy.toFixed(2)}), adjusted size: ${awidthAdjusted.toFixed(2)}x${aheightAdjusted.toFixed(2)}
- Enemy center: (${bcx.toFixed(2)}, ${bcy.toFixed(2)}), adjusted size: ${bwidthAdjusted.toFixed(2)}x${bheightAdjusted.toFixed(2)}
- Overlap X: ${Math.min(a_right, b_right) - Math.max(a_left, b_left)}
- Overlap Y: ${Math.min(a_bottom, b_bottom) - Math.max(a_top, b_top)}
`);
        }
        
        // Add collision visualization if debug flag is enabled
        if (colliding && window.DEBUG_COLLISIONS) {
            this.drawCollisionDebug(a_left, a_top, awidthAdjusted, aheightAdjusted, 
                                    b_left, b_top, bwidthAdjusted, bheightAdjusted);
        }
        
        return colliding;
    }
    
    /**
     * Draw collision debug visualization
     * @param {number} ax - First rect left
     * @param {number} ay - First rect top
     * @param {number} awidth - First rect width
     * @param {number} aheight - First rect height
     * @param {number} bx - Second rect left
     * @param {number} by - Second rect top
     * @param {number} bwidth - Second rect width
     * @param {number} bheight - Second rect height
     */
    drawCollisionDebug(ax, ay, awidth, aheight, bx, by, bwidth, bheight) {
        // This will be called when a collision is detected, so need to draw in world space
        // Get canvas context if available
        const canvas = document.getElementById('debugCanvas');
        if (!canvas) return;
        
        // Check if debug canvas exists, create it if not
        if (!canvas) {
            const newCanvas = document.createElement('canvas');
            newCanvas.id = 'debugCanvas';
            newCanvas.style.position = 'absolute';
            newCanvas.style.top = '0';
            newCanvas.style.left = '0';
            newCanvas.style.pointerEvents = 'none';
            newCanvas.width = window.innerWidth;
            newCanvas.height = window.innerHeight;
            document.body.appendChild(newCanvas);
            return; // Skip this frame, will draw on next collision
        }
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Convert world positions to screen positions
        const camera = window.gameState?.camera;
        if (!camera || !camera.worldToScreen) return;
        
        const screenWidth = canvas.width;
        const screenHeight = canvas.height;
        
        // Draw bullet hitbox
        const bulletPos = camera.worldToScreen(ax + awidth/2, ay + aheight/2, screenWidth, screenHeight);
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            bulletPos.x - awidth/2,
            bulletPos.y - aheight/2,
            awidth,
            aheight
        );
        
        // Draw enemy hitbox
        const enemyPos = camera.worldToScreen(bx + bwidth/2, by + bheight/2, screenWidth, screenHeight);
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            enemyPos.x - bwidth/2,
            enemyPos.y - bheight/2,
            bwidth,
            bheight
        );
        
        // Draw overlap area
        const overlapLeft = Math.max(ax, bx);
        const overlapTop = Math.max(ay, by);
        const overlapRight = Math.min(ax + awidth, bx + bwidth);
        const overlapBottom = Math.min(ay + aheight, by + bheight);
        
        if (overlapRight > overlapLeft && overlapBottom > overlapTop) {
            const overlapWidth = overlapRight - overlapLeft;
            const overlapHeight = overlapBottom - overlapTop;
            
            const overlapPos = camera.worldToScreen(
                overlapLeft + overlapWidth/2, 
                overlapTop + overlapHeight/2,
                screenWidth, 
                screenHeight
            );
            
            ctx.fillStyle = 'rgba(255, 255, 0, 0.5)';
            ctx.fillRect(
                overlapPos.x - overlapWidth/2,
                overlapPos.y - overlapHeight/2,
                overlapWidth,
                overlapHeight
            );
        }
        
        // Set timeout to clear the debug visualization
        setTimeout(() => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }, 500);
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
        
        // Get bullet data for verification
        const bulletData = {
            id: bulletId,
            x: this.bulletManager.x[bulletIndex],
            y: this.bulletManager.y[bulletIndex],
            ownerId: this.bulletManager.ownerId[bulletIndex]
        };
        
        // Get enemy data for verification
        const enemyData = {
            id: enemyId,
            x: this.enemyManager.x[enemyIndex],
            y: this.enemyManager.y[enemyIndex],
            health: this.enemyManager.health[enemyIndex]
        };
        
        // Log collision detected with more details for debugging
        console.log(`Collision detected: Bullet ${bulletId} (${bulletData.x.toFixed(2)},${bulletData.y.toFixed(2)}) hit Enemy ${enemyId} (${enemyData.x.toFixed(2)},${enemyData.y.toFixed(2)})`);
        
        // Mark as processed immediately to prevent duplicates
        this.processedCollisions.set(collisionId, Date.now());
        
        // Apply client-side prediction for immediate feedback
        this.applyClientPrediction(bulletIndex, enemyIndex);
        
        // Try to get networkManager from gameState if not directly available
        let networkManager = this.networkManager;
        if (!networkManager && window.gameState && window.gameState.networkManager) {
            networkManager = window.gameState.networkManager;
            console.log("Using networkManager from gameState");
            // Store for future use
            this.networkManager = networkManager;
        }
        
        // Report collision to server if we have network manager
        if (networkManager) {
            if (networkManager.isConnected()) {
                try {
                    console.log(`Reporting collision to server: Bullet ${bulletId} hit Enemy ${enemyId}`);
                    networkManager.sendCollision({
                        bulletId,
                        enemyId,
                        clientId: this.localPlayerId || (window.gameState?.character?.id),
                        timestamp: Date.now(),
                        // Include additional position data for server validation
                        bulletPos: { x: bulletData.x, y: bulletData.y },
                        enemyPos: { x: enemyData.x, y: enemyData.y }
                    });
                } catch (error) {
                    console.error("Error sending collision to server:", error);
                }
            } else {
                console.warn(`Cannot report collision to server: NetworkManager disconnected`);
            }
        } else {
            console.warn(`Cannot report collision to server: NetworkManager not available. Make sure it's properly initialized.`);
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