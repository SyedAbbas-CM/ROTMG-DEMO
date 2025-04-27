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
        
        // Add debug flag for coordinate system debugging
        this.debugCoordinates = true; // Enable coordinate debugging
        
        // Add debug flags for collision visualization
        this.debugWallCollisions = true; // Enable wall collision debugging
        this.debugEntityCollisions = true; // Enable entity collision debugging
        
        // Debug logging frequency (0-1)
        this.debugLogFrequency = 0.1; // Log 10% of collisions
        
        // Check for network manager
        if (!this.networkManager) {
            console.warn("No networkManager provided to CollisionManager. Will attempt to get from gameState when needed.");
        } else {
            console.log("NetworkManager successfully initialized in CollisionManager");
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
     * Debug method to check coordinate systems
     * @param {number} x - X coordinate to check
     * @param {number} y - Y coordinate to check
     */
    debugCoordinateSystem(x, y) {
        if (!this.debugCoordinates) return;
        
        // Get map manager reference
        const mapManager = this.mapManager || window.gameState?.map;
        if (!mapManager) {
            console.warn("Cannot debug coordinates: No map manager available");
            return;
        }
        
        // Get tile size
        const tileSize = mapManager.tileSize || 12;
        
        // Calculate tile coordinates from world coordinates
        const tileX = Math.floor(x / tileSize);
        const tileY = Math.floor(y / tileSize);
        
        // Get tile info at this position if available
        let tileInfo = "Unknown";
        if (mapManager && mapManager.getTile) {
            const tile = mapManager.getTile(tileX, tileY);
            tileInfo = tile ? `Type: ${tile.type}` : "No tile";
        }
        
        // Get chunk info if available
        let chunkInfo = "Unknown";
        if (mapManager && mapManager.getChunkCoordinates) {
            const chunkCoords = mapManager.getChunkCoordinates(x, y);
            chunkInfo = `Chunk (${chunkCoords.x}, ${chunkCoords.y})`;
        } else {
            // Calculate chunk coordinates if method not available
            const chunkSize = mapManager.chunkSize || 16;
            const chunkX = Math.floor(tileX / chunkSize);
            const chunkY = Math.floor(tileY / chunkSize);
            chunkInfo = `Estimated Chunk (${chunkX}, ${chunkY})`;
        }
        
        console.log(`COORDINATE DEBUG at (${x.toFixed(2)}, ${y.toFixed(2)}):
- World to Tile: (${tileX}, ${tileY}) [using tileSize=${tileSize}]
- Tile: ${tileInfo}
- ${chunkInfo}
- Is Wall/Obstacle: ${mapManager.isWallOrObstacle?.(x, y) ? 'Yes' : 'No'}`);
        
        // If window.gameState exists, check camera coordinates
        if (window.gameState && window.gameState.camera) {
            const camera = window.gameState.camera;
            console.log(`- Camera at (${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)})`);
            console.log(`- Distance from camera: ${Math.sqrt(
                Math.pow(x - camera.position.x, 2) + 
                Math.pow(y - camera.position.y, 2)
            ).toFixed(2)} units`);
        }
        
        // Check nearby walls/obstacles
        this.debugNearbyWalls(x, y, tileX, tileY, mapManager);
    }
    
    /**
     * Debug nearby walls around a position
     * @param {number} worldX - World X coordinate
     * @param {number} worldY - World Y coordinate
     * @param {number} tileX - Tile X coordinate
     * @param {number} tileY - Tile Y coordinate
     * @param {Object} mapManager - Map manager reference
     */
    debugNearbyWalls(worldX, worldY, tileX, tileY, mapManager) {
        if (!mapManager || !mapManager.isWallOrObstacle) return;
        
        console.log("Checking nearby walls...");
        const tileSize = mapManager.tileSize || 12;
        const searchRadius = 3;
        
        // Check walls in a grid around the position
        let wallsFound = 0;
        const nearbyWalls = [];
        
        for (let dy = -searchRadius; dy <= searchRadius; dy++) {
            for (let dx = -searchRadius; dx <= searchRadius; dx++) {
                const checkTileX = tileX + dx;
                const checkTileY = tileY + dy;
                const checkWorldX = checkTileX * tileSize + tileSize/2;
                const checkWorldY = checkTileY * tileSize + tileSize/2;
                
                if (mapManager.isWallOrObstacle(checkWorldX, checkWorldY)) {
                    wallsFound++;
                    nearbyWalls.push({
                        tile: {x: checkTileX, y: checkTileY},
                        world: {x: checkWorldX, y: checkWorldY},
                        distance: Math.sqrt(
                            Math.pow(worldX - checkWorldX, 2) + 
                            Math.pow(worldY - checkWorldY, 2)
                        ).toFixed(2)
                    });
                }
            }
        }
        
        if (wallsFound > 0) {
            console.log(`Found ${wallsFound} nearby walls:`);
            nearbyWalls.sort((a, b) => a.distance - b.distance);
            nearbyWalls.slice(0, 5).forEach(wall => {
                console.log(`- Wall at tile (${wall.tile.x}, ${wall.tile.y}), world (${wall.world.x.toFixed(2)}, ${wall.world.y.toFixed(2)}), distance: ${wall.distance}`);
            });
        } else {
            console.log("No walls found nearby");
        }
    }
    
    /**
     * AABB collision test with precise square hitboxes
     * This is the main collision detection logic for entities
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
        
        // For debugging, log collision details
        if (colliding && (Math.random() < this.debugLogFrequency || this.debugCoordinates)) {
            console.log(`ENTITY COLLISION DETAILS:
- Bullet center: (${acx.toFixed(2)}, ${acy.toFixed(2)}), adjusted size: ${awidthAdjusted.toFixed(2)}x${aheightAdjusted.toFixed(2)}
- Enemy center: (${bcx.toFixed(2)}, ${bcy.toFixed(2)}), adjusted size: ${bwidthAdjusted.toFixed(2)}x${bheightAdjusted.toFixed(2)}
- Overlap X: ${Math.min(a_right, b_right) - Math.max(a_left, b_left)}
- Overlap Y: ${Math.min(a_bottom, b_bottom) - Math.max(a_top, b_top)}
`);
            
            // Check coordinate system for the bullet position
            if (this.debugCoordinates) {
                this.debugCoordinateSystem(acx, acy);
            }
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
        
        const totalBullets = this.bulletManager.bulletCount;
        const tileSize = this.mapManager.tileSize || 12; // Get tile size for debugging
        let wallCollisionsDetected = 0;
        
        for (let i = 0; i < totalBullets; i++) {
            const x = this.bulletManager.x[i];
            const y = this.bulletManager.y[i];
            
            // Skip if invalid position
            if (x === undefined || y === undefined) continue;
            
            // Calculate tile position for debugging
            const tileX = Math.floor(x / tileSize);
            const tileY = Math.floor(y / tileSize);
            
            // Check if position is wall or out of bounds
            if (this.mapManager.isWallOrObstacle) {
                const isWall = this.mapManager.isWallOrObstacle(x, y);
                
                // Log wall collision details for debugging
                if (isWall && (Math.random() < this.debugLogFrequency || this.debugWallCollisions)) {
                    console.log(`WALL COLLISION DETECTED:
- Bullet world position: (${x.toFixed(2)}, ${y.toFixed(2)})
- Bullet tile position: (${tileX}, ${tileY}) [using tileSize=${tileSize}]
- Bullet ID: ${this.bulletManager.id[i]}
- Bullet owner: ${this.bulletManager.ownerId[i]}`);
                    
                    // Get tile type for more detail if possible
                    if (this.mapManager.getTile) {
                        const tile = this.mapManager.getTile(tileX, tileY);
                        console.log(`- Tile at collision: ${JSON.stringify(tile)}`);
                    }
                    
                    // Debug the coordinate system at this position
                    if (this.debugCoordinates) {
                        this.debugCoordinateSystem(x, y);
                    }
                    
                    wallCollisionsDetected++;
                }
                
                // Mark bullet for removal if collision detected
                if (isWall) {
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
            } else {
                // If method doesn't exist, log warning only occasionally to prevent spam
                if (Math.random() < 0.01) {
                    console.warn('Warning: mapManager.isWallOrObstacle() method not available for bullet-wall collision detection');
                }
            }
        }
        
        // Log summary if wall collisions were detected
        if (wallCollisionsDetected > 0 && Math.random() < 0.2) {
            console.log(`Detected ${wallCollisionsDetected} bullet-wall collisions this frame`);
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
        
        // Calculate tile coordinates for debugging
        const mapManager = this.mapManager || window.gameState?.map;
        const tileSize = mapManager?.tileSize || 12;
        const bulletTileX = Math.floor(bulletData.x / tileSize);
        const bulletTileY = Math.floor(bulletData.y / tileSize);
        const enemyTileX = Math.floor(enemyData.x / tileSize);
        const enemyTileY = Math.floor(enemyData.y / tileSize);
        
        // Enhanced collision logging with tile coordinates
        console.log(`COLLISION DETECTED: 
- Bullet ${bulletId} at world (${bulletData.x.toFixed(2)},${bulletData.y.toFixed(2)}), tile (${bulletTileX},${bulletTileY})
- Enemy ${enemyId} at world (${enemyData.x.toFixed(2)},${enemyData.y.toFixed(2)}), tile (${enemyTileX},${enemyTileY})
- Using tileSize=${tileSize}`);
        
        // Track entity collision in global stats
        if (window.COLLISION_STATS) {
            window.COLLISION_STATS.entityCollisions++;
            
            // Store collision details for analysis
            if (!window.COLLISION_STATS.lastEntityCollisions) {
                window.COLLISION_STATS.lastEntityCollisions = [];
            }
            
            const collisionDetails = {
                timestamp: Date.now(),
                bullet: {
                    id: bulletId,
                    x: bulletData.x,
                    y: bulletData.y,
                    tileX: bulletTileX,
                    tileY: bulletTileY,
                    ownerId: bulletData.ownerId
                },
                enemy: {
                    id: enemyId,
                    x: enemyData.x,
                    y: enemyData.y,
                    tileX: enemyTileX,
                    tileY: enemyTileY,
                    health: enemyData.health
                }
            };
            
            // Keep only the last 10 collisions
            window.COLLISION_STATS.lastEntityCollisions.unshift(collisionDetails);
            if (window.COLLISION_STATS.lastEntityCollisions.length > 10) {
                window.COLLISION_STATS.lastEntityCollisions.pop();
            }
        }
        
        // Mark as processed immediately to prevent duplicates
        this.processedCollisions.set(collisionId, Date.now());
        
        // Apply client-side prediction for immediate feedback
        this.applyClientPrediction(bulletIndex, enemyIndex);
        
        // UPDATED: Better network manager handling
        // First check direct reference in this instance
        let networkManager = this.networkManager;
        
        // If not available, try to get from gameState global
        if (!networkManager && window.gameState) {
            networkManager = window.gameState.networkManager;
            
            // If we found it in gameState, save for future use
            if (networkManager) {
                console.log("Found networkManager in gameState, storing for future use");
                this.networkManager = networkManager;
            }
        }
        
        // Report collision to server if we have network manager
        if (networkManager) {
            if (networkManager.isConnected && networkManager.isConnected()) {
                try {
                    console.log(`Reporting collision to server: Bullet ${bulletId} hit Enemy ${enemyId}`);
                    networkManager.sendCollision({
                        bulletId,
                        enemyId,
                        clientId: this.localPlayerId || (window.gameState?.character?.id),
                        timestamp: Date.now(),
                        // Include additional position data for server validation
                        bulletPos: { 
                            x: bulletData.x, 
                            y: bulletData.y,
                            tileX: bulletTileX,
                            tileY: bulletTileY
                        },
                        enemyPos: { 
                            x: enemyData.x, 
                            y: enemyData.y,
                            tileX: enemyTileX,
                            tileY: enemyTileY
                        }
                    });
                } catch (error) {
                    console.error("Error sending collision to server:", error);
                }
            } else {
                console.warn(`Cannot report collision to server: NetworkManager not connected`);
            }
        } else {
            console.warn(`Cannot report collision to server: NetworkManager not available. Make sure it's properly initialized.`);
            console.warn(`Possible solutions: 1) Check if networkManager is created in game.js 2) Make sure it's assigned to gameState.networkManager`);
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