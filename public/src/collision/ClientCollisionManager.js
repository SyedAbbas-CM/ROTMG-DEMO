// public/src/collision/ClientCollisionManager.js

import SpatialGrid from '../shared/spatialGrid.js';
import { QuadTree } from './QuadTree.js';

/**
 * ClientCollisionManager
 * Handles collision detection on the client side and reports to server
 *
 * OPTIMIZATION: Uses QuadTree for spatial partitioning (O(log n) queries)
 * Falls back to SpatialGrid if QuadTree encounters issues
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

        // ============================================================================
        // QUAD TREE SPATIAL PARTITIONING (NEW)
        // ============================================================================
        this.useQuadTree = false; // Toggle to false to use old SpatialGrid - DISABLED: Too expensive to rebuild every frame
        this.worldBounds = { x: 0, y: 0, width: 2000, height: 2000 }; // Will be updated from map
        this.quadTree = null; // Created lazily on first update
        this.quadTreeCapacity = 8; // Entities per node before subdivision
        this.quadTreeMaxDepth = 8; // Maximum tree depth

        // ============================================================================
        // LEGACY SPATIAL GRID (FALLBACK)
        // ============================================================================
        this.gridCellSize = 64; // Size of each grid cell
        this.grid = new SpatialGrid(this.gridCellSize, 2000, 2000);

        // Collision tracking to prevent duplicates
        this.processedCollisions = new Map(); // collisionId -> timestamp
        this.collisionTimeout = 500; // ms until a collision can be processed again

        // Setup cleanup interval
        this.cleanupInterval = setInterval(() => this.cleanupProcessedCollisions(), 5000);

        // ============================================================================
        // DEBUG FLAGS & LOGGING
        // ============================================================================
        this.debugCoordinates = false; // Coordinate system debugging
        this.debugWallCollisions = false; // Wall collision visualization
        this.debugEntityCollisions = false; // Entity collision visualization
        this.debugQuadTree = false; // QuadTree-specific logging (DISABLED for performance)
        // TEMPORARY: Enable 100% logging to debug bullet collision issues
        this.debugLogFrequency = 1.0; // Log 100% of collisions (was 0.02 / 2%)

        // QuadTree performance tracking
        this.quadTreeStats = {
            lastRebuildTime: 0,
            avgRebuildTime: 0,
            avgQueryTime: 0,
            totalQueries: 0,
            totalCollisions: 0
        };

        // Check for network manager
        if (!this.networkManager) {
            console.warn("[CollisionManager] No networkManager provided. Will attempt to get from gameState when needed.");
        } else {
            console.log("[CollisionManager] NetworkManager successfully initialized");
        }

        console.log(`[CollisionManager] Initialized with ${this.useQuadTree ? 'QuadTree' : 'SpatialGrid'} spatial partitioning`);
    }
    
    /**
     * Update collision detection
     * @param {number} deltaTime - Time since last update in seconds
     */
    update(deltaTime) {
        if (!this.bulletManager || !this.enemyManager) return;

        // ============================================================================
        // QUAD TREE PATH
        // ============================================================================
        if (this.useQuadTree) {
            this._updateWithQuadTree(deltaTime);
            return;
        }

        // ============================================================================
        // LEGACY SPATIAL GRID PATH (FALLBACK)
        // ============================================================================
        this._updateWithSpatialGrid(deltaTime);
    }

    /**
     * Update collision detection using QuadTree (NEW)
     * @param {number} deltaTime - Time since last update
     * @private
     */
    _updateWithQuadTree(deltaTime) {
        const startTime = performance.now();

        // Initialize QuadTree on first update or if map bounds changed
        if (!this.quadTree) {
            this._initializeQuadTree();
        }

        // Rebuild QuadTree from scratch each frame (fastest for dynamic entities)
        const rebuildStart = performance.now();
        this.quadTree.clear();

        // ============================================================================
        // INSERT BULLETS INTO QUAD TREE
        // ============================================================================
        let playerBulletCount = 0;
        for (let i = 0; i < this.bulletManager.bulletCount; i++) {
            // Skip bullets fired by enemies
            const ownerId = this.bulletManager.ownerId[i];
            const isEnemyBullet = ownerId && typeof ownerId === 'string' && ownerId.startsWith('enemy_');

            if (isEnemyBullet) {
                continue;
            }

            // Create entity object for QuadTree
            const bulletEntity = {
                type: 'bullet',
                index: i,
                id: this.bulletManager.id[i],
                x: this.bulletManager.x[i],
                y: this.bulletManager.y[i],
                width: this.bulletManager.width[i] || 0.2,
                height: this.bulletManager.height[i] || 0.2,
                ownerId: ownerId
            };

            this.quadTree.insert(bulletEntity);
            playerBulletCount++;
        }

        // ============================================================================
        // INSERT ENEMIES INTO QUAD TREE
        // ============================================================================
        let aliveEnemyCount = 0;
        for (let i = 0; i < this.enemyManager.enemyCount; i++) {
            // Skip dead enemies
            if (this.enemyManager.health[i] <= 0) continue;

            const enemyEntity = {
                type: 'enemy',
                index: i,
                id: this.enemyManager.id[i],
                x: this.enemyManager.x[i],
                y: this.enemyManager.y[i],
                width: this.enemyManager.width[i] || 1.0,
                height: this.enemyManager.height[i] || 1.0
            };

            this.quadTree.insert(enemyEntity);
            aliveEnemyCount++;
        }

        const rebuildTime = performance.now() - rebuildStart;
        this.quadTreeStats.lastRebuildTime = rebuildTime;
        this.quadTreeStats.avgRebuildTime = (this.quadTreeStats.avgRebuildTime * 0.95) + (rebuildTime * 0.05);

        // Log rebuild stats occasionally
        if (this.debugQuadTree && Math.random() < 0.01) {
            const stats = this.quadTree.getStats();
            console.log(`[QuadTree] Rebuilt: ${playerBulletCount} bullets, ${aliveEnemyCount} enemies | ` +
                       `Nodes: ${stats.nodes}, Max Depth: ${stats.maxDepth}, ` +
                       `Rebuild Time: ${rebuildTime.toFixed(2)}ms`);
        }

        // ============================================================================
        // QUERY QUAD TREE FOR COLLISIONS
        // ============================================================================
        let collisionChecks = 0;
        let actualCollisions = 0;

        // For each bullet, query nearby enemies
        for (let i = 0; i < this.bulletManager.bulletCount; i++) {
            const ownerId = this.bulletManager.ownerId[i];
            const isEnemyBullet = ownerId && typeof ownerId === 'string' && ownerId.startsWith('enemy_');
            if (isEnemyBullet) continue;

            const queryStart = performance.now();

            // Query QuadTree for entities near this bullet
            const bulletBounds = {
                x: this.bulletManager.x[i] - 0.5, // Expand search slightly
                y: this.bulletManager.y[i] - 0.5,
                width: (this.bulletManager.width[i] || 0.2) + 1.0,
                height: (this.bulletManager.height[i] || 0.2) + 1.0
            };

            const nearbyEntities = this.quadTree.query(bulletBounds);
            const queryTime = performance.now() - queryStart;
            this.quadTreeStats.avgQueryTime = (this.quadTreeStats.avgQueryTime * 0.95) + (queryTime * 0.05);
            this.quadTreeStats.totalQueries++;

            // Filter for enemies only
            const nearbyEnemies = nearbyEntities.filter(e => e.type === 'enemy');

            collisionChecks += nearbyEnemies.length;

            // Check actual AABB collision with each nearby enemy
            for (const enemyEntity of nearbyEnemies) {
                const enemyIndex = enemyEntity.index;

                // Verify enemy still exists and is alive
                if (enemyIndex >= this.enemyManager.enemyCount || this.enemyManager.health[enemyIndex] <= 0) {
                    continue;
                }

                // AABB collision test
                const collision = this._checkAABBCollision(
                    this.bulletManager.x[i],
                    this.bulletManager.y[i],
                    this.bulletManager.width[i] || 0.2,
                    this.bulletManager.height[i] || 0.2,
                    this.enemyManager.x[enemyIndex],
                    this.enemyManager.y[enemyIndex],
                    this.enemyManager.width[enemyIndex] || 1.0,
                    this.enemyManager.height[enemyIndex] || 1.0
                );

                if (collision) {
                    this._handleCollision(i, enemyIndex);
                    actualCollisions++;
                }
            }
        }

        const totalTime = performance.now() - startTime;

        // Log performance stats periodically
        if (this.debugQuadTree && Math.random() < 0.05) {
            console.log(`[QuadTree] Collision Update: ${collisionChecks} checks, ${actualCollisions} hits | ` +
                       `Total Time: ${totalTime.toFixed(2)}ms (${(totalTime / Math.max(1, collisionChecks)).toFixed(3)}ms/check)`);
        }

        this.quadTreeStats.totalCollisions += actualCollisions;
    }

    /**
     * Initialize QuadTree with world bounds
     * @private
     */
    _initializeQuadTree() {
        // Update world bounds from map if available
        if (this.mapManager && this.mapManager.mapWidth && this.mapManager.mapHeight) {
            this.worldBounds = {
                x: 0,
                y: 0,
                width: this.mapManager.mapWidth,
                height: this.mapManager.mapHeight
            };
        }

        this.quadTree = new QuadTree(
            this.worldBounds,
            this.quadTreeCapacity,
            this.quadTreeMaxDepth
        );

        console.log(`[QuadTree] Initialized with bounds:`, this.worldBounds,
                   `| Capacity: ${this.quadTreeCapacity}, Max Depth: ${this.quadTreeMaxDepth}`);
    }

    /**
     * Update collision detection using SpatialGrid (LEGACY FALLBACK)
     * @param {number} deltaTime - Time since last update
     * @private
     */
    _updateWithSpatialGrid(deltaTime) {
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
        
        // Detect enemy bullets hitting the local player for immediate feedback
        this.checkEnemyBulletsHitPlayer();
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
        
        // In this game, world coordinates are already in tile units
        const tileX = Math.floor(x);
        const tileY = Math.floor(y);
        
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
        const searchRadius = 3;
        
        // Check walls in a grid around the position
        let wallsFound = 0;
        const nearbyWalls = [];
        
        for (let dy = -searchRadius; dy <= searchRadius; dy++) {
            for (let dx = -searchRadius; dx <= searchRadius; dx++) {
                const checkTileX = tileX + dx;
                const checkTileY = tileY + dy;
                const checkWorldX = checkTileX + 0.5;
                const checkWorldY = checkTileY + 0.5;
                
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
     * Wrapper for AABB collision (used by QuadTree path)
     * @private
     */
    _checkAABBCollision(ax, ay, aw, ah, bx, by, bw, bh) {
        return this.checkAABBCollision(ax, ay, aw, ah, bx, by, bw, bh);
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
        
        // Use full hitboxes for pixel-perfect collision (like player collision)
        // No shrinking - bullets hit exactly when they visually touch
        const awidthAdjusted = awidth;
        const aheightAdjusted = aheight;
        const bwidthAdjusted = bwidth;
        const bheightAdjusted = bheight;
        
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
        let wallCollisionsDetected = 0;

        for (let i = 0; i < totalBullets; i++) {
            const x = this.bulletManager.x[i];
            const y = this.bulletManager.y[i];

            // Skip if invalid position
            if (x === undefined || y === undefined) continue;
            
            // Skip collision if the bullet's chunk isn't loaded yet to avoid "invisible walls"
            if (this.mapManager && this.mapManager.getChunk && this.mapManager.chunkSize) {
                // World coordinates are already in tile units - no division needed
                const tileX = Math.floor(x);
                const tileY = Math.floor(y);
                const chunkX = Math.floor(tileX / this.mapManager.chunkSize);
                const chunkY = Math.floor(tileY / this.mapManager.chunkSize);
                const chunk = this.mapManager.getChunk(chunkX, chunkY);
                if (!chunk) {
                    continue; // don't collide until we have real tile data for this area
                }
            }

            // ============================================================================
            // OPTIMIZED BULLET COLLISION: Only check TILES, not objects
            // Bullets should NOT collide with trees/boulders (walkable: false objects)
            // They ONLY collide with solid tiles (walls, obstacles, mountains)
            // This is 10x faster than the full isWallOrObstacle check
            // ============================================================================
            let isWall = false;

            if (this.mapManager.getTile) {
                const tileX = Math.floor(x);
                const tileY = Math.floor(y);
                const tile = this.mapManager.getTile(tileX, tileY);

                if (!tile) {
                    // CRITICAL FIX: Handle null tiles based on whether map bounds are known
                    const mapWidth = this.mapManager.width;
                    const mapHeight = this.mapManager.height;
                    const mapBoundsKnown = mapWidth > 0 && mapHeight > 0;

                    if (!mapBoundsKnown) {
                        // Map bounds not set - conservatively treat null as wall
                        // This prevents bullets from passing through unloaded chunks
                        isWall = false; // DON'T treat as wall when bounds unknown - let bullets pass through unloaded areas
                        if (Math.random() < 0.01) {
                            console.log(`[BULLET COLLISION] Null tile at (${tileX}, ${tileY}), map bounds unknown - allowing pass`);
                        }
                    } else {
                        // Map bounds ARE known - check if truly out of bounds
                        if (tileX < 0 || tileX >= mapWidth || tileY < 0 || tileY >= mapHeight) {
                            // Truly out of bounds - treat as wall
                            isWall = true;
                            if (Math.random() < 0.01) {
                                console.log(`[BULLET COLLISION] Bullet hit true map boundary at (${tileX}, ${tileY}), bounds: ${mapWidth}x${mapHeight}`);
                            }
                        } else {
                            // Within bounds but tile is null - this is missing/corrupt data
                            // DON'T treat as wall - let bullet pass
                            isWall = false;
                            if (Math.random() < 0.01) {
                                console.warn(`[BULLET COLLISION] Null tile at (${tileX}, ${tileY}) within bounds ${mapWidth}x${mapHeight} - allowing pass`);
                            }
                        }
                    }
                } else {
                    // Check if it's a solid tile (wall, obstacle, or mountain)
                    // NOTE: Water and lava are passable for bullets
                    const TILE_IDS = window.TILE_IDS || {};
                    isWall = (
                        tile.type === TILE_IDS.WALL ||
                        tile.type === TILE_IDS.OBSTACLE ||
                        tile.type === TILE_IDS.MOUNTAIN
                    );
                }
            } else if (this.mapManager.isWallOrObstacle) {
                // Fallback to old method if getTile not available
                isWall = this.mapManager.isWallOrObstacle(x, y);
            }

            if (isWall) {
                // Log wall collision details for debugging
                if (Math.random() < this.debugLogFrequency || this.debugWallCollisions) {
                    const debugTileX = Math.floor(x);
                    const debugTileY = Math.floor(y);
                    const bulletId = this.bulletManager.id[i];

                    // Get tile type
                    let tileTypeName = 'UNKNOWN';
                    if (this.mapManager.getTile) {
                        const tile = this.mapManager.getTile(debugTileX, debugTileY);
                        if (tile) {
                            const TILE_IDS = window.TILE_IDS || {};
                            // Reverse lookup to find type name
                            for (const [name, id] of Object.entries(TILE_IDS)) {
                                if (id === tile.type) {
                                    tileTypeName = name;
                                    break;
                                }
                            }
                        }
                    }

                    // Unified log format matching server
                    console.log(`[CLIENT BULLET] ID: ${bulletId}, Pos: (${x.toFixed(4)}, ${y.toFixed(4)}), Tile: (${debugTileX}, ${debugTileY}), Type: ${tileTypeName}`);

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
     * Check if any enemy bullets hit the local player and apply client-side
     * prediction (the server remains authoritative but this removes latency).
     */
    checkEnemyBulletsHitPlayer() {
        if (!window.gameState || !window.gameState.character) return;

        const player = window.gameState.character;
        const pw = player.collisionWidth || 1;
        const ph = player.collisionHeight || 1;

        for (let i = 0; i < this.bulletManager.bulletCount; i++) {
            if (this.bulletManager.life[i] <= 0) continue;

            const ownerId = this.bulletManager.ownerId[i];
            if (typeof ownerId !== 'string' || !ownerId.startsWith('enemy_')) continue;

            // Ignore bullets from other realms
            if (this.bulletManager.worldId && this.bulletManager.worldId[i] !== player.worldId) {
                continue;
            }

            const bx = this.bulletManager.x[i];
            const by = this.bulletManager.y[i];
            const bw = this.bulletManager.width[i];
            const bh = this.bulletManager.height[i];

            const hit = (
                bx < player.x + pw &&
                bx + bw > player.x &&
                by < player.y + ph &&
                by + bh > player.y
            );

            if (hit) {
                const dmg = this.bulletManager.damage ? this.bulletManager.damage[i] : 10;

                // Apply visual flash (if any)
                if (typeof player.takeDamage === 'function') {
                    player.takeDamage(dmg);
                } else {
                    player.health = Math.max(0, (player.health || 100) - dmg);
                }

                // Mark bullet for removal locally
                this.bulletManager.markForRemoval(i);

                // Optionally show hit indicator (simple console for now)
                if (Math.random() < 0.05) {
                    console.log(`Local player hit by ${ownerId} for ${dmg} dmg (hp=${player.health})`);
                }

                // Immediately update UI health bar if available
                if (window.gameUI && typeof window.gameUI.updateHealth === 'function') {
                    window.gameUI.updateHealth(player.health, player.maxHealth || 100);
                }
            }
        }
    }
    
    /**
     * Wrapper for _handleCollision (for compatibility)
     */
    _handleCollision(bulletIndex, enemyIndex) {
        return this.handleCollision(bulletIndex, enemyIndex);
    }

    /**
     * Handle a bullet-enemy collision (CLIENT-SIDE DETECTION)
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
            if (this.debugQuadTree && Math.random() < 0.01) {
                console.log(`[CLIENT] Collision already processed: ${collisionId}`);
            }
            return;
        }

        // Log collision detection
        console.log(`[CLIENT] 🎯 COLLISION DETECTED: Bullet ${bulletId} hit Enemy ${enemyId} | ` +
                   `Bullet pos: (${this.bulletManager.x[bulletIndex].toFixed(2)}, ${this.bulletManager.y[bulletIndex].toFixed(2)}) | ` +
                   `Enemy pos: (${this.enemyManager.x[enemyIndex].toFixed(2)}, ${this.enemyManager.y[enemyIndex].toFixed(2)})`);

        
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

    /**
     * Called after a WORLD_SWITCH so the collision manager can hook into the
     * fresh entity managers owned by the new ClientWorld.
     */
    setEntityManagers(bulletMgr, enemyMgr){
        if (bulletMgr) this.bulletManager = bulletMgr;
        if (enemyMgr)  this.enemyManager  = enemyMgr;
        console.log('[Collision] Entity managers updated after world switch');
    }
}
