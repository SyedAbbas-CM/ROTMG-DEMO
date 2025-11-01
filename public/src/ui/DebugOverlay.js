// public/src/ui/DebugOverlay.js

/**
 * Debug overlay for monitoring game status and performance
 */
export class DebugOverlay {
    /**
     * Create a new debug overlay
     */
    constructor() {
        this.container = document.createElement('div');
        this.container.style.position = 'absolute';
        this.container.style.top = '40px';
        this.container.style.left = '10px';
        this.container.style.backgroundColor = 'rgba(0, 0, 0, 0.7)'; // Slightly darker for readability
        this.container.style.color = '#fff';
        this.container.style.padding = '10px';
        this.container.style.borderRadius = '5px';
        this.container.style.fontFamily = 'monospace';
        this.container.style.fontSize = '12px';
        this.container.style.zIndex = '1000';
        this.container.style.maxWidth = '400px'; // Wider to accommodate more info
        this.container.style.maxHeight = '80vh'; // Limit height
        this.container.style.overflowY = 'auto'; // Allow scrolling
        this.container.style.display = 'none'; // Hidden by default
        
        // Create sections
        this.fpsSection = this.createSection('FPS');
        this.networkSection = this.createSection('Network');
        this.entitiesSection = this.createSection('Entities');
        this.playerSection = this.createSection('Player');
        this.mapSection = this.createSection('Map');
        this.collisionSection = this.createSection('Collision');
        this.coordinatesSection = this.createSection('Coordinates');
        
        // Add to document
        document.body.appendChild(this.container);
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.fps = 0;
        
        // Network ping tracking
        this.pingHistory = [];
        this.lastPingTime = 0;
        this.currentPing = 0;
        this.pingInterval = 2000; // Send ping every 2 seconds
        
        // Collision tracking
        this.collisionStats = {
            totalWallChecks: 0,
            wallCollisions: 0,
            entityCollisions: 0,
            lastWallCollision: null,
            nearbyWalls: []
        };
        
        // Toggle with F3 key
        window.addEventListener('keydown', (e) => {
            if (e.code === 'F3') {
                this.toggle();
            }
        });

        // Basic constructor
        this.enabled = true; // Enabled by default (calculations happen)
        this.hide(); // Ensure it's hidden visually
    }
    
    /**
     * Create a section in the debug overlay
     * @param {string} title - Section title
     * @returns {Object} Section elements
     */
    createSection(title) {
        const section = document.createElement('div');
        section.style.marginBottom = '5px';
        
        const titleElem = document.createElement('div');
        titleElem.textContent = title;
        titleElem.style.borderBottom = '1px solid #aaa';
        titleElem.style.marginBottom = '2px';
        titleElem.style.color = '#8df'; // Blue title for better distinction
        
        const content = document.createElement('div');
        content.style.marginLeft = '10px';
        content.style.wordWrap = 'break-word'; // Ensure long lines wrap
        
        section.appendChild(titleElem);
        section.appendChild(content);
        this.container.appendChild(section);
        
        return { section, title: titleElem, content };
    }
    
    /**
     * Show the debug overlay
     */
    show() {
        this.container.style.display = 'block';
    }
    
    /**
     * Hide the debug overlay
     */
    hide() {
        this.container.style.display = 'none';
    }
    
    /**
     * Toggle the debug overlay
     */
    toggle() {
        if (this.container.style.display === 'none') {
            this.show();
        } else {
            this.hide();
        }
    }
    
    /**
     * Update the debug overlay with current game state
     * @param {Object} gameState - Current game state
     * @param {number} time - Current timestamp
     */
    update(gameState, time) {
        if (!this.enabled) return;
        
        // Update FPS counter
        this.frameCount++;
        
        if (time - this.lastFpsUpdate >= 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (time - this.lastFpsUpdate));
            this.frameCount = 0;
            this.lastFpsUpdate = time;
        }
        
        // Send ping request periodically
        if (gameState.networkManager && time - this.lastPingTime > this.pingInterval) {
            this.lastPingTime = time;
            this.sendPingRequest(gameState.networkManager);
        }
        
        // Update sections
        this.updateFpsSection();
        this.updateNetworkSection(gameState);
        this.updateEntitiesSection(gameState);
        this.updatePlayerSection(gameState);
        this.updateMapSection(gameState);
        this.updateCollisionSection(gameState);
        this.updateCoordinatesSection(gameState);
    }
    
    /**
     * Send a ping request to server
     * @param {Object} networkManager - Network manager instance
     */
    sendPingRequest(networkManager) {
        if (!networkManager || !networkManager.isConnected || !networkManager.isConnected()) {
            return;
        }
        
        const startTime = performance.now();
        
        try {
            // Use the standard network manager ping if available
            if (typeof networkManager.sendPing === 'function') {
                networkManager.sendPing();
                this._lastPingSentTime = startTime;
                
                // Register a handler for PONG message type if needed
                if (!this._pingHandlerAdded && networkManager.on) {
                    const MessageType = window.MessageType || { PONG: 4 };
                    
                    networkManager.on(MessageType.PONG, (data) => {
                        const pingTime = Math.round(performance.now() - this._lastPingSentTime);
                        this.updatePingTime(pingTime);
                    });
                    
                    this._pingHandlerAdded = true;
                    console.log('Ping handler registered for DebugOverlay');
                }
            } 
            // Fall back to socket.emit if needed
            else if (networkManager.socket && typeof networkManager.socket.emit === 'function') {
                // Add a ping handler if it doesn't exist
                if (!this._pingHandlerAdded) {
                    networkManager.socket.on('pong', () => {
                        const pingTime = Math.round(performance.now() - startTime);
                        this.updatePingTime(pingTime);
                    });
                    this._pingHandlerAdded = true;
                }
                
                networkManager.socket.emit('ping', { timestamp: Date.now() });
            }
            // Last resort, direct message sending
            else if (typeof networkManager.send === 'function') {
                // Try to get MessageType from window
                const MessageType = window.MessageType || { PING: 3, PONG: 4 };
                
                // Register for PONG once
                if (!this._pingHandlerAdded && networkManager.on) {
                    networkManager.on(MessageType.PONG, (data) => {
                        const pingTime = performance.now() - startTime;
                        this.updatePingTime(Math.round(pingTime));
                    });
                    this._pingHandlerAdded = true;
                }
                
                // Send ping with current timestamp
                networkManager.send(MessageType.PING, { time: Date.now() });
                this._lastPingSentTime = startTime;
            }
        } catch (error) {
            console.error('Error sending ping request:', error);
        }
    }
    
    /**
     * Update the ping time measurement
     * @param {number} pingTime - Ping time in ms
     */
    updatePingTime(pingTime) {
        // Validate ping time (ignore negative values or unreasonable values)
        if (pingTime < 0 || pingTime > 10000) {
            return;
        }
        
        this.currentPing = pingTime;
        this.pingHistory.push(pingTime);
        
        // Keep only the last 10 measurements
        if (this.pingHistory.length > 10) {
            this.pingHistory.shift();
        }
    }
    
    /**
     * Update FPS section
     */
    updateFpsSection() {
        this.fpsSection.content.textContent = `${this.fps} FPS`;
        
        // Color code based on performance
        if (this.fps >= 50) {
            this.fpsSection.content.style.color = '#8f8';
        } else if (this.fps >= 30) {
            this.fpsSection.content.style.color = '#ff8';
        } else {
            this.fpsSection.content.style.color = '#f88';
        }
    }
    
    /**
     * Update network section
     * @param {Object} gameState - Current game state
     */
    updateNetworkSection(gameState) {
        const network = gameState.networkManager;
        
        if (!network) {
            this.networkSection.content.textContent = 'Network manager not initialized';
            return;
        }
        
        const status = network.isConnected() ? 'Connected' : 'Disconnected';
        const clientId = network.getClientId() || 'Unknown';
        
        // Calculate ping statistics
        const averagePing = this.pingHistory.length > 0 ? 
            Math.round(this.pingHistory.reduce((sum, ping) => sum + ping, 0) / this.pingHistory.length) : 
            'N/A';
        
        const minPing = this.pingHistory.length > 0 ?
            Math.min(...this.pingHistory) :
            'N/A';
            
        const maxPing = this.pingHistory.length > 0 ?
            Math.max(...this.pingHistory) :
            'N/A';
        
        // Color code the ping value based on quality
        let pingColor = '#aaa';
        if (this.currentPing < 50) {
            pingColor = '#4ade80'; // Green for good ping
        } else if (this.currentPing < 100) {
            pingColor = '#facc15'; // Yellow for medium ping
        } else {
            pingColor = '#f87171'; // Red for poor ping
        }
        
        this.networkSection.content.innerHTML = 
            `Status: ${status}<br>` +
            `Client ID: ${clientId}<br>` +
            `Ping: <span style="color:${pingColor}">${this.currentPing}ms</span> (avg: ${averagePing}ms)<br>` +
            `Min: ${minPing}ms / Max: ${maxPing}ms<br>` +
            `Chunks loaded: ${gameState.map?.chunks?.size || 0}<br>` +
            `Chunks pending: ${gameState.map?.pendingChunks?.size || 0}`;
    }
    
    /**
     * Update entities section
     * @param {Object} gameState - Current game state
     */
    updateEntitiesSection(gameState) {
        const bulletCount = gameState.bulletManager ? gameState.bulletManager.bulletCount : 0;
        const enemyCount = gameState.enemyManager ? gameState.enemyManager.enemyCount : 0;
        
        // Check multiple places where other players might be stored
        let otherPlayerCount = 0;
        
        // Check in gameState.otherPlayers
        if (gameState.otherPlayers) {
            otherPlayerCount = Object.keys(gameState.otherPlayers).length;
        }
        // Check in gameState.players
        else if (gameState.players) {
            // Subtract 1 for the local player if it's included
            otherPlayerCount = Object.keys(gameState.players).length;
            if (gameState.clientId && gameState.players[gameState.clientId]) {
                otherPlayerCount--;
            }
        }
        // Check in gameState.playerManager
        else if (gameState.playerManager && gameState.playerManager.players) {
            otherPlayerCount = gameState.playerManager.players.size;
            // Subtract 1 for the local player if it's included
            if (gameState.clientId && gameState.playerManager.players.has(gameState.clientId)) {
                otherPlayerCount--;
            }
        }
        
        this.entitiesSection.content.innerHTML = 
            `Bullets: ${bulletCount}<br>` +
            `Enemies: ${enemyCount}<br>` +
            `Other Players: ${otherPlayerCount}`;
    }
    
    /**
     * Update player section
     * @param {Object} gameState - Current game state
     */
    updatePlayerSection(gameState) {
        const character = gameState.character;
        
        if (!character) {
            this.playerSection.content.textContent = 'Player not initialized';
            return;
        }
        
        // Get tile coordinates if map exists
        const tileSize = gameState.map?.tileSize || 12;
        const tileX = Math.floor(character.x / tileSize);
        const tileY = Math.floor(character.y / tileSize);
        
        // Calculate percentage within current tile
        const tilePercentX = ((character.x % tileSize) / tileSize).toFixed(2);
        const tilePercentY = ((character.y % tileSize) / tileSize).toFixed(2);
        
        // Calculate velocity
        const velocity = character.moveDirection ? 
            Math.sqrt(character.moveDirection.x ** 2 + character.moveDirection.y ** 2).toFixed(2) : 
            '0.00';
            
        this.playerSection.content.innerHTML = 
            `World: (${character.x.toFixed(2)}, ${character.y.toFixed(2)})<br>` +
            `Tile: (${tileX}, ${tileY}) [${tilePercentX}, ${tilePercentY}]<br>` +
            `Health: ${character.health}/${character.maxHealth || 100}<br>` +
            `Speed: ${character.speed || 'default'} (velocity: ${velocity})<br>` +
            `Moving: ${character.isMoving ? 'yes' : 'no'}`;
    }
    
    /**
     * Update map section
     * @param {Object} gameState - Current game state
     */
    updateMapSection(gameState) {
        const map = gameState.map;
        
        if (!map) {
            this.mapSection.content.textContent = 'Map not initialized';
            return;
        }
        
        // Get more detailed map info
        const tileSize = map.tileSize || 12;
        const chunkSize = map.chunkSize || 16;
        const worldWidth = map.width * tileSize;
        const worldHeight = map.height * tileSize;
        
        // Calculate chunk coordinates
        const chunkX = Math.floor(gameState.character.x / (chunkSize * tileSize));
        const chunkY = Math.floor(gameState.character.y / (chunkSize * tileSize));
        const chunksLoaded = map.chunks ? map.chunks.size : 0;
        
        // Get current chunk tiles if available
        const chunkKey = `${chunkX},${chunkY}`;
        const currentChunk = map.chunks ? map.chunks.get(chunkKey) : null;
        const chunkTileCount = currentChunk ? currentChunk.length * currentChunk[0].length : 'N/A';
        
        this.mapSection.content.innerHTML = 
            `Map ID: ${map.activeMapId || 'unknown'}<br>` +
            `Size: ${map.width}x${map.height} tiles (${worldWidth}x${worldHeight} units)<br>` +
            `Tile Size: ${tileSize}, Chunk Size: ${chunkSize}<br>` +
            `Current Chunk: (${chunkX}, ${chunkY}) - ${chunkTileCount} tiles<br>` +
            `Chunks Loaded: ${chunksLoaded}<br>` +
            `View: ${gameState.camera.viewType}`;
    }
    
    /**
     * Update collision section
     * @param {Object} gameState - Current game state
     */
    updateCollisionSection(gameState) {
        // Get current collision stats
        if (!this.collisionStats) {
            this.collisionStats = {
                totalWallChecks: 0,
                wallCollisions: 0,
                entityCollisions: 0,
                lastWallCollision: null,
                nearbyWalls: []
            };
        }
        
        // Update from global tracking if available
        if (window.COLLISION_STATS) {
            this.collisionStats = window.COLLISION_STATS;
        }
        
        // Get current character position
        const character = gameState.character;
        const map = gameState.map;
        
        if (character && map) {
            // Find nearby walls for debugging
            this.updateNearbyWalls(character.x, character.y, map);
        }
        
        // Format the last wall collision time
        const lastCollisionTime = this.collisionStats.lastWallCollision ? 
            `${Math.round((Date.now() - this.collisionStats.lastWallCollision) / 1000)}s ago` : 
            'none';
        
        // Wall collision percentage
        const collisionPercentage = this.collisionStats.totalWallChecks > 0 ? 
            ((this.collisionStats.wallCollisions / this.collisionStats.totalWallChecks) * 100).toFixed(1) + '%' : 
            'N/A';
        
        // Build collision display
        let collisionHtml = 
            `Wall Checks: ${this.collisionStats.totalWallChecks}<br>` +
            `Wall Collisions: ${this.collisionStats.wallCollisions} (${collisionPercentage})<br>` +
            `Entity Collisions: ${this.collisionStats.entityCollisions}<br>` +
            `Last Wall Collision: ${lastCollisionTime}<br>`;
        
        // Add nearby walls info if available
        if (this.collisionStats.nearbyWalls && this.collisionStats.nearbyWalls.length > 0) {
            collisionHtml += `Nearby Walls: ${this.collisionStats.nearbyWalls.length}<br>`;
            
            // Show closest wall
            const closestWall = this.collisionStats.nearbyWalls[0];
            if (closestWall) {
                collisionHtml += `Closest Wall: ${closestWall.distance.toFixed(2)} units away at tile (${closestWall.tileX}, ${closestWall.tileY})`;
            }
        }
        
        this.collisionSection.content.innerHTML = collisionHtml;
    }
    
    /**
     * Update nearby walls information
     * @param {number} x - Character X position in world
     * @param {number} y - Character Y position in world 
     * @param {Object} map - Map manager instance
     */
    updateNearbyWalls(x, y, map) {
        if (!map || !map.isWallOrObstacle) return;
        
        const tileSize = map.tileSize || 12;
        const tileX = Math.floor(x / tileSize);
        const tileY = Math.floor(y / tileSize);
        const searchRadius = 3;
        
        const nearbyWalls = [];
        
        // Check surrounding tiles for walls
        for (let dy = -searchRadius; dy <= searchRadius; dy++) {
            for (let dx = -searchRadius; dx <= searchRadius; dx++) {
                const checkTileX = tileX + dx;
                const checkTileY = tileY + dy;
                
                // Skip center tile
                if (dx === 0 && dy === 0) continue;
                
                // Skip out of bounds tiles
                if (checkTileX < 0 || checkTileY < 0 || 
                    (map.width > 0 && checkTileX >= map.width) || 
                    (map.height > 0 && checkTileY >= map.height)) {
                    continue;
                }
                
                // Get world coordinates of tile center
                const checkWorldX = (checkTileX + 0.5) * tileSize;
                const checkWorldY = (checkTileY + 0.5) * tileSize;
                
                // Check if it's a wall
                if (map.isWallOrObstacle(checkWorldX, checkWorldY)) {
                    // Calculate distance to player
                    const distance = Math.sqrt(
                        Math.pow(x - checkWorldX, 2) + 
                        Math.pow(y - checkWorldY, 2)
                    );
                    
                    nearbyWalls.push({
                        tileX: checkTileX,
                        tileY: checkTileY,
                        worldX: checkWorldX,
                        worldY: checkWorldY,
                        distance: distance
                    });
                }
            }
        }
        
        // Sort by distance
        nearbyWalls.sort((a, b) => a.distance - b.distance);
        
        // Store only the closest 5 walls
        this.collisionStats.nearbyWalls = nearbyWalls.slice(0, 5);
    }
    
    /**
     * Update coordinates section with detailed position info
     * @param {Object} gameState - Current game state
     */
    updateCoordinatesSection(gameState) {
        const character = gameState.character;
        const map = gameState.map;
        const camera = gameState.camera;
        
        if (!character || !map) {
            this.coordinatesSection.content.textContent = 'Character or map not initialized';
            return;
        }
        
        // Get detailed position info
        const worldX = character.x;
        const worldY = character.y;
        const worldZ = character.z || 0; // Height/jump coordinate
        const tileSize = map.tileSize || 12;
        
        // Calculate tile coordinates using different methods
        const floorTileX = Math.floor(worldX / tileSize);
        const floorTileY = Math.floor(worldY / tileSize);
        const roundTileX = Math.round(worldX / tileSize);
        const roundTileY = Math.round(worldY / tileSize);
        
        // Tile center in world coordinates
        const tileCenterX = (floorTileX + 0.5) * tileSize;
        const tileCenterY = (floorTileY + 0.5) * tileSize;
        
        // Distance from tile boundaries
        const tileLeft = floorTileX * tileSize;
        const tileRight = (floorTileX + 1) * tileSize;
        const tileTop = floorTileY * tileSize;
        const tileBottom = (floorTileY + 1) * tileSize;
        
        // Distance to each boundary
        const distToLeft = worldX - tileLeft;
        const distToRight = tileRight - worldX;
        const distToTop = worldY - tileTop;
        const distToBottom = tileBottom - worldY;
        
        // Calculate camera-related info
        let cameraSection = '';
        if (camera) {
            const cameraX = camera.position.x;
            const cameraY = camera.position.y;
            const distToCamera = Math.sqrt(
                Math.pow(worldX - cameraX, 2) + 
                Math.pow(worldY - cameraY, 2)
            );
            
            cameraSection = 
                `Camera Position: (${cameraX.toFixed(1)}, ${cameraY.toFixed(1)})<br>` +
                `Distance to Camera: ${distToCamera.toFixed(1)} units<br>`;
        }
        
        // Format coordinate and tile data
        this.coordinatesSection.content.innerHTML =
            `World: (${worldX.toFixed(3)}, ${worldY.toFixed(3)}, ${worldZ.toFixed(3)})<br>` +
            `Tile [floor]: (${floorTileX}, ${floorTileY})<br>` +
            `Tile [round]: (${roundTileX}, ${roundTileY})<br>` +
            `Tile Percent: ${(distToLeft / tileSize).toFixed(3)}x, ${(distToTop / tileSize).toFixed(3)}y<br>` +
            `Tile Center: (${tileCenterX.toFixed(1)}, ${tileCenterY.toFixed(1)})<br>` +
            `Boundaries: L:${distToLeft.toFixed(1)} R:${distToRight.toFixed(1)} T:${distToTop.toFixed(1)} B:${distToBottom.toFixed(1)}<br>` +
            cameraSection;
    }
}

// Create a singleton instance
export const debugOverlay = new DebugOverlay();

// Initialize global collision stats tracker
if (!window.COLLISION_STATS) {
    window.COLLISION_STATS = {
        totalWallChecks: 0,
        wallCollisions: 0,
        entityCollisions: 0,
        lastWallCollision: null,
        nearbyWalls: []
    };
}