import { spriteManager } from '../assets/spriteManager.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { gameState } from '../game/gamestate.js';
import { EntityAnimator } from './EntityAnimator.js';

/**
 * Utility for throttling log messages
 */
const logThrottles = {};
function throttledLog(key, message, data, interval = 1000) {
    const now = Date.now();
    if (!logThrottles[key] || now - logThrottles[key] >= interval) {
        logThrottles[key] = now;
        if (data !== undefined) {
            console.log(message, data);
        } else {
            console.log(message);
        }
        return true;
    }
    return false;
}

/**
 * PlayerManager - Manages all players in the game
 */
export class PlayerManager {
    /**
     * Create a new player manager
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        this.players = new Map(); // Map of player ID to player data
        this.localPlayerId = options.localPlayerId || null;
        this.maxPlayers = options.maxPlayers || 100;
        
        // Animation tracking for other players
        this.playerAnimators = new Map(); // Map of player ID to animator
        
        // Debug
        this.debug = true; // Enable debug by default for troubleshooting
        
        // Debug visualization - makes other players more obvious
        this.visualDebug = false; // Turn off the pink borders
        
        console.log("PlayerManager initialized");
    }
    
    /**
     * Update players with data from server
     * @param {Object} playersData - Player data from server
     */
    updatePlayers(playersData) {
        if (!playersData) return;
        
        // Get array of current player IDs
        const currentPlayerIds = Array.from(this.players.keys());
        
        // Add or update players from server data
        for (const [playerId, playerData] of Object.entries(playersData)) {
            // Skip local player, it's handled separately
            if (playerId === this.localPlayerId) continue;
            
            if (this.players.has(playerId)) {
                // Update existing player
                const player = this.players.get(playerId);
                
                // Save previous position for movement detection
                const prevX = player.x;
                const prevY = player.y;
                
                // Update player data
                Object.assign(player, playerData);
                
                // Update player's animation if it exists
                if (this.playerAnimators.has(playerId)) {
                    const animator = this.playerAnimators.get(playerId);
                    
                    // Calculate if player is moving based on position change
                    const isMoving = prevX !== player.x || prevY !== player.y;
                    
                    // Calculate velocity direction based on position change
                    const velocity = {
                        x: player.x - prevX,
                        y: player.y - prevY
                    };
                    
                    // Update animator
                    animator.update(0.016, isMoving, velocity); // Use 1/60 as delta time
                }
            } else {
                // Add new player with default sprite properties if they're missing
                const enhancedPlayerData = {
                    ...playerData,
                    // Ensure these properties are set for rendering
                    width: playerData.width || 10,
                    height: playerData.height || 10,
                    spriteX: playerData.spriteX !== undefined ? playerData.spriteX : 0,
                    spriteY: playerData.spriteY !== undefined ? playerData.spriteY : 0,
                    maxHealth: playerData.maxHealth || 100,
                    name: playerData.name || `Player ${playerId}`,
                    lastUpdate: Date.now(),
                    // Store the current position as both current and target for interpolation
                    _prevX: playerData.x,
                    _prevY: playerData.y, 
                    _targetX: playerData.x,
                    _targetY: playerData.y,
                    // Timestamps for interpolation
                    lastPositionUpdate: Date.now()
                };
                
                this.players.set(playerId, enhancedPlayerData);
                
                // Create an animator for this player
                this.playerAnimators.set(playerId, new EntityAnimator({
                    defaultState: 'idle',
                    frameCount: 4,
                    frameDuration: 0.15,
                    spriteWidth: TILE_SIZE,
                    spriteHeight: TILE_SIZE,
                    spriteSheet: 'character_sprites'
                }));
                
                console.log(`Added new player: ${playerId} at position (${playerData.x}, ${playerData.y})`, enhancedPlayerData);
            }
        }
        
        // Remove players that no longer exist
        for (const playerId of currentPlayerIds) {
            if (playerId !== this.localPlayerId && !playersData[playerId]) {
                console.log(`Removing player: ${playerId}`);
                this.players.delete(playerId);
                this.playerAnimators.delete(playerId);
            }
        }
        
        // Always log player count for debugging, but throttled
        throttledLog('player-count', `Player Manager: Now tracking ${this.players.size} other players`);
        if (this.players.size > 0 && throttledLog('player-ids', 'Players being tracked:', Array.from(this.players.keys()), 5000)) {
            // Log player IDs only every 5 seconds
        }
    }
    
    /**
     * Update animations for all players
     * @param {number} deltaTime - Time elapsed since last update in seconds
     */
    updatePlayerAnimations(deltaTime) {
        // Update all player animations
        for (const [playerId, player] of this.players.entries()) {
            if (this.playerAnimators.has(playerId)) {
                const animator = this.playerAnimators.get(playerId);
                
                // Skip update if we don't have position data
                if (player._prevX === undefined || player._targetX === undefined) continue;
                
                // Calculate if player is moving based on interpolation targets
                const isMoving = player._prevX !== player._targetX || player._prevY !== player._targetY;
                
                // Calculate velocity direction based on movement targets
                const velocity = {
                    x: player._targetX - player._prevX,
                    y: player._targetY - player._prevY
                };
                
                // Update animator
                animator.update(deltaTime, isMoving, velocity);
            }
        }
        
        // Clean up stale player positions after some time
        // This helps prevent "ghost" player artifacts
        const now = Date.now();
        for (const [playerId, player] of this.players.entries()) {
            if (now - player.lastPositionUpdate > 10000) { // 10 seconds of no updates
                throttledLog('stale-player', `Removing stale player ${playerId} - no updates for 10 seconds`, null, 5000);
                this.players.delete(playerId);
                this.playerAnimators.delete(playerId);
            }
        }
    }
    
    /**
     * Get all players for rendering
     * @returns {Array} Array of player objects
     */
    getPlayersForRender() {
        const allPlayers = Array.from(this.players.values());
        
        // Log player data and IDs, but throttled to reduce spam
        throttledLog('players-available', `[PlayerManager] Players available: ${allPlayers.length}, Local ID: ${this.localPlayerId}`);
        
        if (allPlayers.length > 0 && throttledLog('player-debug', 'Player debug info', null, 10000)) {
            // Only log detailed player data every 10 seconds
            console.log(`[PlayerManager] Player IDs in Map: ${Array.from(this.players.keys()).join(', ')}`);
            console.log(`[PlayerManager] First player object:`, allPlayers[0]);
        }
        
        // Enhanced filter logic with more checks and debug output
        const playersToRender = allPlayers.filter(player => {
            // Skip players without an ID (shouldn't happen but check anyway)
            if (!player.id) {
                throttledLog('player-no-id', `[PlayerManager] Player without ID found, will render:`, player, 5000);
                return true;
            }
            
            // Convert both IDs to strings for comparison (in case of type mismatch)
            const playerId = String(player.id);
            const localId = this.localPlayerId ? String(this.localPlayerId) : null;
            
            // Skip the local player - this is the key filtering logic
            const isLocalPlayer = localId && playerId === localId;
            
            if (isLocalPlayer) {
                throttledLog('filter-local', `[PlayerManager] Filtering out local player with ID: ${playerId}`, null, 5000);
                return false;
            }
            
            // Keep this player for rendering
            return true;
        });
        
        // Log filtering results with throttling
        throttledLog('render-count', `[PlayerManager] Rendering ${playersToRender.length} out of ${allPlayers.length} players`);
        
        if (playersToRender.length > 0 && throttledLog('render-ids', 'Players to render IDs', null, 5000)) {
            console.log(`[PlayerManager] Players to render IDs: ${playersToRender.map(p => p.id).join(', ')}`);
        } else if (allPlayers.length > 0 && playersToRender.length === 0 && throttledLog('all-filtered', 'All players filtered warning', null, 5000)) {
            console.log(`[PlayerManager] WARNING: All players filtered out! Check if localPlayerId (${this.localPlayerId}) matches all player IDs`);
        }
        
        return playersToRender;
    }
    
    /**
     * Set the client ID to exclude from rendering (since local player is rendered separately)
     * @param {string} clientId - The local player's client ID
     */
    setLocalPlayerId(clientId) {
        this.localPlayerId = clientId;
        console.log(`PlayerManager: Set local player ID to ${clientId}`);
    }
    
    /**
     * Render players on canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} cameraPosition - Camera position
     */
    render(ctx, cameraPosition) {
        if (!ctx) {
            console.warn("Cannot render players: no canvas context provided");
            return;
        }
        
        // Throttle this log to reduce spam
        throttledLog('render-players', `PlayerManager rendering ${this.players.size} players`, null, 5000);
        
        // Get list of players to render (filtered)
        const playersToRender = this.getPlayersForRender();
        
        // Get character sprite sheet - try both possible names for the sprite sheet
        let spriteSheetObj = spriteManager.getSpriteSheet('character_sprites');
        if (!spriteSheetObj) {
            spriteSheetObj = spriteManager.getSpriteSheet('enemy_sprites'); // Fallback to enemy sprites
            
            // As a last resort, try to use any available sprite sheet
            if (!spriteSheetObj && spriteManager.spriteSheets) {
                const sheetNames = Object.keys(spriteManager.spriteSheets);
                if (sheetNames.length > 0) {
                    spriteSheetObj = spriteManager.getSpriteSheet(sheetNames[0]);
                    console.log(`Falling back to first available sprite sheet: ${sheetNames[0]}`);
                }
            }
        }

        // Debug logging for sprite sheet only if missing
        if (!spriteSheetObj) {
            console.error("Character sprite sheet not loaded - switching to fallback rendering!");
            // Use fallback rendering
            this.renderPlayersFallback(ctx, cameraPosition);
            return;
        }
        
        // We have a sprite sheet, render players with it
        const characterSpriteSheet = spriteSheetObj.image;
        
        // Get screen dimensions
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        
        // Define the scale factor
        const scaleFactor = gameState.camera?.viewType === 'strategic' ? 0.5 : 1;
        
        // Get the view type for adjusted culling
        const isStrategicView = gameState.camera?.viewType === 'strategic';
        
        // Draw each player
        for (const player of playersToRender) {
            try {
                // Apply the same scale factor used for the main character (SCALE = 3)
                const width = player.width * SCALE;
                const height = player.height * SCALE;
                
                // Calculate screen position
                const screenX = (player.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
                const screenY = (player.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
                
                // Improved offscreen culling - use larger culling distance in strategic view
                // to prevent players from "popping" when moving quickly
                const cullingDistance = isStrategicView ? Math.max(screenWidth, screenHeight) : width * 2;
                
                // Skip if off screen (with appropriate buffer)
                if (screenX < -cullingDistance || screenX > screenWidth + cullingDistance || 
                    screenY < -cullingDistance || screenY > screenHeight + cullingDistance) {
                    continue;
                }
                
                // Save context for rotation
                ctx.save();
                
                // Translate to player position
                ctx.translate(screenX, screenY);
                
                // If player has a rotation, use it
                if (typeof player.rotation === 'number') {
                    ctx.rotate(player.rotation);
                }
                
                // Get player animator
                const animator = this.playerAnimators.get(player.id);
                
                if (animator) {
                    // Get animation source rect
                    const sourceRect = animator.getSourceRect();
                    
                    // Draw player sprite with animation
                    ctx.drawImage(
                        characterSpriteSheet,
                        sourceRect.x, sourceRect.y,
                        sourceRect.width, sourceRect.height,
                        -width/2, -height/2,
                        width, height
                    );
                } else {
                    // Fallback to old rendering without animation
                    ctx.drawImage(
                        characterSpriteSheet,
                        player.spriteX || 0, player.spriteY || 0, 
                        TILE_SIZE, TILE_SIZE,
                        -width/2, -height/2,
                        width, height
                    );
                }

                // Draw player name
                if (player.name) {
                    ctx.fillStyle = '#ffffff';
                    ctx.strokeStyle = '#000000';
                    ctx.lineWidth = 2;
                    ctx.textAlign = 'center';
                    ctx.font = '12px Arial';
                    ctx.strokeText(player.name, 0, -height/2 - 5);
                    ctx.fillText(player.name, 0, -height/2 - 5);
                }
                
                // Draw health bar if health is defined
                if (player.health !== undefined && player.maxHealth !== undefined) {
                    const healthPercent = player.health / player.maxHealth;
                    const barWidth = width;
                    const barHeight = 3;
                    const barY = height/2 + 5;
                    
                    // Background
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                    ctx.fillRect(-barWidth/2, barY, barWidth, barHeight);
                    
                    // Health
                    ctx.fillStyle = healthPercent > 0.6 ? 'green' : healthPercent > 0.3 ? 'yellow' : 'red';
                    ctx.fillRect(-barWidth/2, barY, barWidth * healthPercent, barHeight);
                }
                
                // Restore context
                ctx.restore();
            } catch (error) {
                // Only log errors occasionally to prevent spam
                if (Math.random() < 0.1) {
                    console.error("Error rendering player:", error, player);
                }
            }
        }
        
        // Clear stale players if moving in strategic view
        if (isStrategicView && gameState.character) {
            const staticThreshold = 0.1; // Distance threshold to consider movement significant
            const characterMoved = 
                Math.abs(gameState.character.x - (gameState.lastUpdateX || 0)) > staticThreshold || 
                Math.abs(gameState.character.y - (gameState.lastUpdateY || 0)) > staticThreshold;
                
            if (characterMoved) {
                gameState.lastUpdateX = gameState.character.x;
                gameState.lastUpdateY = gameState.character.y;
                
                // Force clear canvas next frame to remove any ghost artifacts
                if (!this._lastClearCanvas) {
                    this._lastClearCanvas = Date.now();
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                    ctx.fillRect(0, 0, screenWidth, screenHeight);
                }
            }
        }
    }
    
    /**
     * Fallback rendering for players when sprite sheet is not available
     * @param {CanvasRenderingContext2D} ctx 
     * @param {Object} cameraPosition 
     */
    renderPlayersFallback(ctx, cameraPosition) {
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        const scaleFactor = gameState.camera.viewType === 'strategic' ? 0.5 : 1;
        
        // Log only once when falling back
        console.log(`Fallback rendering for ${this.players.size} players - sprite sheet missing`);
        
        for (const player of this.players.values()) {
            try {
                // Use player's actual dimensions with proper scaling
                const width = player.width * SCALE;
                const height = player.height * SCALE;
                
                // Calculate screen position
                const screenX = (player.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
                const screenY = (player.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
                
                // Skip if off screen
                if (screenX < -width || screenX > screenWidth + width || 
                    screenY < -height || screenY > screenHeight + height) {
                    continue;
                }
                
                // Save context for rotation
                ctx.save();
                
                // Translate to player position for proper rotation
                ctx.translate(screenX, screenY);
                
                // Apply rotation if player has it
                if (typeof player.rotation === 'number') {
                    ctx.rotate(player.rotation);
                }
                
                // Draw simple colored rectangle
                ctx.fillStyle = 'blue';
                ctx.fillRect(-width/2, -height/2, width, height);
                
                // Draw simple direction indicator
                ctx.fillStyle = 'white';
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(width/2, 0);
                ctx.stroke();
                
                // Draw player name
                if (player.name) {
                    ctx.fillStyle = '#ffffff';
                    ctx.strokeStyle = '#000000';
                    ctx.lineWidth = 2;
                    ctx.textAlign = 'center';
                    ctx.font = '12px Arial';
                    ctx.strokeText(player.name, 0, -height/2 - 5);
                    ctx.fillText(player.name, 0, -height/2 - 5);
                }
                
                // Restore context
                ctx.restore();
            } catch (error) {
                console.error("Error rendering player fallback:", error);
            }
        }
    }
} 