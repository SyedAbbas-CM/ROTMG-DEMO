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
        // Comment out all logging to reduce console spam
        // if (data !== undefined) {
        //     console.log(message, data);
        // } else {
        //     console.log(message);
        // }
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
        this.debug = false; // Disable debug
        
        // Debug visualization - makes other players more obvious
        this.visualDebug = false; // Disable visual debug
    }
    
    /**
     * Update players based on server data
     * @param {Object} playersData - Player data from server
     */
    updatePlayers(playersData) {
        if (!playersData) return;
        
        // Track players in current update for cleanup
        const updatedPlayers = new Set();
        
        // Process each player
        for (const [id, data] of Object.entries(playersData)) {
            // Handle filtering of local player and ensuring player is valid
            if (id === this.localPlayerId) {
                throttledLog('skip-local', `Skipping local player ID: ${id}`, null, 5000);
                continue;
            }
            
            // Skip if invalid data
            if (!data || typeof data !== 'object' || data.x === undefined || data.y === undefined) {
                throttledLog('invalid-data', `Invalid player data for ID: ${id}`, data, 5000);
                continue;
            }
            
            updatedPlayers.add(id);
            
            // Get existing player or create new one
            let player = this.players.get(id);
            
            if (player) {
                // Update existing player with our new method
                this.updatePlayerData(player, data);
            } else {
                // Create new player
                player = {
                    id: id,
                    x: data.x,
                    y: data.y,
                    _targetX: data.x,
                    _targetY: data.y,
                    _prevX: data.x,
                    _prevY: data.y,
                    rotation: data.rotation || 0,
                    health: data.health !== undefined ? data.health : 100,
                    maxHealth: data.maxHealth || 100,
                    name: data.name || `Player ${id}`,
                    width: data.width || 10,
                    height: data.height || 10,
                    lastUpdate: Date.now(),
                    lastPositionUpdate: Date.now(),
                    lastServerUpdate: Date.now(),
                    vx: 0,
                    vy: 0
                };
                
                // Add to players map
                this.players.set(id, player);
                
                // Initialize animator
                this.createAnimatorForPlayer(id);
                
                throttledLog('new-player', `Added new player: ${id} at (${data.x.toFixed(1)}, ${data.y.toFixed(1)})`);
            }
        }
        
        // Remove players not in update after a timeout
        // This helps with temporary network issues
        if (updatedPlayers.size > 0) {
            // Clean up players that haven't been updated for too long
            const staleTimeout = 10000; // 10 seconds
            const now = Date.now();
            
            for (const [id, player] of this.players.entries()) {
                if (!updatedPlayers.has(id) && now - player.lastServerUpdate > staleTimeout) {
                    this.players.delete(id);
                    this.playerAnimators.delete(id);
                    throttledLog('remove-player', `Removed stale player: ${id}`);
                }
            }
        }
    }
    
    /**
     * Update animations for all players
     * @param {number} deltaTime - Time since last frame in seconds
     */
    updatePlayerAnimations(deltaTime) {
        // Skip if no delta time
        if (!deltaTime) return;
        
        // Early return on edge cases
        if (!this.players || this.players.size === 0) return;
        
        // Get current time for calculations
        const now = Date.now();
        
        // Update each player's animation
        for (const [playerId, player] of this.players.entries()) {
            // Skip local player
            if (playerId === this.localPlayerId) continue;
            
            // Skip if no animator
            const animator = this.playerAnimators.get(playerId);
            if (!animator) {
                // Try to create animator if it doesn't exist
                this.createAnimatorForPlayer(playerId);
                continue;
            }
            
            // Interpolate position
            if (player._targetX !== undefined && player._prevX !== undefined) {
                // Store old position to detect movement
                const oldX = player.x;
                const oldY = player.y;
                
                // Calculate interpolation progress
                const t = Math.min((now - player.lastPositionUpdate) / 100, 1.0);
                
                // Apply interpolation
                player.x = player._prevX + (player._targetX - player._prevX) * t;
                player.y = player._prevY + (player._targetY - player._prevY) * t;
                
                // Calculate movement this frame
                const dx = player.x - oldX;
                const dy = player.y - oldY;
                
                // Consider the player moving only if there's substantial movement
                const movementThreshold = 0.0001;
                player.isMoving = Math.abs(dx) > movementThreshold || Math.abs(dy) > movementThreshold;
                
                // Store movement direction for animation
                if (player.isMoving) {
                    // Update movement velocity (for direction calculation)
                    player.moveVelocity = { 
                        x: dx / deltaTime, 
                        y: dy / deltaTime 
                    };
                    
                    // Determine dominant direction for when player stops
                    if (Math.abs(dx) > Math.abs(dy)) {
                        player.lastFacingDirection = dx > 0 ? 3 : 1; // right or left
                    } else {
                        player.lastFacingDirection = dy > 0 ? 0 : 2; // down or up
                    }
                }
                
                // Check if we've reached the target position
                const reachedTarget = Math.abs(player.x - player._targetX) < 0.01 && 
                                     Math.abs(player.y - player._targetY) < 0.01;
                
                // If we've reached the target, stop moving
                if (reachedTarget) {
                    player.isMoving = false;
                }
            }
            
            // Update animator state
            if (player.isMoving && player.moveVelocity) {
                // Moving - update animation with movement direction
                animator.update(deltaTime, true, player.moveVelocity);
            } else {
                // Not moving - update with stationary state
                // Pass false for movement, but make sure we're facing the last direction
                animator.update(deltaTime, false);
                
                // Set animator direction to the last facing direction
                if (player.lastFacingDirection !== undefined) {
                    animator.direction = player.lastFacingDirection;
                }
                
                // Explicitly set to idle state to prevent any walk animation
                animator.currentState = animator.states.IDLE;
                animator.frameIndex = 0;
            }
        }
    }
    
    /**
     * Get all players for rendering
     * @returns {Array} Array of player objects
     */
    getPlayersForRender() {
        const allPlayers = Array.from(this.players.values());
        
        // Enhanced filter logic with more checks
        const playersToRender = allPlayers.filter(player => {
            // Skip players without an ID (shouldn't happen but check anyway)
            if (!player.id) {
                return true;
            }
            
            // Convert both IDs to strings for comparison (in case of type mismatch)
            const playerId = String(player.id);
            const localId = this.localPlayerId ? String(this.localPlayerId) : null;
            
            // Skip the local player - this is the key filtering logic
            const isLocalPlayer = localId && playerId === localId;
            
            if (isLocalPlayer) {
                return false;
            }
            
            // Keep this player for rendering
            return true;
        });
        
        return playersToRender;
    }
    
    /**
     * Set the client ID to exclude from rendering (since local player is rendered separately)
     * @param {string} clientId - The local player's client ID
     */
    setLocalPlayerId(clientId) {
        this.localPlayerId = clientId;
    }
    
    /**
     * Render players on canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Object} cameraPosition - Camera position
     */
    render(ctx, cameraPosition) {
        if (!ctx) {
            return;
        }
        
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
                }
            }
        }

        // Debug logging for sprite sheet only if missing
        if (!spriteSheetObj) {
            // Use fallback rendering
            this.renderPlayersFallback(ctx, cameraPosition);
            return;
        }
        
        // We have a sprite sheet, render players with it
        const characterSpriteSheet = spriteSheetObj.image;
        
        // Get screen dimensions
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        
        // Define the scale factor based on view type 
        const viewType = gameState.camera?.viewType || 'top-down';
        const viewScaleFactor = viewType === 'strategic' ? 0.25 : 1.0;
        
        // Get the view type for adjusted culling
        const isStrategicView = viewType === 'strategic';
        
        // Use the camera's worldToScreen method if available for consistent coordinates
        const useCamera = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
        
        // Draw each player
        for (const player of playersToRender) {
            try {
                // Apply the same scale factor used for the main character (SCALE = 3)
                const width = player.width * SCALE * viewScaleFactor;
                const height = player.height * SCALE * viewScaleFactor;
                
                // Get screen coordinates
                let screenX, screenY;
                
                if (useCamera) {
                    // Use camera's consistent transformation method
                    const screenPos = gameState.camera.worldToScreen(
                        player.x, 
                        player.y, 
                        screenWidth, 
                        screenHeight, 
                        TILE_SIZE
                    );
                    screenX = screenPos.x;
                    screenY = screenPos.y;
                } else {
                    // Fallback to direct calculation if camera method not available
                    screenX = (player.x - cameraPosition.x) * TILE_SIZE * viewScaleFactor + screenWidth / 2;
                    screenY = (player.y - cameraPosition.y) * TILE_SIZE * viewScaleFactor + screenHeight / 2;
                }
                
                // Skip if off screen (with appropriate buffer)
                const cullingDistance = isStrategicView ? Math.max(screenWidth, screenHeight) : width * 2;
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
                // Silently handle errors
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
        const viewType = gameState.camera?.viewType || 'top-down';
        const viewScaleFactor = viewType === 'strategic' ? 0.25 : 1.0;
        
        // Use the camera's worldToScreen method if available for consistent coordinates
        const useCamera = gameState.camera && typeof gameState.camera.worldToScreen === 'function';
        
        for (const player of this.players.values()) {
            // Skip local player
            if (player.id === this.localPlayerId) continue;
            
            try {
                // Use player's actual dimensions with proper scaling
                const width = player.width * SCALE * viewScaleFactor;
                const height = player.height * SCALE * viewScaleFactor;
                
                // Get screen coordinates
                let screenX, screenY;
                
                if (useCamera) {
                    // Use camera's consistent transformation method
                    const screenPos = gameState.camera.worldToScreen(
                        player.x, 
                        player.y, 
                        screenWidth, 
                        screenHeight, 
                        TILE_SIZE
                    );
                    screenX = screenPos.x;
                    screenY = screenPos.y;
                } else {
                    // Fallback to direct calculation if camera method not available
                    screenX = (player.x - cameraPosition.x) * TILE_SIZE * viewScaleFactor + screenWidth / 2;
                    screenY = (player.y - cameraPosition.y) * TILE_SIZE * viewScaleFactor + screenHeight / 2;
                }
                
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
                // Silently handle errors
            }
        }
    }

    /**
     * Update player data with latest from server
     * @param {Object} player - Local player data to update
     * @param {Object} data - New player data from server
     */
    updatePlayerData(player, data) {
        // Calculate if this is a position change
        const positionChanged = data.x !== player.x || data.y !== player.y;
        
        // Store the previous position for interpolation before updating
        player._prevX = player.x;
        player._prevY = player.y;
        
        // Calculate velocity based on position changes
        if (player._targetX !== undefined) {
            const dx = data.x - player._targetX;
            const dy = data.y - player._targetY;
            const timeDiff = Date.now() - player.lastPositionUpdate;
            
            if (timeDiff > 0) {
                // Units per millisecond
                player.vx = dx / timeDiff;
                player.vy = dy / timeDiff;
                
                // Calculate if player is moving significantly
                const speed = Math.sqrt(player.vx * player.vx + player.vy * player.vy);
                const MOVEMENT_THRESHOLD = 0.0001; // Lower threshold to detect smaller movements
                const wasMoving = player.isMoving;
                player.isMoving = speed > MOVEMENT_THRESHOLD;
                
                // Always store movement velocity for animation direction
                player.moveVelocity = { x: player.vx, y: player.vy };
                
                // Update player's facing direction
                if (player.isMoving) {
                    // Determine dominant direction and save it
                    if (Math.abs(player.vx) > Math.abs(player.vy)) {
                        player.lastFacingDirection = player.vx > 0 ? 3 : 1; // right or left
                    } else {
                        player.lastFacingDirection = player.vy > 0 ? 0 : 2; // down or up
                    }
                }
            }
        }
        
        // Set target position for interpolation
        player._targetX = data.x;
        player._targetY = data.y;
        player.lastPositionUpdate = Date.now();
        
        // Update other properties directly
        player.rotation = data.rotation !== undefined ? data.rotation : player.rotation;
        player.health = data.health !== undefined ? data.health : player.health;
        player.maxHealth = data.maxHealth !== undefined ? data.maxHealth : (player.maxHealth || 100);
        
        // Update last server update time
        player.lastServerUpdate = Date.now();
    }

    /**
     * Create an animator for a player
     * @param {string} playerId - Player ID
     */
    createAnimatorForPlayer(playerId) {
        this.playerAnimators.set(playerId, new EntityAnimator({
            defaultState: 'idle',
            frameCount: 4,
            frameDuration: 0.15,
            spriteWidth: TILE_SIZE,
            spriteHeight: TILE_SIZE,
            spriteSheet: 'character_sprites'
        }));
    }
} 