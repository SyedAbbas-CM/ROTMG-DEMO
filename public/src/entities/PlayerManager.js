import { spriteManager } from '../assets/spriteManager.js';
import { TILE_SIZE, SCALE } from '../constants/constants.js';
import { gameState } from '../game/gamestate.js';

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
                Object.assign(player, playerData);
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
                console.log(`Added new player: ${playerId} at position (${playerData.x}, ${playerData.y})`, enhancedPlayerData);
            }
        }
        
        // Remove players that no longer exist
        for (const playerId of currentPlayerIds) {
            if (playerId !== this.localPlayerId && !playersData[playerId]) {
                console.log(`Removing player: ${playerId}`);
                this.players.delete(playerId);
            }
        }
        
        // Always log player count for debugging
        console.log(`Player Manager: Now tracking ${this.players.size} other players`);
        if (this.players.size > 0) {
            console.log("Players being tracked:", Array.from(this.players.keys()));
        }
    }
    
    /**
     * Get all players for rendering
     * @returns {Array} Array of player objects
     */
    getPlayersForRender() {
        const allPlayers = Array.from(this.players.values());
        
        // Log only player count to reduce console noise
        if (Math.random() < 0.1) { // Only log 10% of the time
            console.log(`[PlayerManager] Players available: ${allPlayers.length}, Local ID: ${this.localPlayerId}`);
        }
        
        // Filter out ONLY the local player by exact ID match
        // This is the critical fix - we only want to exclude the exact local player ID
        const playersToRender = allPlayers.filter(player => {
            // Check for invalid IDs first
            if (!player.id) return true; // Keep players with undefined IDs
            
            // Keep all players EXCEPT the one matching the local player ID
            return player.id !== this.localPlayerId;
        });
        
        // Brief log of filtered results
        if (Math.random() < 0.1 && playersToRender.length > 0) { // Only log 10% of the time
            console.log(`[PlayerManager] Rendering ${playersToRender.length} players`);
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
        
        // Minimal logging for performance
        if (Math.random() < 0.01) { // Just 1% of calls
            console.log(`PlayerManager rendering ${this.players.size} players`);
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
        
        // Draw each player
        for (const player of playersToRender) {
            try {
                // Apply the same scale factor used for the main character (SCALE = 3)
                const width = player.width * SCALE;
                const height = player.height * SCALE;
                
                // Calculate screen position
                const screenX = (player.x - cameraPosition.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
                const screenY = (player.y - cameraPosition.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
                
                // Skip if off screen (with a buffer)
                if (screenX < -width*2 || screenX > screenWidth + width*2 || 
                    screenY < -height*2 || screenY > screenHeight + height*2) {
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
                
                // Draw player sprite - center it properly
                ctx.drawImage(
                    characterSpriteSheet,
                    player.spriteX || 0, player.spriteY || 0, 
                    TILE_SIZE, TILE_SIZE, // Use TILE_SIZE for source rectangle to match main character
                    -width/2, -height/2, width, height
                );

                // Visual debugging - add bright highlight around player in debug mode
                if (this.visualDebug) {
                    ctx.strokeStyle = 'magenta';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(-width/2, -height/2, width, height);
                    
                    // Draw a line pointing up to make more obvious
                    ctx.beginPath();
                    ctx.moveTo(0, -height/2);
                    ctx.lineTo(0, -height/2 - 20);
                    ctx.stroke();
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
                // Skip local player
                if (player.id === this.localPlayerId) continue;
                
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
                
                // Draw a colored rectangle instead (more professional than magenta circles)
                ctx.fillStyle = 'rgba(0, 128, 255, 0.8)'; // Semi-transparent blue
                ctx.fillRect(-width/2, -height/2, width, height);
                
                // Add border for better visibility
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.strokeRect(-width/2, -height/2, width, height);
                
                // Add direction indicator
                ctx.beginPath();
                ctx.moveTo(0, 0);
                ctx.lineTo(0, -height/2 - 5);
                ctx.strokeStyle = '#ffff00';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw player name
                if (player.name) {
                    ctx.fillStyle = 'white';
                    ctx.textAlign = 'center';
                    ctx.font = '12px Arial';
                    ctx.fillText(player.name, 0, -height/2 - 10);
                }
                
                // Restore context
                ctx.restore();
            } catch (error) {
                console.error("Error in fallback rendering:", error);
            }
        }
    }
} 