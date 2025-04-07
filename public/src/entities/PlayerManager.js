import { spriteManager } from '../assets/spriteManager.js';
import { TILE_SIZE } from '../constants/constants.js';
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
                    lastUpdate: Date.now()
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
        return Array.from(this.players.values());
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
        
        // Get list of players to render
        const playersToRender = this.getPlayersForRender();
        
        // Log player count occasionally for debugging
        if (Math.random() < 0.05) {
            console.log(`PlayerManager rendering ${playersToRender.length} other players`);
            
            // If we have players but render is failing, log player data
            if (playersToRender.length > 0 && this.debug) {
                console.log("Players to render:", playersToRender.map(p => ({
                    id: p.id,
                    pos: `(${p.x.toFixed(0)},${p.y.toFixed(0)})`,
                    lastUpdate: new Date(p.lastUpdate || 0).toISOString()
                })));
            }
        }
        
        // Get character sprite sheet - try both possible names for the sprite sheet
        let spriteSheetObj = window.spriteManager?.getSpriteSheet('character_sprites');
        if (!spriteSheetObj) {
            spriteSheetObj = window.spriteManager?.getSpriteSheet('enemy_sprites'); // Fallback to enemy sprites
            
            // As a last resort, try to use any available sprite sheet
            if (!spriteSheetObj && window.spriteManager?.spriteSheets) {
                const sheetNames = Object.keys(window.spriteManager.spriteSheets);
                if (sheetNames.length > 0) {
                    spriteSheetObj = window.spriteManager.getSpriteSheet(sheetNames[0]);
                    console.log(`Falling back to first available sprite sheet: ${sheetNames[0]}`);
                }
            }
        }

        // Debug logging for sprite sheet
        if (!spriteSheetObj) {
            // Only log rarely to avoid console spam
            if (Math.random() < 0.01) { // 1% chance to log
                console.error("Character sprite sheet not loaded - players won't be visible!");
                if (this.debug && window.spriteManager) {
                    // Log available sprite sheets for debugging
                    console.log("Available sprite sheets:", Object.keys(window.spriteManager.spriteSheets || {}));
                }
            }
            this.renderPlayersFallback(ctx, cameraPosition);
            return;
        }
        
        // We have a sprite sheet, render players with it
        const characterSpriteSheet = spriteSheetObj.image;
        
        // Only log rarely to prevent console spam
        if (Math.random() < 0.01) { // 1% chance
            console.log(`Rendering players with sprite sheet: ${spriteSheetObj.config.name}`);
        }
        
        // Get screen dimensions
        const screenWidth = ctx.canvas.width;
        const screenHeight = ctx.canvas.height;
        
        // Define the scale factor
        const scaleFactor = window.gameState?.camera?.viewType === 'strategic' ? 0.5 : 1;
        
        // Draw each player
        for (const player of playersToRender) {
            try {
                // Use consistent sizes for all players
                const width = 24 * scaleFactor;
                const height = 24 * scaleFactor;
                
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
                    player.width || 10, player.height || 10,
                    -width/2, -height/2, width, height
                );
                
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
        
        for (const player of this.players.values()) {
            try {
                const width = 24 * scaleFactor;
                const height = 24 * scaleFactor;
                
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
                
                // Draw a colored circle for the player
                ctx.fillStyle = 'blue';
                ctx.beginPath();
                ctx.arc(screenX, screenY, width/2, 0, Math.PI * 2);
                ctx.fill();
                
                // Draw player name
                if (player.name) {
                    ctx.fillStyle = 'white';
                    ctx.textAlign = 'center';
                    ctx.font = '12px Arial';
                    ctx.fillText(player.name, screenX, screenY - height/2 - 5);
                }
                
                // Restore context
                ctx.restore();
            } catch (error) {
                console.error("Error rendering player fallback:", error, player);
            }
        }
    }
}

// Export a default instance for easy access
export default new PlayerManager(); 