import { gameState } from '../game/gamestate.js';
import { PlayerManager } from './PlayerManager.js';
import { Player } from '../entities/player.js';
import { TILE_SIZE } from '../constants/constants.js';

// Create a playerManager instance
const playerManager = new PlayerManager();

// Track all bullets in the game
const bullets = [];

/**
 * Update players based on server data
 * @param {Object} playerData - Player data from server
 */
export function updatePlayers(playerData) {
    if (!playerData) {
        console.warn("updatePlayers called with no playerData");
        return;
    }
    
    // Log raw data received
    console.log(`[updatePlayers] Raw data received: ${JSON.stringify(playerData)}`);
    console.log(`[updatePlayers] Player IDs in data: ${Object.keys(playerData).join(', ')}`);
    console.log(`[updatePlayers] Local player ID: ${playerManager.localPlayerId}, Character ID: ${gameState.character?.id}`);
    
    // CRITICAL FIX: Filter out the metadata properties that aren't players
    // Skip properties like 'timestamp' and 'players' which aren't player objects
    const actualPlayerData = {};
    for (const [id, data] of Object.entries(playerData)) {
        // Verify this is actually a player object with coordinates, not metadata
        if (data && typeof data === 'object' && data.x !== undefined && data.y !== undefined) {
            actualPlayerData[id] = data;
        } else {
            console.log(`Skipping non-player property: ${id}`);
        }
    }
    
    // Print the filtered player data
    const playerCount = Object.keys(actualPlayerData).length;
    console.log(`[updatePlayers] Processing ${playerCount} players after filtering metadata properties`);
    console.log(`[updatePlayers] Player IDs after filtering: ${Object.keys(actualPlayerData).join(', ')}`);
    
    // Set local player ID for playerManager if not already set
    if (!playerManager.localPlayerId && gameState.character && gameState.character.id) {
        playerManager.setLocalPlayerId(gameState.character.id);
        console.log(`Set local player ID to ${gameState.character.id} in playerManager`);
    }
    
    // Debug output of our current known local player ID
    if (Math.random() < 0.05) {
        console.log(`Current local player ID: ${playerManager.localPlayerId || 'not set'}, character ID: ${gameState.character?.id || 'not set'}`);
    }
    
    // IMPORTANT: Make sure playerManager is registered with gameState
    if (gameState && !gameState.playerManager) {
        gameState.playerManager = playerManager;
        console.log("Registered playerManager with gameState");
    }
    
    // Clear old players that aren't in the update
    const removedPlayers = [];
    for (let id of playerManager.players.keys()) {
        if (!actualPlayerData[id] && id !== gameState.character?.id && id !== playerManager.localPlayerId) {
            console.log(`Removing player ${id} - no longer in updates`);
            playerManager.players.delete(id);
            removedPlayers.push(id);
        }
    }
    
    // Track new and updated players
    const newPlayers = [];
    const updatedPlayers = [];
    
    // Update or add new players
    for (let [id, data] of Object.entries(actualPlayerData)) {
        // Skip local player since it's handled separately
        if (id === gameState.character?.id || id === playerManager.localPlayerId) {
            if (Math.random() < 0.05) {
                console.log(`Skipping local player update for ID: ${id}`);
            }
            continue;
        }
        
        // Verify the player data has required properties - should be redundant now with filtering
        if (!data || typeof data !== 'object' || data.x === undefined || data.y === undefined) {
            console.warn(`Invalid player data for ID ${id}:`, data);
            continue;
        }
        
        if (!playerManager.players.has(id)) {
            // Log when adding a new player
            console.log(`Adding new player: ${id} at position (${data.x}, ${data.y})`);
            newPlayers.push(id);
            
            // Enhance player data with rendering info if missing
            const enhancedData = {
                id: id, // IMPORTANT: Make sure ID is set
                ...data,
                // Use same dimensions as local player (10x10)
                width: data.width || 10,
                height: data.height || 10,
                // Use TILE_SIZE (usually 8) for sprite dimensions
                spriteX: data.spriteX !== undefined ? data.spriteX : 0,
                spriteY: data.spriteY !== undefined ? data.spriteY : 0,
                // Cosmetic properties
                health: data.health || 100,
                maxHealth: data.maxHealth || 100,
                // Store the current position as both current and target for interpolation
                _prevX: data.x,
                _prevY: data.y, 
                _targetX: data.x,
                _targetY: data.y,
                // Timestamps for interpolation
                lastPositionUpdate: Date.now(),
                lastUpdate: Date.now()
            };
            
            // Create a proper data object
            playerManager.players.set(id, enhancedData);
        } else {
            // Update existing player with new data
            updatedPlayers.push(id);
            const player = playerManager.players.get(id);
            
            // Save previous position for interpolation
            player._prevX = player.x;
            player._prevY = player.y;
            
            // Update position directly for now
            Object.assign(player, data);
            
            // Store new position as target position
            player._targetX = data.x;
            player._targetY = data.y;
            
            // Update timestamps
            player.lastPositionUpdate = Date.now();
            player.lastUpdate = Date.now();
            
            // Make sure ID is properly set
            player.id = id;
        }
    }
    
    // Summary log with useful details
    if (newPlayers.length > 0 || updatedPlayers.length > 0 || removedPlayers.length > 0) {
        console.log(`Player update complete: ${newPlayers.length} new, ${updatedPlayers.length} updated, ${removedPlayers.length} removed`);
        console.log(`Current players in manager: ${playerManager.players.size}`);
    }
}

/**
 * Update bullets based on server data
 * @param {Array} bulletsData - Bullet data from server
 */
export function updateBullets(bulletsData) {
    if (!bulletsData || !Array.isArray(bulletsData)) return;
    
    // Clear bullets array and repopulate with server data
    bullets.length = 0;
    
    for (const bulletData of bulletsData) {
        // Add sprite properties if missing
        const enhancedBullet = {
            ...bulletData,
            lastUpdated: Date.now(),
            // Add these properties if missing
            width: bulletData.width || 8,
            height: bulletData.height || 8,
            // For rendering
            spriteSheet: bulletData.spriteSheet || 'bullet_sprites',
            spriteX: bulletData.spriteX !== undefined ? bulletData.spriteX : 8 * 10, // Default X position in sprite sheet
            spriteY: bulletData.spriteY !== undefined ? bulletData.spriteY : 8 * 11, // Default Y position in sprite sheet
        };
        
        bullets.push(enhancedBullet);
    }
    
    // Log for debugging (only occasionally to reduce console spam)
    if (bulletsData.length > 0 && Math.random() < 0.05) {
        console.log(`Updated ${bulletsData.length} bullets in entities module`);
        console.log(`Current bullet owners: ${[...new Set(bulletsData.map(b => b.ownerId))].join(', ')}`);
    }
}

/**
 * Render other players - called from the render system
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 */
export function renderOtherPlayers(ctx) {
    if (!playerManager) return;
    
    // Use the PlayerManager for rendering
    playerManager.render(ctx, gameState.camera.position);
}

/**
 * Render bullets from all players - can be used as an alternative to bulletManager rendering
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 */
export function renderAllBullets(ctx) {
    if (!ctx || bullets.length === 0) return;
    
    const screenCenterX = ctx.canvas.width / 2;
    const screenCenterY = ctx.canvas.height / 2;
    const scaleFactor = gameState.camera.viewType === 'strategic' ? 0.5 : 1;
    
    // Get sprite manager
    const spriteManager = window.spriteManager || null;
    
    // Only log bullet rendering very rarely to reduce spam
    if (Math.random() < 0.001) {  // Log only 0.1% of the time
        console.log(`Rendering ${bullets.length} bullets through renderAllBullets`);
    }
    
    for (const bullet of bullets) {
        try {
            // Calculate screen coordinates - use TILE_SIZE to scale properly
            const screenX = (bullet.x - gameState.camera.position.x) * TILE_SIZE * scaleFactor + screenCenterX;
            const screenY = (bullet.y - gameState.camera.position.y) * TILE_SIZE * scaleFactor + screenCenterY;
            
            // Skip if the bullet is way off screen
            if (screenX < -50 || screenX > ctx.canvas.width + 50 || 
                screenY < -50 || screenY > ctx.canvas.height + 50) {
                continue;
            }
            
            // Get bullet color based on owner
            const isLocalBullet = bullet.ownerId === playerManager.localPlayerId;
            
            // Check if we have sprite information for this bullet
            let spriteSuccess = false;
            if (spriteManager && bullet.spriteSheet) {
                try {
                    // Width and height in pixels for rendering
                    const width = (bullet.width || 8) * scaleFactor;
                    const height = (bullet.height || 8) * scaleFactor;
                    
                    // Render with sprite
                    spriteManager.drawSprite(
                        ctx,
                        bullet.spriteSheet,
                        bullet.spriteX, 
                        bullet.spriteY,
                        screenX - width / 2,
                        screenY - height / 2,
                        width,
                        height
                    );
                    spriteSuccess = true;
                } catch (spriteError) {
                    // Silently handle sprite errors - will use fallback
                    spriteSuccess = false;
                }
            }
            
            // Fallback to a more visible circle if sprite fails
            if (!spriteSuccess) {
                const bulletSize = (bullet.width || 8) * scaleFactor;
                
                // Draw a glow effect for better visibility
                const gradient = ctx.createRadialGradient(
                    screenX, screenY, 0,
                    screenX, screenY, bulletSize
                );
                
                if (isLocalBullet) {
                    // Yellow/orange glow for player bullets
                    gradient.addColorStop(0, 'rgb(255, 255, 120)');
                    gradient.addColorStop(0.7, 'rgb(255, 160, 0)');
                    gradient.addColorStop(1, 'rgba(255, 100, 0, 0)');
                } else {
                    // Red/purple glow for enemy bullets
                    gradient.addColorStop(0, 'rgb(255, 100, 255)');
                    gradient.addColorStop(0.7, 'rgb(255, 0, 100)');
                    gradient.addColorStop(1, 'rgba(200, 0, 0, 0)');
                }
                
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(screenX, screenY, bulletSize, 0, Math.PI * 2);
                ctx.fill();
                
                // Add a white center for better visibility
                ctx.fillStyle = 'white';
                ctx.beginPath();
                ctx.arc(screenX, screenY, bulletSize * 0.3, 0, Math.PI * 2);
                ctx.fill();
            }
        } catch (error) {
            // Silently handle errors to avoid console spam
        }
    }
}

/**
 * Get bullet data (for other systems that need it)
 * @returns {Array} Array of bullet objects
 */
export function getBullets() {
    return bullets;
}

/**
 * Initialize player system
 * @param {string} localPlayerId - ID of the local player
 */
export function initializePlayers(localPlayerId) {
    if (playerManager) {
        playerManager.setLocalPlayerId(localPlayerId);
        console.log(`Initialized player system with local player ID: ${localPlayerId}`);
    }
}

/**
 * Update player positions using interpolation for smooth movement
 * @param {number} delta - Time elapsed since last frame in seconds
 */
export function updatePlayerInterpolation(delta) {
    if (!playerManager || playerManager.players.size === 0) return;

    // How fast to move toward target position (higher = faster)
    const INTERPOLATION_SPEED = 10;

    // Interpolate each player's position
    for (const player of playerManager.players.values()) {
        if (!player._targetX || !player._targetY) continue;
        
        // Calculate interpolation factor based on time and speed
        // This produces smoother motion than just using a fixed step
        const t = Math.min(delta * INTERPOLATION_SPEED, 1);
        
        // Smoothly interpolate position
        player.x = player.x + (player._targetX - player.x) * t;
        player.y = player.y + (player._targetY - player.y) * t;
    }
}

// Export bullets array for direct access if needed
export { bullets, playerManager }; 