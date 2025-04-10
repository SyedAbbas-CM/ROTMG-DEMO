import { gameState } from '../game/gamestate.js';
import { PlayerManager } from './PlayerManager.js';
import { Player } from '../entities/player.js';
import { TILE_SIZE } from '../constants/constants.js';

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
    
    // Log raw data received (throttled)
    if (throttledLog('update-players-raw', '[updatePlayers] Raw data received:', null, 5000)) {
        console.log(JSON.stringify(playerData));
        console.log(`[updatePlayers] Player IDs in data: ${Object.keys(playerData).join(', ')}`);
        console.log(`[updatePlayers] Local player ID: ${playerManager.localPlayerId}, Character ID: ${gameState.character?.id}`);
    }
    
    // CRITICAL FIX: Filter out the metadata properties that aren't players
    // Skip properties like 'timestamp' and 'players' which aren't player objects
    const actualPlayerData = {};
    for (const [id, data] of Object.entries(playerData)) {
        // Verify this is actually a player object with coordinates, not metadata
        if (data && typeof data === 'object' && data.x !== undefined && data.y !== undefined) {
            actualPlayerData[id] = data;
        } else {
            throttledLog('skip-non-player', `Skipping non-player property: ${id}`, null, 5000);
        }
    }
    
    // Print the filtered player data (throttled)
    const playerCount = Object.keys(actualPlayerData).length;
    throttledLog('player-count', `[updatePlayers] Processing ${playerCount} players after filtering metadata properties`);
    
    if (throttledLog('player-ids-filtered', 'Filtered player IDs', null, 5000)) {
        console.log(`[updatePlayers] Player IDs after filtering: ${Object.keys(actualPlayerData).join(', ')}`);
    }
    
    // IMPORTANT: Ensure localPlayerId is properly set
    if (gameState.character && gameState.character.id) {
        if (!playerManager.localPlayerId || playerManager.localPlayerId !== gameState.character.id) {
            playerManager.setLocalPlayerId(gameState.character.id);
            console.log(`[updatePlayers] Updated playerManager.localPlayerId to match character.id: ${gameState.character.id}`);
        }
    }
    
    // IMPORTANT: Make sure playerManager is registered with gameState
    if (gameState && !gameState.playerManager) {
        gameState.playerManager = playerManager;
        console.log("Registered playerManager with gameState");
    }
    
    // Update players
    playerManager.updatePlayers(actualPlayerData);
    
    // Log summary of current player count after update (throttled)
    throttledLog('update-complete', `[updatePlayers] Update complete, playerManager now has ${playerManager.players.size} players`);
    
    if (playerManager.players.size > 0 && throttledLog('player-ids-current', 'Current player IDs', null, 5000)) {
        console.log(`[updatePlayers] Current player IDs: ${Array.from(playerManager.players.keys()).join(', ')}`);
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
 * Update player position interpolation
 * @param {number} deltaTime - Delta time in seconds
 */
export function updatePlayerInterpolation(deltaTime) {
    if (!playerManager) return;
    
    // Interpolate positions for smooth movement
    for (const player of playerManager.players.values()) {
        // Skip if we don't have interpolation data
        if (player._prevX === undefined || player._targetX === undefined) continue;
        
        // Current position - use lerp for interpolation between prev and target
        player.x = lerp(player._prevX, player._targetX, Math.min(1, (Date.now() - player.lastPositionUpdate) / 100));
        player.y = lerp(player._prevY, player._targetY, Math.min(1, (Date.now() - player.lastPositionUpdate) / 100));
    }
    
    // Update player animations
    playerManager.updatePlayerAnimations(deltaTime);
}

// Linear interpolation helper
function lerp(start, end, t) {
    return start + t * (end - start);
}

// Export bullets array for direct access if needed
export { bullets, playerManager }; 