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
        return;
    }
    
    // Filter out the metadata properties that aren't players
    // Skip properties like 'timestamp' and 'players' which aren't player objects
    const actualPlayerData = {};
    for (const [id, data] of Object.entries(playerData)) {
        // Verify this is actually a player object with coordinates, not metadata
        if (data && typeof data === 'object' && data.x !== undefined && data.y !== undefined) {
            // Server positions are already in correct world coordinates - use them directly
            actualPlayerData[id] = data;
        }
    }
    
    // IMPORTANT: Ensure localPlayerId is properly set
    if (gameState.character && gameState.character.id) {
        if (!playerManager.localPlayerId || playerManager.localPlayerId !== gameState.character.id) {
            playerManager.setLocalPlayerId(gameState.character.id);
        }
    }
    
    // IMPORTANT: Make sure playerManager is registered with gameState
    if (gameState && !gameState.playerManager) {
        gameState.playerManager = playerManager;
    }
    
    // Update players
    playerManager.updatePlayers(actualPlayerData);
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
    }
}

/**
 * Update other players position interpolation
 * @param {number} deltaTime - Time elapsed since last update in seconds
 */
export function updatePlayerInterpolation(deltaTime) {
    if (!playerManager) return;
    
    // Skip if delta time is invalid
    if (!deltaTime || isNaN(deltaTime) || deltaTime <= 0) return;
    
    // Get players to update
    const players = Array.from(playerManager.players.values());
    
    if (players.length === 0) {
        return; // No players to update
    }
    
    // Current time for interpolation calculations
    const now = Date.now();
    
    // First pass: interpolate positions - this is critical for smooth movement
    for (const player of players) {
        // Skip position update if we don't have all the data we need
        if (player._targetX === undefined || player._prevX === undefined) {
            continue;
        }
        
        // Store original position to detect movement
        const oldX = player.x;
        const oldY = player.y;
        
        // Calculate time elapsed since last position update
        const timeElapsed = now - player.lastPositionUpdate;
        
        // Get interpolation progress (0 to 1) - use a longer interpolation time for smoother movement
        // Using 300ms for smoother transitions at potentially lower network update rates
        const interpolationTime = 300; // ms
        const t = Math.min(timeElapsed / interpolationTime, 1.0);
        
        // Apply position interpolation with easing for smoother stops
        // Use quadratic easing out for natural movement
        const ease = 1 - Math.pow(1 - t, 2);
        player.x = player._prevX + (player._targetX - player._prevX) * ease;
        player.y = player._prevY + (player._targetY - player._prevY) * ease;
        
        // Calculate actual movement this frame
        const dx = player.x - oldX;
        const dy = player.y - oldY;
        
        // Check for actual movement (not just interpolation artifacts)
        const movementThreshold = 0.00005; // Very small threshold to catch subtle movement
        const isMovingThisFrame = Math.abs(dx) > movementThreshold || Math.abs(dy) > movementThreshold;
        
        // Update movement state - but avoid flickering by requiring a few frames of no movement
        if (isMovingThisFrame) {
            // Currently moving - reset the stop timer
            player.movementStopTimer = 0;
            player.isMoving = true;
            
            // Update velocity for animation direction
            player.moveVelocity = { 
                x: dx / deltaTime, 
                y: dy / deltaTime 
            };
            
            // Save the facing direction based on current movement
            if (Math.abs(dx) > Math.abs(dy)) {
                player.lastFacingDirection = dx > 0 ? 3 : 1; // right or left
            } else {
                player.lastFacingDirection = dy > 0 ? 0 : 2; // down or up
            }
        } else {
            // Not moving this frame - increment stop timer
            player.movementStopTimer = (player.movementStopTimer || 0) + deltaTime;
            
            // Only change to stopped state after a short delay (100ms) to avoid flickering
            if (player.movementStopTimer > 0.1) {
                player.isMoving = false;
            }
        }
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

// Export this module to window so it can be accessed by the renderer
window.entitiesModule = {
    bullets,
    renderAllBullets,
    renderOtherPlayers,
    updateBullets,
    updatePlayers,
    initializePlayers
}; 