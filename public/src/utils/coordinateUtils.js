/**
 * coordinateUtils.js
 * Utility functions for handling world-to-screen coordinate transformations
 */

import { TILE_SIZE, SCALE } from '../constants/constants.js';

/**
 * CoordinateUtils - Helper for transforming between coordinate systems
 */
export class CoordinateUtils {
  constructor() {
    this.tileSize = TILE_SIZE;
    this.debugMode = false;
  }
  
  /**
   * Convert grid coordinates to world coordinates
   * In our game, grid and world coordinates are the same numerically,
   * but they have different semantic meanings
   */
  gridToWorld(gridX, gridY) {
    return {
      x: gridX,
      y: gridY
    };
  }
  
  /**
   * Convert world coordinates to grid coordinates
   * For fractional world coordinates, this returns the grid cell the coordinates are in
   */
  worldToGrid(worldX, worldY) {
    return {
      x: Math.floor(worldX),
      y: Math.floor(worldY)
    };
  }
  
  /**
   * Convert grid coordinates to screen coordinates
   * @param {number} gridX - X position in grid space
   * @param {number} gridY - Y position in grid space
   * @param {Object} camera - Camera with position and view type
   * @param {number} screenWidth - Screen width in pixels
   * @param {number} screenHeight - Screen height in pixels
   * @returns {Object} Screen coordinates {x, y}
   */
  gridToScreen(gridX, gridY, camera, screenWidth, screenHeight) {
    // First convert to world coordinates
    const worldPos = this.gridToWorld(gridX, gridY);
    
    // Then use world to screen
    return this.worldToScreen(worldPos.x, worldPos.y, camera, screenWidth, screenHeight);
  }
  
  /**
   * Convert world coordinates to screen coordinates
   * @param {number} worldX - X position in world space
   * @param {number} worldY - Y position in world space
   * @param {Object} camera - Camera with position and view type
   * @param {number} screenWidth - Screen width in pixels
   * @param {number} screenHeight - Screen height in pixels
   * @returns {Object} Screen coordinates {x, y}
   */
  worldToScreen(worldX, worldY, camera, screenWidth, screenHeight) {
    const viewType = camera.viewType || 'top-down';
    let scaleFactor = 1.0;
    
    // Get scale factor based on view type
    if (viewType === 'strategic') {
      scaleFactor = 0.5;
    } else if (viewType === 'top-down') {
      scaleFactor = 1.0;
    }
    
    // Calculate screen coordinates
    const screenX = (worldX - camera.position.x) * TILE_SIZE * scaleFactor + screenWidth / 2;
    const screenY = (worldY - camera.position.y) * TILE_SIZE * scaleFactor + screenHeight / 2;
    
    if (this.debugMode) {
      console.log(`World (${worldX}, ${worldY}) -> Screen (${screenX}, ${screenY})`);
    }
    
    return { x: screenX, y: screenY };
  }
  
  /**
   * Convert screen coordinates to world coordinates
   * @param {number} screenX - X position in screen space
   * @param {number} screenY - Y position in screen space
   * @param {Object} camera - Camera with position and view type
   * @param {number} screenWidth - Screen width in pixels
   * @param {number} screenHeight - Screen height in pixels
   * @returns {Object} World coordinates {x, y}
   */
  screenToWorld(screenX, screenY, camera, screenWidth, screenHeight) {
    const viewType = camera.viewType || 'top-down';
    let scaleFactor = 1.0;
    
    // Get scale factor based on view type
    if (viewType === 'strategic') {
      scaleFactor = 0.5;
    } else if (viewType === 'top-down') {
      scaleFactor = 1.0;
    }
    
    // Calculate world coordinates
    const worldX = ((screenX - screenWidth / 2) / (this.tileSize * scaleFactor)) + camera.position.x;
    const worldY = ((screenY - screenHeight / 2) / (this.tileSize * scaleFactor)) + camera.position.y;
    
    return { x: worldX, y: worldY };
  }
  
  /**
   * Convert screen coordinates to grid coordinates
   * @param {number} screenX - X position in screen space
   * @param {number} screenY - Y position in screen space
   * @param {Object} camera - Camera with position and view type
   * @param {number} screenWidth - Screen width in pixels
   * @param {number} screenHeight - Screen height in pixels
   * @returns {Object} Grid coordinates {x, y}
   */
  screenToGrid(screenX, screenY, camera, screenWidth, screenHeight) {
    // First convert to world coordinates
    const worldPos = this.screenToWorld(screenX, screenY, camera, screenWidth, screenHeight);
    
    // Then convert to grid
    return this.worldToGrid(worldPos.x, worldPos.y);
  }
  
  /**
   * Toggle debug mode
   * @returns {boolean} New debug mode state
   */
  toggleDebug() {
    this.debugMode = !this.debugMode;
    return this.debugMode;
  }
}

// Create and export singleton instance
export const coordinateUtils = new CoordinateUtils();

/**
 * Initialize coordinate utilities and make them available globally
 * This makes the utilities accessible to all parts of the codebase
 */
export function initCoordinateUtils() {
  console.log('Coordinate utilities initialized');
  
  // Add to window for debugging if needed
  window.coordinateUtils = coordinateUtils;
  
  return coordinateUtils;
}

// Auto-initialize when imported
initCoordinateUtils();

/**
 * Run a visual test of the coordinate system
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {Object} camera - Camera object
 */
export function testCoordinateSystem(ctx, camera) {
  const screenWidth = ctx.canvas.width;
  const screenHeight = ctx.canvas.height;
  
  // Draw a grid of test points
  ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
  ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
  ctx.lineWidth = 1;
  ctx.font = '10px Arial';
  
  // Draw test points in a grid around the camera
  const gridSize = 5; // Draw a 5x5 grid
  for (let x = -gridSize; x <= gridSize; x++) {
    for (let y = -gridSize; y <= gridSize; y++) {
      // Get grid position relative to camera
      const gridX = Math.floor(camera.position.x) + x;
      const gridY = Math.floor(camera.position.y) + y;
      
      // Convert to screen coordinates
      const screenPos = coordinateUtils.gridToScreen(gridX, gridY, camera, screenWidth, screenHeight);
      
      // Draw point
      ctx.beginPath();
      ctx.arc(screenPos.x, screenPos.y, 3, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw grid coordinates (only on some points to avoid clutter)
      if (x % 2 === 0 && y % 2 === 0) {
        ctx.fillText(`(${gridX},${gridY})`, screenPos.x + 5, screenPos.y - 5);
      }
      
      // Draw tile outline
      const tileSize = TILE_SIZE * (camera.viewType === 'strategic' ? 0.5 : 1.0);
      ctx.strokeRect(
        screenPos.x - tileSize / 2,
        screenPos.y - tileSize / 2,
        tileSize,
        tileSize
      );
    }
  }
  
  // Draw camera position
  const cameraPosScreen = {
    x: screenWidth / 2,
    y: screenHeight / 2
  };
  
  ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
  ctx.beginPath();
  ctx.arc(cameraPosScreen.x, cameraPosScreen.y, 5, 0, Math.PI * 2);
  ctx.fill();
  
  // Draw text for camera position
  ctx.fillText(`Camera: (${camera.position.x.toFixed(1)},${camera.position.y.toFixed(1)})`, 
               cameraPosScreen.x + 10, cameraPosScreen.y);
} 