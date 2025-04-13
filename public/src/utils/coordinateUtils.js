/**
 * coordinateUtils.js
 * Utility functions for handling world-to-screen coordinate transformations
 */

import { TILE_SIZE, SCALE } from '../constants/constants.js';

/**
 * Gets the view scale factor based on camera view type
 * @param {string} viewType - Camera view type ('top-down', 'first-person', 'strategic')
 * @returns {number} The scale factor for the view
 */
export function getViewScaleFactor(viewType) {
  const viewScaleFactors = {
    'top-down': 1.0,
    'first-person': 1.0,
    'strategic': 0.5 // Fixed to match the rest of the codebase (was 0.25)
  };
  return viewScaleFactors[viewType] || 1.0;
}

/**
 * Converts world coordinates to screen coordinates.
 * @param {number} worldX - X coordinate in world space
 * @param {number} worldY - Y coordinate in world space
 * @param {Object} camera - Camera object with position and view type
 * @param {number} screenWidth - Width of the canvas
 * @param {number} screenHeight - Height of the canvas
 * @returns {Object} Screen coordinates {x, y}
 */
export function worldToScreen(worldX, worldY, camera, screenWidth, screenHeight) {
  const scaleFactor = getViewScaleFactor(camera.viewType);
  
  // Ensure we're using camera.position.x and camera.position.y, not camera.x/y
  const cameraX = camera.position ? camera.position.x : camera.x;
  const cameraY = camera.position ? camera.position.y : camera.y;
  
  // Calculate screen coordinates
  const screenX = (worldX - cameraX) * TILE_SIZE * scaleFactor + screenWidth / 2;
  const screenY = (worldY - cameraY) * TILE_SIZE * scaleFactor + screenHeight / 2;
  
  return { x: screenX, y: screenY };
}

/**
 * Get the scaling factor to apply to entity sizes based on view type
 * @param {string} viewType - Camera view type
 * @param {number} baseScale - Base scale factor (usually SCALE constant)
 * @returns {number} Effective scale to use for entity rendering
 */
export function getEntityScaleFactor(viewType, baseScale = SCALE) {
  return baseScale * getViewScaleFactor(viewType);
}

/**
 * Determines if a world coordinate is visible on screen
 * @param {number} worldX - X coordinate in world space
 * @param {number} worldY - Y coordinate in world space
 * @param {number} width - Entity width in world units
 * @param {number} height - Entity height in world units
 * @param {Object} camera - Camera object with position and view type
 * @param {number} screenWidth - Width of the canvas
 * @param {number} screenHeight - Height of the canvas
 * @param {number} buffer - Extra buffer to add around screen (for culling)
 * @returns {boolean} Whether the entity is on screen
 */
export function isOnScreen(worldX, worldY, width, height, camera, screenWidth, screenHeight, buffer = 0) {
  const screen = worldToScreen(worldX, worldY, camera, screenWidth, screenHeight);
  const scaleFactor = getViewScaleFactor(camera.viewType);
  
  // Extend buffer in strategic view
  const viewBuffer = camera.viewType === 'strategic' ? buffer * 2 : buffer;
  
  // Use the scaled entity dimensions
  const scaledWidth = width * TILE_SIZE * scaleFactor;
  const scaledHeight = height * TILE_SIZE * scaleFactor;
  
  return screen.x + scaledWidth/2 + viewBuffer >= 0 && 
         screen.x - scaledWidth/2 - viewBuffer <= screenWidth &&
         screen.y + scaledHeight/2 + viewBuffer >= 0 && 
         screen.y - scaledHeight/2 - viewBuffer <= screenHeight;
}

/**
 * Initialize coordinate utilities and make them available globally
 * This makes the utilities accessible to all parts of the codebase
 */
export function initCoordinateUtils() {
  // Create the global object
  window.coordUtils = {
    getViewScaleFactor,
    worldToScreen,
    getEntityScaleFactor,
    isOnScreen
  };
  
  console.log('Coordinate utilities initialized globally');
  return window.coordUtils;
}

// Auto-initialize when imported
initCoordinateUtils(); 