// src/camera.js
import { createLogger } from './utils/logger.js';

// Create a logger for the camera module
const logger = createLogger('camera');

export class Camera {
  constructor(viewType, position = { x: 0, y: 0 }, zoom = 1) {
    this.viewType = viewType; // 'top-down', 'first-person', 'strategic'
    this.position = position; // { x, y }
    this.zoom = zoom; // For top-down and strategic views
    this.rotation = { pitch: 0, yaw: 0 }; // For first-person view
    
    // View scaling factors based on view type
    this.viewScaleFactors = {
      'top-down': 4.0,
      'first-person': 1.0,
      'strategic': 0.5 // Make strategic view show more of the map
    };
    
    // Debug mode - set to false by default
    this.debugMode = false;
  }

  move(dx, dy) {
    this.position.x += dx;
    this.position.y += dy;
  }

  setZoom(zoomLevel) {
    this.zoom = zoomLevel;
  }

  setRotation(pitch, yaw) {
    this.rotation.pitch = pitch;
    this.rotation.yaw = yaw;
  }

  /**
   * Updates the camera's position to the specified coordinates.
   * @param {Object} newPosition - { x, y } coordinates to set the camera position.
   */
  updatePosition(newPosition) {
    if (newPosition.x !== undefined) this.position.x = newPosition.x;
    if (newPosition.y !== undefined) this.position.y = newPosition.y;
    
    // Log camera position when in debug mode
    if (this.debugMode) {
      console.log(`Camera position updated to: (${this.position.x.toFixed(2)}, ${this.position.y.toFixed(2)})`);
    }
  }

  /**
   * Updates the camera's rotation to the specified values.
   * @param {Object} newRotation - { pitch, yaw } angles in radians.
   */
  updateRotation(newRotation) {
    if (newRotation.pitch !== undefined) this.rotation.pitch = newRotation.pitch;
    if (newRotation.yaw !== undefined) this.rotation.yaw = newRotation.yaw;
  }
  
  /**
   * Gets the current view scale factor based on view type
   * @returns {number} The scale factor for the current view
   */
  getViewScaleFactor() {
    return this.viewScaleFactors[this.viewType] || 1.0;
  }
  
  /**
   * Converts world coordinates to screen coordinates.
   * @param {number} worldX - X coordinate in world space
   * @param {number} worldY - Y coordinate in world space
   * @param {number} screenWidth - Width of the canvas
   * @param {number} screenHeight - Height of the canvas
   * @param {number} tileSize - Tile size in pixels
   * @returns {Object} Screen coordinates {x, y}
   */
  worldToScreen(worldX, worldY, screenWidth, screenHeight, tileSize) {
    const scaleFactor = this.getViewScaleFactor();
    // This formula transforms world coordinates to screen coordinates
    const screenX = (worldX - this.position.x) * tileSize * scaleFactor + screenWidth / 2;
    const screenY = (worldY - this.position.y) * tileSize * scaleFactor + screenHeight / 2;
    
    // Debug logging (only occasionally to avoid spamming console)
    if (this.debugMode && Math.random() < 0.01) {
      console.log(`worldToScreen: 
        World (${worldX}, ${worldY}) 
        Camera (${this.position.x}, ${this.position.y}) 
        Screen (${screenX}, ${screenY})
        TileSize: ${tileSize}, ScaleFactor: ${scaleFactor}
      `);
    }
    
    return { x: screenX, y: screenY };
  }
  
  /**
   * Toggle debug mode
   * @returns {boolean} The new debug mode state
   */
  toggleDebugMode() {
    this.debugMode = !this.debugMode;
    logger.info(`Camera debug mode ${this.debugMode ? 'enabled' : 'disabled'}`);
    return this.debugMode;
  }
  
  /**
   * Get the scaling factor to apply to entity sizes based on view type
   * @param {number} baseScale - Base scale factor (usually SCALE constant)
   * @returns {number} Effective scale to use for entity rendering
   */
  getEntityScaleFactor(baseScale = 1) {
    return baseScale * this.getViewScaleFactor();
  }
  
  /**
   * Determines if a world coordinate is visible on screen
   * @param {number} worldX - X coordinate in world space
   * @param {number} worldY - Y coordinate in world space
   * @param {number} width - Entity width in world units
   * @param {number} height - Entity height in world units
   * @param {number} screenWidth - Width of the canvas
   * @param {number} screenHeight - Height of the canvas
   * @param {number} tileSize - Tile size in pixels
   * @param {number} buffer - Extra buffer to add around screen (for culling)
   * @returns {boolean} Whether the entity is on screen
   */
  isOnScreen(worldX, worldY, width, height, screenWidth, screenHeight, tileSize, buffer = 0) {
    const screen = this.worldToScreen(worldX, worldY, screenWidth, screenHeight, tileSize);
    const scaleFactor = this.getViewScaleFactor();
    
    // Extend buffer in strategic view
    const viewBuffer = this.viewType === 'strategic' ? buffer * 2 : buffer;
    
    // Use the scaled entity dimensions
    const scaledWidth = width * tileSize * scaleFactor;
    const scaledHeight = height * tileSize * scaleFactor;
    
    return screen.x + scaledWidth/2 + viewBuffer >= 0 && 
           screen.x - scaledWidth/2 - viewBuffer <= screenWidth &&
           screen.y + scaledHeight/2 + viewBuffer >= 0 && 
           screen.y - scaledHeight/2 - viewBuffer <= screenHeight;
  }
}
