// src/camera.js

export class Camera {
  constructor(viewType, position = { x: 0, y: 0 }, zoom = 1) {
    this.viewType = viewType; // 'top-down', 'first-person', 'strategic'
    this.position = position; // { x, y }
    this.zoom = zoom; // For top-down and strategic views
    this.rotation = { pitch: 0, yaw: 0 }; // For first-person view
    
    // View scaling factors based on view type
    this.viewScaleFactors = {
      'top-down': 1.0,
      'first-person': 1.0,
      'strategic': 0.5 // Make strategic view show more of the map
    };
    
    // FIXED: Entity position correction is not needed with our new approach
    // This correction is only applied in special cases where direct alignment is required
    this.entityPositionCorrection = { x: 0.0, y: 0.0 };
    
    // FIXED: Add a proper scale ratio between tiles and entities
    // Tiles are now 4x larger than entities in the render
    this.tileToEntityScaleRatio = 4.0;
    
    // Add a correction factor to synchronize entity positions with tile grid
    // This is a multiplier to apply to entity coordinates to align with tiles
    this.entityToTileCorrection = 1.0;
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
   * @param {boolean} isEntity - Whether the coordinates are for an entity (applies correction)
   * @returns {Object} Screen coordinates {x, y}
   */
  worldToScreen(worldX, worldY, screenWidth, screenHeight, tileSize, isEntity = true) {
    const viewScaleFactor = this.getViewScaleFactor();
    
    // Apply entity correction if needed
    let correctedWorldX = worldX;
    let correctedWorldY = worldY;
    
    if (isEntity && this.entityPositionCorrection) {
      // Only apply correction if needed
      correctedWorldX += this.entityPositionCorrection.x;
      correctedWorldY += this.entityPositionCorrection.y;
    }
    
    // FIXED: Account for the different scale between tiles and entities
    // Entities use the base scale, while tiles use a 4x larger scale
    let scaleFactor = viewScaleFactor;
    if (!isEntity) {
      // When rendering tiles, they need to be larger
      scaleFactor *= this.tileToEntityScaleRatio;
    }
    
    // Apply consistent formula for world-to-screen coordinate transformation
    const screenX = (correctedWorldX - this.position.x) * tileSize * scaleFactor + screenWidth / 2;
    const screenY = (correctedWorldY - this.position.y) * tileSize * scaleFactor + screenHeight / 2;
    
    return { x: screenX, y: screenY };
  }
  
  /**
   * Get the scaling factor to apply to entity sizes based on view type
   * @param {number} baseScale - Base scale factor (usually SCALE constant)
   * @param {boolean} isTile - Whether this is for a tile (applies tile-to-entity ratio)
   * @returns {number} Effective scale to use for entity rendering
   */
  getEntityScaleFactor(baseScale = 1, isTile = false) {
    let scale = baseScale * this.getViewScaleFactor();
    if (isTile) {
      scale *= this.tileToEntityScaleRatio;
    }
    return scale;
  }
  
  /**
   * Sets the correction factor to align entity coordinate system with tile coordinate system
   * @param {number} factor - The correction factor (multiplier)
   */
  setEntityToTileCorrection(factor) {
    if (typeof factor === 'number' && !isNaN(factor) && factor > 0) {
      this.entityToTileCorrection = factor;
      console.log(`Camera: Set entity-to-tile correction factor to ${factor}`);
    }
  }
  
  /**
   * Sets the position correction to align entities with tiles
   * @param {Object} correction - Correction values { x, y }
   */
  setEntityPositionCorrection(correction) {
    if (correction && typeof correction.x === 'number' && typeof correction.y === 'number') {
      this.entityPositionCorrection = correction;
      console.log(`Camera: Set entity position correction to x:${correction.x}, y:${correction.y}`);
    }
  }
  
  /**
   * Sets the scale ratio between tiles and entities
   * @param {number} ratio - The scale ratio (tiles:entities)
   */
  setTileToEntityScaleRatio(ratio) {
    if (typeof ratio === 'number' && !isNaN(ratio) && ratio > 0) {
      this.tileToEntityScaleRatio = ratio;
      console.log(`Camera: Set tile-to-entity scale ratio to ${ratio}`);
    }
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
   * @param {boolean} isEntity - Whether this is an entity (uses different scale)
   * @returns {boolean} Whether the entity is on screen
   */
  isOnScreen(worldX, worldY, width, height, screenWidth, screenHeight, tileSize, buffer = 0, isEntity = true) {
    const screen = this.worldToScreen(worldX, worldY, screenWidth, screenHeight, tileSize, isEntity);
    const viewScaleFactor = this.getViewScaleFactor();
    
    // Apply the correct scale factor depending on whether this is an entity or tile
    let scaleFactor = viewScaleFactor;
    if (!isEntity) {
      scaleFactor *= this.tileToEntityScaleRatio;
    }
    
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
