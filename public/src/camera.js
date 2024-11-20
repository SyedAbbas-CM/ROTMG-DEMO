// src/camera.js

export class Camera {
  constructor(viewType, position = { x: 0, y: 0 }, zoom = 1) {
    this.viewType = viewType; // 'top-down', 'first-person', 'strategic'
    this.position = position; // { x, y }
    this.zoom = zoom; // For top-down and strategic views
    this.rotation = { pitch: 0, yaw: 0 }; // For first-person view
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
}
