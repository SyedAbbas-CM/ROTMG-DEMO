
/**
 * MovementBehavior.js
 * 
 * A single class that handles multiple movement "modes" depending on config.
 *   - "approach": Move towards target until within a certain distance
 *   - "maintainDistance": Keep away if closer than a threshold
 *   - "flee": Run away if target is within fleeDistance
 *   - "random": Wander randomly
 *   - "strafe": Orbit around the target at a certain radius
 *   - "none": Do nothing
 *
 * Expand or modify as needed. The key idea is that all movement logic lives here
 * and is chosen by a 'mode' parameter.
 */

export class MovementBehavior {
  /**
   * @param {object} config
   * @param {string} config.mode - "approach" | "maintainDistance" | "flee" | "random" | "strafe" | "none"
   * @param {number} config.speed - Movement speed in units per second (or per update cycle if dt=1).
   * @param {number} [config.distance=5] - Used by "approach" & "maintainDistance" & "strafe"
   * @param {number} [config.fleeDistance=3] - Used by "flee"
   * @param {number} [config.changeInterval=2000] - For "random" movement, how often to pick a new direction (ms)
   */
  constructor(config) {
    this.mode = config.mode || "none";
    this.speed = config.speed ?? 1;

    this.distance = config.distance ?? 5;
    this.fleeDistance = config.fleeDistance ?? 3;
    this.changeInterval = config.changeInterval ?? 2000;

    // For random movement
    this._timeSinceChange = 0;
    this._randDir = { x: 0, y: 0 };

    // For strafe
    this._strafeAngle = Math.random() * 2 * Math.PI;
  }

  /**
   * @param {object} entity - The monster entity (must have x, y, hp, etc.)
   * @param {object} target - The player or other entity we're reacting to (x,y)
   * @param {number} dt - Delta time (ms or s). Must be consistent with speed usage
   */
  update(entity, target, dt) {
    if (!target && this.mode !== "random") return;

    switch (this.mode) {
      case "approach":
        this._approach(entity, target, dt);
        break;
      case "maintainDistance":
        this._maintainDistance(entity, target, dt);
        break;
      case "flee":
        this._flee(entity, target, dt);
        break;
      case "random":
        this._randomMovement(entity, dt);
        break;
      case "strafe":
        this._strafe(entity, target, dt);
        break;
      case "none":
      default:
        // Do nothing
        break;
    }
  }

  _approach(entity, target, dt) {
    const dx = target.x - entity.x;
    const dy = target.y - entity.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > this.distance) {
      const angle = Math.atan2(dy, dx);
      entity.x += Math.cos(angle) * this.speed * dt;
      entity.y += Math.sin(angle) * this.speed * dt;
    }
  }

  _maintainDistance(entity, target, dt) {
    const dx = target.x - entity.x;
    const dy = target.y - entity.y;
    const distSq = dx * dx + dy * dy;
    const desiredSq = this.distance * this.distance;

    // If too close, move away
    if (distSq < desiredSq) {
      const angle = Math.atan2(dy, dx);
      entity.x -= Math.cos(angle) * this.speed * dt;
      entity.y -= Math.sin(angle) * this.speed * dt;
    }
    // If you want them to move closer if they're too far, you could do that here
  }

  _flee(entity, target, dt) {
    const dx = target.x - entity.x;
    const dy = target.y - entity.y;
    const distSq = dx * dx + dy * dy;
    const fleeSq = this.fleeDistance * this.fleeDistance;

    if (distSq < fleeSq) {
      const angle = Math.atan2(dy, dx);
      entity.x -= Math.cos(angle) * this.speed * dt;
      entity.y -= Math.sin(angle) * this.speed * dt;
    }
  }

  _randomMovement(entity, dt) {
    this._timeSinceChange += dt;
    if (this._timeSinceChange >= this.changeInterval) {
      this._timeSinceChange = 0;
      const angle = Math.random() * 2 * Math.PI;
      this._randDir.x = Math.cos(angle);
      this._randDir.y = Math.sin(angle);
    }
    entity.x += this._randDir.x * this.speed * dt;
    entity.y += this._randDir.y * this.speed * dt;
  }

  _strafe(entity, target, dt) {
    // Circle around the target at "distance" radius
    this._strafeAngle += (this.speed / this.distance) * dt;
    entity.x = target.x + Math.cos(this._strafeAngle) * this.distance;
    entity.y = target.y + Math.sin(this._strafeAngle) * this.distance;
  }
}
