/**
 * AdditionalBehaviors.js
 *
 * Some unique or advanced behaviors:
 *   - LayTrapBehavior
 *   - TimeDistortBehavior
 *   - GravityWellBehavior
 */

export class LayTrapBehavior {
    /**
     * @param {object} config
     * @param {number} [config.cooldown=5000]
     * @param {object} [config.trapData] = e.g. { damage: 20, radius: 1, duration: 3000 }
     */
    constructor(config = {}) {
      this.cooldown = config.cooldown ?? 5000;
      this.trapData = config.trapData || { damage: 20, radius: 1, duration: 3000 };
      this._timeSinceTrap = 0;
    }
  
    update(entity, target, dt) {
      this._timeSinceTrap += dt;
      if (this._timeSinceTrap < this.cooldown) return;
  
      // Condition: maybe only lay trap if fleeing or random? We'll just do it on cooldown
      this._timeSinceTrap = 0;
      createTrap({
        x: entity.x,
        y: entity.y,
        ...this.trapData
      });
    }
  }
  
  export class TimeDistortBehavior {
    /**
     * @param {object} config
     * @param {number} [config.radius=3]
     * @param {number} [config.slowFactor=0.5]
     * @param {number} [config.duration=2000]
     * @param {number} [config.cooldown=8000]
     * @param {boolean} [config.hasteSelf=false]
     * @param {number} [config.hasteFactor=1.5]
     */
    constructor(config = {}) {
      this.radius = config.radius ?? 3;
      this.slowFactor = config.slowFactor ?? 0.5;
      this.duration = config.duration ?? 2000;
      this.cooldown = config.cooldown ?? 8000;
      this.hasteSelf = config.hasteSelf ?? false;
      this.hasteFactor = config.hasteFactor ?? 1.5;
  
      this._timeSinceDistort = 0;
    }
  
    update(entity, target, dt) {
      this._timeSinceDistort += dt;
      if (this._timeSinceDistort < this.cooldown) return;
  
      this._timeSinceDistort = 0;
      // Slow everything in radius
      const victims = findEntitiesInRange(entity, this.radius);
      for (const v of victims) {
        if (v !== entity) {
          applyTemporarySlow(v, this.slowFactor, this.duration);
        }
      }
      if (this.hasteSelf) {
        applyTemporaryHaste(entity, this.hasteFactor, this.duration);
      }
    }
  }
  
  export class GravityWellBehavior {
    /**
     * @param {object} config
     * @param {number} [config.cooldown=10000]
     * @param {number} [config.radius=4]
     * @param {number} [config.force=2]
     * @param {number} [config.duration=2000]
     */
    constructor(config = {}) {
      this.cooldown = config.cooldown ?? 10000;
      this.radius = config.radius ?? 4;
      this.force = config.force ?? 2;
      this.duration = config.duration ?? 2000;
  
      this._timeSinceWell = 0;
      this._wellActive = false;
      this._wellTimer = 0;
    }
  
    update(entity, target, dt) {
      this._timeSinceWell += dt;
  
      if (this._wellActive) {
        this._wellTimer += dt;
        if (this._wellTimer >= this.duration) {
          this._wellActive = false;
          return;
        }
        const victims = findEntitiesInRange(entity, this.radius);
        for (const v of victims) {
          // Pull each entity inward
          if (v !== entity) {
            const dx = entity.x - v.x;
            const dy = entity.y - v.y;
            const dist = Math.max(0.1, Math.sqrt(dx * dx + dy * dy));
            if (dist < this.radius) {
              // Force is stronger the closer you are to the center, 
              // or you can invert it to be stronger further away
              const pullStrength = this.force * (1 - dist / this.radius);
              const angle = Math.atan2(dy, dx);
              v.x += Math.cos(angle) * pullStrength * dt;
              v.y += Math.sin(angle) * pullStrength * dt;
            }
          }
        }
        return;
      }
  
      // Check if we can trigger a new well
      if (this._timeSinceWell >= this.cooldown) {
        // Example condition: if target is near or random chance
        this._wellActive = true;
        this._wellTimer = 0;
        this._timeSinceWell = 0;
      }
    }
  }
  
  // Placeholder engine calls
  function createTrap(config) {}
  function findEntitiesInRange(entity, radius) { return []; }
  function applyTemporarySlow(entity, factor, duration) {}
  function applyTemporaryHaste(entity, factor, duration) {}
  