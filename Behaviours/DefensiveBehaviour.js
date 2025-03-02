/**
 * BuffBehavior.js
 *
 * Applies a positive effect (like defense boost, speed boost) to allies in range (including self).
 */

export class BuffBehavior {
    /**
     * @param {object} config
     * @param {string} config.stat - e.g. "defense", "speed", "attack"
     * @param {number} config.amount
     * @param {number} config.radius
     * @param {number} config.duration - ms the buff lasts
     * @param {number} config.cooldown - ms between buffs
     * @param {function} [config.findAlliesInRange] - returns array of allies
     */
    constructor(config = {}) {
      this.stat = config.stat ?? "defense";
      this.amount = config.amount ?? 5;
      this.radius = config.radius ?? 3;
      this.duration = config.duration ?? 2000;
      this.cooldown = config.cooldown ?? 5000;
      this.findAlliesInRange = config.findAlliesInRange || defaultFindAllies;
  
      this._timeSinceBuff = 0;
    }
  
    update(entity, target, dt) {
      this._timeSinceBuff += dt;
      if (this._timeSinceBuff < this.cooldown) return;
  
      // Let's say we always buff on cooldown if there's at least 1 ally (or self).
      this._timeSinceBuff = 0;
  
      const allies = this.findAlliesInRange(entity, this.radius);
      // Also buff self
      allies.push(entity);
  
      for (const ally of allies) {
        applyTemporaryBuff(ally, this.stat, this.amount, this.duration);
      }
    }
  }
  
  function defaultFindAllies(entity, radius) {
    // Return array of ally entities within `radius` of `entity`.
    // e.g. engine.findAlliesInRadius(entity, radius)
    return [];
  }
  
  function applyTemporaryBuff(entity, stat, amount, duration) {
    // Example approach: 
    //  1) entity[stat] += amount
    //  2) set a timer to revert it after 'duration'
    entity[stat] = (entity[stat] || 0) + amount;
  
    setTimeout(() => {
      entity[stat] -= amount;
    }, duration);
  }
  