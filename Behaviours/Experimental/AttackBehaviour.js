/**
 * AttackBehavior.js
 *
 * Handles projectile creation (spread, wave, spiral, random, homing, etc.)
 * Also includes optional on-hit leech (vampire) and explode-on-low-health.
 */

export class AttackBehavior {
    /**
     * @param {object} config
     * @param {number} [config.cooldown=1000]  - ms between shots
     * @param {number} [config.numProjectiles=1]
     * @param {number} [config.spreadAngle=0]  - total radian spread for multiple projectiles
     * @param {number} [config.projectileSpeed=5]
     * @param {number} [config.damage=10]
     * @param {string} [config.pattern="straight"] - "straight"|"wave"|"spiral"|"random"|"homing"
     * @param {boolean} [config.piercing=false]
     *
     * @param {boolean} [config.leechOnHit=false]  - If true, entity regains HP each time a projectile hits
     * @param {number} [config.leechAmount=5]
     *
     * @param {boolean} [config.explodeOnLowHealth=false]
     * @param {number}  [config.explodeThreshold=0.2] - fraction of HP
     * @param {number}  [config.explosionRadius=2]
     * @param {number}  [config.explosionDamage=30]
     */
    constructor(config = {}) {
      this.cooldown = config.cooldown ?? 1000;
      this.numProjectiles = config.numProjectiles ?? 1;
      this.spreadAngle = config.spreadAngle ?? 0;
      this.projectileSpeed = config.projectileSpeed ?? 5;
      this.damage = config.damage ?? 10;
      this.pattern = config.pattern ?? "straight";
      this.piercing = config.piercing ?? false;
  
      this.leechOnHit = config.leechOnHit ?? false;
      this.leechAmount = config.leechAmount ?? 5;
  
      this.explodeOnLowHealth = config.explodeOnLowHealth ?? false;
      this.explodeThreshold = config.explodeThreshold ?? 0.2;
      this.explosionRadius = config.explosionRadius ?? 2;
      this.explosionDamage = config.explosionDamage ?? 30;
  
      this._timeSinceLastShot = 0;
    }
  
    /**
     * Called by your game's main update loop.
     * @param {object} entity - The monster
     * @param {object} target - The player/hero
     * @param {number} dt     - time delta
     */
    update(entity, target, dt) {
      // 1) Check explode on low HP
      if (this.explodeOnLowHealth) {
        const hpFraction = entity.hp / entity.maxHp;
        if (hpFraction <= this.explodeThreshold) {
          this._explode(entity);
          return;
        }
      }
  
      // 2) Attack logic
      this._timeSinceLastShot += dt;
      if (!target) return;
  
      if (this._timeSinceLastShot >= this.cooldown) {
        this._timeSinceLastShot = 0;
        this._shoot(entity, target);
      }
    }
  
    /**
     * Called when the projectile from this entity hits a target
     */
    onProjectileHit(entity, target) {
      if (this.leechOnHit) {
        entity.hp = Math.min(entity.maxHp, entity.hp + this.leechAmount);
      }
    }
  
    // -------------------------------------
    // Internal Helpers
    // -------------------------------------
  
    _shoot(entity, target) {
      const dx = target.x - entity.x;
      const dy = target.y - entity.y;
      let angle = Math.atan2(dy, dx);
  
      // Pattern modifiers
      if (this.pattern === "wave") {
        angle += Math.sin(performance.now() * 0.005) * 0.5;
      } else if (this.pattern === "random") {
        angle += (Math.random() - 0.5) * Math.PI;
      } else if (this.pattern === "spiral") {
        angle += (performance.now() * 0.001);
      }
      // "homing" logic typically goes inside the projectile itself
  
      // Spread
      const startAngle = angle - (this.spreadAngle * (this.numProjectiles - 1)) / 2;
      for (let i = 0; i < this.numProjectiles; i++) {
        const a = startAngle + i * this.spreadAngle;
  
        createProjectile({
          owner: entity,
          x: entity.x,
          y: entity.y,
          angle: a,
          speed: this.projectileSpeed,
          damage: this.damage,
          piercing: this.piercing,
          pattern: this.pattern,
          attackBehavior: this // reference, so projectile can call on hit
        });
      }
    }
  
    _explode(entity) {
      createExplosion({
        x: entity.x,
        y: entity.y,
        radius: this.explosionRadius,
        damage: this.explosionDamage
      });
      entity.destroy?.(); // remove from the game if .destroy() is your engine method
    }
  }
  
  // Placeholder engine calls. Replace with actual implementations:
  function createProjectile(params) {
    // e.g., engine.createProjectile(params)
  }
  function createExplosion(params) {
    // e.g., engine.createExplosion(params)
  }
  