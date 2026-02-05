// src/entities/projectile.js
export class Projectile {
    constructor(x, y, z, velocity, damage, lifespan, owner) {
      this.x = x; // World coordinates
      this.y = y;
      this.z = z; // For 3D rendering
      this.velocity = velocity; // { x: dx, y: dy, z: dz }
      this.damage = damage;
      this.lifespan = lifespan; // Time in seconds before the projectile disappears
      this.owner = owner; // Reference to the entity that fired the projectile
      this.age = 0; // Time since the projectile was fired
      this.sprite = null; // For 3D rendering
    }
  
    update(delta) {
      // Update position
      this.x += this.velocity.x * delta;
      this.y += this.velocity.y * delta;
      this.z += this.velocity.z * delta;
  
      // Update age
      this.age += delta;
  
      // Update sprite position if in 3D view
      if (this.sprite) {
        this.sprite.position.set(this.x, this.z, this.y);
      }
    }
  
    isExpired() {
      return this.age >= this.lifespan;
    }
  
    // Additional methods for collision detection can be added here
  }
  