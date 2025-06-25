/**
 * ECSystemBehaviors.js
 * ECS-compatible behavior functions for the enemy manager
 */

// Behavior functions - each takes entity data arrays and the specific entity index
const behaviors = {
  /**
   * Chase player behavior - moves entity towards player
   * @param {object} entityArrays - Entity data arrays (x, y, speed, etc.)
   * @param {number} index - Entity index
   * @param {object} target - Target entity (usually player)
   * @param {number} dt - Delta time in seconds
   */
  chase: function(entityArrays, index, target, dt) {
    const speed = entityArrays.speed[index];
    const dx = target.x - entityArrays.x[index];
    const dy = target.y - entityArrays.y[index];
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    // Only move if not already at target
    if (dist > 1) {
      entityArrays.x[index] += (dx / dist) * speed * dt;
      entityArrays.y[index] += (dy / dist) * speed * dt;
      
      // Update rotation to face target
      entityArrays.rotation[index] = Math.atan2(dy, dx);
    }
  },
  
  /**
   * Maintain distance behavior - keeps entity at a specific distance from target
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Target entity (usually player)
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters { distance: number }
   */
  maintainDistance: function(entityArrays, index, target, dt, params) {
    const preferredDistance = params?.distance || 5;
    const speed = entityArrays.speed[index];
    const dx = target.x - entityArrays.x[index];
    const dy = target.y - entityArrays.y[index];
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    // Update rotation regardless of movement
    entityArrays.rotation[index] = Math.atan2(dy, dx);
    
    if (Math.abs(dist - preferredDistance) < 0.5) {
      // We're at preferred distance, don't move
      return;
    }
    
    // Direction to move (towards or away from target)
    const moveDir = dist > preferredDistance ? 1 : -1;
    entityArrays.x[index] += moveDir * (dx / dist) * speed * dt;
    entityArrays.y[index] += moveDir * (dy / dist) * speed * dt;
  },
  
  /**
   * Flee behavior - moves entity away from target
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Target entity to flee from
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters { fleeDistance: number }
   */
  flee: function(entityArrays, index, target, dt, params) {
    const fleeDistance = params?.fleeDistance || 7;
    const speed = entityArrays.speed[index];
    const dx = target.x - entityArrays.x[index];
    const dy = target.y - entityArrays.y[index];
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    // Only flee if within flee distance
    if (dist < fleeDistance) {
      // Move away from target
      entityArrays.x[index] -= (dx / dist) * speed * dt;
      entityArrays.y[index] -= (dy / dist) * speed * dt;
    }
  },
  
  /**
   * Patrol behavior - moves entity between waypoints
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Unused, but kept for consistent parameter structure
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters { waypoints: Array<{x,y}> }
   */
  patrol: function(entityArrays, index, target, dt, params) {
    if (!params?.waypoints || params.waypoints.length === 0) return;
    
    // Make sure the entity has a waypoint index
    if (entityArrays.waypointIndex === undefined) {
      entityArrays.waypointIndex = new Int32Array(entityArrays.x.length).fill(0);
    }
    
    const waypoints = params.waypoints;
    const waypointIndex = entityArrays.waypointIndex[index];
    const currentWaypoint = waypoints[waypointIndex];
    const speed = entityArrays.speed[index];
    
    const dx = currentWaypoint.x - entityArrays.x[index];
    const dy = currentWaypoint.y - entityArrays.y[index];
    const dist = Math.sqrt(dx * dx + dy * dy);
    
    // Update rotation to face waypoint
    entityArrays.rotation[index] = Math.atan2(dy, dx);
    
    if (dist < 1) {
      // Reached waypoint, move to next
      entityArrays.waypointIndex[index] = (waypointIndex + 1) % waypoints.length;
    } else {
      // Move towards waypoint
      entityArrays.x[index] += (dx / dist) * speed * dt;
      entityArrays.y[index] += (dy / dist) * speed * dt;
    }
  },
  
  /**
   * Shoot behavior - fires projectiles at target
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Target entity
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters { bulletManager, pattern, cooldown, etc. }
   */
  shoot: function(entityArrays, index, target, dt, params) {
    const bulletManager = params?.bulletManager;
    if (!bulletManager) return;
    
    // Ensure we have cooldown timers
    if (entityArrays.shootCooldown === undefined) {
      entityArrays.shootCooldown = new Float32Array(entityArrays.x.length).fill(0);
    }
    
    // Decrease cooldown
    entityArrays.shootCooldown[index] -= dt;
    
    // Check if can shoot
    if (entityArrays.shootCooldown[index] <= 0) {
      const pattern = params?.pattern || "straight";
      const numProjectiles = params?.numProjectiles || 1;
      const spreadAngle = params?.spreadAngle || 0.3;
      const projectileSpeed = params?.speed || 5;
      const damage = params?.damage || 10;
      const cooldown = params?.cooldown || 2.0;
      
      // Reset cooldown
      entityArrays.shootCooldown[index] = cooldown;
      
      // Calculate angle to target
      const dx = target.x - entityArrays.x[index];
      const dy = target.y - entityArrays.y[index];
      let angle = Math.atan2(dy, dx);
      
      // Apply pattern modifiers
      if (pattern === "wave") {
        angle += Math.sin(Date.now() * 0.005) * 0.5;
      } else if (pattern === "random") {
        angle += (Math.random() - 0.5) * Math.PI;
      } else if (pattern === "spiral") {
        angle += (Date.now() * 0.001);
      }
      
      // Calculate spread start angle
      const startAngle = angle - (spreadAngle * (numProjectiles - 1)) / 2;
      
      // Create projectiles
      for (let i = 0; i < numProjectiles; i++) {
        const shotAngle = startAngle + i * spreadAngle;
        bulletManager.addBullet({
          x: entityArrays.x[index],
          y: entityArrays.y[index],
          vx: Math.cos(shotAngle) * projectileSpeed,
          vy: Math.sin(shotAngle) * projectileSpeed,
          ownerId: entityArrays.id[index],
          damage: damage,
          lifetime: 3.0,
          worldId: entityArrays.worldId ? entityArrays.worldId[index] : undefined
        });
      }
    }
  },
  
  /**
   * Teleport behavior - randomly teleports entity when threatened
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Target entity
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters { range, cooldown, triggerDistance }
   */
  teleport: function(entityArrays, index, target, dt, params) {
    // Ensure we have cooldown timers
    if (entityArrays.teleportCooldown === undefined) {
      entityArrays.teleportCooldown = new Float32Array(entityArrays.x.length).fill(0);
    }
    
    // Decrease cooldown
    entityArrays.teleportCooldown[index] -= dt;
    
    // Check if can teleport
    if (entityArrays.teleportCooldown[index] <= 0) {
      const range = params?.range || 5;
      const cooldown = params?.cooldown || 8.0;
      const triggerDistance = params?.triggerDistance || 3;
      
      // Calculate distance to target
      const dx = target.x - entityArrays.x[index];
      const dy = target.y - entityArrays.y[index];
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // Only teleport if target is close
      if (dist < triggerDistance) {
        // Reset cooldown
        entityArrays.teleportCooldown[index] = cooldown;
        
        // Get random offset
        const angle = Math.random() * Math.PI * 2;
        const distance = range * Math.random() + range / 2;
        
        // Apply teleport
        entityArrays.x[index] += Math.cos(angle) * distance;
        entityArrays.y[index] += Math.sin(angle) * distance;
        
        // Optionally add a teleport effect
        if (params?.onTeleport) {
          params.onTeleport(entityArrays, index);
        }
      }
    }
  },
  
  /**
   * Gravitational pull behavior - pulls nearby entities toward this entity
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Target entity
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters { radius, force, entityManager, duration, cooldown }
   */
  gravityWell: function(entityArrays, index, target, dt, params) {
    // Ensure we have cooldown and active timers
    if (entityArrays.gravityCooldown === undefined) {
      entityArrays.gravityCooldown = new Float32Array(entityArrays.x.length).fill(0);
      entityArrays.gravityActive = new Uint8Array(entityArrays.x.length).fill(0);
      entityArrays.gravityTimer = new Float32Array(entityArrays.x.length).fill(0);
    }
    
    // Check if gravity well is active
    if (entityArrays.gravityActive[index]) {
      entityArrays.gravityTimer[index] -= dt;
      
      if (entityArrays.gravityTimer[index] <= 0) {
        // Deactivate
        entityArrays.gravityActive[index] = 0;
        return;
      }
      
      // Apply gravity to nearby entities
      const radius = params?.radius || 4;
      const force = params?.force || 2;
      const entityManager = params?.entityManager;
      
      if (entityManager) {
        // Loop through all other entities and pull them if they're in range
        for (let i = 0; i < entityManager.enemyCount; i++) {
          if (i !== index) {
            const dx = entityArrays.x[index] - entityManager.x[i];
            const dy = entityArrays.y[index] - entityManager.y[i];
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < radius && dist > 0.1) {
              // Force is stronger the closer you are
              const pullStrength = force * (1 - dist / radius);
              const angle = Math.atan2(dy, dx);
              
              // Apply pull
              entityManager.x[i] += Math.cos(angle) * pullStrength * dt;
              entityManager.y[i] += Math.sin(angle) * pullStrength * dt;
            }
          }
        }
      }
      
      return;
    }
    
    // Decrease cooldown if not active
    entityArrays.gravityCooldown[index] -= dt;
    
    // Check if can activate gravity well
    if (entityArrays.gravityCooldown[index] <= 0) {
      const cooldown = params?.cooldown || 10.0;
      const duration = params?.duration || 2.0;
      
      // Activate gravity well
      entityArrays.gravityActive[index] = 1;
      entityArrays.gravityTimer[index] = duration;
      entityArrays.gravityCooldown[index] = cooldown;
    }
  },
  
  /**
   * Time distortion behavior - slows nearby entities
   * @param {object} entityArrays - Entity data arrays
   * @param {number} index - Entity index
   * @param {object} target - Target entity
   * @param {number} dt - Delta time in seconds
   * @param {object} params - Additional parameters
   */
  timeDistort: function(entityArrays, index, target, dt, params) {
    // Ensure we have cooldown timers
    if (entityArrays.distortCooldown === undefined) {
      entityArrays.distortCooldown = new Float32Array(entityArrays.x.length).fill(0);
    }
    
    // Decrease cooldown
    entityArrays.distortCooldown[index] -= dt;
    
    // Check if can activate time distortion
    if (entityArrays.distortCooldown[index] <= 0) {
      const radius = params?.radius || 3;
      const slowFactor = params?.slowFactor || 0.5;
      const duration = params?.duration || 2.0;
      const cooldown = params?.cooldown || 8.0;
      const hasteSelf = params?.hasteSelf || false;
      const hasteFactor = params?.hasteFactor || 1.5;
      const entityManager = params?.entityManager;
      
      // Reset cooldown
      entityArrays.distortCooldown[index] = cooldown;
      
      if (entityManager) {
        // Find entities in range
        for (let i = 0; i < entityManager.enemyCount; i++) {
          if (i !== index) {
            const dx = entityArrays.x[index] - entityManager.x[i];
            const dy = entityArrays.y[index] - entityManager.y[i];
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < radius) {
              // Apply slow effect
              // This requires a speed multiplier array in your entity system
              if (entityManager.speedMultiplier === undefined) {
                entityManager.speedMultiplier = new Float32Array(entityManager.x.length).fill(1.0);
                entityManager.speedMultiplierTimer = new Float32Array(entityManager.x.length).fill(0);
              }
              
              entityManager.speedMultiplier[i] = slowFactor;
              entityManager.speedMultiplierTimer[i] = duration;
            }
          }
        }
        
        // Haste self if needed
        if (hasteSelf) {
          if (entityArrays.speedMultiplier === undefined) {
            entityArrays.speedMultiplier = new Float32Array(entityArrays.x.length).fill(1.0);
            entityArrays.speedMultiplierTimer = new Float32Array(entityArrays.x.length).fill(0);
          }
          
          entityArrays.speedMultiplier[index] = hasteFactor;
          entityArrays.speedMultiplierTimer[index] = duration;
        }
      }
    }
  }
};

/**
 * Updates speed multipliers across all entities
 * Call this in your main update loop
 * @param {Object} entityArrays - Entity arrays including speedMultiplier and speedMultiplierTimer
 * @param {number} dt - Delta time in seconds
 */
function updateSpeedMultipliers(entityArrays, dt) {
  if (entityArrays.speedMultiplier && entityArrays.speedMultiplierTimer) {
    for (let i = 0; i < entityArrays.speedMultiplierTimer.length; i++) {
      if (entityArrays.speedMultiplierTimer[i] > 0) {
        entityArrays.speedMultiplierTimer[i] -= dt;
        
        if (entityArrays.speedMultiplierTimer[i] <= 0) {
          // Reset multiplier when timer expires
          entityArrays.speedMultiplier[i] = 1.0;
        }
      }
    }
  }
}

// Export behaviors
module.exports = {
  behaviors,
  updateSpeedMultipliers
};

// ES Module export for client
if (typeof window !== 'undefined') {
  window.ECSBehaviors = { behaviors, updateSpeedMultipliers };
}

export { behaviors, updateSpeedMultipliers };
