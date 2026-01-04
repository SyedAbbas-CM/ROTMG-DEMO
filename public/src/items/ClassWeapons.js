/**
 * ClassWeapons.js - Client-side class weapon definitions
 * Each class has a default weapon with unique attack patterns
 */

// Projectile patterns define how bullets are created
export const ProjectilePatterns = {
  SINGLE: 'single',
  DOUBLE: 'double',
  TRIPLE: 'triple',
  SPREAD: 'spread',
  BURST: 'burst',
  WAVE: 'wave',
  SPIRAL: 'spiral'
};

// Class weapon definitions with attack characteristics
export const ClassWeapons = {
  warrior: {
    name: 'Sword of the Warrior',
    damage: 15,
    range: 3.5,           // Short range melee
    projectileSpeed: 12,
    rateOfFire: 1.8,      // Attacks per second
    pattern: ProjectilePatterns.SINGLE,
    projectileSize: 0.8,  // Bigger sword slash
    projectileColor: '#ffcc00',
    projectileLifetime: 0.3,
    spriteRow: 0          // Use row 0 of bullet sprites
  },

  archer: {
    name: 'Longbow',
    damage: 10,
    range: 9,             // Long range
    projectileSpeed: 14,
    rateOfFire: 1.5,
    pattern: ProjectilePatterns.SINGLE,
    projectileSize: 0.5,
    projectileColor: '#8b4513',
    projectileLifetime: 0.65,
    spriteRow: 1
  },

  mage: {
    name: 'Staff of Fire',
    damage: 18,
    range: 7,
    projectileSpeed: 10,
    rateOfFire: 1.2,
    pattern: ProjectilePatterns.DOUBLE,  // Two parallel shots
    patternConfig: {
      count: 2,
      offsetPerpendicular: 0.3
    },
    projectileSize: 0.6,
    projectileColor: '#ff4400',
    projectileLifetime: 0.7,
    spriteRow: 2
  },

  wizard: {
    name: 'Crystal Wand',
    damage: 22,
    range: 8,
    projectileSpeed: 11,
    rateOfFire: 1.0,
    pattern: ProjectilePatterns.TRIPLE,  // Three-way spread
    patternConfig: {
      count: 3,
      spreadAngle: Math.PI / 10  // 18 degree spread
    },
    projectileSize: 0.5,
    projectileColor: '#00ccff',
    projectileLifetime: 0.75,
    spriteRow: 2  // Same sprite as mage
  },

  rogue: {
    name: 'Shadow Dagger',
    damage: 20,
    range: 4,             // Short-medium range
    projectileSpeed: 16,  // Fast projectiles
    rateOfFire: 2.5,      // Very fast attack speed
    pattern: ProjectilePatterns.SINGLE,
    projectileSize: 0.4,  // Small daggers
    projectileColor: '#9932cc',
    projectileLifetime: 0.25,
    spriteRow: 3
  },

  knight: {
    name: 'Broadsword',
    damage: 12,
    range: 3,             // Very short range
    projectileSpeed: 10,
    rateOfFire: 1.5,
    pattern: ProjectilePatterns.SPREAD,  // Wide arc slash
    patternConfig: {
      count: 3,
      spreadAngle: Math.PI / 4  // 45 degree spread
    },
    projectileSize: 0.7,
    projectileColor: '#c0c0c0',
    projectileLifetime: 0.3,
    spriteRow: 4
  },

  necromancer: {
    name: 'Staff of Decay',
    damage: 14,
    range: 6.5,
    projectileSpeed: 8,   // Slower projectiles
    rateOfFire: 1.3,
    pattern: ProjectilePatterns.WAVE,
    patternConfig: {
      waveAmplitude: 0.8,
      waveFrequency: 6
    },
    projectileSize: 0.6,
    projectileColor: '#00ff00',
    projectileLifetime: 0.8,
    spriteRow: 5
  },

  priest: {
    name: 'Holy Wand',
    damage: 8,
    range: 7,
    projectileSpeed: 12,
    rateOfFire: 1.4,
    pattern: ProjectilePatterns.SINGLE,
    projectileSize: 0.5,
    projectileColor: '#ffffff',
    projectileLifetime: 0.6,
    spriteRow: 5  // Shares with necro for now
  }
};

/**
 * Get weapon for a class
 * @param {string} className - Class name (warrior, archer, etc.)
 * @returns {Object} Weapon definition
 */
export function getClassWeapon(className) {
  const normalized = className?.toLowerCase() || 'warrior';
  return ClassWeapons[normalized] || ClassWeapons.warrior;
}

/**
 * Calculate projectiles for a weapon shot
 * @param {Object} weapon - Weapon definition
 * @param {number} x - Player X position
 * @param {number} y - Player Y position
 * @param {number} angle - Aim angle in radians
 * @returns {Array} Array of bullet objects to create
 */
export function calculateProjectiles(weapon, x, y, angle) {
  const bullets = [];
  const config = weapon.patternConfig || {};
  const spawnOffset = 0.4;

  switch (weapon.pattern) {
    case ProjectilePatterns.SINGLE:
      bullets.push(createBullet(weapon, x, y, angle, spawnOffset));
      break;

    case ProjectilePatterns.DOUBLE: {
      const perpAngle = angle + Math.PI / 2;
      const offset = config.offsetPerpendicular || 0.3;

      // Two parallel bullets
      bullets.push(createBullet(weapon,
        x + Math.cos(perpAngle) * offset,
        y + Math.sin(perpAngle) * offset,
        angle, spawnOffset));
      bullets.push(createBullet(weapon,
        x - Math.cos(perpAngle) * offset,
        y - Math.sin(perpAngle) * offset,
        angle, spawnOffset));
      break;
    }

    case ProjectilePatterns.TRIPLE:
    case ProjectilePatterns.SPREAD: {
      const count = config.count || 3;
      const spreadAngle = config.spreadAngle || (Math.PI / 8);
      const startAngle = angle - spreadAngle / 2;
      const angleStep = count > 1 ? spreadAngle / (count - 1) : 0;

      for (let i = 0; i < count; i++) {
        const bulletAngle = count === 1 ? angle : startAngle + angleStep * i;
        bullets.push(createBullet(weapon, x, y, bulletAngle, spawnOffset));
      }
      break;
    }

    case ProjectilePatterns.BURST: {
      const count = config.count || 3;
      const delay = config.burstDelay || 100; // ms between shots

      for (let i = 0; i < count; i++) {
        const bullet = createBullet(weapon, x, y, angle, spawnOffset);
        bullet.delay = i * delay;
        bullets.push(bullet);
      }
      break;
    }

    case ProjectilePatterns.WAVE: {
      const bullet = createBullet(weapon, x, y, angle, spawnOffset);
      bullet.waveAmp = config.waveAmplitude || 0.5;
      bullet.waveFreq = config.waveFrequency || 4;
      bullets.push(bullet);
      break;
    }

    case ProjectilePatterns.SPIRAL: {
      const bullet = createBullet(weapon, x, y, angle, spawnOffset);
      bullet.angularVel = config.angularVelocity || 2;
      bullets.push(bullet);
      break;
    }

    default:
      bullets.push(createBullet(weapon, x, y, angle, spawnOffset));
  }

  return bullets;
}

/**
 * Create a single bullet object
 */
function createBullet(weapon, x, y, angle, spawnOffset) {
  const bulletX = x + Math.cos(angle) * spawnOffset;
  const bulletY = y + Math.sin(angle) * spawnOffset;

  return {
    x: bulletX,
    y: bulletY,
    vx: Math.cos(angle) * weapon.projectileSpeed,
    vy: Math.sin(angle) * weapon.projectileSpeed,
    angle: angle,
    speed: weapon.projectileSpeed,
    damage: weapon.damage,
    lifetime: weapon.projectileLifetime || (weapon.range / weapon.projectileSpeed),
    width: weapon.projectileSize || 0.6,
    height: weapon.projectileSize || 0.6,
    color: weapon.projectileColor || '#ffffff',
    spriteRow: weapon.spriteRow || 0,
    // Motion modifiers (set by pattern)
    waveAmp: 0,
    waveFreq: 0,
    angularVel: 0,
    delay: 0
  };
}

/**
 * Check if enough time has passed since last shot (rate limiting)
 * @param {number} lastShotTime - Timestamp of last shot
 * @param {Object} weapon - Weapon definition
 * @returns {boolean} True if can shoot
 */
export function canShoot(lastShotTime, weapon) {
  const now = Date.now();
  const fireInterval = 1000 / weapon.rateOfFire;
  return (now - lastShotTime) >= fireInterval;
}
