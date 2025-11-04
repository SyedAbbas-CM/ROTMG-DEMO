/**
 * Projectile/Attack Definition System
 *
 * Modular building blocks for all projectiles in the game
 * Used by: Players, Enemies, Bosses, LLM-generated attacks
 *
 * This is the SINGLE SOURCE OF TRUTH for all attack patterns
 */

/**
 * Projectile Types - Base templates
 */
export const ProjectileTypes = {

  // ========================================
  // BASIC PROJECTILES
  // ========================================

  basic_shot: {
    id: 'basic_shot',
    name: 'Basic Shot',
    description: 'Simple straight projectile',
    sprite: 'projectile_basic',
    speed: 20,              // tiles/second
    lifetime: 3.0,          // seconds
    damage: 10,
    size: 0.4,              // tiles (width/height)
    piercing: false,        // Does it go through enemies?
    bounces: 0,            // Number of bounces off walls
    homing: false,         // Does it track enemies?
    effects: [],           // Status effects (poison, slow, etc.)
    color: '#ffffff',      // Tint color
    trail: false,          // Leave particle trail?
  },

  player_bullet: {
    id: 'player_bullet',
    name: 'Player Bullet',
    description: 'Standard player projectile',
    sprite: 'player_bullet',
    speed: 25,
    lifetime: 2.5,
    damage: 15,
    size: 0.5,
    piercing: false,
    bounces: 0,
    homing: false,
    effects: [],
    color: '#00ff00',
    trail: true,
    trailColor: '#00ff0040',
  },

  // ========================================
  // ENEMY PROJECTILES
  // ========================================

  goblin_dart: {
    id: 'goblin_dart',
    name: 'Goblin Dart',
    description: 'Weak poison dart',
    sprite: 'goblin_dart',
    speed: 15,
    lifetime: 2.0,
    damage: 8,
    size: 0.3,
    piercing: false,
    bounces: 0,
    homing: false,
    effects: ['poison_weak'],
    color: '#90EE90',
    trail: false,
  },

  orc_arrow: {
    id: 'orc_arrow',
    name: 'Orc Arrow',
    description: 'Heavy piercing arrow',
    sprite: 'orc_arrow',
    speed: 22,
    lifetime: 2.5,
    damage: 12,
    size: 0.4,
    piercing: true,        // Goes through first enemy
    maxPierces: 1,
    bounces: 0,
    homing: false,
    effects: [],
    color: '#8B4513',
    trail: false,
  },

  demon_fire: {
    id: 'demon_fire',
    name: 'Demon Fireball',
    description: 'Burning projectile',
    sprite: 'red_demon_fire',
    speed: 12,
    lifetime: 2.0,
    damage: 20,
    size: 0.6,
    piercing: false,
    bounces: 0,
    homing: false,
    effects: ['burn'],
    color: '#FF4500',
    trail: true,
    trailColor: '#FF450080',
    particleEffect: 'fire',
  },

  // ========================================
  // ADVANCED PROJECTILES
  // ========================================

  homing_missile: {
    id: 'homing_missile',
    name: 'Homing Missile',
    description: 'Tracks nearest target',
    sprite: 'missile',
    speed: 18,
    lifetime: 4.0,
    damage: 25,
    size: 0.5,
    piercing: false,
    bounces: 0,
    homing: true,
    homingStrength: 3.0,   // Turn rate
    homingRange: 15,       // Max tracking range
    effects: [],
    color: '#FF6B6B',
    trail: true,
    trailColor: '#FF6B6B60',
  },

  bouncing_orb: {
    id: 'bouncing_orb',
    name: 'Bouncing Orb',
    description: 'Bounces off walls',
    sprite: 'orb',
    speed: 16,
    lifetime: 5.0,
    damage: 15,
    size: 0.5,
    piercing: false,
    bounces: 3,            // Bounce 3 times
    bounceDamping: 0.9,    // Lose 10% speed per bounce
    homing: false,
    effects: [],
    color: '#4ECDC4',
    trail: true,
    trailColor: '#4ECDC440',
  },

  explosive_shot: {
    id: 'explosive_shot',
    name: 'Explosive Shot',
    description: 'Explodes on impact',
    sprite: 'explosive',
    speed: 20,
    lifetime: 3.0,
    damage: 30,
    size: 0.6,
    piercing: false,
    bounces: 0,
    homing: false,
    effects: [],
    color: '#FFA500',
    trail: true,
    trailColor: '#FFA50060',
    onHit: {
      type: 'explosion',
      radius: 2.0,         // tiles
      damage: 15,          // AOE damage
      effect: 'explosion_particle',
    }
  },

  piercing_laser: {
    id: 'piercing_laser',
    name: 'Piercing Laser',
    description: 'Goes through everything',
    sprite: 'laser',
    speed: 35,
    lifetime: 2.0,
    damage: 8,
    size: 0.3,
    piercing: true,
    maxPierces: 999,       // Infinite piercing
    bounces: 0,
    homing: false,
    effects: [],
    color: '#00BFFF',
    trail: true,
    trailColor: '#00BFFF80',
    glow: true,
  },

  // ========================================
  // BOSS/SPECIAL PROJECTILES
  // ========================================

  spiral_bolt: {
    id: 'spiral_bolt',
    name: 'Spiral Bolt',
    description: 'Spirals towards target',
    sprite: 'spiral',
    speed: 15,
    lifetime: 4.0,
    damage: 18,
    size: 0.5,
    piercing: false,
    bounces: 0,
    homing: false,
    motion: 'spiral',
    spiralRadius: 1.5,
    spiralSpeed: 2.0,
    effects: [],
    color: '#9370DB',
    trail: true,
    trailColor: '#9370DB40',
  },

  wave_beam: {
    id: 'wave_beam',
    name: 'Wave Beam',
    description: 'Sine wave movement',
    sprite: 'wave',
    speed: 20,
    lifetime: 3.0,
    damage: 12,
    size: 0.4,
    piercing: true,
    maxPierces: 3,
    bounces: 0,
    homing: false,
    motion: 'wave',
    waveAmplitude: 1.0,
    waveFrequency: 2.0,
    effects: [],
    color: '#48D1CC',
    trail: true,
    trailColor: '#48D1CC40',
  },

};

/**
 * Attack Patterns - Pre-defined attack compositions
 * These are groups of projectiles fired together
 */
export const AttackPatterns = {

  // Single shot
  single: {
    id: 'single',
    name: 'Single Shot',
    projectiles: [
      { type: 'basic_shot', angleOffset: 0, delay: 0 }
    ]
  },

  // Double shot
  double: {
    id: 'double',
    name: 'Double Shot',
    projectiles: [
      { type: 'basic_shot', angleOffset: -0.1, delay: 0 },
      { type: 'basic_shot', angleOffset: 0.1, delay: 0 }
    ]
  },

  // Triple shot spread
  triple_spread: {
    id: 'triple_spread',
    name: 'Triple Spread',
    projectiles: [
      { type: 'basic_shot', angleOffset: -0.3, delay: 0 },
      { type: 'basic_shot', angleOffset: 0, delay: 0 },
      { type: 'basic_shot', angleOffset: 0.3, delay: 0 }
    ]
  },

  // 5-shot spread (like red demon)
  five_spread: {
    id: 'five_spread',
    name: 'Five Spread',
    projectiles: [
      { type: 'demon_fire', angleOffset: -0.4, delay: 0 },
      { type: 'demon_fire', angleOffset: -0.2, delay: 0 },
      { type: 'demon_fire', angleOffset: 0, delay: 0 },
      { type: 'demon_fire', angleOffset: 0.2, delay: 0 },
      { type: 'demon_fire', angleOffset: 0.4, delay: 0 }
    ]
  },

  // 8-shot ring
  eight_ring: {
    id: 'eight_ring',
    name: 'Eight Ring',
    projectiles: Array.from({ length: 8 }, (_, i) => ({
      type: 'basic_shot',
      angleOffset: (Math.PI * 2 / 8) * i,
      delay: 0
    }))
  },

  // Spiral burst
  spiral_burst: {
    id: 'spiral_burst',
    name: 'Spiral Burst',
    projectiles: Array.from({ length: 12 }, (_, i) => ({
      type: 'spiral_bolt',
      angleOffset: (Math.PI * 2 / 12) * i,
      delay: i * 0.05  // Slight delay per projectile
    }))
  },

  // Shotgun blast
  shotgun: {
    id: 'shotgun',
    name: 'Shotgun Blast',
    projectiles: Array.from({ length: 7 }, (_, i) => ({
      type: 'basic_shot',
      angleOffset: -0.3 + (0.6 / 6) * i,
      speedMultiplier: 0.8 + Math.random() * 0.4,  // Vary speed
      delay: 0
    }))
  },

};

/**
 * Status Effects - Buffs/Debuffs that projectiles can apply
 */
export const StatusEffects = {

  poison_weak: {
    id: 'poison_weak',
    name: 'Poison',
    duration: 3.0,
    damagePerSecond: 2,
    color: '#90EE90',
    stackable: false,
  },

  poison_strong: {
    id: 'poison_strong',
    name: 'Strong Poison',
    duration: 5.0,
    damagePerSecond: 5,
    color: '#228B22',
    stackable: false,
  },

  burn: {
    id: 'burn',
    name: 'Burning',
    duration: 4.0,
    damagePerSecond: 3,
    color: '#FF4500',
    stackable: true,
    maxStacks: 3,
  },

  slow: {
    id: 'slow',
    name: 'Slowed',
    duration: 2.0,
    speedMultiplier: 0.5,
    color: '#4682B4',
    stackable: false,
  },

  stun: {
    id: 'stun',
    name: 'Stunned',
    duration: 1.0,
    disableMovement: true,
    disableAttack: true,
    color: '#FFD700',
    stackable: false,
  },

  bleed: {
    id: 'bleed',
    name: 'Bleeding',
    duration: 6.0,
    damagePerSecond: 1,
    color: '#DC143C',
    stackable: true,
    maxStacks: 5,
  },

};

/**
 * Helper function to create a projectile instance from a template
 */
export function createProjectile(typeId, overrides = {}) {
  const template = ProjectileTypes[typeId];
  if (!template) {
    console.warn(`Unknown projectile type: ${typeId}`);
    return ProjectileTypes.basic_shot;
  }

  return {
    ...template,
    ...overrides,
    // Ensure required fields exist
    id: overrides.id || `${typeId}_${Date.now()}`,
    type: typeId,
  };
}

/**
 * Helper function to create an attack pattern
 */
export function createAttackPattern(patternId, projectileTypeOverride = null) {
  const pattern = AttackPatterns[patternId];
  if (!pattern) {
    console.warn(`Unknown attack pattern: ${patternId}`);
    return AttackPatterns.single;
  }

  return {
    ...pattern,
    projectiles: pattern.projectiles.map(p => ({
      ...p,
      type: projectileTypeOverride || p.type
    }))
  };
}

/**
 * Export everything for easy access
 */
export default {
  ProjectileTypes,
  AttackPatterns,
  StatusEffects,
  createProjectile,
  createAttackPattern,
};
