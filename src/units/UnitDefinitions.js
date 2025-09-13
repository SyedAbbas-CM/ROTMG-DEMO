/**
 * UnitDefinitions.js - Comprehensive unit definitions with sprites, stats, and combat data
 * This replaces the basic UnitTypes with full game-ready definitions
 */

export const UnitDefinitions = {
  // ===========================================
  // INFANTRY UNITS
  // ===========================================
  LIGHT_INFANTRY: {
    id: 0,
    name: "Light Infantry",
    displayName: "Light Infantry",
    category: "infantry",
    description: "Fast-moving foot soldiers with basic armor and weapons",
    
    // Visual Properties
    sprite: {
      sheet: "Mixed_Units",
      name: "Mixed_Units_0_0", // Top-left sprite for light infantry
      width: 80,
      height: 80,
      scale: 0.5, // Scale down for game rendering
      animationFrames: 1,
      offsetX: 0,
      offsetY: 0
    },
    
    // Physical Stats
    stats: {
      maxHealth: 85,
      armor: 3,
      speed: 65, // pixels per second
      acceleration: 200,
      mass: 75,
      size: { width: 12, height: 12 }
    },
    
    // Combat Stats
    combat: {
      attackDamage: 12,
      attackRange: 25,
      attackCooldown: 0.8,
      projectileSpeed: 0, // Melee
      accuracy: 0.85,
      criticalChance: 0.05,
      armorPenetration: 2
    },
    
    // Morale & Behavior
    morale: {
      baseValue: 65,
      stability: 50,
      discipline: 0.6,
      retreatThreshold: 20,
      rallyRange: 30
    },
    
    // Tactical Properties
    tactical: {
      role: "frontline",
      formationPreference: "line",
      chargeBonus: 0.05,
      momentum: 1.0,
      preferredTargets: ["archer", "crossbowman"],
      weakAgainst: ["heavy_cavalry"],
      strongAgainst: ["archer"]
    },
    
    // Costs & Upkeep
    cost: {
      recruitment: 50,
      upkeep: 2,
      trainingTime: 30
    }
  },

  HEAVY_INFANTRY: {
    id: 1,
    name: "Heavy Infantry",
    displayName: "Heavy Infantry",
    category: "infantry",
    description: "Well-armored warriors with shields and heavy weapons",
    
    sprite: {
      sheet: "Mixed_Units",
      name: "Mixed_Units_0_1",
      width: 80,
      height: 80,
      scale: 0.5,
      animationFrames: 1,
      offsetX: 0,
      offsetY: 0
    },
    
    stats: {
      maxHealth: 140,
      armor: 12,
      speed: 45,
      acceleration: 150,
      mass: 95,
      size: { width: 14, height: 14 }
    },
    
    combat: {
      attackDamage: 18,
      attackRange: 28,
      attackCooldown: 1.0,
      projectileSpeed: 0,
      accuracy: 0.80,
      criticalChance: 0.08,
      armorPenetration: 5
    },
    
    morale: {
      baseValue: 80,
      stability: 75,
      discipline: 0.8,
      retreatThreshold: 15,
      rallyRange: 25
    },
    
    tactical: {
      role: "tank",
      formationPreference: "wedge",
      chargeBonus: 0.08,
      momentum: 1.4,
      preferredTargets: ["light_cavalry", "archer"],
      weakAgainst: ["crossbowman"],
      strongAgainst: ["light_infantry", "light_cavalry"]
    },
    
    cost: {
      recruitment: 80,
      upkeep: 4,
      trainingTime: 45
    }
  },

  // ===========================================
  // CAVALRY UNITS
  // ===========================================
  LIGHT_CAVALRY: {
    id: 2,
    name: "Light Cavalry",
    displayName: "Light Cavalry",
    category: "cavalry",
    description: "Fast-moving mounted units excellent for flanking and pursuit",
    
    sprite: {
      sheet: "Mixed_Units",
      name: "Mixed_Units_0_2",
      width: 80,
      height: 80,
      scale: 0.6, // Slightly larger for cavalry
      animationFrames: 1,
      offsetX: 0,
      offsetY: 0
    },
    
    stats: {
      maxHealth: 95,
      armor: 6,
      speed: 95,
      acceleration: 280,
      mass: 220, // Include horse weight
      size: { width: 16, height: 16 }
    },
    
    combat: {
      attackDamage: 15,
      attackRange: 30,
      attackCooldown: 0.9,
      projectileSpeed: 0,
      accuracy: 0.75,
      criticalChance: 0.12,
      armorPenetration: 3
    },
    
    morale: {
      baseValue: 70,
      stability: 45,
      discipline: 0.5,
      retreatThreshold: 25,
      rallyRange: 40
    },
    
    tactical: {
      role: "flanker",
      formationPreference: "loose",
      chargeBonus: 0.25,
      momentum: 2.0,
      preferredTargets: ["archer", "crossbowman"],
      weakAgainst: ["heavy_infantry", "spearman"],
      strongAgainst: ["archer", "crossbowman", "light_infantry"]
    },
    
    cost: {
      recruitment: 120,
      upkeep: 8,
      trainingTime: 60
    }
  },

  HEAVY_CAVALRY: {
    id: 3,
    name: "Heavy Cavalry",
    displayName: "Heavy Cavalry", 
    category: "cavalry",
    description: "Heavily armored knights on warhorses, devastating in charges",
    
    sprite: {
      sheet: "Mixed_Units",
      name: "Mixed_Units_0_3",
      width: 80,
      height: 80,
      scale: 0.7, // Largest units
      animationFrames: 1,
      offsetX: 0,
      offsetY: 0
    },
    
    stats: {
      maxHealth: 160,
      armor: 15,
      speed: 75,
      acceleration: 220,
      mass: 350,
      size: { width: 18, height: 18 }
    },
    
    combat: {
      attackDamage: 24,
      attackRange: 32,
      attackCooldown: 1.1,
      projectileSpeed: 0,
      accuracy: 0.78,
      criticalChance: 0.15,
      armorPenetration: 8
    },
    
    morale: {
      baseValue: 85,
      stability: 65,
      discipline: 0.7,
      retreatThreshold: 18,
      rallyRange: 35
    },
    
    tactical: {
      role: "charger",
      formationPreference: "wedge",
      chargeBonus: 0.45,
      momentum: 3.0,
      preferredTargets: ["heavy_infantry", "archer"],
      weakAgainst: ["spearman", "crossbowman"],
      strongAgainst: ["light_infantry", "archer"]
    },
    
    cost: {
      recruitment: 200,
      upkeep: 15,
      trainingTime: 90
    }
  },

  // ===========================================
  // RANGED UNITS
  // ===========================================
  ARCHER: {
    id: 4,
    name: "Archer",
    displayName: "Archer",
    category: "ranged",
    description: "Skilled bowmen providing ranged support and harassment",
    
    sprite: {
      sheet: "Mixed_Units", 
      name: "Mixed_Units_0_4",
      width: 80,
      height: 80,
      scale: 0.5,
      animationFrames: 1,
      offsetX: 0,
      offsetY: 0
    },
    
    stats: {
      maxHealth: 65,
      armor: 2,
      speed: 55,
      acceleration: 190,
      mass: 70,
      size: { width: 11, height: 11 }
    },
    
    combat: {
      attackDamage: 10,
      attackRange: 180,
      attackCooldown: 1.2,
      projectileSpeed: 150,
      accuracy: 0.80,
      criticalChance: 0.10,
      armorPenetration: 1,
      projectileSprite: "arrow"
    },
    
    morale: {
      baseValue: 55,
      stability: 35,
      discipline: 0.4,
      retreatThreshold: 30,
      rallyRange: 20
    },
    
    tactical: {
      role: "ranged",
      formationPreference: "line",
      chargeBonus: 0.0,
      momentum: 0.6,
      preferredTargets: ["light_cavalry", "heavy_cavalry"],
      weakAgainst: ["light_cavalry", "heavy_cavalry"],
      strongAgainst: ["heavy_infantry", "light_infantry"]
    },
    
    cost: {
      recruitment: 75,
      upkeep: 3,
      trainingTime: 40
    }
  },

  CROSSBOWMAN: {
    id: 5,
    name: "Crossbowman",
    displayName: "Crossbowman",
    category: "ranged", 
    description: "Elite marksmen with powerful crossbows and high accuracy",
    
    sprite: {
      sheet: "Mixed_Units",
      name: "Mixed_Units_0_5",
      width: 80,
      height: 80,
      scale: 0.5,
      animationFrames: 1,
      offsetX: 0,
      offsetY: 0
    },
    
    stats: {
      maxHealth: 75,
      armor: 4,
      speed: 50,
      acceleration: 170,
      mass: 75,
      size: { width: 12, height: 12 }
    },
    
    combat: {
      attackDamage: 16,
      attackRange: 200,
      attackCooldown: 1.8,
      projectileSpeed: 180,
      accuracy: 0.90,
      criticalChance: 0.12,
      armorPenetration: 6,
      projectileSprite: "bolt"
    },
    
    morale: {
      baseValue: 65,
      stability: 45,
      discipline: 0.6,
      retreatThreshold: 28,
      rallyRange: 25
    },
    
    tactical: {
      role: "ranged",
      formationPreference: "staggered",
      chargeBonus: 0.0,
      momentum: 0.7,
      preferredTargets: ["heavy_cavalry", "heavy_infantry"],
      weakAgainst: ["light_cavalry"],
      strongAgainst: ["heavy_cavalry", "heavy_infantry"]
    },
    
    cost: {
      recruitment: 100,
      upkeep: 5,
      trainingTime: 55
    }
  }
};

// ===========================================
// UTILITY FUNCTIONS
// ===========================================

/**
 * Get unit definition by ID
 */
export function getUnitDefinition(id) {
  const definitions = Object.values(UnitDefinitions);
  return definitions.find(def => def.id === id) || definitions[0];
}

/**
 * Get all unit definitions as array
 */
export function getAllUnitDefinitions() {
  return Object.values(UnitDefinitions);
}

/**
 * Get unit definitions by category
 */
export function getUnitsByCategory(category) {
  return Object.values(UnitDefinitions).filter(def => def.category === category);
}

/**
 * Legacy compatibility - convert to old UnitTypes format
 */
export function toLegacyUnitTypes() {
  const legacy = {};
  Object.values(UnitDefinitions).forEach(def => {
    legacy[def.name.toUpperCase().replace(/ /g, '_')] = {
      width: def.stats.size.width,
      height: def.stats.size.height,
      maxHealth: def.stats.maxHealth,
      armor: def.stats.armor,
      moveSpeed: def.stats.speed,
      acceleration: def.stats.acceleration,
      attackRange: def.combat.attackRange,
      attackCooldown: def.combat.attackCooldown,
      attackDamage: def.combat.attackDamage,
      momentum: def.tactical.momentum,
      morale: def.morale.baseValue,
      projectileSpeed: def.combat.projectileSpeed || 0
    };
  });
  return legacy;
}

// Export for backwards compatibility
export const UnitTypes = toLegacyUnitTypes();
export const UnitTypeKeys = Object.keys(UnitTypes);