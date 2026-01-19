/**
 * PlayerClasses - Character class definitions with stats and abilities
 */

export const PlayerClasses = {
  WARRIOR: {
    id: 'warrior',
    name: 'Warrior',
    description: 'Melee fighter with high health and charge ability',
    stats: {
      health: 200,
      maxHealth: 200,
      damage: 15,
      speed: 5.5,
      defense: 10,
      mana: 100,
      maxMana: 100
    },
    ability: {
      id: 'charge',
      name: 'Warrior Charge',
      description: 'Dash forward dealing damage to enemies in path',
      cooldown: 8,
      manaCost: 30,
      effect: 'charge',
      chargeDistance: 5,
      chargeDamage: 50
    },
    sprite: { sheet: 'characters', row: 0 }
  },

  ARCHER: {
    id: 'archer',
    name: 'Archer',
    description: 'Ranged attacker with multishot ability',
    stats: {
      health: 135,
      maxHealth: 135,
      damage: 12,
      speed: 6.5,
      defense: 5,
      mana: 100,
      maxMana: 100
    },
    ability: {
      id: 'multishot',
      name: 'Multishot',
      description: 'Fire 5 arrows in a spread pattern',
      cooldown: 6,
      manaCost: 25,
      effect: 'multishot',
      arrowCount: 5,
      spreadAngle: Math.PI / 6
    },
    sprite: { sheet: 'characters', row: 1 }
  },

  MAGE: {
    id: 'mage',
    name: 'Mage',
    description: 'Spellcaster with powerful AoE attacks',
    stats: {
      health: 100,
      maxHealth: 100,
      damage: 20,
      speed: 5.0,
      defense: 3,
      mana: 150,
      maxMana: 150
    },
    ability: {
      id: 'fireball',
      name: 'Fireball',
      description: 'Launch an explosive fireball dealing AoE damage',
      cooldown: 5,
      manaCost: 40,
      effect: 'aoe_explosion',
      explosionRadius: 3,
      explosionDamage: 80
    },
    sprite: { sheet: 'characters', row: 2 }
  },

  ROGUE: {
    id: 'rogue',
    name: 'Rogue',
    description: 'Fast assassin with stealth ability',
    stats: {
      health: 115,
      maxHealth: 115,
      damage: 18,
      speed: 7.5,
      defense: 4,
      mana: 100,
      maxMana: 100
    },
    ability: {
      id: 'stealth',
      name: 'Cloak',
      description: 'Become invisible for 3 seconds',
      cooldown: 10,
      manaCost: 35,
      effect: 'stealth',
      duration: 3
    },
    sprite: { sheet: 'characters', row: 3 }
  },

  KNIGHT: {
    id: 'knight',
    name: 'Knight',
    description: 'Defensive tank with shield ability',
    stats: {
      health: 250,
      maxHealth: 250,
      damage: 12,
      speed: 4.5,
      defense: 20,
      mana: 100,
      maxMana: 100
    },
    ability: {
      id: 'shield',
      name: 'Shield Wall',
      description: 'Block all damage for 2 seconds',
      cooldown: 12,
      manaCost: 50,
      effect: 'shield',
      duration: 2
    },
    sprite: { sheet: 'characters', row: 4 }
  },

  NECROMANCER: {
    id: 'necromancer',
    name: 'Necromancer',
    description: 'Dark mage who drains life from enemies',
    stats: {
      health: 110,
      maxHealth: 110,
      damage: 14,
      speed: 5.0,
      defense: 4,
      mana: 120,
      maxMana: 120
    },
    ability: {
      id: 'lifedrain',
      name: 'Life Drain',
      description: 'Drain health from nearby enemies',
      cooldown: 7,
      manaCost: 45,
      effect: 'lifedrain',
      drainRadius: 4,
      drainAmount: 30
    },
    sprite: { sheet: 'characters', row: 5 }
  },

  WIZARD: {
    id: 'wizard',
    name: 'Wizard',
    description: 'Powerful spellcaster with spread attacks',
    stats: {
      health: 100,
      maxHealth: 100,
      damage: 22,
      speed: 5.0,
      defense: 2,
      mana: 175,
      maxMana: 175
    },
    ability: {
      id: 'spellbomb',
      name: 'Spell Bomb',
      description: 'Launch a powerful explosive spell',
      cooldown: 6,
      manaCost: 50,
      effect: 'aoe_explosion',
      explosionRadius: 4,
      explosionDamage: 100
    },
    sprite: { sheet: 'characters', row: 2 }  // shares with mage
  },

  PRIEST: {
    id: 'priest',
    name: 'Priest',
    description: 'Healer with low damage but support abilities',
    stats: {
      health: 120,
      maxHealth: 120,
      damage: 8,
      speed: 5.5,
      defense: 5,
      mana: 150,
      maxMana: 150
    },
    ability: {
      id: 'heal',
      name: 'Divine Heal',
      description: 'Heal yourself and nearby allies',
      cooldown: 5,
      manaCost: 40,
      effect: 'heal',
      healRadius: 5,
      healAmount: 50
    },
    sprite: { sheet: 'characters', row: 5 }  // shares with necromancer
  }
};

export function getClassById(classId) {
  const normalized = classId?.toUpperCase?.() || 'WARRIOR';
  return PlayerClasses[normalized] || PlayerClasses.WARRIOR;
}

export function getAllClasses() {
  return Object.values(PlayerClasses);
}

export function getClassList() {
  return Object.keys(PlayerClasses).map(k => ({
    id: PlayerClasses[k].id,
    name: PlayerClasses[k].name,
    description: PlayerClasses[k].description
  }));
}
