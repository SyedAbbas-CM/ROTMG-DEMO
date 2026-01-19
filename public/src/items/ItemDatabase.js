/**
 * ItemDatabase.js - Shared item definitions for client and server
 * Items can modify player stats and attack patterns
 */

// Item slots
export const ItemSlot = {
  WEAPON: 'weapon',
  ARMOR: 'armor',
  ABILITY: 'ability',
  RING: 'ring'
};

// Item tiers (affects base stats)
export const ItemTier = {
  T0: 0,  // Starting gear
  T1: 1,
  T2: 2,
  T3: 3,
  T4: 4,
  T5: 5,
  T6: 6,  // Top tier normal
  UT: 10  // Untiered (unique)
};

/**
 * Item definitions
 * Stats are ADDITIVE to base class stats
 * Pattern overrides replace the class weapon pattern
 */
export const Items = {
  // ==================== WEAPONS ====================

  // Swords (Warrior, Knight)
  sword_t0: {
    id: 'sword_t0',
    name: 'Rusty Sword',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T0,
    classes: ['warrior', 'knight'],
    stats: { damage: 5 },
    description: 'A basic sword.'
  },
  sword_t3: {
    id: 'sword_t3',
    name: 'Steel Blade',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T3,
    classes: ['warrior', 'knight'],
    stats: { damage: 15 },
    description: 'A well-forged steel blade.'
  },
  sword_t6: {
    id: 'sword_t6',
    name: 'Archon Sword',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T6,
    classes: ['warrior', 'knight'],
    stats: { damage: 30 },
    description: 'A legendary blade of immense power.'
  },
  sword_doom: {
    id: 'sword_doom',
    name: 'Doom Blade',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.UT,
    classes: ['warrior', 'knight'],
    stats: { damage: 40, speed: -1 },
    pattern: {
      type: 'spread',
      config: { count: 2, spreadAngle: 0.15 }
    },
    description: 'A cursed blade that fires two arcs of destruction.'
  },

  // Bows (Archer)
  bow_t0: {
    id: 'bow_t0',
    name: 'Short Bow',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T0,
    classes: ['archer'],
    stats: { damage: 3 },
    description: 'A simple bow.'
  },
  bow_t6: {
    id: 'bow_t6',
    name: 'Bow of Covert Havens',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T6,
    classes: ['archer'],
    stats: { damage: 18, attackSpeed: 0.2 },
    description: 'A masterwork bow.'
  },
  bow_doom: {
    id: 'bow_doom',
    name: 'Doom Bow',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.UT,
    classes: ['archer'],
    stats: { damage: 50, attackSpeed: -0.8 },
    bulletMods: { lifetimeBonus: 0.5, speedBonus: 5 },
    description: 'Fires a single devastating arrow with extreme range.'
  },

  // Staves (Mage, Necromancer)
  staff_t0: {
    id: 'staff_t0',
    name: 'Wooden Staff',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T0,
    classes: ['mage', 'necromancer'],
    stats: { damage: 4 },
    description: 'A basic magical staff.'
  },
  staff_t6: {
    id: 'staff_t6',
    name: 'Staff of the Cosmic Whole',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T6,
    classes: ['mage', 'necromancer'],
    stats: { damage: 20 },
    description: 'Channels cosmic energy.'
  },

  // Wands (Wizard, Priest)
  wand_t0: {
    id: 'wand_t0',
    name: 'Wooden Wand',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T0,
    classes: ['wizard', 'priest'],
    stats: { damage: 5 },
    description: 'A simple wand.'
  },
  wand_t6: {
    id: 'wand_t6',
    name: 'Wand of Recompense',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T6,
    classes: ['wizard', 'priest'],
    stats: { damage: 25 },
    description: 'A powerful wand of judgment.'
  },
  wand_bulwark: {
    id: 'wand_bulwark',
    name: 'Bulwark',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.UT,
    classes: ['wizard', 'priest'],
    stats: { damage: 15 },
    pattern: {
      type: 'spread',
      config: { count: 5, spreadAngle: 0.8 }
    },
    bulletMods: { piercing: 3 },
    description: 'Fires a wide spread of piercing projectiles.'
  },

  // Daggers (Rogue)
  dagger_t0: {
    id: 'dagger_t0',
    name: 'Iron Dagger',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T0,
    classes: ['rogue'],
    stats: { damage: 6, attackSpeed: 0.3 },
    description: 'A quick iron dagger.'
  },
  dagger_t6: {
    id: 'dagger_t6',
    name: 'Dagger of Foul Malevolence',
    slot: ItemSlot.WEAPON,
    tier: ItemTier.T6,
    classes: ['rogue'],
    stats: { damage: 20, attackSpeed: 0.5 },
    description: 'A wickedly sharp dagger.'
  },

  // ==================== ARMOR ====================

  robe_t0: {
    id: 'robe_t0',
    name: 'Simple Robe',
    slot: ItemSlot.ARMOR,
    tier: ItemTier.T0,
    classes: ['mage', 'wizard', 'necromancer', 'priest'],
    stats: { defense: 2, maxMana: 10 },
    description: 'A basic cloth robe.'
  },
  robe_t6: {
    id: 'robe_t6',
    name: 'Robe of the Grand Sorcerer',
    slot: ItemSlot.ARMOR,
    tier: ItemTier.T6,
    classes: ['mage', 'wizard', 'necromancer', 'priest'],
    stats: { defense: 14, maxMana: 40, damage: 3 },
    description: 'Robes worn by the greatest sorcerers.'
  },

  leather_t0: {
    id: 'leather_t0',
    name: 'Leather Armor',
    slot: ItemSlot.ARMOR,
    tier: ItemTier.T0,
    classes: ['archer', 'rogue'],
    stats: { defense: 4 },
    description: 'Basic leather protection.'
  },
  leather_t6: {
    id: 'leather_t6',
    name: 'Hydra Skin Armor',
    slot: ItemSlot.ARMOR,
    tier: ItemTier.T6,
    classes: ['archer', 'rogue'],
    stats: { defense: 18, speed: 2 },
    description: 'Made from hydra scales.'
  },

  heavy_t0: {
    id: 'heavy_t0',
    name: 'Chainmail',
    slot: ItemSlot.ARMOR,
    tier: ItemTier.T0,
    classes: ['warrior', 'knight'],
    stats: { defense: 6 },
    description: 'Basic chain armor.'
  },
  heavy_t6: {
    id: 'heavy_t6',
    name: 'Acropolis Armor',
    slot: ItemSlot.ARMOR,
    tier: ItemTier.T6,
    classes: ['warrior', 'knight'],
    stats: { defense: 24, maxHealth: 20 },
    description: 'Legendary heavy armor.'
  },

  // ==================== RINGS ====================

  ring_attack: {
    id: 'ring_attack',
    name: 'Ring of Attack',
    slot: ItemSlot.RING,
    tier: ItemTier.T3,
    classes: null,  // Any class
    stats: { damage: 4 },
    description: 'Increases attack power.'
  },
  ring_defense: {
    id: 'ring_defense',
    name: 'Ring of Defense',
    slot: ItemSlot.RING,
    tier: ItemTier.T3,
    classes: null,
    stats: { defense: 6 },
    description: 'Increases defense.'
  },
  ring_speed: {
    id: 'ring_speed',
    name: 'Ring of Speed',
    slot: ItemSlot.RING,
    tier: ItemTier.T3,
    classes: null,
    stats: { speed: 3 },
    description: 'Increases movement speed.'
  },
  ring_health: {
    id: 'ring_health',
    name: 'Ring of Health',
    slot: ItemSlot.RING,
    tier: ItemTier.T3,
    classes: null,
    stats: { maxHealth: 40 },
    description: 'Increases maximum health.'
  },
  ring_mana: {
    id: 'ring_mana',
    name: 'Ring of Magic',
    slot: ItemSlot.RING,
    tier: ItemTier.T3,
    classes: null,
    stats: { maxMana: 40 },
    description: 'Increases maximum mana.'
  },
  ring_omnipotence: {
    id: 'ring_omnipotence',
    name: 'Ring of Omnipotence',
    slot: ItemSlot.RING,
    tier: ItemTier.UT,
    classes: null,
    stats: { damage: 8, defense: 8, speed: 2, maxHealth: 60, maxMana: 60 },
    description: 'The ultimate ring of power.'
  },
  ring_piercing: {
    id: 'ring_piercing',
    name: 'Ring of Piercing',
    slot: ItemSlot.RING,
    tier: ItemTier.UT,
    classes: null,
    stats: { damage: 2 },
    bulletMods: { piercing: 2 },
    description: 'Projectiles pierce through enemies.'
  }
};

/**
 * Get item by ID
 */
export function getItem(itemId) {
  return Items[itemId] || null;
}

/**
 * Get all items for a specific slot
 */
export function getItemsBySlot(slot) {
  return Object.values(Items).filter(item => item.slot === slot);
}

/**
 * Get items usable by a specific class
 */
export function getItemsForClass(className, slot = null) {
  return Object.values(Items).filter(item => {
    if (slot && item.slot !== slot) return false;
    if (item.classes === null) return true;  // Universal item
    return item.classes.includes(className.toLowerCase());
  });
}

/**
 * Check if a class can use an item
 */
export function canClassUseItem(className, itemId) {
  const item = getItem(itemId);
  if (!item) return false;
  if (item.classes === null) return true;
  return item.classes.includes(className.toLowerCase());
}
