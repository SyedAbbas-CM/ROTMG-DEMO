/**
 * Test database for items and enemies
 */

// Item Types (copied from ItemManager for convenience)
const ItemType = {
    WEAPON: 1,
    ARMOR: 2,
    CONSUMABLE: 3,
    MATERIAL: 4,
    QUEST: 5,
    DEFAULT: 0 // Added default type
};

// Item Rarities (copied from ItemManager for convenience)
const ItemRarity = {
    COMMON: 1,
    UNCOMMON: 2,
    RARE: 3,
    EPIC: 4,
    LEGENDARY: 5
};

// Item definitions
export const ITEM_DEFINITIONS = {
    // Default Item (Placeholder)
    DEFAULT: {
        id: 0,
        name: "Default Item",
        type: ItemType.DEFAULT,
        rarity: ItemRarity.COMMON,
        spriteSheet: "items", // Assuming a default/error sprite exists
        spriteX: 0,
        spriteY: 0, 
        spriteWidth: 8, // Small default
        spriteHeight: 8,
        maxStackSize: 1,
        description: "A default item placeholder."
    },
    // Weapons
    SWORD: {
        id: 1,
        name: "Sword",
        type: ItemType.WEAPON,
        rarity: ItemRarity.COMMON,
        spriteSheet: "items",
        spriteX: 0,
        spriteY: 0,
        spriteWidth: 32,
        spriteHeight: 32,
        maxStackSize: 1,
        maxDurability: 100,
        baseStats: {
            damage: 10,
            speed: 1.0
        },
        randomStats: {
            damage: [0, 5],
            speed: [-0.1, 0.1]
        },
        description: "A basic sword."
    },
    
    BOW: {
        id: 2,
        name: "Bow",
        type: ItemType.WEAPON,
        rarity: ItemRarity.UNCOMMON,
        spriteSheet: "items",
        spriteX: 32,
        spriteY: 0,
        spriteWidth: 32,
        spriteHeight: 32,
        maxStackSize: 1,
        maxDurability: 100,
        baseStats: {
            damage: 8,
            speed: 1.2,
            range: 5
        },
        randomStats: {
            damage: [0, 4],
            speed: [-0.1, 0.1],
            range: [0, 2]
        },
        description: "A standard bow."
    },
    
    // Armor
    LEATHER_ARMOR: {
        id: 3,
        name: "Leather Armor",
        type: ItemType.ARMOR,
        rarity: ItemRarity.COMMON,
        spriteSheet: "items",
        spriteX: 0,
        spriteY: 32,
        spriteWidth: 32,
        spriteHeight: 32,
        maxStackSize: 1,
        maxDurability: 100,
        baseStats: {
            defense: 5,
            speed: -0.1
        },
        randomStats: {
            defense: [0, 3],
            speed: [-0.05, 0.05]
        },
        description: "Simple leather armor."
    },
    
    // Consumables
    HEALTH_POTION: {
        id: 4,
        name: "Health Potion",
        type: ItemType.CONSUMABLE,
        rarity: ItemRarity.COMMON,
        spriteSheet: "items",
        spriteX: 32,
        spriteY: 32,
        spriteWidth: 32,
        spriteHeight: 32,
        maxStackSize: 10,
        baseStats: {
            heal: 50
        },
        description: "Restores a small amount of health."
    },
    
    MANA_POTION: {
        id: 5,
        name: "Mana Potion",
        type: ItemType.CONSUMABLE,
        rarity: ItemRarity.COMMON,
        spriteSheet: "items",
        spriteX: 64,
        spriteY: 32,
        spriteWidth: 32,
        spriteHeight: 32,
        maxStackSize: 10,
        baseStats: {
            mana: 50
        },
        description: "Restores a small amount of mana."
    }
};

// Enemy definitions
export const ENEMY_DEFINITIONS = {
    SLIME: {
        id: 1,
        name: "Slime",
        spriteSheet: "enemies",
        spriteX: 0,
        spriteY: 0,
        spriteWidth: 32,
        spriteHeight: 32,
        health: 50,
        speed: 1.0,
        damage: 5,
        exp: 10,
        dropTable: [
            { itemId: 4, chance: 0.3 }, // Health Potion
            { itemId: 5, chance: 0.2 }  // Mana Potion
        ]
    },
    
    SKELETON: {
        id: 2,
        name: "Skeleton",
        spriteSheet: "enemies",
        spriteX: 32,
        spriteY: 0,
        spriteWidth: 32,
        spriteHeight: 32,
        health: 100,
        speed: 1.2,
        damage: 8,
        exp: 20,
        dropTable: [
            { itemId: 1, chance: 0.1 }, // Sword
            { itemId: 3, chance: 0.2 }, // Leather Armor
            { itemId: 4, chance: 0.2 }  // Health Potion
        ]
    },
    
    ZOMBIE: {
        id: 3,
        name: "Zombie",
        spriteSheet: "enemies",
        spriteX: 64,
        spriteY: 0,
        spriteWidth: 32,
        spriteHeight: 32,
        health: 150,
        speed: 0.8,
        damage: 12,
        exp: 30,
        dropTable: [
            { itemId: 2, chance: 0.05 }, // Bow
            { itemId: 3, chance: 0.3 },  // Leather Armor
            { itemId: 4, chance: 0.3 },  // Health Potion
            { itemId: 5, chance: 0.2 }   // Mana Potion
        ]
    }
};

// Map Object definitions
export const MAP_OBJECT_DEFINITIONS = {
    CHEST: {
        id: 1,
        name: "Chest",
        type: "container",
        spriteSheet: "objects", // Needs an objects.png spritesheet
        spriteX: 0,
        spriteY: 0,
        spriteWidth: 32,
        spriteHeight: 32,
        inventorySize: 20,
        interactionType: "open_inventory"
    }
}; 