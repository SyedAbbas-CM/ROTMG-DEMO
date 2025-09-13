/**
 * itemConstants.js - Client-side item constants shared with server
 * These constants must match the server-side ItemManager.js
 */

// Item type constants for efficient type checking
export const ItemType = {
    WEAPON: 1,
    ARMOR: 2,
    CONSUMABLE: 3,
    MATERIAL: 4,
    QUEST: 5
};

// Rarity constants
export const ItemRarity = {
    COMMON: 1,
    UNCOMMON: 2,
    RARE: 3,
    EPIC: 4,
    LEGENDARY: 5
};