/**
 * ItemManager.js
 * Optimized item management system with binary serialization
 */

// Item type constants for efficient type checking
const ItemType = {
    WEAPON: 1,
    ARMOR: 2,
    CONSUMABLE: 3,
    MATERIAL: 4,
    QUEST: 5
};

// Rarity constants
const ItemRarity = {
    COMMON: 1,
    UNCOMMON: 2,
    RARE: 3,
    EPIC: 4,
    LEGENDARY: 5
};

/**
 * BinaryItemSerializer - Optimized binary serialization for items
 */
class BinaryItemSerializer {
    static encode(item) {
        const buffer = new ArrayBuffer(40); // Increased size for sprite data
        const view = new DataView(buffer);
        
        // Encode item properties
        view.setUint16(0, item.id, true);
        view.setUint8(2, item.type);
        view.setUint8(3, item.rarity);
        view.setFloat32(4, item.x, true);
        view.setFloat32(8, item.y, true);
        view.setUint32(12, item.ownerId || 0, true);
        view.setUint32(16, item.stackSize || 1, true);
        view.setUint32(20, item.durability || 0, true);
        
        // Encode sprite data
        view.setUint8(24, item.spriteSheet.charCodeAt(0));
        view.setUint8(25, item.spriteSheet.charCodeAt(1));
        view.setUint8(26, item.spriteSheet.charCodeAt(2));
        view.setUint16(27, item.spriteX, true);
        view.setUint16(29, item.spriteY, true);
        view.setUint16(31, item.spriteWidth, true);
        view.setUint16(33, item.spriteHeight, true);
        
        // Encode stats (up to 3 stats)
        const stats = item.stats || {};
        view.setUint8(35, Object.keys(stats).length);
        let offset = 36;
        for (const [stat, value] of Object.entries(stats)) {
            if (offset >= 40) break;
            view.setUint8(offset++, stat.charCodeAt(0));
            view.setUint16(offset, value, true);
            offset += 2;
        }
        
        return buffer;
    }
    
    static decode(buffer) {
        const view = new DataView(buffer);
        const item = {
            id: view.getUint16(0, true),
            type: view.getUint8(2),
            rarity: view.getUint8(3),
            x: view.getFloat32(4, true),
            y: view.getFloat32(8, true),
            ownerId: view.getUint32(12, true) || null,
            stackSize: view.getUint32(16, true),
            durability: view.getUint32(20, true),
            spriteSheet: String.fromCharCode(
                view.getUint8(24),
                view.getUint8(25),
                view.getUint8(26)
            ),
            spriteX: view.getUint16(27, true),
            spriteY: view.getUint16(29, true),
            spriteWidth: view.getUint16(31, true),
            spriteHeight: view.getUint16(33, true),
            stats: {}
        };
        
        // Decode stats
        const statCount = view.getUint8(35);
        let offset = 36;
        for (let i = 0; i < statCount && offset < 40; i++) {
            const stat = String.fromCharCode(view.getUint8(offset++));
            const value = view.getUint16(offset, true);
            offset += 2;
            item.stats[stat] = value;
        }
        
        return item;
    }
}

/**
 * ItemManager - Manages all items in the game
 */
class ItemManager {
    constructor() {
        this.items = new Map(); // Using Map for O(1) lookups
        this.nextItemId = 1;
        this.itemDefinitions = new Map();
        this.spawnedItems = new Set();
    }
    
    /**
     * Register an item definition
     * @param {Object} definition - Item definition
     */
    registerItemDefinition(definition) {
        // Validate sprite data - check for existence, not truthiness (spriteX/Y can be 0)
        if (!definition.spriteSheet || definition.spriteX === undefined || definition.spriteY === undefined) {
            console.error('Item definition missing required sprite data:', definition);
            return;
        }
        
        this.itemDefinitions.set(definition.id, definition);
    }
    
    /**
     * Create a new item instance
     * @param {number} definitionId - Item definition ID
     * @param {Object} options - Additional options
     * @returns {Object} Created item
     */
    createItem(definitionId, options = {}) {
        const definition = this.itemDefinitions.get(definitionId);
        if (!definition) return null;
        
        const item = {
            id: this.nextItemId++,
            definitionId,
            type: definition.type,
            rarity: options.rarity || definition.rarity,
            x: options.x || 0,
            y: options.y || 0,
            ownerId: options.ownerId || null,
            stackSize: options.stackSize || 1,
            maxStackSize: definition.maxStackSize || 1,
            durability: options.durability || definition.maxDurability,
            spriteSheet: definition.spriteSheet,
            spriteX: definition.spriteX,
            spriteY: definition.spriteY,
            spriteWidth: definition.spriteWidth || 32,
            spriteHeight: definition.spriteHeight || 32,
            stats: { ...definition.baseStats }
        };
        
        // Apply random stats if defined
        if (definition.randomStats) {
            this._applyRandomStats(item, definition);
        }
        
        this.items.set(item.id, item);
        return item;
    }
    
    /**
     * Spawn an item in the world
     * @param {number} definitionId - Item definition ID
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {Object} Spawned item
     */
    spawnItem(definitionId, x, y) {
        const item = this.createItem(definitionId, { x, y });
        if (item) {
            this.spawnedItems.add(item.id);
        }
        return item;
    }
    
    /**
     * Remove an item from the world
     * @param {number} itemId - Item ID
     */
    removeItem(itemId) {
        this.items.delete(itemId);
        this.spawnedItems.delete(itemId);
    }
    
    /**
     * Get all spawned items in a region
     * @param {number} x - Center X
     * @param {number} y - Center Y
     * @param {number} radius - Search radius
     * @returns {Array} Items in range
     */
    getItemsInRange(x, y, radius) {
        const items = [];
        for (const itemId of this.spawnedItems) {
            const item = this.items.get(itemId);
            if (!item) continue;
            
            const dx = item.x - x;
            const dy = item.y - y;
            if (dx * dx + dy * dy <= radius * radius) {
                items.push(item);
            }
        }
        return items;
    }
    
    /**
     * Apply random stats to an item
     * @private
     */
    _applyRandomStats(item, definition) {
        const { randomStats } = definition;
        for (const [stat, range] of Object.entries(randomStats)) {
            const [min, max] = range;
            const value = Math.floor(Math.random() * (max - min + 1)) + min;
            item.stats[stat] = (item.stats[stat] || 0) + value;
        }
    }
    
    /**
     * Get binary data for network transmission
     * @returns {ArrayBuffer} Binary data
     */
    getBinaryData() {
        const items = Array.from(this.spawnedItems)
            .map(id => this.items.get(id))
            .filter(item => item);
            
        const buffer = new ArrayBuffer(4 + items.length * 40);
        const view = new DataView(buffer);
        view.setUint32(0, items.length, true);
        
        let offset = 4;
        for (const item of items) {
            const itemBuffer = BinaryItemSerializer.encode(item);
            new Uint8Array(buffer, offset, 40).set(new Uint8Array(itemBuffer));
            offset += 40;
        }
        
        return buffer;
    }
}

export { ItemManager, ItemType, ItemRarity, BinaryItemSerializer }; 