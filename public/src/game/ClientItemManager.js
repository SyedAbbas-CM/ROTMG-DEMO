/**
 * ClientItemManager.js
 * Client-side item management and rendering
 */

import { SpriteManager } from '../sprite/SpriteManager.js';
import { ItemType, ItemRarity } from '../../src/ItemManager.js';

class ClientItemManager {
    constructor() {
        this.items = new Map();
        this.itemSprites = new Map();
        this.itemDefinitions = new Map();
        this.pickupRadius = 1.5; // Tile radius for item pickup
        this.spriteManager = null;
    }
    
    /**
     * Initialize the item manager
     * @param {Object} spriteManager - Sprite manager instance
     */
    init(spriteManager) {
        this.spriteManager = spriteManager;
        this._loadItemSprites();
    }
    
    /**
     * Load item sprites
     * @private
     */
    _loadItemSprites() {
        /*
         * Updated item sprite loading – two different sheets were provided:
         *  1) stacked_spritesheet.png  – appears to be a square atlas of small 8×8 icons
         *  2) merged_spritesheet_vertical.png – a column-oriented sheet (height >> width)
         *
         * We use autoDetect so irregular icon sizes are still captured, falling back
         * to an 8×8 grid for stacked and a 10×10 guess for merged if auto-detection
         * yields no regions.
         */

        // Stacked grid – 8×8 tiles by default
        this.spriteManager.loadSpriteSheet({
            name: 'items_stacked',
            path: 'assets/images/items/stacked_spritesheet.png',
            defaultSpriteWidth: 8,
            defaultSpriteHeight: 8,
            autoDetect: true
        });

        // Vertical merged – guess 10×10 tiles (adjust at runtime if needed)
        this.spriteManager.loadSpriteSheet({
            name: 'items_merged',
            path: 'assets/images/items/merged_spritesheet_vertical.png',
            defaultSpriteWidth: 10,
            defaultSpriteHeight: 10,
            autoDetect: true
        });
        
        // Rarity overlay sheet kept (path unchanged)
        this.spriteManager.loadSpriteSheet({
            name: 'rarity',
            path: 'assets/sprites/rarity.png',
            defaultSpriteWidth: 32,
            defaultSpriteHeight: 32,
            spritesPerRow: 5,
            spritesPerColumn: 1,
            autoDetect: false
        });
    }
    
    /**
     * Register an item definition
     * @param {Object} definition - Item definition
     */
    registerItemDefinition(definition) {
        this.itemDefinitions.set(definition.id, definition);
    }
    
    /**
     * Add an item to the world
     * @param {Object} item - Item data
     */
    addItem(item) {
        this.items.set(item.id, item);
        this._createItemSprite(item);
    }
    
    /**
     * Remove an item from the world
     * @param {number} itemId - Item ID
     */
    removeItem(itemId) {
        const sprite = this.itemSprites.get(itemId);
        if (sprite) {
            sprite.destroy();
            this.itemSprites.delete(itemId);
        }
        this.items.delete(itemId);
    }
    
    /**
     * Update item positions and states
     * @param {number} deltaTime - Time since last update
     */
    update(deltaTime) {
        for (const [id, item] of this.items) {
            const sprite = this.itemSprites.get(id);
            if (sprite) {
                // Update sprite position
                sprite.x = item.x * 32; // Convert tile to pixel coordinates
                sprite.y = item.y * 32;
                
                // Add floating animation
                sprite.y += Math.sin(Date.now() * 0.002) * 2;
            }
        }
    }
    
    /**
     * Create a sprite for an item
     * @private
     */
    _createItemSprite(item) {
        const definition = this.itemDefinitions.get(item.definitionId);
        if (!definition) return;
        
        // Create base sprite using SpriteManager
        const sprite = this.spriteManager.createSprite(
            definition.spriteSheet,
            definition.spriteX,
            definition.spriteY,
            definition.spriteWidth,
            definition.spriteHeight
        );
        
        // Set initial position
        sprite.setPosition(item.x * 32, item.y * 32);
        
        // Add rarity overlay
        const raritySprite = this.spriteManager.createSprite(
            'rarity',
            (item.rarity - 1) * 32, // Assuming rarity sprites are in a row
            0,
            32,
            32
        );
        raritySprite.setPosition(item.x * 32, item.y * 32);
        
        // Store sprites
        this.itemSprites.set(item.id, {
            base: sprite,
            rarity: raritySprite
        });
    }
    
    /**
     * Get items in pickup range of a position
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @returns {Array} Items in range
     */
    getItemsInRange(x, y) {
        const items = [];
        for (const [id, item] of this.items) {
            const dx = item.x - x;
            const dy = item.y - y;
            if (dx * dx + dy * dy <= this.pickupRadius * this.pickupRadius) {
                items.push(item);
            }
        }
        return items;
    }
    
    /**
     * Handle item pickup
     * @param {number} itemId - Item ID
     */
    pickupItem(itemId) {
        const item = this.items.get(itemId);
        if (!item) return;
        
        // Remove item from world
        this.removeItem(itemId);
        
        // Play pickup animation
        this._playPickupAnimation(item);
    }
    
    /**
     * Play pickup animation
     * @private
     */
    _playPickupAnimation(item) {
        const sprites = this.itemSprites.get(item.id);
        if (!sprites) return;
        
        // Fade out and scale up
        const duration = 500; // ms
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const alpha = 1 - progress;
            const scale = 1 + progress;
            
            sprites.base.alpha = alpha;
            sprites.rarity.alpha = alpha;
            sprites.base.scale = scale;
            sprites.rarity.scale = scale;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                sprites.base.destroy();
                sprites.rarity.destroy();
            }
        };
        
        animate();
    }
}

export { ClientItemManager }; 