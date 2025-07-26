/**
 * InventoryManager.js
 * Optimized inventory management system with binary serialization
 */

import { BinaryItemSerializer } from './ItemManager.js';

/**
 * BinaryInventorySerializer - Optimized binary serialization for inventories
 */
class BinaryInventorySerializer {
    static encode(inventory) {
        const { slots, size } = inventory;
        const buffer = new ArrayBuffer(4 + size * 40); // 4 bytes for size + 40 bytes per slot (updated item size)
        const view = new DataView(buffer);
        
        view.setUint32(0, size, true);
        
        let offset = 4;
        for (let i = 0; i < size; i++) {
            const item = slots[i];
            if (item) {
                const itemBuffer = BinaryItemSerializer.encode(item);
                // Ensure the buffer sizes match
                if (itemBuffer.byteLength === 40) {
                    new Uint8Array(buffer, offset, 40).set(new Uint8Array(itemBuffer));
                } else {
                    console.error(`InventoryManager: Encoded item buffer size mismatch. Expected 40, got ${itemBuffer.byteLength}`);
                    // Handle error or padding if necessary
                }
            }
            // If slot is empty, ensure it's zeroed out (ArrayBuffer initializes to 0)
            offset += 40;
        }
        
        return buffer;
    }
    
    static decode(buffer) {
        const view = new DataView(buffer);
        const size = view.getUint32(0, true);
        const inventory = {
            size,
            slots: new Array(size).fill(null)
        };
        
        let offset = 4;
        const itemSize = 40; // Updated item size
        for (let i = 0; i < size; i++) {
            const itemBuffer = buffer.slice(offset, offset + itemSize);
            // Check if the buffer slice actually contains data (not all zeros)
            if (new Uint8Array(itemBuffer).some(byte => byte !== 0)) {
                try {
                    inventory.slots[i] = BinaryItemSerializer.decode(itemBuffer);
                } catch (e) {
                    console.error(`InventoryManager: Failed to decode item in slot ${i}:`, e);
                    inventory.slots[i] = null; // Set to null on decode error
                }
            }
            offset += itemSize;
        }
        
        return inventory;
    }
}

/**
 * InventoryManager - Manages player and object inventories
 */
class InventoryManager {
    constructor() {
        this.inventories = new Map(); // ownerId (playerId or objectId string) -> inventory
        this.defaultSize = 20; // Default inventory size
    }
    
    /**
     * Create a new inventory for an owner (player or object)
     * @param {number|string} ownerId - Player ID (number) or Object ID (string like 'chest_1')
     * @param {number} size - Inventory size
     * @returns {Object} Created inventory
     */
    createInventory(ownerId, size = this.defaultSize) {
        if (this.inventories.has(ownerId)) {
            console.warn(`Inventory already exists for owner: ${ownerId}. Returning existing one.`);
            return this.inventories.get(ownerId);
        }
        const inventory = {
            ownerId,
            size,
            slots: new Array(size).fill(null)
        };
        this.inventories.set(ownerId, inventory);
        console.log(`Created inventory for ${ownerId} with size ${size}`);
        return inventory;
    }
    
    /**
     * Remove an inventory
     * @param {number|string} ownerId - Player ID or Object ID
     */
    removeInventory(ownerId) {
        if (this.inventories.has(ownerId)) {
            this.inventories.delete(ownerId);
            console.log(`Removed inventory for ${ownerId}`);
        } else {
            console.warn(`Attempted to remove non-existent inventory for ${ownerId}`);
        }
    }
    
    /**
     * Get an owner's inventory
     * @param {number|string} ownerId - Player ID or Object ID
     * @returns {Object} Owner's inventory (creates if doesn't exist for players)
     */
    getInventory(ownerId) {
        if (!this.inventories.has(ownerId)) {
            // Only auto-create for players (assuming numeric IDs)
            if (typeof ownerId === 'number') {
                console.log(`Inventory not found for player ${ownerId}, creating default.`);
                return this.createInventory(ownerId);
            } else {
                console.error(`Inventory not found for non-player owner: ${ownerId}`);
                return null; // Or throw an error, depending on desired behavior
            }
        }
        return this.inventories.get(ownerId);
    }
    
    /**
     * Add an item to an owner's inventory
     * @param {number|string} ownerId - Player ID or Object ID
     * @param {Object} item - Item to add
     * @returns {boolean} Success
     */
    addItem(ownerId, item) {
        const inventory = this.getInventory(ownerId);
        if (!inventory) return false;
        
        // Pull missing stack metadata from the canonical item definition
        if (typeof item.maxStackSize === 'undefined') {
            const def = globalThis?.itemManager?.itemDefinitions?.get?.(item.definitionId);
            if (def) {
                item.maxStackSize = def.maxStackSize || 1;
            } else {
                // Fallback if definition missing – assume non-stackable (1)
                item.maxStackSize = 1;
                console.warn(`[InventoryManager] Definition not found for item ${item.definitionId}; assuming maxStackSize = 1`);
            }
        }

        // Try to merge with existing stacks whenever possible
        if (item.maxStackSize > 1) {
            for (let i = 0; i < inventory.size && item.stackSize > 0; i++) {
                const slot = inventory.slots[i];
                if (slot && slot.definitionId === item.definitionId && slot.stackSize < slot.maxStackSize) {
                    const space = slot.maxStackSize - slot.stackSize;
                    const toAdd = Math.min(space, item.stackSize);
                    slot.stackSize += toAdd;
                    item.stackSize -= toAdd;
                }
            }
            if (item.stackSize <= 0) return true; // fully merged
        }
        
        // Find empty slot
        for (let i = 0; i < inventory.size; i++) {
            if (!inventory.slots[i]) {
                inventory.slots[i] = item; // Place remaining item (or the whole item if not stackable/stacked)
                return true;
            }
        }
        
        console.log(`Inventory full for ${ownerId}`);
        return false; // Inventory full
    }
    
    /**
     * Remove an item from an owner's inventory
     * @param {number|string} ownerId - Player ID or Object ID
     * @param {number} slotIndex - Slot index
     * @param {number} amount - Amount to remove
     * @returns {Object|null} Removed item stack (or null if failed)
     */
    removeItem(ownerId, slotIndex, amount = 1) {
        const inventory = this.getInventory(ownerId);
        if (!inventory) return null;

        const slot = inventory.slots[slotIndex];
        
        if (!slot) return null; // Slot is empty
        if (amount <= 0) return null; // Invalid amount

        const amountToRemove = Math.min(amount, slot.stackSize);

        if (amountToRemove >= slot.stackSize) {
            // Remove the entire stack
            inventory.slots[slotIndex] = null;
            return slot; // Return the original item object
        } else {
            // Remove partial stack
            const removedStack = { ...slot }; // Clone the item
            removedStack.stackSize = amountToRemove;
            slot.stackSize -= amountToRemove; // Decrease stack size in inventory
            return removedStack;
        }
    }
    
    /**
     * Move an item between slots within the same inventory
     * @param {number|string} ownerId - Player ID or Object ID
     * @param {number} fromSlot - Source slot index
     * @param {number} toSlot - Destination slot index
     * @returns {boolean} Success
     */
    moveItem(ownerId, fromSlot, toSlot) {
        const inventory = this.getInventory(ownerId);
        if (!inventory) return false;
        
        if (fromSlot < 0 || fromSlot >= inventory.size ||
            toSlot < 0 || toSlot >= inventory.size ||
            fromSlot === toSlot) { // Cannot move to the same slot
            return false;
        }
        
        const source = inventory.slots[fromSlot];
        const target = inventory.slots[toSlot];
        
        if (!source) return false; // Cannot move an empty slot
        
        // Try to stack if same item type and target exists and has space
        if (target && target.definitionId === source.definitionId && target.stackSize < target.maxStackSize) {
            const space = target.maxStackSize - target.stackSize;
            const toMove = Math.min(space, source.stackSize);
            
            if (toMove > 0) {
                target.stackSize += toMove;
                source.stackSize -= toMove;
                
                // If source stack is empty after moving, clear the slot
                if (source.stackSize <= 0) {
                    inventory.slots[fromSlot] = null;
                }
                return true; // Stack successful
            }
        }
        
        // If stacking didn't happen or wasn't possible, swap items
        inventory.slots[fromSlot] = target; // Put target (or null) into source slot
        inventory.slots[toSlot] = source;   // Put source into target slot
        return true;
    }
    
    /**
     * Get binary data for network transmission for a specific inventory
     * @param {number|string} ownerId - Player ID or Object ID
     * @returns {ArrayBuffer|null} Binary data or null if inventory not found
     */
    getBinaryData(ownerId) {
        const inventory = this.getInventory(ownerId);
        if (!inventory) {
            console.error(`Cannot get binary data: Inventory not found for ${ownerId}`);
            return null;
        }
        return BinaryInventorySerializer.encode(inventory);
    }
}

export { InventoryManager, BinaryInventorySerializer }; 