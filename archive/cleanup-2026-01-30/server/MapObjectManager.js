/**
 * MapObjectManager.js
 * Manages static objects placed on the map (chests, signs, etc.)
 */

import { MAP_OBJECT_DEFINITIONS } from '../database.js';
import { BinaryItemSerializer } from '../ItemManager.js'; // For potential future use with object states

/**
 * BinaryObjectSerializer - For sending object data efficiently
 */
class BinaryObjectSerializer {
    static encode(object) {
        // Basic encoding: ID, definition ID, x, y
        const buffer = new ArrayBuffer(16);
        const view = new DataView(buffer);
        view.setUint32(0, object.id, true);
        view.setUint16(4, object.definitionId, true);
        view.setFloat32(6, object.x, true);
        view.setFloat32(10, object.y, true);
        // Future: Add more state like open/closed for chests
        return buffer;
    }

    static decode(buffer) {
        const view = new DataView(buffer);
        return {
            id: view.getUint32(0, true),
            definitionId: view.getUint16(4, true),
            x: view.getFloat32(6, true),
            y: view.getFloat32(10, true)
        };
    }
}

class MapObjectManager {
    constructor(inventoryManager) {
        this.objects = new Map(); // object.id -> object data
        this.nextObjectId = 1;
        this.definitions = MAP_OBJECT_DEFINITIONS;
        this.inventoryManager = inventoryManager; // To manage inventories for containers
    }

    /**
     * Place an object on the map
     * @param {number} definitionId - ID from MAP_OBJECT_DEFINITIONS
     * @param {number} x - Tile X coordinate
     * @param {number} y - Tile Y coordinate
     * @param {object} options - Optional initial state or properties
     * @returns {Object|null} The created map object instance
     */
    placeObject(definitionId, x, y, options = {}) {
        const definition = Object.values(this.definitions).find(def => def.id === definitionId);
        if (!definition) {
            console.error(`Map object definition not found for ID: ${definitionId}`);
            return null;
        }

        const object = {
            id: this.nextObjectId++,
            definitionId: definition.id,
            name: definition.name,
            type: definition.type,
            x,
            y,
            spriteSheet: definition.spriteSheet,
            spriteX: definition.spriteX,
            spriteY: definition.spriteY,
            spriteWidth: definition.spriteWidth,
            spriteHeight: definition.spriteHeight,
            interactionType: definition.interactionType,
            ...options // Allow overriding properties or adding initial state
        };

        this.objects.set(object.id, object);

        // If it's a container, create an inventory for it
        if (definition.type === 'container' && definition.inventorySize > 0) {
            // Use a unique identifier for the chest's inventory
            const inventoryId = `chest_${object.id}`;
            this.inventoryManager.createInventory(inventoryId, definition.inventorySize);
            object.inventoryId = inventoryId; // Store reference to its inventory
        }

        console.log(`Placed object '${object.name}' (ID: ${object.id}) at (${x}, ${y})`);
        return object;
    }

    /**
     * Remove an object from the map
     * @param {number} objectId - The ID of the object instance
     */
    removeObject(objectId) {
        const object = this.objects.get(objectId);
        if (object) {
            // If it had an inventory, remove it
            if (object.inventoryId) {
                this.inventoryManager.removeInventory(object.inventoryId);
            }
            this.objects.delete(objectId);
            console.log(`Removed object ID: ${objectId}`);
        } else {
            console.warn(`Attempted to remove non-existent object ID: ${objectId}`);
        }
    }

    /**
     * Get an object by its ID
     * @param {number} objectId
     * @returns {Object|undefined}
     */
    getObject(objectId) {
        return this.objects.get(objectId);
    }

    /**
     * Get all objects currently placed on the map
     * @returns {Array<Object>}
     */
    getAllObjects() {
        return Array.from(this.objects.values());
    }

    /**
     * Get binary data for all objects for network transmission
     * @returns {ArrayBuffer}
     */
    getBinaryData() {
        const allObjects = this.getAllObjects();
        const objectSize = 16; // Size of each encoded object
        const buffer = new ArrayBuffer(4 + allObjects.length * objectSize);
        const view = new DataView(buffer);

        view.setUint32(0, allObjects.length, true);
        let offset = 4;

        for (const obj of allObjects) {
            const objBuffer = BinaryObjectSerializer.encode(obj);
            new Uint8Array(buffer, offset, objectSize).set(new Uint8Array(objBuffer));
            offset += objectSize;
        }

        return buffer;
    }
}

export { MapObjectManager, BinaryObjectSerializer }; 