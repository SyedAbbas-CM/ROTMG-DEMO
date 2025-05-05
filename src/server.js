// Import the BehaviorSystem
import BehaviorSystem from './BehaviorSystem.js';
import ItemManager, { ItemType, ItemRarity } from './ItemManager.js';
import InventoryManager from './InventoryManager.js';
import { MapManager } from './MapManager.js';
import { MapObjectManager } from './MapObjectManager.js';
import { ITEM_DEFINITIONS, ENEMY_DEFINITIONS, MAP_OBJECT_DEFINITIONS } from './database.js';
import NetworkManager from './NetworkManager.js';
import EnemyManager from './EnemyManager.js';

class Server {
  constructor() {
    this.networkManager = new NetworkManager(this);
    this.mapManager = new MapManager({ mapStoragePath: './maps' });
    this.itemManager = new ItemManager();
    this.inventoryManager = new InventoryManager();
    this.mapObjectManager = new MapObjectManager(this.inventoryManager);
    this.enemyManager = new EnemyManager(1000);
    this.behaviorSystem = new BehaviorSystem();
    this.gameState = {
        players: new Map(),
    };
    this.bulletManager = {};

    // Register definitions
    this._registerItemDefinitions();

    // Load the fixed map and place objects
    this.initializeMapAndObjects();
  }

  async initializeMapAndObjects() {
      try {
          console.log('Loading map: maps/fat_plus_playable_tilemap.json');
          const mapId = await this.mapManager.loadFixedMap('maps/fat_plus_playable_tilemap.json');
          console.log(`Map loaded with ID: ${mapId}`);

          // Place a chest near the center (example coordinates)
          const mapData = this.mapManager.getMapMetadata(mapId);
          const chestX = mapData ? Math.floor(mapData.width / 2) + 2 : 32;
          const chestY = mapData ? Math.floor(mapData.height / 2) : 32;
          
          const chestDefinition = MAP_OBJECT_DEFINITIONS.CHEST;
          const chestObject = this.mapObjectManager.placeObject(chestDefinition.id, chestX, chestY);

          if (chestObject && chestObject.inventoryId) {
              console.log(`Populating chest (ID: ${chestObject.id}, InvID: ${chestObject.inventoryId})`);
              this._populateChest(chestObject.inventoryId);
          }

      } catch (error) {
          console.error("FATAL: Failed to initialize map and objects:", error);
      }
  }

  _populateChest(inventoryId) {
    Object.values(ITEM_DEFINITIONS).forEach(definition => {
        if (definition.id === 0) return;

        const itemInstance = this.itemManager.createItem(definition.id);
        if (itemInstance) {
            const added = this.inventoryManager.addItem(inventoryId, itemInstance);
            if (!added) {
                console.warn(`Could not add item ${definition.name} (ID: ${definition.id}) to chest ${inventoryId} - inventory full?`);
            }
        }
    });
    console.log(`Finished populating chest ${inventoryId}`);
  }

  // Register item definitions
  _registerItemDefinitions() {
    Object.values(ITEM_DEFINITIONS).forEach(definition => {
      this.itemManager.registerItemDefinition(definition);
    });
    console.log(`Registered ${Object.keys(ITEM_DEFINITIONS).length} item definitions.`);
  }

  // Update method
  update(deltaTime) {
    if (this.gameState && this.gameState.players && this.gameState.players.size > 0) {
        const targetPlayer = this.gameState.players.values().next().value;
        if (targetPlayer) {
             const target = { x: targetPlayer.x, y: targetPlayer.y };
             this.enemyManager.update(deltaTime, this.bulletManager, target);
        }
    } else {
        this.enemyManager.update(deltaTime, this.bulletManager, null);
    }
  }

  // Broadcast game state
  broadcastGameState() {
    const enemiesData = this.enemyManager.getEnemiesData();
    const itemsData = this.itemManager.getBinaryData();
    const objectsData = this.mapObjectManager.getBinaryData();
    const playersData = {};
    for (const [id, player] of this.gameState.players.entries()) {
        playersData[id] = { x: player.x, y: player.y };
    }

    const fullGameState = {
      players: playersData, 
      enemies: enemiesData,
      items: itemsData,
      objects: objectsData
    };

    this.networkManager.broadcastGameState(fullGameState);
  }

  // Handle item pickup
  handleItemPickup(playerId, itemId) {
    const item = this.itemManager.items.get(itemId);
    if (!item) return false;
    
    const player = this.gameState.players.get(playerId);
    if (!player) return false;
    
    const pickupRadiusSq = 1.5 * 1.5;
    const dx = item.x - player.x;
    const dy = item.y - player.y;
    if (dx * dx + dy * dy > pickupRadiusSq) return false;
    
    const success = this.inventoryManager.addItem(playerId, item);
    if (success) {
      this.itemManager.removeItem(itemId);
      const inventoryData = this.inventoryManager.getBinaryData(playerId);
       if (inventoryData) {
            this.networkManager.sendToPlayer(playerId, { type: 'inventory-update', data: inventoryData });
       }
      this.networkManager.broadcast({ type: 'item-remove', id: itemId });
    } else {
        this.networkManager.sendToPlayer(playerId, { type: 'notify', message: 'Inventory full!' });
    }
    return success;
  }

  // Handle inventory move
  handleInventoryMove(playerId, { fromSlot, toSlot }) {
    const success = this.inventoryManager.moveItem(playerId, fromSlot, toSlot);
    if (success) {
      const inventoryData = this.inventoryManager.getBinaryData(playerId);
      if (inventoryData) {
            this.networkManager.sendToPlayer(playerId, { type: 'inventory-update', data: inventoryData });
      }
    }
    return success;
  }

  // Handle object interaction
  handleObjectInteraction(playerId, objectId) {
      const player = this.gameState.players.get(playerId);
      const mapObject = this.mapObjectManager.getObject(objectId);

      if (!player || !mapObject) return;

      if (mapObject.type === 'container' && mapObject.inventoryId) {
          const inventoryData = this.inventoryManager.getBinaryData(mapObject.inventoryId);
          if (inventoryData) {
              this.networkManager.sendToPlayer(playerId, {
                  type: 'container-inventory-update', 
                  containerId: mapObject.inventoryId,
                  data: inventoryData 
              });
              console.log(`Sent inventory for container ${mapObject.inventoryId} to player ${playerId}`);
          } else {
               console.error(`Could not get inventory data for container ${mapObject.inventoryId}`);
          }
      } else {
          console.log(`Player ${playerId} interacted with object ${objectId} (Type: ${mapObject.type})`);
      }
  }

  // Handle inventory transfer
  handleInventoryTransfer(playerId, { fromInventoryId, toInventoryId, fromSlot, toSlot, amount = 1 }) {
      const player = this.gameState.players.get(playerId);
      if (!player) return false;

      if (fromInventoryId !== playerId && toInventoryId !== playerId) {
          console.warn(`Player ${playerId} attempted invalid transfer between external inventories.`);
          return false;
      }
      
      const fromInventory = this.inventoryManager.getInventory(fromInventoryId);
      const toInventory = this.inventoryManager.getInventory(toInventoryId);

      if (!fromInventory || !toInventory) {
          console.error(`Inventory transfer failed: One or both inventories not found.`);
          return false;
      }

      const removedItem = this.inventoryManager.removeItem(fromInventoryId, fromSlot, amount);
      if (!removedItem) {
          console.log(`Inventory transfer failed: Could not remove item from slot ${fromSlot} in ${fromInventoryId}`);
          return false;
      }

      const added = this.inventoryManager.addItem(toInventoryId, removedItem);

      if (!added) {
          console.warn(`Inventory transfer failed: Could not add item to ${toInventoryId}. Refunding.`);
          this.inventoryManager.addItem(fromInventoryId, removedItem);
          this.networkManager.sendToPlayer(playerId, { type: 'notify', message: 'Target inventory full!'});
          return false;
      }

      console.log(`Transfer success: ${amount} of item ${removedItem.definitionId} from ${fromInventoryId}[${fromSlot}] to ${toInventoryId}`);
      const fromInvData = this.inventoryManager.getBinaryData(fromInventoryId);
      const toInvData = this.inventoryManager.getBinaryData(toInventoryId);

      const sendUpdate = (invId, invData) => {
          if (!invData) return;
          if (invId === playerId) {
              this.networkManager.sendToPlayer(playerId, { type: 'inventory-update', data: invData });
          } else {
              this.networkManager.sendToPlayer(playerId, { type: 'container-inventory-update', containerId: invId, data: invData });
          }
      };

      sendUpdate(fromInventoryId, fromInvData);
      sendUpdate(toInventoryId, toInvData);

      return true;
  }

  // Handle enemy death and drops
  handleEnemyDeath(enemyId) {
    const enemy = this.enemyManager.getEnemy(enemyId);
    if (!enemy) return;
    
    const definition = Object.values(ENEMY_DEFINITIONS).find(def => def.id === enemy.definitionId); 
    if (!definition || !definition.dropTable) return;
    
    console.log(`Processing drops for enemy ${enemyId} (Type: ${definition.name})`);
    definition.dropTable.forEach(drop => {
      if (Math.random() < drop.chance) {
        const itemInstance = this.itemManager.createItem(drop.itemId, {
          x: enemy.x,
          y: enemy.y
        });
        
        if (itemInstance) {
           this.itemManager.spawnItem(itemInstance.definitionId, itemInstance.x, itemInstance.y);
           console.log(`Dropped item ${itemInstance.id} at (${itemInstance.x}, ${itemInstance.y})`);
           this.networkManager.broadcast({ type: 'item-spawn', item: BinaryItemSerializer.encode(itemInstance) });
        } else {
            console.error(`Failed to create drop item instance for ID: ${drop.itemId}`);
        }
      }
    });

    this.enemyManager.removeEnemy(enemyId);
    this.networkManager.broadcast({ type: 'enemy-remove', id: enemyId });
  }
}

export default Server; 