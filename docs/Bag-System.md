# Bag System Documentation

## Overview
The Bag System manages loot bags that appear when enemies die, providing temporary storage for items in the world. It uses Structure of Arrays (SoA) layout for performance and integrates with the Drop System, Item System, and network layer for complete loot bag functionality.

## Core Architecture

### 1. BagManager Class (`/src/BagManager.js`)

#### **Data Structure (SoA Layout)**
```javascript
class BagManager {
  constructor(maxBags = 500) {
    this.maxBags = maxBags;
    this.bagCount = 0;
    this.nextBagId = 1;

    // Core bag properties (SoA for performance)
    this.id = new Array(maxBags);              // "bag_1", "bag_2", etc
    this.x = new Float32Array(maxBags);        // World X position
    this.y = new Float32Array(maxBags);        // World Y position
    this.creationTime = new Float32Array(maxBags); // Spawn timestamp (seconds)
    this.lifetime = new Float32Array(maxBags);     // TTL duration (seconds)
    this.itemSlots = new Array(maxBags);           // Array<itemInstanceId>
    this.bagType = new Uint8Array(maxBags);       // Color type (0-6)
    this.owners = new Array(maxBags);              // Array<clientId> or null
    this.worldId = new Array(maxBags);             // World/map context
  }
}
```

#### **Key Design Decisions**
- **SoA Layout**: Mirrors EnemyManager and BulletManager for cache efficiency
- **Fixed Capacity**: Prevents runaway entity counts
- **Ownership System**: Supports soulbound items and player-specific visibility
- **TTL Management**: Automatic cleanup prevents world clutter
- **World Isolation**: Bags are filtered by world context

### 2. Core Functions

#### **`spawnBag(x, y, itemIds, worldId, ttl, bagType, owners)`**

**Purpose**: Creates a new loot bag containing items
**Parameters**:
- `x, y`: World position (enemy death location)
- `itemIds`: Array of item instance IDs from ItemManager
- `worldId`: World context (default: 'default')
- `ttl`: Time-to-live in seconds (default: 300 = 5 minutes)
- `bagType`: Color type 0-6 (determined by Drop System)
- `owners`: Array of client IDs or null for public bags

**Process**:
```javascript
1. Check capacity (max 500 bags)
2. Generate unique bag ID: `bag_${nextBagId++}`
3. Store position and metadata in SoA arrays
4. Limit item slots to 8 items max
5. Set creation timestamp and lifetime
6. Assign ownership and world context
7. Return bag ID for reference
```

**Implementation**:
```javascript
spawnBag(x, y, itemIds = [], worldId = 'default', ttl = 300, bagType = 0, owners = null) {
  if (this.bagCount >= this.maxBags) {
    console.warn('[BagManager] Max bag capacity reached');
    return null;
  }
  
  const idx = this.bagCount++;
  const bagId = `bag_${this.nextBagId++}`;
  
  this.id[idx] = bagId;
  this.x[idx] = x;
  this.y[idx] = y;
  this.creationTime[idx] = Date.now() / 1000; // Convert to seconds
  this.lifetime[idx] = ttl;
  this.itemSlots[idx] = itemIds.slice(0, 8);   // Max 8 items per bag
  this.bagType[idx] = bagType;
  this.owners[idx] = owners;                   // null = public bag
  this.worldId[idx] = worldId;
  
  return bagId;
}
```

#### **`update(nowSec)`**

**Purpose**: Per-frame cleanup of expired bags
**Process**:
```javascript
1. Iterate through all active bags
2. Check if (currentTime - creationTime) >= lifetime
3. Remove expired bags using swap-remove technique
4. Decrement index after removal to recheck swapped bag
```

**Implementation**:
```javascript
update(nowSec) {
  for (let i = 0; i < this.bagCount; i++) {
    if (nowSec - this.creationTime[i] >= this.lifetime[i]) {
      this._swapRemove(i);
      i--; // Re-check the swapped index
    }
  }
}
```

#### **`removeItemFromBag(bagId, itemInstanceId)`**

**Purpose**: Handle item pickup from bags
**Returns**: `true` if bag became empty and was removed
**Process**:
```javascript
1. Find bag by ID
2. Locate item in itemSlots array
3. Remove item using array.splice()
4. If bag empty, remove entire bag
5. Return cleanup status
```

**Usage in Player Interaction**:
```javascript
// When player picks up item
const bagEmpty = bagManager.removeItemFromBag("bag_123", itemInstanceId);
if (bagEmpty) {
  // Bag was removed, update client displays
  broadcastBagRemoval("bag_123");
}
```

#### **`getBagsData(filterWorldId, viewerId)`**

**Purpose**: Serialize bag data for network transmission
**Parameters**:
- `filterWorldId`: Only include bags from specific world
- `viewerId`: Client ID for ownership filtering

**Visibility Logic**:
```javascript
1. Filter by world if specified
2. Check ownership: if bag.owners exists, ensure viewerId is included
3. Public bags (owners = null) are visible to everyone
4. Return sanitized bag data for network
```

**Output Format**:
```javascript
[{
  id: "bag_123",
  x: 45.2,
  y: 23.8,
  bagType: 2,           // Purple bag
  items: [456, 789]     // Item instance IDs
}, ...]
```

### 3. Integration with Drop System

#### **Bag Spawning from Enemy Death**

**Called from**: `EnemyManager.onDeath()` (`EnemyManager.js:559`)
```javascript
// After rolling drop table and creating items
this._bagManager.spawnBag(
  this.x[index],           // Enemy X position
  this.y[index],           // Enemy Y position  
  itemInstanceIds,         // Items from ItemManager
  this.worldId[index],     // Enemy's world context
  300,                     // 5 minute TTL
  bagType                  // Color from Drop System priority
);
```

#### **Bag Color Integration**

**Color Determination**:
```javascript
// From DropSystem.rollDropTable()
const {items, bagType} = rollDropTable(dropTable);
// bagType = highest priority color from all dropped items

// Passed to bag spawning
spawnBag(x, y, itemInstanceIds, worldId, 300, bagType);
```

**Color-to-Sprite Mapping** (DropSystem.js):
```javascript
function getBagColourSprite(bagType) {
  switch(bagType) {
    case 0: return 'items_sprite_lootbag_white';    // Common
    case 1: return 'items_sprite_lootbag_brown';    // Uncommon
    case 2: return 'items_sprite_lootbag_purple';   // Rare
    case 3: return 'items_sprite_lootbag_orange';   // Epic
    case 4: return 'items_sprite_lootbag_cyan';     // Legendary
    case 5: return 'items_sprite_lootbag_blue';     // Special
    case 6: return 'items_sprite_lootbag_red';      // Ultimate
    default: return 'items_sprite_lootbag_white';
  }
}
```

### 4. Network Integration

#### **Server-Side Broadcasting**

**Integration with Server.js World Updates** (`Server.js:796-850`):
```javascript
// Include bags in world update packets
const bags = ctx.bagMgr.getBagsData(mapId, client.id);

// Apply interest management (distance filtering)
const visibleBags = bags.filter(bag => {
  const dx = bag.x - playerX;
  const dy = bag.y - playerY;
  return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
});

// Include in world update message
sendToClient(client.socket, MessageType.WORLD_UPDATE, {
  enemies: visibleEnemies,
  bullets: visibleBullets,
  players: visiblePlayers,
  objects: visibleObjects,
  bags: visibleBags.slice(0, MAX_ENTITIES_PER_PACKET)
});
```

#### **Client-Side Message Handling**

**WORLD_UPDATE Handler**:
```javascript
this.handlers[MessageType.WORLD_UPDATE] = (data) => {
  if (this.game.updateWorld) {
    this.game.updateWorld(
      data.enemies, 
      data.bullets, 
      data.players, 
      data.objects, 
      data.bags     // Client receives bag data
    );
  }
};
```

**Client Bag Management** (Typical pattern):
```javascript
// In client game loop
updateWorld(enemies, bullets, players, objects, bags) {
  // Update bag entities
  this.clientBagManager.updateBags(bags);
  
  // Render bags with appropriate sprites
  bags.forEach(bag => {
    const sprite = getBagColourSprite(bag.bagType);
    this.renderer.drawSprite(sprite, bag.x, bag.y);
  });
}
```

### 5. Performance Optimizations

#### **Memory Management**

**Swap-Remove Pattern** (`_swapRemove(idx)`):
```javascript
_swapRemove(idx) {
  const last = this.bagCount - 1;
  if (idx !== last) {
    // Swap with last element
    this.id[idx] = this.id[last];
    this.x[idx] = this.x[last];
    this.y[idx] = this.y[last];
    this.creationTime[idx] = this.creationTime[last];
    this.lifetime[idx] = this.lifetime[last];
    this.itemSlots[idx] = this.itemSlots[last];
    this.bagType[idx] = this.bagType[last];
    this.owners[idx] = this.owners[last];
    this.worldId[idx] = this.worldId[last];
  }
  this.bagCount--;
}
```

**Benefits**:
- O(1) removal time
- No array shifting required
- Maintains cache-friendly SoA layout

#### **Network Efficiency**

**Interest Management**:
- Only send bags within player view distance
- Limit entities per packet to prevent oversized messages
- Filter by world context to avoid cross-world leakage

**Ownership Filtering**:
- Server-side filtering reduces client-side processing
- Soulbound items only sent to eligible players
- Public bags visible to all players in range

### 6. Ownership and Visibility System

#### **Public Bags** (`owners = null`)
```javascript
// Anyone can see and interact
spawnBag(x, y, itemIds, worldId, 300, bagType, null);
```

#### **Soulbound Bags** (`owners = [clientId, ...]`)
```javascript
// Only specific players can see
const soulboundOwners = [player1.id, player2.id]; // Party members
spawnBag(x, y, itemIds, worldId, 300, bagType, soulboundOwners);
```

#### **Visibility Check Implementation**:
```javascript
// In getBagsData()
if(this.owners[i] && viewerId && !this.owners[i].includes(viewerId)) {
  continue; // Skip this bag for this viewer
}
```

### 7. World Context Management

#### **Per-World Bag Isolation**

**Server Context Creation** (`Server.js:267-280`):
```javascript
function getWorldCtx(mapId) {
  if (!worldContexts.has(mapId)) {
    const bagMgr = new BagManager(500);
    // ... other managers
    worldContexts.set(mapId, { 
      enemyMgr, bulletMgr, bagMgr, collMgr 
    });
  }
  return worldContexts.get(mapId);
}
```

**World-Filtered Updates**:
```javascript
// Only bags from player's current world
const bags = ctx.bagMgr.getBagsData(client.mapId, client.id);
```

### 8. Integration with Item System

#### **Item Instance Storage**

**Bag Contents**:
```javascript
// Bags store item instance IDs, not definition IDs
this.itemSlots[bagIdx] = [123, 456, 789]; // ItemManager instance IDs
```

**Item Lookup for Display**:
```javascript
// When client needs item details
const bagItems = bag.items.map(instanceId => {
  return itemManager.items.get(instanceId);
}).filter(Boolean);

// Render items with sprites and stats
bagItems.forEach(item => {
  renderItemIcon(item.spriteSheet, item.spriteX, item.spriteY);
  showItemTooltip(item.stats, item.rarity);
});
```

#### **Item Lifecycle**

**Creation Flow**:
```
Enemy Death → Drop Roll → Item Creation → Bag Spawn → Player Pickup
     ↓             ↓           ↓            ↓           ↓
EnemyManager → DropSystem → ItemManager → BagManager → ItemManager
```

**Cleanup Flow**:
```
Bag Expiry → Item Removal → Memory Cleanup
     ↓             ↓             ↓
BagManager → ItemManager → Garbage Collection
```

### 9. Configuration and Extensibility

#### **Capacity Tuning**
```javascript
// Adjust based on server capacity
const bagMgr = new BagManager(1000); // Higher capacity for busy servers
```

#### **TTL Configuration**
```javascript
// Different TTL for different bag types
const ttl = bagType >= 3 ? 600 : 300; // Epic+ bags last 10 minutes
spawnBag(x, y, itemIds, worldId, ttl, bagType);
```

#### **Custom Ownership Logic**
```javascript
// Guild-based ownership
const guildMembers = getGuildMembers(killerGuildId);
spawnBag(x, y, itemIds, worldId, 300, bagType, guildMembers);
```

### 10. Future Enhancements

#### **Bag Interaction System**
```javascript
// Advanced bag mechanics
class BagManager {
  openBag(bagId, playerId) {
    // Show bag contents UI
    // Handle item transfers
    // Apply pickup restrictions
  }
  
  mergeBags(bagId1, bagId2) {
    // Combine nearby bags of same type
    // Optimize bag density
  }
}
```

#### **Advanced Ownership**
```javascript
// Time-based ownership transitions
spawnBag(x, y, itemIds, worldId, 300, bagType, owners, {
  publicAfter: 60,  // Becomes public after 60 seconds
  despawnWarning: 30 // Warning 30 seconds before despawn
});
```

### 11. Advanced Lifecycle Management and State Tracking

#### **Complete Bag Lifecycle Analysis**

The bag system implements a sophisticated lifecycle with multiple states and transition events:

**Lifecycle States** (Enhanced Implementation):
```javascript
class AdvancedBagManager extends BagManager {
  constructor(maxBags = 500) {
    super(maxBags);
    
    // Enhanced state tracking
    this.bagState = new Uint8Array(maxBags);      // Lifecycle state
    this.lastInteraction = new Float32Array(maxBags); // Last access time
    this.interactionCount = new Uint16Array(maxBags); // Number of interactions
    this.spawnReason = new Array(maxBags);         // Why bag was created
    this.originalContents = new Array(maxBags);   // Initial item list
    this.stateTransitions = new Array(maxBags);   // State change history
  }
}

// Bag lifecycle states
const BagState = {
  SPAWNING: 0,      // Just created, not yet visible
  ACTIVE: 1,        // Normal state, visible to players
  ACCESSED: 2,      // Player has opened/interacted
  PARTIALLY_EMPTY: 3, // Some items taken
  WARNING: 4,       // Near expiration
  EXPIRING: 5,      // Grace period before removal
  REMOVED: 6        // Marked for cleanup
};

// State transition logic
function updateBagState(bagIndex, nowSec) {
  const currentState = this.bagState[bagIndex];
  const age = nowSec - this.creationTime[bagIndex];
  const remainingTime = this.lifetime[bagIndex] - age;
  const itemCount = this.itemSlots[bagIndex].length;
  const originalCount = this.originalContents[bagIndex].length;
  
  let newState = currentState;
  
  switch (currentState) {
    case BagState.SPAWNING:
      if (age > 0.1) { // 100ms spawn delay
        newState = BagState.ACTIVE;
      }
      break;
      
    case BagState.ACTIVE:
      if (this.interactionCount[bagIndex] > 0) {
        newState = BagState.ACCESSED;
      } else if (remainingTime < 30) {
        newState = BagState.WARNING;
      }
      break;
      
    case BagState.ACCESSED:
      if (itemCount < originalCount) {
        newState = BagState.PARTIALLY_EMPTY;
      } else if (remainingTime < 30) {
        newState = BagState.WARNING;
      }
      break;
      
    case BagState.PARTIALLY_EMPTY:
      if (itemCount === 0) {
        newState = BagState.EXPIRING;
      } else if (remainingTime < 30) {
        newState = BagState.WARNING;
      }
      break;
      
    case BagState.WARNING:
      if (remainingTime < 5) {
        newState = BagState.EXPIRING;
      }
      break;
      
    case BagState.EXPIRING:
      if (remainingTime <= 0) {
        newState = BagState.REMOVED;
      }
      break;
  }
  
  if (newState !== currentState) {
    this.transitionBagState(bagIndex, currentState, newState, nowSec);
  }
}

// State transition with event logging
function transitionBagState(bagIndex, fromState, toState, timestamp) {
  this.bagState[bagIndex] = toState;
  
  // Log transition
  if (!this.stateTransitions[bagIndex]) {
    this.stateTransitions[bagIndex] = [];
  }
  
  this.stateTransitions[bagIndex].push({
    from: fromState,
    to: toState,
    timestamp,
    age: timestamp - this.creationTime[bagIndex]
  });
  
  // Trigger state-specific actions
  this.onBagStateChange(bagIndex, fromState, toState);
}
```

#### **Advanced TTL Management with Dynamic Extensions**

**Smart TTL System**:
```javascript
class SmartTTLManager {
  constructor(bagManager) {
    this.bagManager = bagManager;
    this.ttlRules = new Map();
    this.extensionHistory = new Map();
  }
  
  // Register TTL rules based on bag properties
  registerTTLRule(condition, ttlModifier) {
    this.ttlRules.set(condition.name, { condition, ttlModifier });
  }
  
  // Calculate dynamic TTL for new bags
  calculateInitialTTL(bagType, itemIds, context) {
    let baseTTL = 300; // 5 minutes default
    
    // Bag type modifiers
    const typeMultipliers = {
      0: 1.0,   // White: 5 minutes
      1: 1.2,   // Brown: 6 minutes
      2: 1.5,   // Purple: 7.5 minutes
      3: 2.0,   // Orange: 10 minutes
      4: 3.0,   // Cyan: 15 minutes
      5: 4.0,   // Blue: 20 minutes
      6: 6.0    // Red: 30 minutes
    };
    
    baseTTL *= (typeMultipliers[bagType] || 1.0);
    
    // Item count modifier
    const itemCountBonus = Math.min(itemIds.length * 30, 180); // Max 3 minutes bonus
    baseTTL += itemCountBonus;
    
    // Context modifiers
    if (context.bossKill) {
      baseTTL *= 2.0; // Boss drops last twice as long
    }
    
    if (context.playerCount > 5) {
      baseTTL *= 1.5; // Longer in crowded areas
    }
    
    return Math.floor(baseTTL);
  }
  
  // Extend TTL based on player activity
  considerTTLExtension(bagIndex, interactionType) {
    const bagId = this.bagManager.id[bagIndex];
    const currentTTL = this.bagManager.lifetime[bagIndex];
    const age = Date.now() / 1000 - this.bagManager.creationTime[bagIndex];
    const remainingTime = currentTTL - age;
    
    // Don't extend if bag still has plenty of time
    if (remainingTime > 120) return false;
    
    // Track extension history to prevent abuse
    if (!this.extensionHistory.has(bagId)) {
      this.extensionHistory.set(bagId, []);
    }
    
    const history = this.extensionHistory.get(bagId);
    const recentExtensions = history.filter(ext => 
      Date.now() - ext.timestamp < 60000 // Last minute
    ).length;
    
    // Limit extensions
    if (recentExtensions >= 2) return false;
    
    // Calculate extension amount
    let extensionSeconds = 0;
    
    switch (interactionType) {
      case 'PLAYER_NEARBY':
        extensionSeconds = 30;
        break;
      case 'BAG_OPENED':
        extensionSeconds = 60;
        break;
      case 'ITEM_EXAMINED':
        extensionSeconds = 15;
        break;
      case 'COMBAT_NEARBY':
        extensionSeconds = 45;
        break;
    }
    
    if (extensionSeconds > 0) {
      this.bagManager.lifetime[bagIndex] += extensionSeconds;
      
      // Log extension
      history.push({
        type: interactionType,
        extension: extensionSeconds,
        timestamp: Date.now()
      });
      
      console.log(`[BagManager] Extended TTL for ${bagId} by ${extensionSeconds}s (${interactionType})`);
      return true;
    }
    
    return false;
  }
}
```

#### **Comprehensive Ownership and Visibility System**

**Multi-Tiered Ownership Model**:
```javascript
class AdvancedOwnershipManager {
  constructor() {
    this.ownershipTiers = new Map();
    this.visibilityRules = new Map();
    this.accessHistory = new Map();
  }
  
  // Define ownership tiers with different privileges
  defineTiers() {
    return {
      KILLER: {
        priority: 0,
        duration: 30,  // 30 seconds exclusive access
        canSee: true,
        canTake: true,
        label: 'Killer'
      },
      PARTY_MEMBER: {
        priority: 1,
        duration: 60,  // 1 minute after killer period
        canSee: true,
        canTake: true,
        label: 'Party'
      },
      GUILD_MEMBER: {
        priority: 2,
        duration: 120, // 2 minutes after party period
        canSee: true,
        canTake: true,
        label: 'Guild'
      },
      DAMAGE_CONTRIBUTOR: {
        priority: 3,
        duration: 180, // 3 minutes total
        canSee: true,
        canTake: true,
        label: 'Contributor'
      },
      PUBLIC: {
        priority: 4,
        duration: Infinity,
        canSee: true,
        canTake: true,
        label: 'Public'
      }
    };
  }
  
  // Calculate ownership based on context
  calculateOwnership(context) {
    const ownership = {
      tiers: [],
      schedule: []
    };
    
    const tiers = this.defineTiers();
    let cumulativeTime = 0;
    
    // Add killer tier
    if (context.killerId) {
      ownership.tiers.push({
        type: 'KILLER',
        playerIds: [context.killerId],
        startTime: cumulativeTime,
        endTime: cumulativeTime + tiers.KILLER.duration
      });
      cumulativeTime += tiers.KILLER.duration;
    }
    
    // Add party tier
    if (context.partyMembers && context.partyMembers.length > 1) {
      ownership.tiers.push({
        type: 'PARTY_MEMBER',
        playerIds: context.partyMembers,
        startTime: cumulativeTime,
        endTime: cumulativeTime + tiers.PARTY_MEMBER.duration
      });
      cumulativeTime += tiers.PARTY_MEMBER.duration;
    }
    
    // Add guild tier
    if (context.guildMembers && context.guildMembers.length > 0) {
      ownership.tiers.push({
        type: 'GUILD_MEMBER',
        playerIds: context.guildMembers,
        startTime: cumulativeTime,
        endTime: cumulativeTime + tiers.GUILD_MEMBER.duration
      });
      cumulativeTime += tiers.GUILD_MEMBER.duration;
    }
    
    // Add contributor tier
    if (context.damageContributors && context.damageContributors.length > 0) {
      ownership.tiers.push({
        type: 'DAMAGE_CONTRIBUTOR',
        playerIds: context.damageContributors,
        startTime: cumulativeTime,
        endTime: cumulativeTime + tiers.DAMAGE_CONTRIBUTOR.duration
      });
      cumulativeTime += tiers.DAMAGE_CONTRIBUTOR.duration;
    }
    
    // Always end with public access
    ownership.tiers.push({
      type: 'PUBLIC',
      playerIds: null, // null means everyone
      startTime: cumulativeTime,
      endTime: Infinity
    });
    
    return ownership;
  }
  
  // Check if player can see/access bag at current time
  checkAccess(bagIndex, playerId, currentTime) {
    const bagId = this.bagManager.id[bagIndex];
    const ownership = this.ownershipTiers.get(bagId);
    
    if (!ownership) {
      return { canSee: true, canTake: true, tier: 'PUBLIC' };
    }
    
    const bagAge = currentTime - this.bagManager.creationTime[bagIndex];
    
    // Find current tier
    for (const tier of ownership.tiers) {
      if (bagAge >= tier.startTime && bagAge < tier.endTime) {
        const canAccess = !tier.playerIds || tier.playerIds.includes(playerId);
        
        return {
          canSee: canAccess,
          canTake: canAccess,
          tier: tier.type,
          timeRemaining: tier.endTime - bagAge
        };
      }
    }
    
    // Default to public access
    return { canSee: true, canTake: true, tier: 'PUBLIC' };
  }
}
```

#### **Advanced Spatial and Interest Management**

**Efficient Spatial Indexing**:
```javascript
class BagSpatialIndex {
  constructor(bagManager, cellSize = 10) {
    this.bagManager = bagManager;
    this.cellSize = cellSize;
    this.grid = new Map(); // cellKey -> Set<bagIndex>
    this.bagToCell = new Map(); // bagIndex -> cellKey
  }
  
  // Convert world coordinates to grid cell
  getCellKey(x, y) {
    const cellX = Math.floor(x / this.cellSize);
    const cellY = Math.floor(y / this.cellSize);
    return `${cellX},${cellY}`;
  }
  
  // Add bag to spatial index
  addBag(bagIndex) {
    const x = this.bagManager.x[bagIndex];
    const y = this.bagManager.y[bagIndex];
    const cellKey = this.getCellKey(x, y);
    
    if (!this.grid.has(cellKey)) {
      this.grid.set(cellKey, new Set());
    }
    
    this.grid.get(cellKey).add(bagIndex);
    this.bagToCell.set(bagIndex, cellKey);
  }
  
  // Remove bag from spatial index
  removeBag(bagIndex) {
    const cellKey = this.bagToCell.get(bagIndex);
    if (cellKey && this.grid.has(cellKey)) {
      this.grid.get(cellKey).delete(bagIndex);
      
      // Clean up empty cells
      if (this.grid.get(cellKey).size === 0) {
        this.grid.delete(cellKey);
      }
    }
    this.bagToCell.delete(bagIndex);
  }
  
  // Find bags within radius using spatial index
  findBagsInRadius(centerX, centerY, radius) {
    const results = [];
    const radiusSq = radius * radius;
    
    // Calculate which cells to check
    const minCellX = Math.floor((centerX - radius) / this.cellSize);
    const maxCellX = Math.floor((centerX + radius) / this.cellSize);
    const minCellY = Math.floor((centerY - radius) / this.cellSize);
    const maxCellY = Math.floor((centerY + radius) / this.cellSize);
    
    // Check each relevant cell
    for (let cellX = minCellX; cellX <= maxCellX; cellX++) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY++) {
        const cellKey = `${cellX},${cellY}`;
        const cell = this.grid.get(cellKey);
        
        if (cell) {
          for (const bagIndex of cell) {
            const dx = this.bagManager.x[bagIndex] - centerX;
            const dy = this.bagManager.y[bagIndex] - centerY;
            const distSq = dx * dx + dy * dy;
            
            if (distSq <= radiusSq) {
              results.push({
                index: bagIndex,
                distance: Math.sqrt(distSq),
                bagId: this.bagManager.id[bagIndex]
              });
            }
          }
        }
      }
    }
    
    return results.sort((a, b) => a.distance - b.distance);
  }
  
  // Update bag position in spatial index
  updateBagPosition(bagIndex, newX, newY) {
    this.removeBag(bagIndex);
    this.bagManager.x[bagIndex] = newX;
    this.bagManager.y[bagIndex] = newY;
    this.addBag(bagIndex);
  }
}
```

#### **Performance Monitoring and Analytics**

**Comprehensive Bag Analytics**:
```javascript
class BagAnalytics {
  constructor(bagManager) {
    this.bagManager = bagManager;
    this.metrics = {
      spawnCount: 0,
      expiredCount: 0,
      emptyCount: 0,
      interactionCount: 0,
      averageLifetime: 0,
      peakBagCount: 0,
      spatialHotspots: new Map(),
      itemPopularity: new Map(),
      playerInteractions: new Map()
    };
    
    this.performanceHistory = [];
    this.lastMetricsUpdate = Date.now();
  }
  
  // Record bag spawn event
  recordBagSpawn(bagIndex, context) {
    this.metrics.spawnCount++;
    this.metrics.peakBagCount = Math.max(this.metrics.peakBagCount, this.bagManager.bagCount);
    
    // Track spatial hotspots
    const x = this.bagManager.x[bagIndex];
    const y = this.bagManager.y[bagIndex];
    const region = `${Math.floor(x / 50)},${Math.floor(y / 50)}`; // 50-unit regions
    
    this.metrics.spatialHotspots.set(region, 
      (this.metrics.spatialHotspots.get(region) || 0) + 1
    );
    
    // Track item popularity
    const items = this.bagManager.itemSlots[bagIndex];
    items.forEach(itemId => {
      this.metrics.itemPopularity.set(itemId,
        (this.metrics.itemPopularity.get(itemId) || 0) + 1
      );
    });
  }
  
  // Record bag expiration
  recordBagExpiration(bagIndex, reason) {
    if (reason === 'TTL_EXPIRED') {
      this.metrics.expiredCount++;
    } else if (reason === 'EMPTY') {
      this.metrics.emptyCount++;
    }
    
    // Update average lifetime
    const lifetime = Date.now() / 1000 - this.bagManager.creationTime[bagIndex];
    const totalLifetime = this.metrics.averageLifetime * (this.metrics.expiredCount + this.metrics.emptyCount - 1);
    this.metrics.averageLifetime = (totalLifetime + lifetime) / (this.metrics.expiredCount + this.metrics.emptyCount);
  }
  
  // Record player interaction
  recordInteraction(bagIndex, playerId, interactionType) {
    this.metrics.interactionCount++;
    
    if (!this.metrics.playerInteractions.has(playerId)) {
      this.metrics.playerInteractions.set(playerId, {
        bagCount: 0,
        itemCount: 0,
        interactionTypes: new Map()
      });
    }
    
    const playerStats = this.metrics.playerInteractions.get(playerId);
    playerStats.bagCount++;
    playerStats.interactionTypes.set(interactionType,
      (playerStats.interactionTypes.get(interactionType) || 0) + 1
    );
  }
  
  // Generate performance report
  generateReport() {
    const currentTime = Date.now();
    const timeSinceLastUpdate = currentTime - this.lastMetricsUpdate;
    
    const report = {
      timestamp: currentTime,
      period: timeSinceLastUpdate,
      
      bagMetrics: {
        current: this.bagManager.bagCount,
        peak: this.metrics.peakBagCount,
        spawned: this.metrics.spawnCount,
        expired: this.metrics.expiredCount,
        emptied: this.metrics.emptyCount,
        averageLifetime: this.metrics.averageLifetime.toFixed(2) + 's'
      },
      
      interactionMetrics: {
        total: this.metrics.interactionCount,
        uniquePlayers: this.metrics.playerInteractions.size,
        ratePerMinute: (this.metrics.interactionCount / (timeSinceLastUpdate / 60000)).toFixed(2)
      },
      
      spatialAnalysis: {
        hotspots: Array.from(this.metrics.spatialHotspots.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10), // Top 10 hotspots
        coverage: this.metrics.spatialHotspots.size
      },
      
      itemAnalysis: {
        uniqueItems: this.metrics.itemPopularity.size,
        mostPopular: Array.from(this.metrics.itemPopularity.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10) // Top 10 items
      }
    };
    
    this.lastMetricsUpdate = currentTime;
    return report;
  }
}
```

### 12. Integration Points Summary

#### **Enhanced Input Dependencies**
- **Drop System**: Bag color priority, item instances, and contextual spawn data
- **Enemy System**: Death events, world context, damage contribution tracking
- **Item System**: Item instance IDs, lifecycle management, and ownership data
- **Player System**: Party/guild memberships, proximity tracking, interaction history
- **Network System**: Interest management, spatial filtering, and ownership visibility
- **Combat System**: Damage contribution data for soulbound calculations
- **World System**: Map boundaries, safe zones, and environmental context

#### **Enhanced Output Products**
- **World Entities**: Spatially-indexed bags with lifecycle states
- **Network Data**: Ownership-filtered bag information with state metadata
- **Cleanup Events**: Expired bag notifications with analytics data
- **Spatial Queries**: Efficient radius-based bag finding with caching
- **Analytics Data**: Comprehensive metrics for balancing and optimization
- **State Events**: Bag lifecycle transitions for client synchronization

#### **Advanced Data Flow**
```
Enemy Death → Context Analysis → Ownership Calculation → Item Creation → Bag Spawn → Spatial Index → State Tracking → Network Sync → Client Display → Player Interaction → Lifecycle Management
     ↓              ↓                    ↓                  ↓             ↓           ↓              ↓               ↓            ↓               ↓                    ↓
CombatSystem → OwnershipMgr → TTLManager → ItemManager → BagManager → SpatialIndex → StateTracker → Server.js → ClientGame → InteractionMgr → AnalyticsSystem
```

This enhanced bag system provides enterprise-grade lifecycle management, sophisticated ownership models, efficient spatial indexing, and comprehensive analytics while maintaining the performance characteristics required for real-time multiplayer gaming. The multi-tiered ownership system ensures fair loot distribution, while the spatial indexing and state tracking enable scalable performance even with hundreds of concurrent bags.