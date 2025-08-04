# Drop System Documentation

## Overview
The Drop System manages loot generation when enemies die, implementing probabilistic drop tables with bag color priority mechanics. It integrates tightly with the Enemy System, Item System, and Bag System to provide the complete loot experience.

## Core Architecture

### 1. Drop Table Evaluation (`/src/DropSystem.js`)

#### **Core Function: `rollDropTable(dropTable, rng=Math.random)`**

**Purpose**: Evaluates enemy drop table and determines which items to drop
**Parameters**:
- `dropTable`: Array of drop definitions
- `rng`: Optional RNG function (defaults to Math.random)

**Drop Definition Structure**:
```javascript
{
  "id": 1001,           // Item definition ID
  "prob": 0.3,          // Drop probability (0.0-1.0)
  "bagType": 0,         // Bag color type (0-6)
  "soulbound": false    // Whether item is soulbound (optional)
}
```

**Process**:
```javascript
1. Iterate through each drop definition
2. Roll RNG against probability threshold
3. If roll succeeds, add item ID to results
4. Track highest bag type from successful rolls
5. Return {items: [itemIds], bagType: maxBagType}
```

**Example Usage**:
```javascript
const dropTable = [
  { id: 1001, prob: 0.3, bagType: 0 },   // 30% white bag
  { id: 1004, prob: 0.1, bagType: 1 },   // 10% brown bag  
  { id: 1007, prob: 0.02, bagType: 2 }   // 2% purple bag
];

const result = rollDropTable(dropTable);
// result = { items: [1001, 1004], bagType: 1 }
```

#### **Bag Color Priority System**

**Color Hierarchy** (`BAG_COLOUR_PRIORITY`):
```javascript
[0, 1, 2, 3, 4, 5, 6] // White → Brown → Purple → Orange → Cyan → Blue → Red
```

**Bag Type to Sprite Mapping** (`getBagColourSprite(bagType)`):
```javascript
0: 'items_sprite_lootbag_white'    // Common drops
1: 'items_sprite_lootbag_brown'    // Uncommon drops
2: 'items_sprite_lootbag_purple'   // Rare drops
3: 'items_sprite_lootbag_orange'   // Epic drops
4: 'items_sprite_lootbag_cyan'     // Legendary drops
5: 'items_sprite_lootbag_blue'     // Special drops
6: 'items_sprite_lootbag_red'      // Ultimate drops
```

**Priority Logic**:
- Multiple items can drop from one enemy death
- Final bag color = highest priority from all dropped items
- Higher numbers = higher priority bags

### 2. Integration with Enemy System

#### **Death Event Handling** (`EnemyManager.js:547-560`)

**Function**: `onDeath(index, killedBy)`
**Integration Points**:

```javascript
// Called when enemy health <= 0
this.onDeath(index, killedBy);

// Implementation in EnemyManager
onDeath(index, killedBy) {
  // 1. Get enemy template and drop table
  const enemyType = this.type[index];
  const template = this.enemyTypes[enemyType] || {};
  const dropTable = template.dropTable || [];
  
  // 2. Roll drops using DropSystem
  const {items, bagType} = rollDropTable(dropTable);
  if(items.length === 0) return;
  
  // 3. Create item instances via ItemManager
  const itemInstanceIds = items.map(defId => {
    const inst = globalThis.itemManager.createItem(defId, {
      x: this.x[index], 
      y: this.y[index]
    });
    return inst?.id;
  }).filter(Boolean);
  
  // 4. Spawn loot bag via BagManager
  if(itemInstanceIds.length === 0) return;
  this._bagManager.spawnBag(
    this.x[index],           // Position
    this.y[index], 
    itemInstanceIds,         // Item instance IDs
    this.worldId[index],     // World context
    300,                     // TTL (5 minutes)
    bagType                  // Bag color
  );
}
```

#### **Enemy Type Drop Tables**

**Hardcoded Example** (`EnemyManager.js:116-120`):
```javascript
{
  id: 0,
  name: 'Goblin',
  // ... other properties
  dropTable: [
    { id: 1001, prob: 0.3, bagType: 0 },   // Ironveil Sword (30% white)
    { id: 1004, prob: 0.1, bagType: 1 },   // Greenwatch Sword (10% brown)
    { id: 1007, prob: 0.02, bagType: 2 }   // Skysteel Sword (2% purple)
  ]
}
```

**JSON-Loaded Enemies** (`EntityDatabase.js`):
- Drop tables defined in external JSON files
- Loaded via `_loadExternalEnemyDefs()`
- Allows designers to modify loot without code changes

### 3. Integration with Item System

#### **Item Creation Pipeline**

**Step 1**: Drop table evaluation produces item definition IDs
**Step 2**: ItemManager creates item instances with properties
**Step 3**: Item instances get unique instance IDs for tracking

```javascript
// From dropTable roll
const items = [1001, 1004]; // Item definition IDs

// Create instances via ItemManager
const itemInstanceIds = items.map(defId => {
  const inst = globalThis.itemManager.createItem(defId, {
    x: enemyX,
    y: enemyY,
    // Optional: ownerId for soulbound items
  });
  return inst?.id; // Instance ID, not definition ID
}).filter(Boolean);
```

#### **Item Definition Requirements**

**Required Properties** for drop table items:
```javascript
{
  id: 1001,                    // Must match drop table ID
  type: ItemType.WEAPON,       // Item category
  rarity: ItemRarity.COMMON,   // Visual rarity
  spriteSheet: "items",        // Sprite sheet reference
  spriteX: 32,                 // Sprite coordinates
  spriteY: 64,
  spriteWidth: 32,             // Sprite dimensions
  spriteHeight: 32,
  baseStats: {                 // Item statistics
    attack: 10,
    defense: 5
  }
}
```

### 4. Integration with Bag System

#### **Bag Spawning Process**

**Function**: `BagManager.spawnBag(x, y, itemIds, worldId, ttl, bagType, owners)`

**Parameters from Drop System**:
- `x, y`: Enemy death position
- `itemIds`: Array of item instance IDs from ItemManager
- `worldId`: World context from enemy
- `ttl`: Time-to-live (default 300 seconds)
- `bagType`: Color priority from drop evaluation
- `owners`: Optional player ownership (for soulbound items)

**Bag Data Structure** (`BagManager.js:19-29`):
```javascript
// SoA layout for performance
this.id = new Array(maxBags);           // "bag_1", "bag_2", etc
this.x = new Float32Array(maxBags);     // World position
this.y = new Float32Array(maxBags);     
this.creationTime = new Float32Array(maxBags); // Spawn timestamp
this.lifetime = new Float32Array(maxBags);     // TTL duration
this.itemSlots = new Array(maxBags);           // Array<itemInstanceId>
this.bagType = new Uint8Array(maxBags);       // Color (0-6)
this.owners = new Array(maxBags);              // Array<clientId>
this.worldId = new Array(maxBags);             // World context
```

#### **Bag Visibility System**

**Ownership Logic** (`BagManager.js:116-131`):
```javascript
getBagsData(filterWorldId = null, viewerId = null) {
  const out = [];
  for (let i = 0; i < this.bagCount; i++) {
    // World filtering
    if (filterWorldId && this.worldId[i] !== filterWorldId) continue;
    
    // Visibility check: if owners defined, ensure viewer included
    if(this.owners[i] && viewerId && !this.owners[i].includes(viewerId)) continue;
    
    out.push({
      id: this.id[i],
      x: this.x[i],
      y: this.y[i],
      bagType: this.bagType[i],
      items: this.itemSlots[i],
    });
  }
  return out;
}
```

### 5. Network Integration

#### **Server-Side Data Flow**

**Drop Event Trigger** (`Server.js` collision handling):
```javascript
// When bullet hits enemy and kills it
if (enemyHealth <= 0) {
  // Triggers EnemyManager.onDeath()
  // → DropSystem.rollDropTable()
  // → ItemManager.createItem() for each drop
  // → BagManager.spawnBag()
}
```

**Bag Broadcasting** (`Server.js:796-850`):
```javascript
// Include bags in world updates
const bags = ctx.bagMgr.getBagsData(mapId, client.id);
const visibleBags = bags.filter(bag => {
  const dx = bag.x - playerX;
  const dy = bag.y - playerY;
  return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
});

sendToClient(client.socket, MessageType.WORLD_UPDATE, {
  enemies: visibleEnemies,
  bullets: visibleBullets,
  bags: visibleBags,           // Include bag data
  // ... other data
});
```

#### **Client-Side Handling**

**WORLD_UPDATE Message Handler**:
```javascript
this.handlers[MessageType.WORLD_UPDATE] = (data) => {
  if (this.game.updateWorld) {
    this.game.updateWorld(
      data.enemies, 
      data.bullets, 
      data.players, 
      data.objects, 
      data.bags    // Client receives bag updates
    );
  }
};
```

### 6. Performance Optimizations

#### **Drop Calculation Efficiency**
- Single pass through drop table per enemy death
- Early exit if no items drop
- Minimal object allocation during evaluation

#### **Memory Management**
- Item instances reuse definition data via references
- Bag auto-cleanup via TTL system
- SoA layout for cache-friendly bag storage

#### **Network Optimization**
- Bags included in existing WORLD_UPDATE packets
- Interest management filters bags by distance
- Binary serialization for item data

### 7. Configuration and Extensibility

#### **Drop Table Sources**

**1. Hardcoded in EnemyManager**:
```javascript
// Direct definition in enemyTypes array
dropTable: [
  { id: 1001, prob: 0.3, bagType: 0 }
]
```

**2. EntityDatabase JSON**:
```javascript
// public/assets/entities/enemies.json
{
  "id": "red_demon",
  "drops": [
    { "item": 1005, "probability": 0.15, "bag": 1 }
  ]
}
```

**3. Custom Enemy Definitions**:
```javascript
// public/assets/enemies-custom/*.enemy.json
{
  "id": "boss_dragon",
  "dropTable": [
    { "id": 2001, "prob": 1.0, "bagType": 6 }  // Guaranteed red bag
  ]
}
```

#### **Bag Color Configuration**

**Adding New Bag Types**:
1. Extend `BAG_COLOUR_PRIORITY` array
2. Add sprite mapping in `getBagColourSprite()`
3. Update client-side rendering code

### 8. Advanced Probability Mechanics and Statistical Analysis

#### **Sophisticated Drop Probability System**

The current implementation supports basic fixed probabilities, but the system is designed for extensibility:

**Current Implementation** (`DropSystem.js:16-27`):
```javascript
export function rollDropTable(dropTable, rng=Math.random){
  if(!Array.isArray(dropTable) || dropTable.length===0) return {items:[],bagType:0};
  const rolled=[];
  let maxBag=0;
  dropTable.forEach(def=>{
    // Simple probability check - each item rolled independently
    if(rng()< (def.prob ?? 1)){
      rolled.push(def.id);
      // Track highest bag type for final bag color
      if((def.bagType??0) > maxBag) maxBag = def.bagType??0;
    }
  });
  return {items:rolled, bagType:maxBag};
}
```

**Enhanced Probability Models**:

```javascript
// Advanced drop table with conditional probability
const advancedDropTable = [
  {
    id: 1001,
    prob: 0.3,
    bagType: 0,
    conditions: {
      minPlayerLevel: 10,
      maxDropsPerPlayer: 5,
      cooldownMinutes: 30
    }
  },
  {
    id: 1007,
    prob: 0.02,
    bagType: 2,
    scalingFactors: {
      bossHealthPercent: 0.8,  // Higher chance if boss killed at full health
      playerCount: 1.2,        // Scales with number of participants
      damageDonePercent: 0.5   // Scales with player contribution
    }
  },
  {
    id: 2001,
    prob: 0.001,
    bagType: 6,
    guaranteedConditions: {
      firstKillOfDay: true,     // Guaranteed on first daily kill
      perfectExecution: true    // No damage taken during fight
    }
  }
];

// Enhanced probability calculation
function calculateDynamicProbability(dropDef, context) {
  let baseProb = dropDef.prob || 0;
  
  // Apply guaranteed conditions
  if (dropDef.guaranteedConditions) {
    for (const [condition, required] of Object.entries(dropDef.guaranteedConditions)) {
      if (context[condition] === required) {
        return 1.0; // Guaranteed drop
      }
    }
  }
  
  // Apply scaling factors
  if (dropDef.scalingFactors) {
    for (const [factor, multiplier] of Object.entries(dropDef.scalingFactors)) {
      const contextValue = context[factor] || 0;
      baseProb *= (1 + (contextValue * multiplier));
    }
  }
  
  // Apply conditions
  if (dropDef.conditions) {
    for (const [condition, requirement] of Object.entries(dropDef.conditions)) {
      if (!context[condition] || context[condition] < requirement) {
        return 0; // Condition not met
      }
    }
  }
  
  return Math.min(baseProb, 1.0); // Cap at 100%
}
```

#### **Statistical Drop Analysis and Balancing**

**Drop Rate Monitoring System**:
```javascript
class DropStatistics {
  constructor() {
    this.dropHistory = new Map(); // enemyType -> drop records
    this.playerDrops = new Map();  // playerId -> recent drops
    this.rarityDistribution = new Map(); // bagType -> count
  }
  
  recordDrop(enemyType, droppedItems, killedBy) {
    const timestamp = Date.now();
    
    // Record for enemy type analysis
    if (!this.dropHistory.has(enemyType)) {
      this.dropHistory.set(enemyType, []);
    }
    
    this.dropHistory.get(enemyType).push({
      timestamp,
      items: droppedItems,
      killedBy,
      context: this.captureContext()
    });
    
    // Track player-specific drops for anti-farming protection
    if (!this.playerDrops.has(killedBy)) {
      this.playerDrops.set(killedBy, []);
    }
    
    const playerHistory = this.playerDrops.get(killedBy);
    playerHistory.push({ timestamp, enemyType, items: droppedItems });
    
    // Maintain sliding window (last 24 hours)
    const dayAgo = timestamp - (24 * 60 * 60 * 1000);
    playerHistory.splice(0, playerHistory.findIndex(record => record.timestamp >= dayAgo));
    
    // Update rarity distribution
    droppedItems.forEach(item => {
      const bagType = this.getBagTypeForItem(item);
      this.rarityDistribution.set(bagType, (this.rarityDistribution.get(bagType) || 0) + 1);
    });
  }
  
  // Anti-farming: detect unusual drop patterns
  detectSuspiciousActivity(playerId) {
    const playerHistory = this.playerDrops.get(playerId) || [];
    const recentDrops = playerHistory.filter(drop => 
      Date.now() - drop.timestamp < (60 * 60 * 1000) // Last hour
    );
    
    // Check for excessive rare drops
    const rareDrops = recentDrops.filter(drop => 
      drop.items.some(item => this.getBagTypeForItem(item) >= 4)
    );
    
    if (rareDrops.length > 3) {
      console.warn(`[DropStatistics] Suspicious activity detected for player ${playerId}: ${rareDrops.length} rare drops in 1 hour`);
      return true;
    }
    
    // Check for rapid-fire kills of same enemy type
    const enemyKillCounts = new Map();
    recentDrops.forEach(drop => {
      enemyKillCounts.set(drop.enemyType, (enemyKillCounts.get(drop.enemyType) || 0) + 1);
    });
    
    for (const [enemyType, count] of enemyKillCounts) {
      if (count > 20) { // More than 20 kills of same enemy in 1 hour
        console.warn(`[DropStatistics] Potential farming detected: ${count} ${enemyType} kills by ${playerId}`);
        return true;
      }
    }
    
    return false;
  }
  
  // Generate statistical report for balancing
  generateBalanceReport(timeWindow = 24 * 60 * 60 * 1000) {
    const cutoff = Date.now() - timeWindow;
    const report = {
      totalDrops: 0,
      enemyDropRates: new Map(),
      rarityDistribution: new Map(),
      playerDistribution: new Map(),
      recommendations: []
    };
    
    // Analyze drop rates by enemy type
    for (const [enemyType, history] of this.dropHistory) {
      const recentDrops = history.filter(drop => drop.timestamp >= cutoff);
      const dropCount = recentDrops.length;
      const avgItemsPerDrop = recentDrops.reduce((sum, drop) => sum + drop.items.length, 0) / dropCount || 0;
      
      report.enemyDropRates.set(enemyType, {
        totalKills: dropCount,
        avgItemsPerDrop,
        rarityBreakdown: this.analyzeRarityBreakdown(recentDrops)
      });
      
      report.totalDrops += dropCount;
    }
    
    // Generate balancing recommendations
    report.recommendations = this.generateRecommendations(report);
    
    return report;
  }
  
  generateRecommendations(report) {
    const recommendations = [];
    
    // Check for enemies that drop too frequently
    for (const [enemyType, stats] of report.enemyDropRates) {
      if (stats.avgItemsPerDrop > 2.5) {
        recommendations.push({
          type: 'REDUCE_DROP_RATE',
          enemyType,
          current: stats.avgItemsPerDrop,
          suggested: 2.0,
          reason: 'Excessive drop rate may devalue loot'
        });
      }
      
      if (stats.avgItemsPerDrop < 0.5) {
        recommendations.push({
          type: 'INCREASE_DROP_RATE',
          enemyType,
          current: stats.avgItemsPerDrop,
          suggested: 1.0,
          reason: 'Drop rate too low, may frustrate players'
        });
      }
    }
    
    return recommendations;
  }
}
```

#### **Dynamic Bag Color Assignment with Rarity Economics**

**Enhanced Bag Color Logic**:
```javascript
// Economic balance for bag colors
const BAG_ECONOMICS = {
  // Target percentage distribution of bag colors
  targetDistribution: {
    0: 0.70,  // White: 70% of all bags
    1: 0.15,  // Brown: 15%
    2: 0.08,  // Purple: 8%
    3: 0.04,  // Orange: 4%
    4: 0.02,  // Cyan: 2%
    5: 0.008, // Blue: 0.8%
    6: 0.002  // Red: 0.2%
  },
  
  // Economic value per bag color (for balancing)
  economicValue: {
    0: 1,     // White
    1: 5,     // Brown
    2: 25,    // Purple
    3: 100,   // Orange
    4: 500,   // Cyan
    5: 2500,  // Blue
    6: 10000  // Red
  }
};

// Dynamic bag color adjustment based on current distribution
function adjustBagColorProbabilities(currentDistribution, targetDistribution) {
  const adjustments = new Map();
  
  for (const [bagType, currentRatio] of Object.entries(currentDistribution)) {
    const targetRatio = targetDistribution[bagType] || 0;
    const deviation = currentRatio - targetRatio;
    
    if (Math.abs(deviation) > 0.02) { // 2% tolerance
      // If we have too many of this bag type, reduce probability
      // If we have too few, increase probability
      const adjustment = deviation > 0 ? 0.8 : 1.25;
      adjustments.set(parseInt(bagType), adjustment);
    }
  }
  
  return adjustments;
}

// Enhanced bag color selection
function selectBagColorWithEconomics(droppedItems, enemyType, context) {
  let maxBagType = 0;
  let totalEconomicValue = 0;
  
  // Calculate base bag type and economic value
  droppedItems.forEach(itemId => {
    const itemDef = getItemDefinition(itemId);
    const itemBagType = itemDef.bagType || 0;
    const itemValue = itemDef.economicValue || BAG_ECONOMICS.economicValue[itemBagType];
    
    maxBagType = Math.max(maxBagType, itemBagType);
    totalEconomicValue += itemValue;
  });
  
  // Apply economic adjustments
  const currentDistribution = getRecentBagDistribution();
  const adjustments = adjustBagColorProbabilities(currentDistribution, BAG_ECONOMICS.targetDistribution);
  
  // Consider upgrading bag color based on economic value
  if (totalEconomicValue > BAG_ECONOMICS.economicValue[maxBagType + 1]) {
    const upgradeChance = Math.min(totalEconomicValue / BAG_ECONOMICS.economicValue[maxBagType + 1], 1.0);
    if (Math.random() < upgradeChance * 0.1) { // 10% base upgrade chance
      maxBagType = Math.min(maxBagType + 1, 6);
    }
  }
  
  // Apply distribution adjustments
  const adjustment = adjustments.get(maxBagType) || 1.0;
  if (adjustment < 1.0 && Math.random() > adjustment) {
    // Downgrade bag color to maintain distribution
    maxBagType = Math.max(maxBagType - 1, 0);
  }
  
  return maxBagType;
}
```

#### **Soulbound System and Ownership Mechanics**

**Advanced Soulbound Logic**:
```javascript
class SoulboundManager {
  constructor() {
    this.damageContributions = new Map(); // enemyId -> Map<playerId, damage>
    this.soulboundThresholds = {
      DAMAGE_PERCENT: 0.1,    // Must deal 10% of enemy's HP
      TIME_WINDOW: 30000,     // Must have dealt damage in last 30 seconds
      MIN_PARTICIPANTS: 1     // Minimum players to qualify for soulbound
    };
  }
  
  recordDamage(enemyId, playerId, damage) {
    if (!this.damageContributions.has(enemyId)) {
      this.damageContributions.set(enemyId, new Map());
    }
    
    const enemyDamage = this.damageContributions.get(enemyId);
    const currentDamage = enemyDamage.get(playerId) || 0;
    enemyDamage.set(playerId, currentDamage + damage);
  }
  
  calculateSoulboundEligibility(enemyId, enemyMaxHealth) {
    const damageMap = this.damageContributions.get(enemyId);
    if (!damageMap) return [];
    
    const eligiblePlayers = [];
    const minDamageRequired = enemyMaxHealth * this.soulboundThresholds.DAMAGE_PERCENT;
    
    for (const [playerId, damage] of damageMap) {
      if (damage >= minDamageRequired) {
        eligiblePlayers.push(playerId);
      }
    }
    
    // If too few participants, make non-soulbound
    if (eligiblePlayers.length < this.soulboundThresholds.MIN_PARTICIPANTS) {
      return [];
    }
    
    return eligiblePlayers;
  }
  
  cleanupEnemy(enemyId) {
    this.damageContributions.delete(enemyId);
  }
}

// Enhanced drop table with soulbound rules
const advancedDropTableWithSoulbound = [
  {
    id: 1001,
    prob: 0.3,
    bagType: 0,
    soulbound: false  // Always public
  },
  {
    id: 1007,
    prob: 0.02,
    bagType: 2,
    soulbound: true,  // Always soulbound
    soulboundRules: {
      minDamagePercent: 0.05,  // Must deal 5% damage
      maxEligiblePlayers: 3    // Max 3 players can see this drop
    }
  },
  {
    id: 2001,
    prob: 0.001,
    bagType: 6,
    soulbound: 'conditional',
    soulboundConditions: {
      playerCount: { operator: '>', value: 1 },      // Soulbound only if multiple players
      damageContribution: { operator: '>', value: 0.2 } // Must deal 20% damage to qualify
    }
  }
];
```

### 9. Performance Optimization and Caching

#### **Drop Calculation Caching**

```javascript
class OptimizedDropSystem {
  constructor() {
    this.dropTableCache = new Map();     // enemyType -> compiled drop table
    this.probabilityCache = new Map();   // dropTableHash -> pre-calculated ranges
    this.recentDrops = new LRUCache(1000); // Cache recent drop results
  }
  
  // Pre-compile drop tables for faster runtime execution
  compileDropTable(enemyType, dropTable) {
    const compiled = {
      totalWeight: 0,
      ranges: [],
      items: [],
      bagTypes: []
    };
    
    let cumulativeWeight = 0;
    dropTable.forEach(drop => {
      const weight = drop.prob * 1000; // Convert to integer for faster math
      cumulativeWeight += weight;
      
      compiled.ranges.push(cumulativeWeight);
      compiled.items.push(drop.id);
      compiled.bagTypes.push(drop.bagType || 0);
    });
    
    compiled.totalWeight = cumulativeWeight;
    this.dropTableCache.set(enemyType, compiled);
    return compiled;
  }
  
  // Ultra-fast drop rolling using pre-compiled tables
  rollDropTableOptimized(enemyType, compiledTable, rng = Math.random) {
    const cacheKey = `${enemyType}_${Date.now() >> 10}`; // 1-second cache granularity
    
    if (this.recentDrops.has(cacheKey)) {
      return this.recentDrops.get(cacheKey);
    }
    
    const rolled = [];
    let maxBagType = 0;
    
    // Binary search for faster range finding
    for (let i = 0; i < compiledTable.ranges.length; i++) {
      const roll = rng() * compiledTable.totalWeight;
      
      // Binary search to find which item was rolled
      let left = 0, right = compiledTable.ranges.length - 1;
      while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (roll <= compiledTable.ranges[mid]) {
          rolled.push(compiledTable.items[mid]);
          maxBagType = Math.max(maxBagType, compiledTable.bagTypes[mid]);
          break;
        }
        left = mid + 1;
      }
    }
    
    const result = { items: rolled, bagType: maxBagType };
    this.recentDrops.set(cacheKey, result);
    return result;
  }
}
```

### 10. Integration Points Summary

#### **Input Dependencies**
- **EnemyManager**: Enemy death events, position, world context, damage contributions
- **ItemManager**: Item definition registry, instance creation, economic values
- **BagManager**: Bag spawning and management, ownership controls
- **StatisticsManager**: Drop rate monitoring, anti-farming detection
- **SoulboundManager**: Damage tracking, eligibility calculation

#### **Output Products**
- **Item Instances**: Created with unique IDs, properties, and ownership
- **Loot Bags**: Spawned with economically-balanced colors and contents
- **Network Data**: Bag information with ownership filtering for client updates
- **Statistical Data**: Drop rate analytics for balancing and fraud detection

#### **Enhanced Data Flow**
```
Enemy Death → Damage Analysis → Drop Table Roll → Economic Adjustment → Item Creation → Soulbound Check → Bag Spawn → Network Sync
     ↓              ↓                ↓                 ↓                ↓              ↓             ↓           ↓
EnemyManager → SoulboundMgr → DropSystem → EconomicBalancer → ItemManager → OwnershipMgr → BagManager → Server.js
                    ↓                        ↓
               StatisticsMgr          BalanceReporter
```

This enhanced drop system provides sophisticated probability mechanics, economic balancing, anti-farming protection, and comprehensive statistical analysis while maintaining the classic ROTMG experience and supporting modern scalable game development practices.