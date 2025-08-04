# Enemy System Documentation

## Overview
The enemy system is a comprehensive, high-performance game entity management system that forms the backbone of ROTMG-DEMO's gameplay. It combines Structure of Arrays (SoA) data layout for optimal memory access patterns, sophisticated AI behavior systems, and efficient network synchronization to support 1000+ concurrent enemies with smooth 60 FPS performance.

### Key Innovations
- **Structure of Arrays (SoA)** layout for cache-efficient data access
- **Comprehensive Behavior System** with 30+ AI behavior components
- **Multi-World Context Support** with isolated enemy populations
- **Advanced Network Synchronization** with interest management
- **Integration with LLM Boss System** for dynamic AI behaviors
- **Binary-Optimized Network Protocol** for minimal bandwidth usage

## Core Architecture

### 1. Backend: EnemyManager (`/src/EnemyManager.js`)

#### **Data Structure (SoA Layout)**
```javascript
class EnemyManager {
  constructor(maxEnemies = 1000) {
    // Core identification
    this.id = new Array(maxEnemies);              // "enemy_1", "enemy_2", etc
    this.nextEnemyId = 1;                         // Auto-incrementing ID counter
    
    // Spatial data
    this.x = new Float32Array(maxEnemies);        // World X position
    this.y = new Float32Array(maxEnemies);        // World Y position  
    this.width = new Float32Array(maxEnemies);    // Collision width
    this.height = new Float32Array(maxEnemies);   // Collision height
    
    // Combat properties
    this.health = new Float32Array(maxEnemies);       // Current HP
    this.maxHealth = new Float32Array(maxEnemies);    // Maximum HP
    this.damage = new Float32Array(maxEnemies);       // Bullet damage
    this.bulletSpeed = new Float32Array(maxEnemies);  // Projectile speed
    
    // Behavior properties
    this.type = new Uint8Array(maxEnemies);           // Enemy type index (0-4)
    this.moveSpeed = new Float32Array(maxEnemies);    // Movement speed
    this.chaseRadius = new Float32Array(maxEnemies);  // Detection range
    this.shootRange = new Float32Array(maxEnemies);   // Attack range
    this.cooldown = new Float32Array(maxEnemies);     // Attack cooldown
    this.currentCooldown = new Float32Array(maxEnemies); // Current timer
    
    // Visual effects
    this.isFlashing = new Uint8Array(maxEnemies);     // Hit flash state
    this.flashTimer = new Float32Array(maxEnemies);   // Flash duration
    this.isDying = new Uint8Array(maxEnemies);        // Death animation state
    this.deathTimer = new Float32Array(maxEnemies);   // Death animation timer
    
    // World management
    this.worldId = new Array(maxEnemies);             // Map/world assignment
  }
}
```

#### **Core Functions**

##### `spawnEnemy(type, x, y, worldId='default')`
**Purpose**: Creates a new enemy instance
**Process**:
```javascript
1. Capacity check (max 1000 enemies)
2. Type validation (0 ≤ type < enemyTypes.length)
3. Get template from enemyTypes[type]
4. Assign unique ID: `enemy_${nextEnemyId++}`
5. Populate SoA arrays with template values
6. Initialize behavior system via behaviorSystem.initBehavior()
7. Register with world context
8. Return enemyId
```

**Integration Points**:
- Called by `spawnMapEnemies()` in Server.js:585
- Triggered during map loading/switching
- Used by behavior system for spawning minions

##### `spawnEnemyById(entityId, x, y, worldId)`
**Purpose**: Spawns enemy by JSON entity ID rather than type index
**Process**:
```javascript
1. Lookup type index via enemyIdToTypeIndex map
2. Fallback to type 0 if entityId not found
3. Delegate to spawnEnemy(typeIdx, x, y, worldId)
```

##### `update(deltaTime, bulletManager, target, mapManager)`
**Purpose**: Main update loop for all enemies
**Process**:
```javascript
1. Iterate through all active enemies (0 to enemyCount)
2. Update behavior system for each enemy
3. Process visual effects (flash, death animations)
4. Handle enemy death and cleanup
5. Return total active enemy count
```

**Integration Points**:
- Called by Server.js:743 in main game loop
- Integrates with BehaviorSystem for AI
- Coordinates with CollisionManager for hit detection

##### `getEnemiesData(filterWorldId=null)`
**Purpose**: Serializes enemy data for network transmission
**Returns**:
```javascript
[{
  id: "enemy_123",
  x: 45.2, y: 23.8,
  width: 1.0, height: 1.0,
  type: 2,
  spriteName: "goblin",
  health: 85, maxHealth: 100,
  isFlashing: 0, isDying: 0,
  deathStage: 0,
  worldId: "map_1"
}, ...]
```

**Integration Points**:
- Called by Server.js:796 for WORLD_UPDATE messages
- Called by Server.js:970 for initial ENEMY_LIST
- Filtered by worldId for multi-map support

##### `onDeath(index, killedBy)`
**Purpose**: Handles enemy death and loot generation
**Process**:
```javascript
1. Get enemy type template and dropTable
2. Roll drops using DropSystem.rollDropTable()
3. Create item instances via ItemManager
4. Spawn loot bag via BagManager
5. Clean up enemy entity
```

**Integration Points**:
- Triggered by collision system when health ≤ 0
- Integrates with DropSystem for loot generation
- Coordinates with BagManager for loot bag creation

#### **Enemy Type System**

##### Hardcoded Types (`EnemyManager.js:112-175`)
```javascript
this.enemyTypes = [
  {
    id: 0, name: "Basic Enemy",
    maxHealth: 100, speed: 2, damage: 10,
    shootRange: 8, shootCooldown: 1.5,
    dropTable: [
      { id: 1001, prob: 0.3, bagType: 0 },   // 30% white bag
      { id: 1004, prob: 0.1, bagType: 1 },   // 10% brown bag
      { id: 1007, prob: 0.02, bagType: 2 }   // 2% purple bag
    ]
  },
  // ... 4 more types (Archer, Tank, Mage, Boss)
];
```

##### JSON Entity Loading
- **Source**: `/public/assets/entities/enemies.json`
- **Loader**: `_loadExternalEnemyDefs()`
- **Mapping**: `enemyIdToTypeIndex` for ID→type lookup

##### Custom Enemy Definitions
- **Source**: `*.enemy.json` files in `/public/assets/enemies-custom/`
- **Loader**: `EnemyDefinitionLoader.js`
- **Features**: Behavior trees, schema validation

### 2. Frontend: ClientEnemyManager (`/public/src/game/ClientEnemyManager.js`)

#### **Data Structure**
```javascript
class ClientEnemyManager {
  constructor(maxEnemies = 1000) {
    // Mirror server SoA structure
    this.id = new Array(maxEnemies);
    this.x = new Float32Array(maxEnemies);
    this.y = new Float32Array(maxEnemies);
    this.health = new Float32Array(maxEnemies);
    
    // Client-specific rendering data
    this.sprite = new Array(maxEnemies);           // Sprite references
    this.spriteName = new Array(maxEnemies);       // Sprite identifiers
    this.animFrame = new Uint8Array(maxEnemies);   // Animation frame
    this.animTime = new Float32Array(maxEnemies);  // Animation timer
    
    // Visual effects
    this.flashTime = new Float32Array(maxEnemies); // Hit flash effect
    this.deathTime = new Float32Array(maxEnemies); // Death animation
    
    // Movement interpolation
    this.prevX = new Float32Array(maxEnemies);     // Previous position
    this.prevY = new Float32Array(maxEnemies);
    this.targetX = new Float32Array(maxEnemies);   // Target position
    this.targetY = new Float32Array(maxEnemies);
    this.interpTime = new Float32Array(maxEnemies); // Lerp timer
  }
}
```

#### **Key Functions**

##### `setEnemies(enemiesData)`
**Purpose**: Updates client state from server ENEMY_LIST message
**Process**:
```javascript
1. Clear existing enemy data
2. Populate SoA arrays from server data
3. Initialize rendering properties
4. Set up sprites and animations
```

##### `updateEnemies(enemiesData)`
**Purpose**: Updates client state from server WORLD_UPDATE message
**Process**:
```javascript
1. Merge new enemy data with existing
2. Handle enemy additions/removals
3. Update positions with interpolation
4. Trigger visual effects
```

##### `update(deltaTime)`
**Purpose**: Client-side animation and interpolation
**Process**:
```javascript
1. Update animation frames and timers
2. Process visual effects (flash, death)
3. Interpolate movement for smooth rendering
4. Update sprite states
```

### 3. Network Integration

#### **Server → Client Data Flow**

##### Initial State (`Server.js:970`)
```javascript
// Send complete enemy list on client connect
const enemies = getWorldCtx(newClient.mapId).enemyMgr.getEnemiesData(newClient.mapId);
sendToClient(socket, MessageType.ENEMY_LIST, {
  enemies,
  timestamp: Date.now()
});
```

##### Continuous Updates (`Server.js:796-850`)
```javascript
// Send filtered enemy updates in main game loop
const enemies = ctx.enemyMgr.getEnemiesData(mapId);
const visibleEnemies = enemies.filter(e => {
  const dx = e.x - playerX;
  const dy = e.y - playerY;
  return (dx * dx + dy * dy) <= UPDATE_RADIUS_SQ;
});

sendToClient(client.socket, MessageType.WORLD_UPDATE, {
  enemies: visibleEnemies.slice(0, MAX_ENTITIES_PER_PACKET),
  // ... other data
});
```

#### **Client Message Handlers**

##### ENEMY_LIST Handler
```javascript
this.handlers[MessageType.ENEMY_LIST] = (data) => {
  if (this.game.setEnemies && data.enemies) {
    this.game.setEnemies(data.enemies);
  }
};
```

##### WORLD_UPDATE Handler
```javascript
this.handlers[MessageType.WORLD_UPDATE] = (data) => {
  if (this.game.updateWorld) {
    this.game.updateWorld(data.enemies, data.bullets, data.players, data.objects, data.bags);
  }
};
```

### 4. Advanced Behavior System Integration

The behavior system is one of the most sophisticated parts of the enemy architecture, providing 30+ distinct AI behaviors that can be composed into complex enemy patterns.

#### **Complete Behavior Catalog** (`/src/Behaviors.js`)

**Movement Behaviors**:
- `Wander` - Random movement with configurable direction changes
- `Chase` - Direct pursuit of target with minimum distance control
- `RunAway` - Flee from target with configurable distance thresholds
- `Orbit` - Circular movement around target with radius and speed control
- `Swirl` - Spiral movement pattern with configurable angular velocity
- `MoveLine` - Linear movement along defined paths
- `MoveTo` - Point-to-point movement with arrival detection
- `BackAndForth` - Oscillating movement between two points
- `Buzz` - Erratic movement with random directional changes
- `StayAbove` - Maintain vertical position relative to target
- `StayBack` - Maintain distance from target with retreat behavior
- `StayCloseToSpawn` - Remain within radius of spawn point
- `ReturnToSpawn` - Active movement back to spawn location

**Combat Behaviors**:
- `Shoot` - Projectile firing with cooldown and accuracy control
- `Charge` - Rapid movement toward target with collision damage
- `Flash` - Teleportation with invulnerability frames
- `Grenade` - Area-of-effect projectile with delayed explosion
- `Aoe` - Immediate area damage around enemy position
- `TalismanAttack` - Multi-projectile burst with spread patterns
- `InvisiToss` - Invisible projectile with surprise timing

**Support Behaviors**:
- `HealSelf` - Self-restoration with cooldown management
- `HealGroup` - Area healing for nearby allies
- `HealEntity` - Targeted healing of specific entities
- `Spawn` - Enemy creation with type and position control
- `SpawnGroup` - Multiple enemy creation with formation patterns
- `RelativeSpawn` - Positional spawning relative to current location

**Advanced Behaviors**:
- `Follow` - Complex following with formation maintenance
- `OldSwirl` - Legacy spiral pattern for compatibility

#### **Behavior System Architecture** (`/src/BehaviorSystem.js`)

```javascript
class BehaviorSystem {
  constructor() {
    this.behaviorTemplates = new Map();     // type -> behavior list
    this.enemyStates = new Map();           // index -> state data
    this.stateTransitions = new Map();      // type -> transition rules
    this.globalCooldowns = new Map();       // behavior -> cooldown
  }
  
  // Initialize behavior for new enemy
  initBehavior(index, type) {
    const template = this.behaviorTemplates.get(type);
    if (!template) {
      console.warn(`No behavior template for enemy type ${type}`);
      return;
    }
    
    this.enemyStates.set(index, {
      currentBehaviors: [...template.behaviors],
      stateData: {},
      lastUpdate: Date.now(),
      transitionCooldown: 0
    });
  }
  
  // Complex behavior update with state management
  updateBehavior(index, enemyManager, bulletManager, target, deltaTime) {
    const state = this.enemyStates.get(index);
    if (!state) return;
    
    // Update cooldowns
    state.transitionCooldown -= deltaTime;
    
    // Execute all active behaviors
    state.currentBehaviors.forEach(behavior => {
      if (this.canExecuteBehavior(behavior, state)) {
        behavior.execute(index, enemyManager, bulletManager, target, deltaTime, state.stateData);
      }
    });
    
    // Check for state transitions
    this.checkStateTransitions(index, enemyManager, target, state);
  }
  
  // Advanced state transition system
  checkStateTransitions(index, enemyManager, target, state) {
    const enemyType = enemyManager.type[index];
    const transitions = this.stateTransitions.get(enemyType);
    
    if (!transitions || state.transitionCooldown > 0) return;
    
    for (const transition of transitions) {
      if (this.evaluateTransitionCondition(transition.condition, index, enemyManager, target)) {
        this.transitionToState(index, transition.targetState, state);
        state.transitionCooldown = transition.cooldown || 1.0;
        break;
      }
    }
  }
}
```

#### **Behavior Composition Examples**

**Aggressive Enemy** (Goblin):
```javascript
{
  behaviors: [
    new Chase(1.2, 0),        // Fast chase with no minimum distance
    new Shoot(1.0, 2.0),      // Regular shooting with 2s cooldown
    new Wander(0.5, 4.0)      // Slow wander when no target
  ],
  transitions: [
    {
      condition: 'target_in_range:5',
      targetState: 'combat',
      behaviors: [new Chase(1.5), new Shoot(1.5, 1.0)]
    },
    {
      condition: 'health_below:0.3',
      targetState: 'retreat',
      behaviors: [new RunAway(2.0, 8.0), new HealSelf(0.1)]
    }
  ]
}
```

**Support Enemy** (Cleric):
```javascript
{
  behaviors: [
    new StayBack(6.0),        // Maintain distance from combat
    new HealGroup(5.0, 3.0),  // Heal allies in 5-tile radius every 3s
    new Orbit(4.0, 0.5)       // Orbit around allies for positioning
  ],
  transitions: [
    {
      condition: 'ally_health_low',
      targetState: 'emergency_heal',
      behaviors: [new MoveTo(), new HealEntity(2.0)]
    }
  ]
}
```

**Boss-Tier Enemy** (Dragon):
```javascript
{
  behaviors: [
    new Charge(2.0, 8.0),     // Powerful charge attacks
    new Aoe(10.0, 5.0),       // Area damage every 5 seconds
    new SpawnGroup(3, 'minion'), // Spawn minions
    new Flash(15.0)           // Teleport every 15 seconds
  ],
  phases: [
    {
      healthThreshold: 1.0,
      behaviors: [new Orbit(8.0, 1.0), new Shoot(1.0, 1.5)]
    },
    {
      healthThreshold: 0.6,
      behaviors: [new Chase(1.5), new Grenade(2.0, 3.0)]
    },
    {
      healthThreshold: 0.3,
      behaviors: [new Swirl(2.0, 2.0), new Aoe(8.0, 2.0)]
    }
  ]
}
```

#### **Dynamic Behavior Registration**

```javascript
// Register behaviors from JSON definitions
registerBehaviorTemplate(enemyType, template) {
  const compiledBehaviors = template.behaviors.map(behaviorDef => {
    switch (behaviorDef.type) {
      case 'chase':
        return new Chase(behaviorDef.speed, behaviorDef.minDistance);
      case 'shoot':
        return new Shoot(behaviorDef.damage, behaviorDef.cooldown);
      case 'orbit':
        return new Orbit(behaviorDef.radius, behaviorDef.speed);
      // ... handle all behavior types
    }
  });
  
  this.behaviorTemplates.set(enemyType, {
    behaviors: compiledBehaviors,
    transitions: template.transitions || [],
    metadata: template.metadata || {}
  });
}

// Load from external JSON
loadBehaviorDefinitions(filePath) {
  const definitions = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  definitions.forEach(def => {
    this.registerBehaviorTemplate(def.enemyType, def.template);
  });
}
```

#### **Behavior State Management**

Each behavior maintains its own state data within the enemy's state object:

```javascript
// Example state data structure
{
  currentBehaviors: [chaseInstance, shootInstance],
  stateData: {
    // Chase behavior state
    chase_lastDirection: { x: 0.5, y: 0.8 },
    chase_stuckTimer: 0.0,
    
    // Shoot behavior state  
    shoot_lastFired: 1234567890,
    shoot_burstCount: 0,
    shoot_targetPrediction: { x: 10.5, y: 15.2 },
    
    // Orbit behavior state
    orbit_angle: 1.57,
    orbit_centerX: 20.0,
    orbit_centerY: 25.0,
    
    // Spawn behavior state
    spawn_lastSpawn: 1234567800,
    spawn_minionCount: 3
  },
  lastUpdate: 1234567890,
  transitionCooldown: 0.5
}
```

#### **Performance Optimizations**

**Behavior Culling**:
```javascript
// Only update behaviors for enemies near players
updateBehavior(index, enemyManager, bulletManager, target, deltaTime) {
  // Distance check for behavior updates
  const dx = enemyManager.x[index] - target.x;
  const dy = enemyManager.y[index] - target.y;
  const distanceSq = dx * dx + dy * dy;
  
  // Skip complex behaviors for distant enemies
  if (distanceSq > BEHAVIOR_UPDATE_RADIUS_SQ) {
    this.updateSimpleBehavior(index, enemyManager, deltaTime);
    return;
  }
  
  // Full behavior update for nearby enemies
  this.updateFullBehavior(index, enemyManager, bulletManager, target, deltaTime);
}
```

**Behavior Batching**:
```javascript
// Batch similar behaviors for SIMD optimization
updateBehaviorsBatched(enemies, deltaTime) {
  // Group enemies by behavior type
  const behaviorGroups = new Map();
  
  enemies.forEach(index => {
    const state = this.enemyStates.get(index);
    state.currentBehaviors.forEach(behavior => {
      const type = behavior.constructor.name;
      if (!behaviorGroups.has(type)) {
        behaviorGroups.set(type, []);
      }
      behaviorGroups.get(type).push({ index, behavior, state });
    });
  });
  
  // Execute behaviors in batches
  behaviorGroups.forEach((group, behaviorType) => {
    this.executeBehaviorBatch(behaviorType, group, deltaTime);
  });
}
```

#### **AI Difficulty Management and Balancing**

The system includes sophisticated difficulty management through the Difficulty Critic system:

**Difficulty Critic Integration** (`/src/critic/DifficultyCritic.js`):
```javascript
const DPS_LIMITS = { easy: 50, mid: 100, hard: 200 };
const UNAVOIDABLE_LIMIT = 0.2;

export function evaluate(kpi, { tier = 'mid', ruleset = 'default' } = {}) {
  const reasons = [];

  // Check if DPS is within acceptable limits for tier
  if (typeof kpi.dpsAvg === 'number' && kpi.dpsAvg > DPS_LIMITS[tier]) {
    reasons.push('dpsTooHigh');
  }

  // Ensure damage remains avoidable
  if (
    typeof kpi.unavoidableDamagePct === 'number' &&
    kpi.unavoidableDamagePct > UNAVOIDABLE_LIMIT
  ) {
    reasons.push('tooMuchUnavoidableDamage');
  }

  return { ok: reasons.length === 0, reasons };
}
```

**Dynamic Difficulty Adjustment**:
```javascript
// BehaviorSystem.js - Adaptive difficulty
class BehaviorSystem {
  adjustDifficultyForPlayer(index, playerSkillLevel) {
    const state = this.enemyStates.get(index);
    if (!state) return;
    
    // Modify behavior parameters based on player skill
    state.currentBehaviors.forEach(behavior => {
      if (behavior instanceof Shoot) {
        // Adjust accuracy and fire rate
        behavior.accuracy *= (0.5 + playerSkillLevel * 0.5);
        behavior.cooldown *= (2.0 - playerSkillLevel);
      } else if (behavior instanceof Chase) {
        // Adjust movement speed
        behavior.speed *= (0.7 + playerSkillLevel * 0.3);
      }
    });
  }
  
  // Headless simulation for pattern validation
  simulatePattern(enemyType, duration = 10) {
    const simulator = new HeadlessSimulator();
    const results = simulator.runSimulation(enemyType, duration);
    
    const kpi = {
      dpsAvg: results.totalDamage / duration,
      unavoidableDamagePct: results.unavoidableDamage / results.totalDamage,
      survivalRate: results.playerSurvivalTime / duration
    };
    
    const evaluation = DifficultyCritic.evaluate(kpi, { tier: enemyType.tier });
    
    if (!evaluation.ok) {
      console.warn(`Pattern ${enemyType.id} failed validation:`, evaluation.reasons);
      this.adjustPatternForBalance(enemyType, evaluation.reasons);
    }
    
    return evaluation;
  }
}
```

#### **Advanced Logging and Telemetry**

**LLM Boss Logging System** (`/src/routes/llmRoutes.js`):
```javascript
// Comprehensive logging for AI boss behavior analysis
const LOG_FILE = path.join(process.cwd(), 'logs', 'boss_llm.jsonl');

// Historical data analysis
router.get('/history', (req, res) => {
  const lines = readLines(50);
  const analysis = {
    totalPlans: lines.length,
    successRate: lines.filter(l => l.success).length / lines.length,
    avgLatency: lines.reduce((sum, l) => sum + (l.latency || 0), 0) / lines.length,
    commonPatterns: analyzePatterns(lines),
    errorFrequency: categorizeErrors(lines)
  };
  
  res.json({ logs: lines, analysis });
});

// Manual rating system for AI behavior quality
router.post('/rate', (req, res) => {
  const { planId, rating, feedback } = req.body || {};
  
  // Store rating with contextual information
  const entry = { 
    type: 'rating_manual', 
    planId, 
    rating, 
    feedback,
    timestamp: Date.now(),
    context: {
      playerCount: getCurrentPlayerCount(),
      bossPhase: getCurrentBossPhase(planId),
      difficultyTier: getCurrentDifficulty()
    }
  };
  
  appendLogEntry(entry);
  
  // Trigger behavior pattern adjustment if rating is low
  if (rating < 3) {
    adjustAIPatterns(planId, feedback);
  }
  
  res.json({ ok: true });
});

function analyzePatterns(logs) {
  const patterns = {};
  logs.forEach(log => {
    if (log.plan && log.plan.actions) {
      const signature = log.plan.actions.map(a => a.type).join('-');
      patterns[signature] = (patterns[signature] || 0) + 1;
    }
  });
  return patterns;
}
```

#### **Complete Integration Testing Framework**

**Behavior Testing Suite**:
```javascript
// test/BehaviorIntegration.test.js
class BehaviorTestSuite {
  constructor() {
    this.enemyManager = new EnemyManager(100);
    this.behaviorSystem = new BehaviorSystem();
    this.testResults = new Map();
  }
  
  // Test individual behavior components
  async testBehaviorIsolation() {
    const behaviors = [
      new Chase(1.0, 0),
      new Shoot(1.0, 2.0),
      new Orbit(5.0, 1.0),
      new Wander(0.5, 3.0)
    ];
    
    for (const behavior of behaviors) {
      const results = await this.runBehaviorTest(behavior);
      this.testResults.set(behavior.constructor.name, results);
    }
  }
  
  // Test behavior composition and state transitions
  async testBehaviorComposition() {
    const compositions = [
      { name: 'AggressiveMelee', behaviors: [new Chase(1.5), new Charge(2.0)] },
      { name: 'SupportCaster', behaviors: [new StayBack(8.0), new HealGroup(5.0)] },
      { name: 'TankBoss', behaviors: [new Aoe(10.0), new Spawn(5, 'minion')] }
    ];
    
    for (const comp of compositions) {
      const enemy = this.spawnTestEnemy(comp.behaviors);
      const results = await this.runCompositionTest(enemy, 30); // 30 second test
      this.testResults.set(comp.name, results);
    }
  }
  
  // Performance stress test
  async testPerformanceAtScale() {
    // Spawn maximum enemy count
    for (let i = 0; i < 1000; i++) {
      this.enemyManager.spawnEnemy(Math.floor(Math.random() * 5), 
                                   Math.random() * 100, 
                                   Math.random() * 100);
    }
    
    const startTime = performance.now();
    
    // Run 1000 update cycles
    for (let cycle = 0; cycle < 1000; cycle++) {
      this.enemyManager.update(0.016, null, { x: 50, y: 50 }, null);
    }
    
    const endTime = performance.now();
    const avgFrameTime = (endTime - startTime) / 1000;
    
    return {
      avgFrameTime,
      fps: 1000 / avgFrameTime,
      memoryUsage: process.memoryUsage(),
      acceptable: avgFrameTime < 16.67 // Must maintain 60 FPS
    };
  }
  
  generateTestReport() {
    const report = {
      timestamp: new Date().toISOString(),
      totalTests: this.testResults.size,
      passed: 0,
      failed: 0,
      details: {}
    };
    
    this.testResults.forEach((result, testName) => {
      if (result.success) {
        report.passed++;
      } else {
        report.failed++;
      }
      report.details[testName] = result;
    });
    
    // Write to file for CI/CD integration
    fs.writeFileSync('test-results/behavior-integration.json', 
                     JSON.stringify(report, null, 2));
    
    return report;
  }
}
```

### 5. World Context Integration

#### **Per-World Enemy Management**
```javascript
// Server.js:267-280
function getWorldCtx(mapId) {
  if (!worldContexts.has(mapId)) {
    const enemyMgr = new EnemyManager(1000);
    // ... other managers
    worldContexts.set(mapId, { enemyMgr, bulletMgr, collMgr, bagMgr });
  }
  return worldContexts.get(mapId);
}
```

#### **World Isolation**
- Each map/world has separate EnemyManager instance
- Enemies are filtered by worldId in network updates
- Prevents cross-world interference

### 6. Performance Optimizations

#### **Structure of Arrays (SoA)**
- Contiguous memory layout for cache efficiency
- Vectorized operations on arrays
- Minimal object allocation

#### **Network Optimization**
- Interest management (only send nearby enemies)
- Entity limits per packet
- Delta compression for position updates

#### **Memory Management**
- Fixed-size arrays to prevent allocation
- Swap-remove pattern for deletions
- Object pooling for temporary data structures

### 7. Integration Points Summary

#### **Server.js Integrations**
- `spawnMapEnemies()` - Map loading spawning
- `updateGame()` - Main game loop updates  
- `broadcastWorldUpdates()` - Network transmission
- `getWorldCtx()` - Multi-world management

#### **Behavior System**
- Enemy AI state machines
- Behavior tree execution
- Complex enemy patterns

#### **Collision System**
- Hit detection
- Damage application
- Death triggering

#### **Drop System**
- Loot table evaluation
- Item creation
- Bag spawning

#### **Network System**
- Binary packet encoding
- Message routing
- Client synchronization

This architecture provides a scalable, performant enemy system capable of handling hundreds of entities while maintaining smooth gameplay and network efficiency.