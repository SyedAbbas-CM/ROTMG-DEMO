# System Architecture Overview

## Introduction
This document provides a comprehensive overview of the ROTMG-DEMO game systems, their interactions, and the overall architecture. This serves as a high-level guide to understanding how all components work together to create the complete gameplay experience, with particular emphasis on the innovative LLM-powered boss system and high-performance multiplayer architecture.

## Documentation Index

### Core Systems Documentation
1. **[Enemy System](./Enemy-System.md)** - Enemy management, AI behaviors, and network synchronization
2. **[LLM Boss System](./LLM-Boss-System.md)** - AI-powered boss architecture and capability system
3. **[Drop System](./Drop-System.md)** - Loot generation, probability mechanics, and bag color priority
4. **[Item System](./Item-System.md)** - Item definitions, instances, and binary serialization
5. **[Bag System](./Bag-System.md)** - Loot bag management, TTL system, and ownership controls
6. **[Frontend-Backend Data Flow](./Frontend-Backend-Data-Flow.md)** - Network protocols and client-server communication

## Complete System Architecture

### Multi-Tier Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CLIENT TIER                                      │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│  Game Loop      │  Entity Mgmt    │  Rendering      │  Network/Input          │
│  ──────────     │  ────────────   │  ─────────      │  ──────────────         │
│ • Game.js       │ • ClientEnemy   │ • WebGL         │ • ClientNetwork         │
│ • GameState     │   Manager       │   Renderers     │   Manager               │
│ • Input         │ • ClientBullet  │ • Sprite        │ • Message               │
│   Handling      │   Manager       │   Database      │   Handlers              │
│ • Entity        │ • ClientBag     │ • Multi-view    │ • WebSocket             │
│   Interpolation │   Manager       │   System        │   Connection            │
│ • Animation     │ • ClientItem    │ • Texture       │ • Prediction            │
│   System        │   Manager       │   Management    │   Reconciliation        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │   NETWORK LAYER   │
                            │ ──────────────── │
                            │ • WebSocket       │
                            │ • HTTP Routes     │
                            │ • Binary Proto    │
                            │ • Interest Mgmt   │
                            │ • Rate Limiting   │
                            │ • Message Queue   │
                            └─────────┬─────────┘
                                      │
┌─────────────────────────────────────▼─────────────────────────────────────────┐
│                                SERVER TIER                                     │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│  Core Game      │  LLM Boss AI    │  World Systems  │  Infrastructure         │
│  ──────────     │  ────────────   │  ─────────────  │  ──────────────         │
│ • EnemyManager  │ • BossManager   │ • MapManager    │ • NetworkManager        │
│ • BulletManager │ • LLMBoss       │ • World         │ • Telemetry             │
│ • Collision     │   Controller    │   Contexts      │ • Logger                │
│   Manager       │ • Script        │ • Spatial       │ • Config                │
│ • ItemManager   │   Behaviour     │   Systems       │ • Routes                │
│ • BagManager    │   Runner        │ • Chunk         │ • Error                 │
│ • DropSystem    │ • Capability    │   Loading       │   Handling              │
│ • Behavior      │   Registry      │ • Map           │ • Performance           │
│   System        │ • DSL           │   Generation    │   Monitoring            │
│ • Inventory     │   Interpreter   │                 │                         │
│   Manager       │                 │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
                                      │
                  ┌───────────────────▼───────────────────┐
                  │            AI PROVIDER LAYER          │
                  │          ─────────────────────        │
                  │ • Google Gemini API                   │
                  │ • Ollama Local Models                 │
                  │ • Provider Factory                    │
                  │ • Response Parsing                    │
                  │ • Retry & Backoff                     │
                  │ • Hash-based Change Detection         │
                  └───────────────────┬───────────────────┘
                                      │
                            ┌─────────▼─────────┐
                            │     DATA LAYER    │
                            │ ──────────────── │
                            │ • Entity Database │
                            │ • Sprite Atlases  │
                            │ • Map Definitions │
                            │ • Capability      │
                            │   Schemas         │
                            │ • LLM Logs        │
                            │ • Configuration   │
                            │ • Runtime State   │
                            └───────────────────┘
```

### Detailed System Integration Flow
```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE GAME LOOP EXECUTION                            │
│                               (60 FPS)                                       │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Server Tick      │
                    │    ───────────      │
                    │ 1. Process Input    │
                    │ 2. Update Physics   │
                    │ 3. Update AI/LLM    │
                    │ 4. Handle Collision │
                    │ 5. Process Deaths   │
                    │ 6. Update Bags      │
                    │ 7. Network Broadcast│
                    │ 8. Telemetry        │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼──────┐    ┌─────────▼────────┐    ┌────────▼────────┐
│ Enemy/Boss   │    │  Loot Pipeline   │    │  Network Sync   │
│ AI Update    │    │  ─────────────   │    │  ────────────   │
├──────────────┤    │ 1. Enemy Death   │    │ 1. Interest     │
│ • Behavior   │    │ 2. Roll Drops    │    │    Management   │
│   System     │    │ 3. Create Items  │    │ 2. Message      │
│ • LLM Boss   │    │ 4. Spawn Bags    │    │    Batching     │
│   Planning   │    │ 5. Network Sync  │    │ 3. Binary       │
│ • Capability │    │ 6. TTL Cleanup   │    │    Encoding     │
│   Execution  │    │ 7. Ownership     │    │ 4. Client       │
│ • Script     │    │    Management    │    │    Filtering    │
│   Runner     │    └──────────────────┘    │ 5. Prediction   │
│ • Hash-based │                            │    Support      │
│   Change     │                            └─────────────────┘
│   Detection  │
└──────────────┘
```

## Core Game Systems

### 1. Entity Management Architecture

#### Structure of Arrays (SoA) Pattern
All major systems use SoA for optimal cache performance:

```javascript
// Example from EnemyManager
class EnemyManager {
  constructor(maxEnemies = 1000) {
    // Hot data (accessed every frame)
    this.x = new Float32Array(maxEnemies);       // Position X
    this.y = new Float32Array(maxEnemies);       // Position Y
    this.health = new Float32Array(maxEnemies);  // Current HP
    
    // Warm data (accessed frequently)
    this.type = new Uint8Array(maxEnemies);      // Enemy type
    this.moveSpeed = new Float32Array(maxEnemies);
    
    // Cold data (accessed occasionally)
    this.id = new Array(maxEnemies);             // String IDs
    this.worldId = new Array(maxEnemies);        // World context
  }
}
```

**Benefits**:
- **Cache Efficiency**: Related data stored contiguously
- **SIMD Potential**: Vectorized operations possible
- **Memory Locality**: Reduced cache misses during iteration
- **Predictable Performance**: Fixed allocation patterns

### 2. World Context System

#### Multi-World Architecture (`Server.js:267-280`)
```javascript
// Each world has isolated systems
const worldContexts = new Map(); // mapId -> context

function getWorldCtx(mapId) {
  if (!worldContexts.has(mapId)) {
    const context = {
      enemyMgr: new EnemyManager(1000),
      bulletMgr: new BulletManager(5000),
      collMgr: new CollisionManager(),
      bagMgr: new BagManager(500),
      itemMgr: new ItemManager(),
      bossMgr: new BossManager(enemyMgr, 1),
      llmController: new LLMBossController(...)
    };
    worldContexts.set(mapId, context);
  }
  return worldContexts.get(mapId);
}
```

**Integration Points**:
- **Player Assignment**: `client.mapId` determines world context
- **Entity Filtering**: All operations filtered by `worldId`
- **Network Isolation**: Updates only sent to clients in same world
- **Resource Isolation**: Memory and processing separated per world

### 3. Advanced Network Architecture

#### Message Type System (`NetworkManager.js`)
```javascript
export const MessageType = {
  // Connection lifecycle
  PLAYER_JOIN: 'player_join',
  PLAYER_LEAVE: 'player_leave',
  
  // World state
  WORLD_UPDATE: 'world_update',
  ENEMY_LIST: 'enemy_list',
  CHUNK_REQUEST: 'chunk_request',
  CHUNK_DATA: 'chunk_data',
  
  // Player actions
  PLAYER_MOVE: 'player_move',
  PLAYER_SHOOT: 'player_shoot',
  PLAYER_CHAT: 'player_chat',
  
  // Loot system
  ITEM_PICKUP: 'item_pickup',
  BAG_OPEN: 'bag_open',
  
  // Boss system
  BOSS_SPEECH: 'boss_speech',
  BOSS_PHASE: 'boss_phase'
};
```

#### Binary Protocol Integration
```javascript
export class BinaryPacket {
  static encode(type, data) {
    const header = Buffer.alloc(8);
    header.writeUInt32LE(type, 0);
    header.writeUInt32LE(data.length, 4);
    return Buffer.concat([header, data]);
  }
  
  static decode(buffer) {
    const type = buffer.readUInt32LE(0);
    const length = buffer.readUInt32LE(4);
    const data = buffer.slice(8, 8 + length);
    return { type, data };
  }
}
```

#### Interest Management System
```javascript
// Server.js:796-850 - Advanced filtering
function broadcastWorldUpdates() {
  worldContexts.forEach((ctx, mapId) => {
    const clients = getClientsInWorld(mapId);
    
    clients.forEach(client => {
      // Distance-based filtering
      const visibleEnemies = ctx.enemyMgr.getEnemiesData(mapId)
        .filter(enemy => {
          const dx = enemy.x - client.x;
          const dy = enemy.y - client.y;
          const distSq = dx * dx + dy * dy;
          return distSq <= UPDATE_RADIUS_SQ;
        })
        .slice(0, MAX_ENTITIES_PER_PACKET);
      
      // Ownership-based filtering for bags
      const visibleBags = ctx.bagMgr.getBagsData(mapId, client.id);
      
      // Binary item data
      const itemData = ctx.itemMgr.getBinaryData();
      
      sendToClient(client.socket, MessageType.WORLD_UPDATE, {
        enemies: visibleEnemies,
        bullets: visibleBullets,
        bags: visibleBags,
        items: itemData,  // Binary format
        timestamp: Date.now()
      });
    });
  });
}
```

## LLM Boss System Deep Dive

### 1. Snapshot Generation and Hashing
```javascript
// BossManager.js - Context generation for AI
buildSnapshot(players) {
  const snapshot = {
    boss: {
      id: this.id[0],
      position: { x: this.x[0], y: this.y[0] },
      health: this.hp[0],
      phase: this.phase[0],
      lastAction: this.lastAction,
      cooldowns: {
        dash: this.cooldownDash[0],
        aoe: this.cooldownAOE[0]
      }
    },
    players: players.map(p => ({
      id: p.id,
      position: { x: p.x, y: p.y },
      health: p.health / p.maxHealth,
      class: p.characterClass,
      distance: Math.sqrt((p.x - this.x[0])**2 + (p.y - this.y[0])**2),
      isMoving: p.velocityX !== 0 || p.velocityY !== 0
    })),
    environment: {
      timestamp: Date.now(),
      worldId: this.worldId[0],
      nearbyEnemies: this.getNearbyEnemyCount(),
      activeBullets: this.getActiveBulletCount()
    },
    history: {
      recentDamage: this.getRecentDamageHistory(),
      playerDeaths: this.getRecentPlayerDeaths(),
      phaseTransitions: this.getPhaseHistory()
    }
  };
  
  return snapshot;
}

// Hash-based change detection
async hashSnapshot(snapshot) {
  const hashApi = await xxhash32();
  const jsonStr = JSON.stringify(snapshot, Object.keys(snapshot).sort());
  return hashApi.hash(Buffer.from(jsonStr), HASH_SEED);
}
```

### 2. Capability Registry Deep Architecture
```javascript
// Registry.js - Advanced capability system
class CapabilityRegistry {
  constructor() {
    this.validators = new Map();      // JSON Schema validators
    this.compilers = new Map();       // Brick → Action compilers  
    this.invokers = new Map();        // Action executors
    this.schemas = new Map();         // Raw schemas
    this.dependencies = new Map();    // Capability dependencies
    this.resourceLimits = new Map();  // Safety constraints
    
    // Hot reload support
    this.fileWatcher = null;
    this.loadTimestamps = new Map();
  }
  
  // Advanced validation with safety limits
  validateWithLimits(brick) {
    const baseValidation = this.validate(brick);
    if (!baseValidation.ok) return baseValidation;
    
    // Apply global safety constraints
    const limits = this.resourceLimits.get(brick.type);
    if (limits) {
      const violations = this.checkResourceLimits(brick, limits);
      if (violations.length > 0) {
        return { ok: false, errors: violations };
      }
    }
    
    return { ok: true };
  }
  
  // Dependency resolution
  resolveDependencies(capabilities) {
    const graph = new Map();
    const resolved = [];
    
    // Build dependency graph
    capabilities.forEach(cap => {
      const deps = this.dependencies.get(cap.type) || [];
      graph.set(cap, deps);
    });
    
    // Topological sort
    return this.topologicalSort(graph);
  }
}
```

### 3. Advanced Telemetry Integration
```javascript
// src/telemetry/index.js - Production monitoring
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'rotmg-demo',
    [SemanticResourceAttributes.SERVICE_VERSION]: '1.0.0',
  }),
});

// Custom span processor for game metrics
class GameMetricsProcessor extends BatchSpanProcessor {
  onEnd(span) {
    super.onEnd(span);
    
    // Extract game-specific metrics
    const attributes = span.attributes;
    if (span.name.startsWith('llm.generate')) {
      this.recordLLMMetrics(attributes);
    } else if (span.name.startsWith('boss.plan')) {
      this.recordBossMetrics(attributes);
    }
  }
  
  recordLLMMetrics(attributes) {
    // Track LLM performance
    this.metrics.llmLatency.record(attributes['llm.latency_ms']);
    this.metrics.llmTokens.record(attributes['llm.tokens_used']);
    this.metrics.llmErrors.add(attributes['llm.error'] ? 1 : 0);
  }
}
```

## Performance Characteristics

### 1. Memory Architecture
```javascript
// Memory layout optimization
Entity Count Limits:
├── Enemies: 1,000 (40 KB SoA arrays)
├── Bullets: 5,000 (200 KB SoA arrays) 
├── Items: 2,000 (80 KB + binary data)
├── Bags: 500 (20 KB SoA arrays)
└── Bosses: 1 per world (minimal overhead)

Total Memory per World: ~340 KB + dynamic allocations
```

### 2. Network Performance
```javascript
// Packet size analysis
Message Type Breakdown:
├── WORLD_UPDATE: ~2-8 KB (varies by entity count)
├── ENEMY_LIST: ~10-40 KB (initial state)
├── Binary Items: 44 bytes per item (header + data)
├── Boss Updates: ~1 KB (LLM state changes)
└── Player Actions: ~100-500 bytes each

Network Bandwidth: ~50-200 KB/s per client (60 FPS)
```

### 3. CPU Performance Profile
```javascript
// Profiling data (60 FPS server)
Frame Budget: 16.67ms
├── Entity Updates: ~3-6ms (SoA iteration)
├── Collision Detection: ~2-4ms (spatial optimization)
├── LLM Processing: ~0-50ms (async, cached)
├── Network I/O: ~1-3ms (binary encoding)
├── Bag/Item Management: ~0.5-1ms
└── Telemetry: ~0.1-0.5ms

Average Frame Time: 8-15ms (40-60% utilization)
```

## Configuration Management

### 1. Environment Configuration
```javascript
// Complete .env configuration
LLM_BACKEND=gemini                    # 'gemini' | 'ollama'
LLM_MODEL=gemini-pro                  # Model identifier
LLM_TEMP=0.7                         # Generation temperature
LLM_MAXTOKENS=256                     # Response limit
LLM_PLAN_PERIOD=2                     # Seconds between LLM calls
LLM_BACKOFF_SEC=15                    # Error cooldown
LLM_SPEECH_PERIOD=6                   # Boss speech frequency

GOOGLE_API_KEY=your_api_key           # Gemini authentication
OLLAMA_HOST=127.0.0.1                 # Local Ollama server
OLLAMA_PORT=11434                     # Ollama port

NETWORK_UPDATE_RADIUS=20              # Tiles visibility
NETWORK_MAX_ENTITIES=50               # Entities per packet
NETWORK_COMPRESSION=true              # Enable compression

DEBUG_ENEMY_SPAWNS=false              # Debug flags
DEBUG_LLM_CALLS=false
DEBUG_COLLISIONS=false
DEBUG_NETWORK=false

TELEMETRY_EXPORT_INTERVAL=5000        # Metrics export interval
TELEMETRY_CONSOLE_EXPORT=true         # Console output
```

### 2. Runtime Configuration
```javascript
// src/config/llmConfig.js - Dynamic settings
export default {
  planPeriodSec: +process.env.LLM_PLAN_PERIOD || 2,
  backoffSec: +process.env.LLM_BACKOFF_SEC || 15,
  speechPeriodSec: +process.env.LLM_SPEECH_PERIOD || 6,
  
  // Provider-specific settings
  providers: {
    gemini: {
      temperature: +process.env.LLM_TEMP || 0.7,
      maxTokens: +process.env.LLM_MAXTOKENS || 256,
      timeout: +process.env.LLM_TIMEOUT || 10000
    },
    ollama: {
      host: process.env.OLLAMA_HOST || '127.0.0.1',
      port: +process.env.OLLAMA_PORT || 11434,
      keepAlive: process.env.OLLAMA_KEEP_ALIVE || '5m'
    }
  },
  
  // Safety constraints
  safety: {
    maxProjectiles: 400,
    maxSpeed: 100,
    maxRadius: 15,
    maxDuration: 10
  }
};
```

## Error Handling and Resilience

### 1. LLM Provider Resilience
```javascript
// LLMBossController.js - Error handling
class LLMBossController {
  async _callLLMProvider(snapshot) {
    const span = tracer.startSpan('boss.plan');
    let retryCount = 0;
    
    while (retryCount < MAX_RETRIES) {
      try {
        const response = await provider.generate(prompt, snapshot);
        this.feedback.push({ success: true, latency: span.duration });
        return this.processResponse(response);
        
      } catch (error) {
        retryCount++;
        
        if (error.code === 'RATE_LIMIT') {
          this.cooldown = Math.max(this.cooldown, 30); // Extended backoff
        } else if (error.code === 'TIMEOUT') {
          this.cooldown = Math.max(this.cooldown, 5);
        }
        
        this.feedback.push({ 
          success: false, 
          error: error.message,
          retryCount 
        });
        
        if (retryCount >= MAX_RETRIES) {
          // Fallback to baseline behavior
          return this.getBaselineBehavior();
        }
        
        await this.sleep(Math.pow(2, retryCount) * 1000); // Exponential backoff
      } finally {
        span.end();
      }
    }
  }
}
```

### 2. Network Resilience
```javascript
// Server.js - Connection handling
function handleClientDisconnect(socket, clientId) {
  const client = clients.get(clientId);
  if (!client) return;
  
  // Cleanup player state
  const ctx = getWorldCtx(client.mapId);
  ctx.playerMgr.removePlayer(clientId);
  
  // Release resources
  clients.delete(clientId);
  
  // Notify other players
  broadcastToWorld(client.mapId, MessageType.PLAYER_LEAVE, {
    playerId: clientId,
    timestamp: Date.now()
  });
  
  console.log(`[Network] Client ${clientId} disconnected from ${client.mapId}`);
}

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('[Server] Graceful shutdown initiated');
  
  // Stop accepting new connections
  server.close();
  
  // Notify all clients
  clients.forEach((client, id) => {
    sendToClient(client.socket, MessageType.SERVER_SHUTDOWN, {
      message: 'Server shutting down',
      countdown: 30
    });
  });
  
  // Cleanup resources
  setTimeout(() => {
    process.exit(0);
  }, 30000);
});
```

This comprehensive architecture provides a scalable, performant, and innovative foundation for multiplayer gaming with AI-powered bosses, demonstrating advanced techniques in game server development, real-time AI integration, and high-performance networking.