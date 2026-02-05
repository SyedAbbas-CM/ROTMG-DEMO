# ROTMG RTS - Critical Fixes TODO List
**Date**: 2025-09-06
**Project**: ROTMG RTS Game
**Priority**: URGENT - Production Architecture Fixes
**Status**: Ready for Implementation

---

## 🔥 PHASE 1: CRITICAL ARCHITECTURE FIXES (Week 1)

### 1. **SPLIT MONOLITHIC SERVER.JS** (CRITICAL - Production Blocker)
**Current State**: 1,384 lines of unmaintainable code
**Target**: Split into focused modules

#### 1.1 Create Server Module Structure
```bash
mkdir -p src/server
```

#### 1.2 Split Server.js Into Modules
```javascript
// src/server/GameServer.js - Core game server setup (150 lines max)
import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';

export class GameServer {
    constructor(config = {}) {
        this.port = config.port || 3000;
        this.app = express();
        this.server = http.createServer(this.app);
        this.wss = new WebSocketServer({ server: this.server });
    }
    
    async start() {
        // Clean server startup logic only
    }
}
```

```javascript
// src/server/AssetServer.js - Asset API routes (200 lines max)
export class AssetServer {
    constructor(app) {
        this.app = app;
        this.setupRoutes();
    }
    
    setupRoutes() {
        // Move lines 117-276 from Server.js here
        this.app.get('/api/assets/images', this.handleImageList.bind(this));
        this.app.get('/api/assets/atlases', this.handleAtlasList.bind(this));
        // ... etc
    }
}
```

```javascript
// src/server/ConnectionManager.js - WebSocket connections (250 lines max)
export class ConnectionManager {
    constructor(wss, worldManager) {
        this.wss = wss;
        this.worldManager = worldManager;
        this.clients = new Map();
        this.setupConnectionHandlers();
    }
    
    setupConnectionHandlers() {
        // Move lines 831-952 from Server.js here
    }
}
```

```javascript
// src/server/WorldManager.js - World/map management (200 lines max)
export class WorldManager {
    constructor() {
        this.worldContexts = new Map();
        this.mapManager = new MapManager();
    }
    
    getWorldCtx(mapId) {
        // Move world context logic here
    }
}
```

#### 1.3 Refactored Server.js (Target: <100 lines)
```javascript
// Server.js - Clean orchestration only
import { GameServer } from './src/server/GameServer.js';
import { AssetServer } from './src/server/AssetServer.js';
import { ConnectionManager } from './src/server/ConnectionManager.js';
import { WorldManager } from './src/server/WorldManager.js';

async function startServer() {
    const gameServer = new GameServer();
    const assetServer = new AssetServer(gameServer.app);
    const worldManager = new WorldManager();
    const connectionManager = new ConnectionManager(gameServer.wss, worldManager);
    
    await gameServer.start();
    console.log('ROTMG RTS Server started successfully');
}

startServer().catch(console.error);
```

### 2. **SPLIT MASSIVE BEHAVIORS.JS** (CRITICAL - 2,058 lines)
**Target**: Split by behavior categories

#### 2.1 Create Behavior Module Structure
```bash
mkdir -p src/Behaviours/combat
mkdir -p src/Behaviours/movement  
mkdir -p src/Behaviours/ai
mkdir -p src/Behaviours/boss
```

#### 2.2 Split by Categories
```javascript
// src/Behaviours/combat/AttackBehaviors.js (300 lines max)
export const ShootBehavior = { /* ... */ };
export const MeleeBehavior = { /* ... */ };
export const SpecialAttackBehavior = { /* ... */ };

// src/Behaviours/movement/MovementBehaviors.js (250 lines max)
export const ChaseBehavior = { /* ... */ };
export const PatrolBehavior = { /* ... */ };
export const FleeBehavior = { /* ... */ };

// src/Behaviours/ai/AIPatterns.js (400 lines max)  
export const AggressiveAI = { /* ... */ };
export const DefensiveAI = { /* ... */ };
export const SwarmAI = { /* ... */ };

// src/Behaviours/boss/BossBehaviors.js (600 lines max)
export const LLMBossBehavior = { /* ... */ };
export const StaticBossBehavior = { /* ... */ };
```

#### 2.3 Update Behavior Index  
```javascript
// src/Behaviours/index.js - Clean exports
export * from './combat/AttackBehaviors.js';
export * from './movement/MovementBehaviors.js';
export * from './ai/AIPatterns.js';
export * from './boss/BossBehaviors.js';
```

### 3. **FIX SECURITY VULNERABILITIES** (CRITICAL)

#### 3.1 Input Validation Middleware
```javascript
// src/middleware/ValidationMiddleware.js
export class ValidationMiddleware {
    static validateMovement(data) {
        if (!data.x || !data.y) return false;
        
        // Speed validation - prevent teleporting
        const MAX_SPEED = 10; // tiles per frame
        const timeDelta = Date.now() - data.timestamp;
        const distance = Math.sqrt((data.x - data.prevX)**2 + (data.y - data.prevY)**2);
        const speed = distance / (timeDelta / 1000);
        
        return speed <= MAX_SPEED;
    }
    
    static sanitizePath(path) {
        // Prevent path traversal
        if (path.includes('..') || path.includes('~') || path.startsWith('/')) {
            throw new Error('Invalid path');
        }
        return path.replace(/[^a-zA-Z0-9.-]/g, '');
    }
    
    static sanitizeCommand(command) {
        // Prevent command injection
        return command.replace(/[;&|`$(){}[\]]/g, '');
    }
}
```

#### 3.2 Rate Limiting
```javascript
// src/middleware/RateLimiter.js
export class RateLimiter {
    constructor() {
        this.clientRates = new Map();
    }
    
    checkRate(clientId, action, limit = 10, window = 1000) {
        const now = Date.now();
        const key = `${clientId}_${action}`;
        
        if (!this.clientRates.has(key)) {
            this.clientRates.set(key, { count: 0, resetTime: now + window });
        }
        
        const rateData = this.clientRates.get(key);
        
        if (now >= rateData.resetTime) {
            rateData.count = 0;
            rateData.resetTime = now + window;
        }
        
        rateData.count++;
        return rateData.count <= limit;
    }
}
```

### 4. **FIX PERFORMANCE ISSUES** (HIGH)

#### 4.1 Remove Global Pollution
```javascript
// BEFORE (BAD):
globalThis.DEBUG = DEBUG;
globalThis.itemManager = itemManager;
window.spriteManager = spriteManager;

// AFTER (GOOD):
// src/core/GlobalState.js
export class GlobalState {
    static instance = null;
    
    static getInstance() {
        if (!GlobalState.instance) {
            GlobalState.instance = new GlobalState();
        }
        return GlobalState.instance;
    }
    
    constructor() {
        this.debug = false;
        this.managers = new Map();
    }
    
    setManager(name, manager) {
        this.managers.set(name, manager);
    }
    
    getManager(name) {
        return this.managers.get(name);
    }
}
```

#### 4.2 Implement Binary Protocols
```javascript  
// src/network/BinaryProtocol.js
export class BinaryProtocol {
    static encodePositionUpdate(playerId, x, y, timestamp) {
        const buffer = new ArrayBuffer(20);
        const view = new DataView(buffer);
        
        view.setUint32(0, playerId);      // 4 bytes
        view.setFloat32(4, x);            // 4 bytes  
        view.setFloat32(8, y);            // 4 bytes
        view.setBigUint64(12, BigInt(timestamp)); // 8 bytes
        
        return buffer;
    }
    
    static decodePositionUpdate(buffer) {
        const view = new DataView(buffer);
        return {
            playerId: view.getUint32(0),
            x: view.getFloat32(4),
            y: view.getFloat32(8),
            timestamp: Number(view.getBigUint64(12))
        };
    }
}
```

#### 4.3 Add Object Pooling
```javascript
// src/core/ObjectPool.js  
export class ObjectPool {
    constructor(factory, reset, maxSize = 1000) {
        this.factory = factory;
        this.reset = reset;
        this.pool = [];
        this.maxSize = maxSize;
    }
    
    acquire() {
        if (this.pool.length > 0) {
            return this.pool.pop();
        }
        return this.factory();
    }
    
    release(obj) {
        if (this.pool.length < this.maxSize) {
            this.reset(obj);
            this.pool.push(obj);
        }
    }
}

// Usage in BulletManager
const bulletPool = new ObjectPool(
    () => ({ x: 0, y: 0, vx: 0, vy: 0, life: 0 }),
    (bullet) => Object.assign(bullet, { x: 0, y: 0, vx: 0, vy: 0, life: 0 })
);
```

---

## 🚀 PHASE 2: INTEGRATION & TESTING (Week 2)

### 5. **OVERWORLD DEMO INTEGRATION** 
**File**: `overworld-demo/` system investigation

#### 5.1 Test Overworld Demo Standalone
```bash
cd overworld-demo
PORT=3002 node server.js
# Test at http://localhost:3002
```

#### 5.2 Analyze Integration Points
- **EfficientWorldManager.js** - 5k×5k world with chunking
- **Spore-like zoom system** - Region drilling capability
- **Memory management** - LRU chunk eviction
- **Procedural generation** - Distance-based terrain

#### 5.3 Integration Strategy
```javascript
// src/world/OverworldIntegration.js
import { EfficientWorldManager } from '../overworld-demo/EfficientWorldManager.js';

export class OverworldGameIntegration {
    constructor(gameServer) {
        this.gameServer = gameServer;
        this.overworldManager = new EfficientWorldManager({
            worldWidth: 10000,
            worldHeight: 10000,
            chunkSize: 100
        });
    }
    
    // Bridge between overworld and main game
    async enterRegion(playerId, regionX, regionY) {
        // Switch player from overworld to detailed region
        const region = await this.generateDetailedRegion(regionX, regionY);
        return this.gameServer.transferPlayer(playerId, region.mapId);
    }
}
```

### 6. **COMPREHENSIVE TESTING**

#### 6.1 Unit Tests
```bash
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/performance
```

#### 6.2 Critical Test Cases
```javascript
// tests/unit/ServerModules.test.js
describe('Server Module Split', () => {
    test('GameServer starts without errors', async () => {
        const server = new GameServer({ port: 0 });
        await expect(server.start()).resolves.not.toThrow();
    });
    
    test('AssetServer routes work', () => {
        // Test asset API endpoints
    });
    
    test('ConnectionManager handles WebSocket', () => {
        // Test connection handling
    });
});

// tests/integration/GameFlow.test.js  
describe('End-to-End Game Flow', () => {
    test('Player can connect, move, shoot, disconnect', async () => {
        // Full game session test
    });
    
    test('Multiple worlds isolated correctly', async () => {
        // Cross-world isolation test
    });
});

// tests/performance/LoadTest.test.js
describe('Performance Benchmarks', () => {
    test('Server handles 100 concurrent connections', async () => {
        // Load test with 100 simulated players
    });
    
    test('Binary protocol 10x faster than JSON', async () => {
        // Protocol performance comparison
    });
});
```

### 7. **ERROR BOUNDARY SYSTEM**
```javascript
// src/core/ErrorBoundary.js
export class ErrorBoundary {
    static wrapLLMController(controller, fallback) {
        return new Proxy(controller, {
            get(target, prop) {
                const original = target[prop];
                if (typeof original === 'function') {
                    return function(...args) {
                        try {
                            return original.apply(target, args);
                        } catch (error) {
                            console.error(`LLM Controller error in ${prop}:`, error);
                            return fallback?.[prop]?.apply(fallback, args) || null;
                        }
                    };
                }
                return original;
            }
        });
    }
}
```

---

## 📋 PHASE 3: OPTIMIZATION & PRODUCTION (Week 3)

### 8. **MEMORY LEAK FIXES**

#### 8.1 World Context Cleanup
```javascript
// src/server/WorldManager.js
export class WorldManager {
    // ... existing code
    
    cleanupEmptyWorlds() {
        for (const [mapId, ctx] of this.worldContexts) {
            const playerCount = this.getPlayersInWorld(mapId).length;
            
            if (playerCount === 0 && this.isWorldExpired(mapId)) {
                // Clean up managers
                ctx.bulletMgr.cleanup?.();
                ctx.enemyMgr.cleanup?.(); 
                ctx.collMgr.cleanup?.();
                ctx.bagMgr.cleanup?.();
                
                this.worldContexts.delete(mapId);
                console.log(`Cleaned up empty world: ${mapId}`);
            }
        }
    }
    
    isWorldExpired(mapId) {
        // World expires after 5 minutes with no players
        const expiry = this.worldExpiry.get(mapId);
        return expiry && Date.now() > expiry;
    }
}
```

#### 8.2 Bullet Array Bounds
```javascript  
// src/entities/BulletManager.js improvements
export default class BulletManager {
    constructor(maxBullets = 10000) {
        // ... existing SoA arrays
        
        this.maxBullets = maxBullets;
        this.bulletCount = 0;
        this.removalQueue = []; // Batch removals
    }
    
    addBullet(bulletData) {
        if (this.bulletCount >= this.maxBullets) {
            console.warn('Bullet limit reached, removing oldest');
            this.removeOldestBullet();
        }
        
        // ... add bullet logic
    }
    
    removeOldestBullet() {
        let oldestIndex = 0;
        let oldestTime = this.spawnTime[0];
        
        for (let i = 1; i < this.bulletCount; i++) {
            if (this.spawnTime[i] < oldestTime) {
                oldestTime = this.spawnTime[i];
                oldestIndex = i;
            }
        }
        
        this.removeBullet(oldestIndex);
    }
}
```

### 9. **CONFIGURATION SYSTEM**
```javascript
// config/GameConfig.js
export const GameConfig = {
    server: {
        port: parseInt(process.env.PORT) || 3000,
        maxConnections: 1000,
        updateRate: 30, // FPS
    },
    
    performance: {
        maxBulletsPerWorld: 10000,
        maxEnemiesPerWorld: 1000,
        chunkSize: 16,
        updateRadius: 20,
        useObjectPooling: true,
        useBinaryProtocol: true,
    },
    
    security: {
        rateLimitMoves: 60,     // moves per second
        rateLimitShots: 20,     // shots per second  
        rateLimitCommands: 10,  // commands per second
        maxMessageSize: 8192,   // bytes
    },
    
    llm: {
        provider: process.env.LLM_BACKEND || 'gemini',
        apiKey: process.env.GOOGLE_API_KEY,
        maxRetries: 3,
        timeout: 5000, // ms
        fallbackBehavior: 'aggressive', // when LLM fails
    }
};
```

### 10. **DEPLOYMENT PREPARATION**
```bash
# package.json scripts
{
  "scripts": {
    "start": "node Server.js",
    "dev": "nodemon Server.js", 
    "test": "jest",
    "test:integration": "jest tests/integration",
    "test:performance": "jest tests/performance",
    "lint": "eslint src/",
    "build": "npm run lint && npm test",
    "docker:build": "docker build -t rotmg-rts .",
    "docker:run": "docker run -p 3000:3000 rotmg-rts"
  }
}
```

---

## ✅ SUCCESS CRITERIA & VALIDATION

### Architecture Quality Gates:
- [ ] No single file >500 lines (except generated)
- [ ] All modules have single responsibility
- [ ] No global state pollution
- [ ] Circular dependencies eliminated
- [ ] Error boundaries implemented

### Performance Benchmarks:
- [ ] Server handles 500+ concurrent players
- [ ] Binary protocol 5x+ faster than JSON
- [ ] Memory usage stable over 24 hours
- [ ] World cleanup prevents memory leaks
- [ ] Response time <100ms 95th percentile

### Security Validation:
- [ ] Input validation on all endpoints
- [ ] Rate limiting prevents abuse
- [ ] No path traversal vulnerabilities
- [ ] Movement validation prevents cheating
- [ ] Command injection impossible

### Integration Success:
- [ ] Overworld demo integrated smoothly
- [ ] Spore-like zoom system working
- [ ] Multi-scale gameplay functional
- [ ] No performance degradation
- [ ] Memory management effective

---

## 🎯 IMPLEMENTATION PRIORITY ORDER

1. **Server.js Split** (Week 1, Days 1-3) - BLOCKING everything else
2. **Behaviors.js Split** (Week 1, Days 4-5) - Required for AI work
3. **Security Fixes** (Week 1, Days 6-7) - Production requirement
4. **Performance Fixes** (Week 2, Days 1-3) - User experience critical
5. **Overworld Integration** (Week 2, Days 4-5) - New feature
6. **Testing Suite** (Week 2, Days 6-7) - Quality assurance
7. **Optimization** (Week 3, Days 1-5) - Polish and production readiness

**CRITICAL PATH**: Server split must complete before any other architectural changes.

This TODO list addresses every critical issue identified in the architecture review and provides specific, actionable steps to transform the ROTMG RTS codebase from a prototype into a production-ready multiplayer game.