# LLM Boss System Documentation

## Overview
The LLM Boss System is the core innovation of this ROTMG-DEMO project - a real-time AI-driven boss that uses Large Language Models (Google Gemini or local Ollama) to plan and execute dynamic attack patterns. This system creates emergent gameplay by allowing AI to analyze player behavior and adapt boss strategies in real-time.

## Architecture Overview

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BossManager   │    │LLMBossController│    │   LLM Provider  │
│  (Physics/HP)   │◄──►│  (Coordinator)  │◄──►│ (Gemini/Ollama) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ScriptBehaviour │    │CapabilityRegistry│    │ OpenTelemetry   │
│    Runner       │    │ (Action System) │    │   (Metrics)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. BossManager (`/src/BossManager.js`)

**Purpose**: Manages deterministic boss physics, health, and basic state
**Architecture**: Structure of Arrays (SoA) for performance

```javascript
class BossManager {
  constructor(enemyMgr, maxBosses = 1) {
    // SoA data layout
    this.id = new Array(maxBosses);           // "boss_1", "boss_2"
    this.x = new Float32Array(maxBosses);     // World position
    this.y = new Float32Array(maxBosses);
    this.hp = new Float32Array(maxBosses);    // 0-1 fraction
    this.phase = new Uint8Array(maxBosses);   // Current phase (0,1,2)
    this.worldId = new Array(maxBosses);      // World context
    
    // Cooldown timers for baseline patterns
    this.cooldownDash = new Float32Array(maxBosses);
    this.cooldownAOE = new Float32Array(maxBosses);
    
    // Action queue (filled by LLM)
    this.actionQueue = Array.from({ length: maxBosses }, () => []);
  }
}
```

**Key Functions**:

- `spawnBoss(defId, x, y, worldId)` - Creates boss instance
- `update(dt, players)` - Physics and baseline attack execution
- `buildSnapshot(players)` - Creates AI context snapshot
- `mirrorToEnemy()` - Syncs with EnemyManager for client visibility

**Integration**: 
- Links with EnemyManager for visual representation
- Provides snapshot data for LLM decision making

### 2. LLMBossController (`/src/LLMBossController.js`)

**Purpose**: Coordinates between BossManager and AI providers
**Core Loop**: Snapshot → Hash → LLM Call → Action Queue → Execute

```javascript
class LLMBossController {
  async tick(dt, players) {
    // 1. Execute queued actions from previous LLM calls
    this._drainActionQueue(dt);
    
    // 2. Update script runner (adds baseline actions)
    this.runner.tick(dt);
    
    // 3. Check if LLM call needed (cooldown + snapshot change)
    if (this.shouldCallLLM(players)) {
      this._callLLMProvider(players);
    }
  }
}
```

**Key Features**:
- **Hash-based Change Detection**: Only calls LLM when game state changes
- **Cooldown Management**: Prevents excessive API calls (configurable)
- **Telemetry Integration**: OpenTelemetry spans for monitoring
- **Difficulty Evaluation**: Analyzes player performance to adjust challenge

### 3. Capability Registry System

**Purpose**: Modular action system for boss abilities
**Architecture**: Plugin-based with JSON schemas and JS implementations

#### Directory Structure
```
capabilities/
├── Core/
│   └── Wait/
│       └── 1.0.0/
│           ├── implementation.js
│           └── schema.json
├── Emitter/
│   ├── RadialBurst/
│   └── ProjectileSpread/
└── Movement/
    └── Dash/
```

#### Schema Example (`Core/Wait/1.0.0/schema.json`)
```json
{
  "$id": "Core:Wait@1.0.0",
  "title": "Core:Wait",
  "type": "object",
  "required": ["type"],
  "properties": {
    "type": { "const": "Core:Wait@1.0.0" },
    "duration": { 
      "type": "number", 
      "minimum": 0, 
      "maximum": 10, 
      "default": 1 
    }
  }
}
```

#### Implementation Example (`Core/Wait/1.0.0/implementation.js`)
```javascript
export function compile(brick = {}) {
  return {
    ability: 'wait',
    args: { duration: brick.duration ?? 1 },
    _capType: brick.type || 'Core:Wait@1.0.0'
  };
}

export function invoke(node, state = {}, { dt }) {
  state.elapsed = (state.elapsed || 0) + dt;
  return state.elapsed >= (node.args.duration ?? 1);
}
```

### 4. LLM Provider System (`/src/llm/`)

**Architecture**: Plugin system supporting multiple AI providers

#### Provider Factory (`ProviderFactory.js`)
```javascript
export function createProvider() {
  const backend = process.env.LLM_BACKEND || 'gemini';
  
  switch (backend) {
    case 'gemini':
      return new GeminiProvider({
        apiKey: process.env.GOOGLE_API_KEY,
        model: process.env.LLM_MODEL || 'gemini-pro'
      });
    case 'ollama':
      return new OllamaProvider({
        host: process.env.OLLAMA_HOST || '127.0.0.1',
        model: process.env.LLM_MODEL || 'llama3'
      });
    default:
      throw new Error(`Unknown LLM backend: ${backend}`);
  }
}
```

#### Gemini Provider (`providers/GeminiProvider.js`)
```javascript
class GeminiProvider extends BaseProvider {
  async generate(prompt, snapshot) {
    const response = await this.client.generateContent({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: this.config.temperature,
        maxOutputTokens: this.config.maxTokens
      }
    });
    
    return this.parseResponse(response.response.text());
  }
}
```

### 5. ScriptBehaviourRunner (`/src/ScriptBehaviourRunner.js`)

**Purpose**: Executes both LLM-generated actions and baseline boss behaviors
**Architecture**: State machine with action queue

```javascript
class ScriptBehaviourRunner {
  tick(dt) {
    // Execute capability-based actions (from LLM)
    if (this.script?.nodes) {
      const nodes = this.interpreter.tick(dt);
      const queue = this.bossMgr.actionQueue[this.bossIdx];
      for (const node of nodes) {
        queue.push(node);
      }
      return;
    }
    
    // Execute timed actions
    this._processTimedActions(dt);
    
    // Check state transitions
    this._checkTransitions(dt);
  }
}
```

## Data Flow Architecture

### 1. Snapshot Generation
```javascript
// BossManager creates AI context
buildSnapshot(players) {
  return {
    boss: {
      id: this.id[0],
      x: this.x[0], y: this.y[0],
      hp: this.hp[0],
      phase: this.phase[0]
    },
    players: players.map(p => ({
      id: p.id,
      x: p.x, y: p.y,
      hp: p.health / p.maxHealth,
      class: p.characterClass
    })),
    environment: {
      timestamp: Date.now(),
      worldId: this.worldId[0]
    }
  };
}
```

### 2. Hash-based Change Detection
```javascript
// Only call LLM when game state actually changes
const snapshot = this.bossMgr.buildSnapshot(players);
const hash = await this.hashSnapshot(snapshot);

if (hash !== this.lastHash) {
  this.lastHash = hash;
  await this._callLLMProvider(snapshot);
}
```

### 3. LLM Response Processing
```javascript
// Parse LLM response into capability actions
parseResponse(llmText) {
  try {
    const parsed = JSON.parse(llmText);
    const actions = [];
    
    for (const brick of parsed.actions || []) {
      // Validate against capability schema
      const validation = registry.validate(brick);
      if (!validation.ok) {
        console.warn('Invalid capability:', validation.errors);
        continue;
      }
      
      // Compile into executable action
      const action = registry.compile(brick);
      actions.push(action);
    }
    
    return { actions, reasoning: parsed.reasoning };
  } catch (err) {
    console.error('Failed to parse LLM response:', err);
    return { actions: [], reasoning: 'Parse error' };
  }
}
```

### 4. Action Execution
```javascript
// Execute compiled actions in game loop
_drainActionQueue(dt) {
  const queue = this.bossMgr.actionQueue[this.bossIdx];
  
  for (let i = 0; i < queue.length; i++) {
    const action = queue[i];
    const ctx = { dt, bossMgr: this.bossMgr, bulletMgr: this.bulletMgr };
    
    // Execute via capability registry
    const completed = registry.invoke(action, ctx);
    
    if (completed) {
      queue.splice(i--, 1); // Remove completed action
    }
  }
}
```

## Available Capabilities

### Core Capabilities
1. **Core:Wait@1.0.0** - Pause execution for specified duration
2. **Movement:Dash@1.0.0** - Rapid movement to target position  
3. **Emitter:RadialBurst@1.0.0** - Circular bullet pattern
4. **Emitter:ProjectileSpread@1.0.0** - Customizable projectile spread

### Capability Parameters
```javascript
// Wait capability
{
  "type": "Core:Wait@1.0.0",
  "duration": 2.5  // seconds
}

// Radial burst capability  
{
  "type": "Emitter:RadialBurst@1.0.0", 
  "projectiles": 12,
  "speed": 8.0
}

// Dash capability
{
  "type": "Movement:Dash@1.0.0",
  "dx": 5.0, "dy": 0.0,
  "speed": 20.0,
  "duration": 1.0
}
```

## Configuration

### Environment Variables (`.env`)
```bash
# LLM Provider Configuration
LLM_BACKEND=gemini          # 'gemini' or 'ollama'
LLM_MODEL=gemini-pro        # Model name
LLM_TEMP=0.7               # Generation temperature
LLM_MAXTOKENS=256          # Max response tokens

# Provider-specific settings
GOOGLE_API_KEY=your_key_here  # Required for Gemini
OLLAMA_HOST=127.0.0.1         # Ollama server host
```

### LLM Configuration (`src/config/llmConfig.js`)
```javascript
export default {
  planPeriodSec: 3.0,      // Minimum seconds between LLM calls
  backoffSec: 1.0,         // Cooldown after failed calls
  maxRetries: 3,           // Max retry attempts
  timeoutMs: 10000,        // Request timeout
  
  // Prompt templates
  systemPrompt: "You are an AI boss in a bullet-hell game...",
  userPromptTemplate: "Current situation: {snapshot}..."
};
```

## Telemetry and Monitoring

### OpenTelemetry Integration
```javascript
// Automatic span creation for key operations
const span = tracer.startSpan('llm.generate');
span.setAttributes({
  'llm.provider': 'gemini',
  'llm.model': 'gemini-pro',
  'llm.temperature': 0.7
});

try {
  const response = await provider.generate(prompt);
  span.setAttributes({
    'llm.response.tokens': response.tokens,
    'llm.response.latency_ms': Date.now() - start
  });
} finally {
  span.end();
}
```

### Metrics Tracked
- `llm.generate.*` - LLM API call latency and token usage
- `boss.plan` - High-level planning operation metrics
- `mutator.<name>` - Individual capability execution time
- `registry.compile` - Capability compilation performance

## Performance Optimizations

### Hash-based Change Detection
- Uses xxhash32 for fast snapshot hashing
- Only calls LLM when game state actually changes
- Dramatically reduces API costs and latency

### Cooldown Management
- Configurable minimum time between LLM calls
- Exponential backoff on API failures
- Prevents rate limiting issues

### Capability Registry
- Pre-compiled JSON schemas for validation
- Cached validators for performance
- Hot-reload support for development

### Action Queue Optimization
- Batched action execution
- Efficient queue management
- Memory pooling for temporary objects

## Development Workflow

### Adding New Capabilities

1. **Create Directory Structure**:
```bash
mkdir -p capabilities/YourCategory/YourCapability/1.0.0
```

2. **Define Schema** (`schema.json`):
```json
{
  "$id": "YourCategory:YourCapability@1.0.0",
  "title": "YourCategory:YourCapability",
  "type": "object",
  "required": ["type"],
  "properties": {
    "type": { "const": "YourCategory:YourCapability@1.0.0" },
    "param1": { "type": "number", "default": 1.0 }
  }
}
```

3. **Implement Logic** (`implementation.js`):
```javascript
export function compile(brick) {
  return {
    ability: 'your_ability',
    args: { param1: brick.param1 ?? 1.0 },
    _capType: brick.type
  };
}

export function invoke(node, state, { dt, bossMgr, bulletMgr }) {
  // Execute your capability logic
  return completed; // boolean
}
```

4. **Generate Types**:
```bash
npm run generate:union
npm run generate:types
```

### Testing Capabilities
```javascript
// Unit test example
import { registry } from '../src/registry/index.js';

test('YourCapability validation', () => {
  const brick = {
    type: 'YourCategory:YourCapability@1.0.0',
    param1: 2.5
  };
  
  const result = registry.validate(brick);
  expect(result.ok).toBe(true);
  
  const compiled = registry.compile(brick);
  expect(compiled.ability).toBe('your_ability');
  expect(compiled.args.param1).toBe(2.5);
});
```

## Troubleshooting

### Common Issues

1. **LLM Not Responding**:
   - Check API key configuration
   - Verify network connectivity
   - Monitor rate limiting

2. **Invalid Capabilities**:
   - Check JSON schema validation
   - Verify capability registration
   - Review parameter bounds

3. **Performance Issues**:
   - Monitor telemetry spans
   - Check hash collision rates
   - Optimize snapshot generation

### Debug Configuration
```javascript
// Enable detailed logging
globalThis.DEBUG = {
  llmCalls: true,
  capabilityExecution: true,
  snapshotGeneration: true
};
```

## Advanced Technical Implementation Details

### 1. Enhanced Hash-based Change Detection System

**Deep Snapshot Architecture** (`LLMBossController.js:52-74`):
```javascript
// Complete snapshot generation with performance optimization
const snap = this.bossMgr.buildSnapshot(players, this.bulletMgr, this.tickCount);
if (!snap) return;

// Attach AI memory system - last 5 rated decisions for learning
snap.feedback = this.feedback.slice(-5);

// Ultra-fast hash generation using xxhash32
const hapi = await hashApiPromise;
const newHash = hapi.h32(JSON.stringify(snap), HASH_SEED);

// Only proceed with expensive LLM call if state actually changed
if (newHash !== this.lastHash) {
  this.lastHash = newHash;
  this.cooldown = PLAN_PERIOD;
  
  // Telemetry span for comprehensive monitoring
  const span = tracer.startSpan('boss.plan');
  const sentAt = Date.now();
  
  // Provider abstraction supports multiple LLM backends
  if (!provider) provider = createProvider();
  
  // Execute LLM call with timeout and retry logic
  const { json: res, deltaMs, tokens } = await provider.generate(snap);
  
  // Comprehensive logging for analysis and debugging
  logLLM({ ts: sentAt, snapshot: snap, result: res, deltaMs, tokens });
  
  // Real-time difficulty validation
  if (res?.metrics) {
    const { ok, reasons } = evaluate(res.metrics, { tier: 'mid' });
    if (!ok) {
      console.warn('[LLMBoss] DifficultyCritic veto', reasons.join(','));
      this.adjustDifficultyBasedOnCritique(reasons);
    }
  }
}
```

**Hash Collision Prevention**:
```javascript
// Deterministic JSON serialization for consistent hashing
function deterministicStringify(obj) {
  return JSON.stringify(obj, Object.keys(obj).sort());
}

// Advanced hash comparison with collision detection  
async hashSnapshot(snapshot) {
  const hapi = await hashApiPromise;
  const jsonStr = deterministicStringify(snapshot);
  const primaryHash = hapi.h32(jsonStr, HASH_SEED);
  const secondaryHash = hapi.h32(jsonStr, HASH_SEED ^ 0xFFFFFFFF);
  
  // Store both hashes to detect unlikely collisions
  return { primary: primaryHash, secondary: secondaryHash, raw: jsonStr.length };
}
```

### 2. Comprehensive Error Handling and Resilience

**Concurrency Control and Circuit Breaker** (`LLMBossController.js:50-99`):
```javascript
// Prevent overlapping LLM calls with sophisticated guarding
if (this.pendingLLM) return; // Only one in-flight call allowed

try {
  this.pendingLLM = true;
  
  // Timeout handling with graceful degradation
  const timeoutPromise = new Promise((_, reject) => 
    setTimeout(() => reject(new Error('LLM_TIMEOUT')), llmConfig.timeoutMs)
  );
  
  const llmPromise = provider.generate(snap);
  const result = await Promise.race([llmPromise, timeoutPromise]);
  
  // Success metrics and adaptive cooldown
  this.successfulCalls++;
  this.cooldown = Math.max(PLAN_PERIOD, this.adaptiveCooldown());
  
} catch (err) {
  // Sophisticated error categorization and response
  this.handleLLMError(err);
  
} finally {
  // Critical: Always clear the pending flag
  this.pendingLLM = false;
}

// Advanced error handling with exponential backoff
handleLLMError(err) {
  this.failedCalls++;
  
  switch (err.code || err.message) {
    case 'RATE_LIMIT':
      this.cooldown = Math.min(BACKOFF_SEC * Math.pow(2, this.consecutiveFailures), 60);
      this.consecutiveFailures++;
      break;
      
    case 'LLM_TIMEOUT':
      this.cooldown = BACKOFF_SEC * 2;
      this.fallbackToBaselineBehavior();
      break;
      
    case 'INVALID_RESPONSE':
      this.cooldown = BACKOFF_SEC;
      this.revertToPreviousValidPlan();
      break;
      
    default:
      this.cooldown = BACKOFF_SEC;
      console.warn('[LLMBoss] Unexpected error:', err.message);
  }
  
  // Circuit breaker: disable LLM temporarily after too many failures
  if (this.consecutiveFailures >= 5) {
    this.circuitBreakerActive = true;
    this.circuitBreakerTimeout = Date.now() + (5 * 60 * 1000); // 5 minutes
    console.warn('[LLMBoss] Circuit breaker activated - falling back to baseline');
  }
}
```

### 3. Advanced Plan Ingestion and Validation

**Sophisticated Plan Processing**:
```javascript
async _ingestPlan(response) {
  const { actions, reasoning, metrics, planId } = response;
  
  // Generate unique plan ID for tracking and rating
  const currentPlanId = planId || `plan_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  // Multi-stage validation pipeline
  const validationResults = await this.validatePlan(actions, metrics);
  
  if (!validationResults.safe) {
    console.warn('[LLMBoss] Plan rejected by safety validation:', validationResults.reasons);
    this.fallbackToSafePlan();
    return;
  }
  
  // Compile actions through capability registry with safety limits
  const compiledActions = [];
  const compilationErrors = [];
  
  for (const [index, action] of actions.entries()) {
    try {
      // Schema validation
      const validation = registry.validate(action);
      
      if (!validation.ok) {
        compilationErrors.push({ index, action, errors: validation.errors });
        continue;
      }
      
      // Safety bounds checking
      const safetyCheck = this.applySafetyConstraints(action);
      if (!safetyCheck.safe) {
        compilationErrors.push({ index, action, safety: safetyCheck.violations });
        continue;
      }
      
      // Compile to executable form
      const compiled = registry.compile(safetyCheck.adjustedAction || action);
      compiled._planId = currentPlanId;
      compiled._originalIndex = index;
      
      compiledActions.push(compiled);
      
    } catch (error) {
      compilationErrors.push({ index, action, error: error.message });
    }
  }
  
  // Log compilation results for analysis
  logLLM({
    type: 'plan_compilation',
    planId: currentPlanId,
    totalActions: actions.length,
    compiledActions: compiledActions.length,
    errors: compilationErrors,
    reasoning,
    timestamp: Date.now()
  });
  
  // Queue compiled actions for execution
  if (compiledActions.length > 0) {
    const queue = this.bossMgr.actionQueue[0]; // Assuming single boss
    queue.push(...compiledActions);
    
    // Track plan execution for quality rating
    this.activePlans.set(currentPlanId, {
      startTime: Date.now(),
      actions: compiledActions,
      reasoning,
      metrics: validationResults.metrics
    });
  } else {
    console.warn('[LLMBoss] No valid actions compiled from plan');
    this.fallbackToBaselineBehavior();
  }
}

// Comprehensive plan validation
async validatePlan(actions, metrics) {
  const results = {
    safe: true,
    reasons: [],
    metrics: {}
  };
  
  // Check action count limits
  if (actions.length > llmConfig.maxActionsPerPlan) {
    results.safe = false;
    results.reasons.push('tooManyActions');
  }
  
  // Resource consumption analysis
  const resourceUsage = this.calculateResourceUsage(actions);
  
  if (resourceUsage.bulletCount > llmConfig.maxBulletsPerPlan) {
    results.safe = false;
    results.reasons.push('tooManyBullets');
  }
  
  if (resourceUsage.movementDistance > llmConfig.maxMovementPerPlan) {
    results.safe = false;
    results.reasons.push('tooMuchMovement');
  }
  
  // DPS and difficulty validation using DifficultyCritic
  if (metrics) {
    const difficultyEval = evaluate(metrics, { tier: 'mid' });
    if (!difficultyEval.ok) {
      results.safe = false;
      results.reasons.push(...difficultyEval.reasons);
    }
  }
  
  // Simulate plan execution for safety
  const simulation = await this.simulatePlan(actions);
  results.metrics = simulation.metrics;
  
  if (simulation.playerSurvivalRate < 0.1) {
    results.safe = false;
    results.reasons.push('unsurvivable');
  }
  
  return results;
}
```

### 4. Adaptive Learning and Feedback System

**AI Quality Rating and Learning**:
```javascript
// Comprehensive feedback collection
collectPlanFeedback(planId, outcome) {
  const plan = this.activePlans.get(planId);
  if (!plan) return;
  
  const executionTime = Date.now() - plan.startTime;
  const feedbackEntry = {
    planId,
    outcome, // 'success', 'failure', 'timeout', 'interrupted'
    executionTime,
    originalReasoning: plan.reasoning,
    playerBehaviorDuringPlan: this.capturePlayerBehavior(),
    bossHealthChange: this.calculateBossHealthChange(plan.startTime),
    playerEngagement: this.measurePlayerEngagement(),
    difficultyRating: this.calculateDifficultyRating(plan),
    timestamp: Date.now()
  };
  
  // Store in memory for immediate LLM context
  this.feedback.push(feedbackEntry);
  
  // Persistent logging for long-term analysis
  logLLM({
    type: 'plan_feedback',
    ...feedbackEntry
  });
  
  // Adaptive parameter adjustment
  this.adjustPlanningParameters(feedbackEntry);
  
  // Clean up
  this.activePlans.delete(planId);
}

// Dynamic parameter adjustment based on success rates
adjustPlanningParameters(feedback) {
  const recentFeedback = this.feedback.slice(-10);
  const successRate = recentFeedback.filter(f => f.outcome === 'success').length / recentFeedback.length;
  
  // Adjust difficulty if success rate is too high/low
  if (successRate > 0.8) {
    // Players are finding it too easy
    this.difficultyMultiplier = Math.min(this.difficultyMultiplier * 1.1, 2.0);
    console.log('[LLMBoss] Increasing difficulty multiplier to', this.difficultyMultiplier);
  } else if (successRate < 0.3) {
    // Players are struggling too much
    this.difficultyMultiplier = Math.max(this.difficultyMultiplier * 0.9, 0.5);
    console.log('[LLMBoss] Decreasing difficulty multiplier to', this.difficultyMultiplier);
  }
  
  // Adjust LLM call frequency based on plan effectiveness
  const avgExecutionTime = recentFeedback.reduce((sum, f) => sum + f.executionTime, 0) / recentFeedback.length;
  
  if (avgExecutionTime < 2000) {
    // Plans are too short - increase planning period
    this.adaptivePlanPeriod = Math.min(this.adaptivePlanPeriod * 1.2, 10);
  } else if (avgExecutionTime > 8000) {
    // Plans are too long - decrease planning period
    this.adaptivePlanPeriod = Math.max(this.adaptivePlanPeriod * 0.8, 1);
  }
}
```

### 5. Advanced Telemetry and Monitoring Integration

**Comprehensive OpenTelemetry Integration**:
```javascript
// Detailed telemetry with custom attributes
const span = tracer.startSpan('boss.plan_execution', {
  attributes: {
    'boss.id': this.bossMgr.id[0],
    'boss.phase': this.bossMgr.phase[0],
    'boss.hp': this.bossMgr.hp[0],
    'players.count': players.length,
    'plan.actions.count': actions.length,
    'llm.provider': provider.name,
    'llm.model': provider.model
  }
});

// Nested spans for granular monitoring
const validationSpan = tracer.startSpan('plan.validation', { parent: span });
validationSpan.setAttributes({
  'validation.duration_ms': validationDuration,
  'validation.errors.count': validationErrors.length,
  'validation.safety.passed': safetyCheck.safe
});
validationSpan.end();

const compilationSpan = tracer.startSpan('plan.compilation', { parent: span });
compilationSpan.setAttributes({
  'compilation.actions.input': actions.length,
  'compilation.actions.output': compiledActions.length,
  'compilation.errors.count': compilationErrors.length
});
compilationSpan.end();

// Custom metrics for business logic
const planMetrics = {
  planGenerationLatency: deltaMs,
  planValidationLatency: validationDuration,
  planCompilationLatency: compilationDuration,
  actionsCompiledRatio: compiledActions.length / actions.length,
  difficultyScore: calculatedDifficulty,
  playerEngagementScore: engagementMetrics.score
};

span.setAttributes(planMetrics);
span.end();
```

### 6. Production Monitoring and Alerting

**Health Check and Monitoring Endpoints**:
```javascript
// Advanced health monitoring
router.get('/health', (req, res) => {
  const health = {
    llmProvider: {
      status: provider?.isHealthy() ? 'healthy' : 'unhealthy',
      lastSuccessfulCall: this.lastSuccessfulCall,
      consecutiveFailures: this.consecutiveFailures,
      circuitBreakerActive: this.circuitBreakerActive
    },
    
    planExecution: {
      activePlans: this.activePlans.size,
      queuedActions: this.bossMgr.actionQueue[0]?.length || 0,
      successRate: this.calculateRecentSuccessRate(),
      avgPlanDuration: this.calculateAvgPlanDuration()
    },
    
    performance: {
      hashCollisions: this.hashCollisionCount,
      avgSnapshotGenerationTime: this.avgSnapshotTime,
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage()
    },
    
    gameState: {
      bossActive: this.bossMgr.count > 0,
      playerCount: this.lastPlayerCount,
      currentPhase: this.bossMgr.phase[0],
      difficultyMultiplier: this.difficultyMultiplier
    }
  };
  
  const overallStatus = this.calculateOverallHealth(health);
  
  res.status(overallStatus === 'healthy' ? 200 : 503).json({
    status: overallStatus,
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    ...health
  });
});

// Real-time metrics for dashboards
router.get('/metrics', (req, res) => {
  const recentFeedback = this.feedback.slice(-50);
  
  const metrics = {
    llm: {
      callsPerMinute: this.calculateCallsPerMinute(),
      averageLatency: this.calculateAverageLatency(),
      tokenUsageRate: this.calculateTokenUsageRate(),
      errorRate: this.calculateErrorRate()
    },
    
    gameplay: {
      planSuccessRate: recentFeedback.filter(f => f.outcome === 'success').length / recentFeedback.length,
      averagePlanDuration: recentFeedback.reduce((sum, f) => sum + f.executionTime, 0) / recentFeedback.length,
      playerSurvivalRate: this.calculatePlayerSurvivalRate(),
      difficultyTrend: this.calculateDifficultyTrend()
    },
    
    system: {
      memoryGrowthRate: this.calculateMemoryGrowthRate(),
      gcPressure: this.calculateGCPressure(),
      eventLoopLag: this.measureEventLoopLag()
    }
  };
  
  res.json(metrics);
});
```

## Future Enhancements

### Planned Features
1. **Multi-Boss Coordination** - Multiple AI bosses working together with shared context
2. **Advanced Learning System** - Boss adaptation based on aggregated player success rates across sessions
3. **Dynamic Capability Loading** - Runtime capability discovery and hot-reload
4. **Visual Scripting Interface** - Web-based GUI for capability composition and testing
5. **A/B Testing Framework** - Compare different AI strategies with statistical significance
6. **Player Skill Profiling** - Individual difficulty adjustment based on player performance history

### Architecture Evolution
1. **Distributed LLM Architecture** - Load balancing across multiple providers with failover
2. **Streaming Response Processing** - Real-time action streaming from LLM for reduced latency
3. **Capability Marketplace** - Community-created boss abilities with validation and rating
4. **Comprehensive Replay System** - Record and analyze boss behaviors for training data
5. **Multi-modal AI Integration** - Image-based state analysis for more sophisticated decision making
6. **Federated Learning** - Aggregate learning across multiple game instances while preserving privacy

This LLM Boss System represents a significant innovation in game AI, providing dynamic, adaptive gameplay that emerges from real-time AI decision making rather than pre-scripted behaviors. The comprehensive error handling, telemetry integration, and feedback systems ensure production-ready reliability while maintaining the creative potential of AI-driven gameplay.