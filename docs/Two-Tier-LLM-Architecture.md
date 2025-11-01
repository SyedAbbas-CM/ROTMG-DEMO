# Two-Tier LLM Boss Architecture

**Status:** Design Complete
**Date:** October 24, 2025

---

## 🎯 Vision

Create an AI boss that **adapts tactically in real-time** AND **learns strategically** to create new attacks based on player behavior.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAME LOOP (60 FPS)                        │
└───────────┬───────────────────────────────────┬─────────────────┘
            │                                   │
            ▼                                   ▼
   ┌────────────────────┐            ┌────────────────────┐
   │  TIER 1: TACTICAL  │            │ TIER 2: STRATEGIC  │
   │  (Immediate)       │            │ (Batched Learning) │
   └────────────────────┘            └────────────────────┘

   Every 10-30 seconds               Every 5-10 minutes
   Uses existing attacks             Creates NEW attacks
   Fast model (Flash-Lite)          Smart model (Pro)
   Low latency (~1s)                High latency (~5s)
   Per-snapshot decision            Multi-snapshot analysis
```

---

## ⚡ Tier 1: Tactical Adaptation (Immediate)

### Purpose
Make the boss **responsive and intelligent** using existing capabilities.

### Characteristics
- **Frequency:** Every 10-30 seconds (adaptive based on combat intensity)
- **Model:** Gemini 2.5 Flash-Lite (15 RPM, 1000 RPD free)
- **Latency:** ~1-2 seconds
- **Input:** Current game snapshot
- **Output:** Sequence of existing attack actions

### Decision Types
1. **Movement:** Where to position based on player locations
2. **Attack Selection:** Which existing attack to use
3. **Timing:** When to be aggressive vs defensive
4. **Phase Transitions:** When to change tactics

### Example Snapshot
```javascript
{
  boss: {
    x: 32, y: 32,
    hp: 450, maxHp: 1000,
    phase: 1,
    lastAttack: "radial_burst",
    timeSinceAttack: 2.5
  },
  players: [
    { id: "p1", x: 40, y: 35, hp: 80, distance: 10.6, velocity: {dx: -1, dy: 0.5} },
    { id: "p2", x: 25, y: 28, hp: 100, distance: 8.4, velocity: {dx: 0.3, dy: -0.8} }
  ],
  recentDamage: {
    taken: 150,  // Last 5 seconds
    dealt: 45
  },
  availableAbilities: ["dash", "radial_burst", "projectile_spread", "wait"]
}
```

### Example Response
```javascript
{
  intent: "Players clustered - burst then reposition",
  priority: "high",  // Clear existing queue
  actions: [
    { ability: "radial_burst", args: { projectiles: 24 } },
    { ability: "dash", args: { dx: -5, dy: 0, speed: 15, duration: 0.4 } },
    { ability: "wait", args: { duration: 1.5 } },
    { ability: "projectile_spread", args: { count: 12, arc: 45 } }
  ],
  self_score: 0.82
}
```

### Optimization
- **Request Compression:** Don't send full player history, just relevant state
- **Caching:** Cache common scenarios (1v1, 2+ players, low HP)
- **Adaptive Frequency:**
  ```javascript
  function getNextCallDelay(gameState) {
    if (gameState.boss.hp < gameState.boss.maxHp * 0.25) return 10;  // Critical
    if (gameState.activePlayers > 2) return 15;  // Combat
    if (gameState.activePlayers === 0) return 60; // Idle
    return 30;  // Default
  }
  ```

---

## 🧠 Tier 2: Strategic Learning (Batched)

### Purpose
**Analyze gameplay patterns** and **generate new attack capabilities** the boss has never used before.

### Characteristics
- **Frequency:** Every 5-10 minutes OR after significant events
- **Model:** Gemini 2.5 Pro (5 RPM, 100 RPD free)
- **Latency:** ~5-10 seconds (acceptable - not real-time)
- **Input:** Batch of gameplay history + metrics
- **Output:** New capability definitions OR strategic adjustments

### Batch Contents
```javascript
{
  sessionId: "game_12345",
  duration: 300,  // 5 minutes
  snapshots: [
    // 5-10 key moments (not every snapshot!)
    {
      timestamp: 0,
      event: "combat_start",
      players: [...],
      bossActions: ["radial_burst", "dash"],
      outcome: { damageDealt: 120, damageTaken: 80 }
    },
    {
      timestamp: 45,
      event: "phase_transition",
      boss: { hp: 750, phase: 1 },
      players: [...]
    },
    // ... more key moments
  ],
  aggregateMetrics: {
    totalDamageDealt: 1200,
    totalDamageTaken: 850,
    playerDeaths: 1,
    mostUsedAttack: "radial_burst" (used 15 times),
    leastEffective: "projectile_spread" (avg 8 dmg per use),
    playerPatterns: {
      averageDistance: 12.5,
      dodgeSuccess: 0.65,
      aggressionLevel: "medium"
    }
  },
  currentCapabilities: 4  // Limited arsenal
}
```

### Strategic Decisions

#### Option A: Generate New Attack
```javascript
{
  analysis: "Players maintain 10-15 unit distance. Current attacks don't punish this range effectively. Need mid-range threat.",
  recommendation: "create_capability",
  define_component: {
    manifest: {
      "$id": "Emitter:Homing@1.0.0",
      "type": "object",
      "properties": {
        "count": { "type": "number", "minimum": 1, "maximum": 8 },
        "speed": { "type": "number", "minimum": 5, "maximum": 20 },
        "turnRate": { "type": "number", "minimum": 0.5, "maximum": 3.0 },
        "lifetime": { "type": "number", "minimum": 2, "maximum": 8 }
      }
    },
    impl: `
      export function compile(brick) {
        return {
          ability: 'homing_missiles',
          args: {
            count: brick.count ?? 4,
            speed: brick.speed ?? 12,
            turnRate: brick.turnRate ?? 1.5,
            lifetime: brick.lifetime ?? 5
          },
          _capType: brick.type
        };
      }

      export function invoke(node, state, { dt, bossMgr, bulletMgr, players }) {
        if (!state.init) {
          state.init = true;
          const bossX = bossMgr.x[0];
          const bossY = bossMgr.y[0];

          for (let i = 0; i < node.args.count; i++) {
            const angle = (Math.PI * 2 * i) / node.args.count;
            bulletMgr.spawn(
              bossX, bossY,
              Math.cos(angle) * node.args.speed,
              Math.sin(angle) * node.args.speed,
              {
                damage: 25,
                homing: true,
                turnRate: node.args.turnRate,
                lifetime: node.args.lifetime
              }
            );
          }
        }
        return true;  // One-shot action
      }
    `
  },
  testPlan: {
    scenario: "2 players at 12-unit distance",
    expectedImprovement: "30% more damage dealt",
    riskLevel: "medium"
  }
}
```

#### Option B: Adjust Strategy
```javascript
{
  analysis: "Boss is too aggressive, taking 40% more damage than dealing. Need more defensive windows.",
  recommendation: "tactical_adjustment",
  adjustments: {
    increaseWaitTime: 2.0,  // More breathing room
    preferredRange: "medium",  // Stay farther from players
    retreatThreshold: 0.3,  // Retreat at 30% HP instead of 15%
    newTactics: [
      "dash_away_when_surrounded",
      "use_projectile_spread_while_retreating"
    ]
  }
}
```

### Batch Triggers
1. **Time-based:** Every 5-10 minutes
2. **Event-based:**
   - Boss dies (analyze what went wrong)
   - All players die (analyze what worked)
   - Phase transition
   - New players join (skill level changed)

---

## 🔄 Request Batching System

### Problem
Current: 28,800 requests/day (one every 3 seconds)
Free tier: 250-1,000 requests/day per model

### Solution: Intelligent Batching

```javascript
class TwoTierLLMController {
  constructor() {
    // Tier 1: Tactical (immediate)
    this.tacticalProvider = new GeminiProvider(apiKey, {
      model: 'gemini-2.5-flash-lite'
    });
    this.tacticalCooldown = 0;
    this.tacticalPeriod = 30;  // Adaptive 10-60s

    // Tier 2: Strategic (batched)
    this.strategicProvider = new GeminiProvider(apiKey, {
      model: 'gemini-2.5-pro'
    });
    this.gameplayHistory = [];
    this.lastStrategicCall = Date.now();
    this.strategicPeriod = 300000;  // 5 minutes
  }

  async tick(dt, gameState) {
    // Tier 1: Tactical decisions (frequent)
    this.tacticalCooldown -= dt;
    if (this.tacticalCooldown <= 0) {
      await this.makeTacticalDecision(gameState);
      this.tacticalCooldown = this.getAdaptiveDelay(gameState);
    }

    // Tier 2: Strategic learning (infrequent)
    if (Date.now() - this.lastStrategicCall >= this.strategicPeriod) {
      await this.makeStrategicDecision();
      this.lastStrategicCall = Date.now();
    }

    // Record history for batching
    if (this.isKeyMoment(gameState)) {
      this.recordSnapshot(gameState);
    }
  }

  isKeyMoment(gameState) {
    return (
      gameState.phaseChanged ||
      gameState.playerJoined ||
      gameState.playerDied ||
      gameState.boss.hpChanged > 100 ||
      Math.random() < 0.05  // Random 5% sampling
    );
  }
}
```

### Request Frequency Comparison

| Configuration | Tactical Calls | Strategic Calls | Total/Day | Fits Free Tier? |
|---------------|----------------|-----------------|-----------|-----------------|
| **Current** | 28,800 | 0 | 28,800 | ❌ No |
| **Two-Tier** | 2,880 | 288 | 3,168 | ✅ Yes (Flash-Lite) |
| **With Caching** | 1,440 | 288 | 1,728 | ✅ Yes (both models) |

---

## 🧱 Foundational Building Blocks

### The Problem
Currently only 4 abilities exist. LLM can request 14 others that fail silently.

### The Solution: Composable Primitives

Instead of pre-defining all possible attacks, define **atomic building blocks** that the LLM can combine.

### Core Primitives (Universal)

#### 1. **Emitter** (Spawn projectiles)
```javascript
{
  type: "emitter",
  pattern: "radial" | "cone" | "spiral" | "grid" | "random",
  count: 1-100,
  spread: 0-360,  // degrees
  velocity: { speed: 1-30, variance: 0-1 },
  projectile: {
    damage: 10-100,
    size: 1-10,
    lifetime: 0.5-10,
    behavior: "straight" | "homing" | "accelerating" | "wavy"
  }
}
```

#### 2. **Movement** (Boss positioning)
```javascript
{
  type: "movement",
  mode: "dash" | "teleport" | "circle" | "retreat" | "chase",
  target: { x, y } | "nearest_player" | "safest_spot",
  speed: 1-30,
  duration: 0.1-5
}
```

#### 3. **Modifier** (Change projectile behavior)
```javascript
{
  type: "modifier",
  target: "existing_bullets" | "next_bullets",
  effect: {
    homing: { enabled: true, turnRate: 0.5-3.0 },
    acceleration: { rate: -10 to 10 },
    split: { count: 2-8, angle: 0-180 },
    bounce: { count: 1-5 },
    gravity: { strength: -10 to 10 }
  }
}
```

#### 4. **Conditional** (Reactive logic)
```javascript
{
  type: "conditional",
  condition: {
    if: "boss_hp < 0.5" | "players_within_range(10)" | "time_since_last > 5",
    then: [ /* actions */ ],
    else: [ /* actions */ ]
  }
}
```

#### 5. **Timer** (Delays and loops)
```javascript
{
  type: "timer",
  mode: "wait" | "repeat" | "interval",
  duration: 0.1-30,
  actions: [ /* repeated actions */ ]
}
```

### Composability Example

**LLM Creates "Spiral Homing Burst":**
```javascript
{
  explain: "Create spiraling homing missiles that accelerate",
  actions: [
    {
      type: "emitter",
      pattern: "spiral",
      count: 16,
      spread: 360,
      velocity: { speed: 8, variance: 0.2 },
      projectile: { damage: 30, lifetime: 6, behavior: "straight" }
    },
    {
      type: "modifier",
      target: "next_bullets",
      effect: {
        homing: { enabled: true, turnRate: 1.2 },
        acceleration: { rate: 2.5 }
      }
    },
    {
      type: "movement",
      mode: "dash",
      target: { x: -10, y: 0 },  // Relative
      speed: 20,
      duration: 0.3
    }
  ]
}
```

This is **infinite variety** from just 5 primitives!

---

## 📊 Implementation Phases

### Phase 1: Two-Tier System (Week 1)
- [x] Fix Gemini API
- [ ] Implement TwoTierLLMController
- [ ] Add tactical call handler
- [ ] Add strategic batch collector
- [ ] Test request frequency

### Phase 2: Building Blocks (Week 2)
- [ ] Design primitive schemas
- [ ] Implement Emitter primitive
- [ ] Implement Movement primitive
- [ ] Implement Modifier primitive
- [ ] Update LLM prompts

### Phase 3: Advanced Features (Week 3)
- [ ] Add Conditional primitive
- [ ] Add Timer primitive
- [ ] Implement capability caching
- [ ] Add safety validation

### Phase 4: Optimization (Week 4)
- [ ] Adaptive frequency tuning
- [ ] Response caching
- [ ] Multi-model routing
- [ ] Performance metrics

---

## 💰 Cost Analysis (24/7 Server)

| Tier | Calls/Day | Model | Cost/Day | Cost/Month |
|------|-----------|-------|----------|------------|
| Tactical | 2,880 | Flash-Lite (free) | $0 | $0 |
| Strategic | 288 | Pro (free) | $0 | $0 |
| **Total** | **3,168** | **Mixed** | **$0** | **$0** |

✅ **Completely free** even with 24/7 operation!

---

## 🎮 Gameplay Impact

### Before (Current)
- Boss uses 4 pre-programmed attacks randomly
- No adaptation to player skill
- Repetitive and predictable
- Fixed difficulty

### After (Two-Tier)
- Boss adapts tactics every 10-30 seconds
- Creates new attacks based on player behavior
- Boss "learns" your playstyle
- Dynamic difficulty scaling
- Infinite attack variety

---

## 🔒 Safety Mechanisms

### Tactical Tier
1. **Response Validation:** Ensure actions reference existing capabilities
2. **Rate Limiting:** Max 1 call per 10 seconds
3. **Timeout:** 5 second API timeout
4. **Fallback:** If API fails, use last successful plan

### Strategic Tier
1. **Sandbox Testing:** Test new capabilities in isolation first
2. **Difficulty Critic:** Validate new attacks aren't too hard/easy
3. **Code Validation:** AST parse before eval
4. **Rollback:** Keep last 3 working capability sets

---

**Next Steps:** Implement TwoTierLLMController and test with real gameplay!
