# 🎮 Complete Game Systems Guide

**Everything you need to know about creating content in this game**

## Table of Contents
1. [Creating New Enemies](#creating-new-enemies)
2. [Spawning Enemies](#spawning-enemies)
3. [Defining Player Attacks](#defining-player-attacks)
4. [Creating Items/Weapons](#creating-itemsweapons)
5. [Configuring Enemy Attacks](#configuring-enemy-attacks)
6. [Building Block System](#building-block-system)
7. [LLM Boss Integration](#llm-boss-integration)

---

## Creating New Enemies

### Step 1: Define Enemy Stats

Edit `public/assets/entities/enemies.json`:

```json
{
  "id": "frost_golem",
  "name": "Frost Golem",
  "sprite": "frost_golem",
  "hp": 500,
  "speed": 8,
  "width": 1.5,
  "height": 1.5,
  "renderScale": 2.5,
  "attack": {
    "bulletId": "ice_shard",        // Projectile type (from config/projectiles.js)
    "cooldown": 1200,                // Milliseconds between shots
    "speed": 15,                     // Projectile speed
    "lifetime": 3000,                // Projectile lifetime (ms)
    "count": 3,                      // Number of projectiles per shot
    "spread": 30,                    // Spread angle in degrees
    "inaccuracy": 5                  // Random angle variation
  },
  "ai": { "behaviorTree": "FrostGolemBT" }
}
```

### Step 2: Create Projectile Type

Edit `config/projectiles.js`:

```javascript
ice_shard: {
  id: 'ice_shard',
  name: 'Ice Shard',
  description: 'Freezing projectile',
  sprite: 'ice_shard',
  speed: 15,
  lifetime: 3.0,
  damage: 18,
  size: 0.5,
  piercing: false,
  bounces: 0,
  homing: false,
  effects: ['slow'],      // Apply slow effect
  color: '#87CEEB',
  trail: true,
  trailColor: '#87CEEB40',
},
```

### Step 3: Create Behavior

Edit `src/Behaviours/BehaviorSystem.js`:

```javascript
// In initDefaultBehaviors():
this.registerBehaviorTemplate(7, this.createFrostGolemBehavior());

// Add method:
createFrostGolemBehavior() {
  const idleState = new BehaviorState('idle', [
    new Behaviors.Wander(0.4)
  ]);

  const attackState = new BehaviorState('attack', [
    new Behaviors.Chase(0.8, 150),
    new Behaviors.Shoot(1.2, 3, Math.PI/6)  // Triple shot
  ]);

  const rageState = new BehaviorState('rage', [
    new Behaviors.Chase(1.2, 100),
    new Behaviors.Shoot(0.8, 5, Math.PI/4)  // Faster, more bullets
  ]);

  // Transitions
  idleState.addTransition(new Transitions.PlayerWithinRange(200, attackState));
  attackState.addTransition(new Transitions.NoPlayerWithinRange(250, idleState));
  attackState.addTransition(new Transitions.HealthBelow(0.3, rageState));
  rageState.addTransition(new Transitions.HealthAbove(0.6, attackState));

  return idleState;
}
```

### Step 4: Register Enemy ID Mapping

The system automatically maps enemy IDs! Just add to `enemies.json` and it works.

---

## Spawning Enemies

### Option 1: Config File (Recommended)

Edit `config/world-spawns.js`:

```javascript
overworld: {
  spawns: [
    { id: 'frost_golem', x: 70, y: 50, comment: 'Ice cave guardian' },
    { id: 'frost_golem', x: 75, y: 52 },
    // ... more spawns
  ]
}
```

Restart server to apply.

### Option 2: Map JSON Files

Edit map files in `public/maps/*.json`:

```json
{
  "enemies": [
    { "id": "frost_golem", "x": 16, "y": 16 }
  ]
}
```

### Option 3: Dynamic Spawning (Code)

In `Server.js` or custom spawn logic:

```javascript
worldCtx.enemyMgr.spawnEnemyById('frost_golem', x, y, worldId);
```

---

## Defining Player Attacks

### Step 1: Create Item/Weapon

Edit `src/assets/items.json`:

```json
{
  "id": 2001,
  "name": "Frostbite Staff",
  "tier": 3,
  "description": "Staff that fires freezing projectiles",
  "type": "staff",
  "damage": 145,
  "rateOfFire": 0.8,                    // Attacks per second
  "projectile": "ice_shard",            // Links to projectiles.js!
  "projectileCount": 3,                  // Fire 3 projectiles
  "projectileSpread": 0.3,              // Spread angle (radians)
  "attackPattern": "triple_spread",      // Use pre-defined pattern
  "bagType": 3,
  "soulbound": true,
  "spriteSheet": "lofi_obj",
  "spriteX": 0,
  "spriteY": 16
}
```

### Step 2: Player Attack System

Player attacks are defined by **equipped weapon**:
- Each weapon has a `projectile` type
- Weapon determines `damage`, `rateOfFire`, `projectileCount`
- When player fires, creates projectiles based on weapon

**Current Flow:**
1. Player presses Space/Click
2. Game checks equipped weapon
3. Spawns projectiles defined by weapon
4. Projectiles use template from `config/projectiles.js`

---

## Creating Items/Weapons

### Weapon Types

| Type | Description | Example |
|------|-------------|---------|
| `sword` | Melee/close range | Single powerful shot |
| `staff` | Magic projectiles | Multi-shot, special effects |
| `bow` | Fast projectiles | Single or double shot |
| `wand` | Rapid fire | Many weak projectiles |
| `dagger` | Very fast | Single precise shot |

### Tier System

- **Tier 1**: Common (bagType: 1) - Basic stats
- **Tier 2**: Uncommon (bagType: 2) - +20% stats
- **Tier 3**: Rare (bagType: 3) - +40% stats, special projectiles
- **Tier 4**: Epic (bagType: 5) - +60% stats, powerful projectiles
- **Tier 5**: Legendary (bagType: 6) - +80% stats, unique abilities

### Example: Creating a New Weapon

```json
{
  "id": 3001,
  "name": "Lightning Arc Staff",
  "tier": 4,
  "description": "Channels the fury of the storm",
  "type": "staff",
  "damage": 175,
  "rateOfFire": 1.2,
  "projectile": "lightning_bolt",        // New projectile!
  "projectileCount": 1,
  "attackPattern": "single",
  "specialAbility": "chain_lightning",   // Future feature
  "bagType": 5,
  "soulbound": true,
  "spriteSheet": "lofi_obj",
  "spriteX": 192,
  "spriteY": 0
}
```

Then define the projectile in `config/projectiles.js`:

```javascript
lightning_bolt: {
  id: 'lightning_bolt',
  name: 'Lightning Bolt',
  sprite: 'lightning',
  speed: 30,
  lifetime: 2.0,
  damage: 40,
  size: 0.4,
  piercing: true,
  maxPierces: 3,
  effects: ['stun'],
  color: '#FFFF00',
  trail: true,
  glow: true,
  particleEffect: 'lightning_spark',
},
```

---

## Configuring Enemy Attacks

### Visual Editor

1. Open `http://localhost:3000/editor/behavior-designer.html`
2. Load enemy type
3. Configure:
   - Projectile type (dropdown from `projectiles.js`)
   - Bullet speed
   - Lifetime
   - Damage
   - Count
   - Spread
   - Fire rate
4. Export JSON and save to `enemies.json`

### Manual Configuration

Edit enemy attack properties directly:

```json
"attack": {
  "bulletId": "demon_fire",      // Choose from projectiles.js
  "cooldown": 800,                // Lower = faster fire rate
  "speed": 12,                    // Projectile speed
  "lifetime": 2000,               // How long projectile lives (ms)
  "count": 5,                     // Number of projectiles per shot
  "spread": 45,                   // Spread angle (degrees)
  "inaccuracy": 5,                // Random variance (degrees)
  "pattern": "five_spread"        // Optional: use pre-defined pattern
}
```

### Attack Patterns

Pre-defined patterns in `config/projectiles.js`:

```javascript
AttackPatterns = {
  single: { /* 1 projectile */ },
  double: { /* 2 projectiles */ },
  triple_spread: { /* 3 spread */ },
  five_spread: { /* 5 spread */ },
  eight_ring: { /* 8 in circle */ },
  spiral_burst: { /* 12 spiral */ },
  shotgun: { /* 7 random spread */ },
}
```

Use in enemy definition:

```json
"attack": {
  "pattern": "eight_ring",
  "cooldown": 2000
}
```

---

## Building Block System

### Core Components

#### 1. Projectiles (`config/projectiles.js`)
- **Basic properties**: speed, damage, size, lifetime
- **Advanced**: piercing, bouncing, homing, trails
- **Effects**: status effects, explosions, particles

#### 2. Attack Patterns
- **Pre-composed** projectile groupings
- **Parameterized**: count, spread, delay
- **Reusable** across all entities

#### 3. Behaviors (`src/Behaviours/Behaviors.js`)
- **Movement**: Chase, RunAway, Wander, Orbit, Swirl
- **Combat**: Shoot, ChargeShot, BurstFire
- **Utility**: Wait, Teleport, Shield

#### 4. Transitions (`src/Behaviours/Transitions.js`)
- **Range-based**: PlayerWithinRange, NoPlayerWithinRange
- **Health-based**: HealthBelow, HealthAbove
- **Time-based**: TimedTransition

### Creating a Complex Attack

**Example: Boss with 3-phase attack**

```javascript
// Phase 1: Slow ring pattern
const phase1 = new BehaviorState('phase1', [
  new Behaviors.Wander(0.5),
  new Behaviors.Shoot(2.0, 8, Math.PI/4)  // 8-shot ring every 2s
]);

// Phase 2: Fast spiral
const phase2 = new BehaviorState('phase2', [
  new Behaviors.Orbit(1.2, 10),           // Circle player
  new Behaviors.Shoot(1.0, 3, Math.PI/12) // Rapid triple shots
]);

// Phase 3: Rage mode
const phase3 = new BehaviorState('phase3', [
  new Behaviors.Chase(1.5, 100),
  new Behaviors.Shoot(0.5, 12, Math.PI/6) // Massive bullet spray
]);

// Transitions
phase1.addTransition(new Transitions.HealthBelow(0.66, phase2));
phase2.addTransition(new Transitions.HealthBelow(0.33, phase3));
```

### Composition Pattern

```javascript
// 1. Define projectile
ProjectileTypes.boss_mega_shot = {
  speed: 25,
  damage: 50,
  size: 1.0,
  effects: ['burn', 'slow'],
  onHit: { type: 'explosion', radius: 3.0 }
};

// 2. Create attack pattern
AttackPatterns.boss_ultimate = {
  projectiles: Array.from({ length: 16 }, (_, i) => ({
    type: 'boss_mega_shot',
    angleOffset: (Math.PI * 2 / 16) * i,
    delay: i * 0.1
  }))
};

// 3. Use in behavior
const ultimateAttack = new BehaviorState('ultimate', [
  new Behaviors.CustomShoot({
    pattern: 'boss_ultimate',
    cooldown: 5.0
  })
]);
```

---

## LLM Boss Integration

### How LLM Will Use This System

The LLM boss system can **dynamically generate attacks** using these building blocks:

```javascript
// LLM receives context
const context = {
  availableProjectiles: ProjectileTypes,
  availablePatterns: AttackPatterns,
  availableBehaviors: Behaviors,
  bossPhase: 2,
  playerHealth: 0.4,
  difficulty: 'hard'
};

// LLM generates attack JSON
const llmAttack = {
  name: "Adaptive Firestorm",
  projectile: "demon_fire",
  pattern: "spiral_burst",
  modifications: {
    speed: 20,
    damage: 35,
    lifetime: 4.0,
    effects: ['burn']
  },
  behavior: "orbit_and_shoot",
  duration: 8.0
};

// System converts to actual attack
bossController.executeGeneratedAttack(llmAttack);
```

### Building Block Versioning

```javascript
// In config/projectiles.js
export const BuildingBlockVersion = "1.0.0";

// LLM knows what's available
export const LLMCapabilityRegistry = {
  version: "1.0.0",
  projectiles: Object.keys(ProjectileTypes),
  patterns: Object.keys(AttackPatterns),
  behaviors: Object.keys(Behaviors),
  transitions: Object.keys(Transitions),
  effects: Object.keys(StatusEffects)
};
```

### Example LLM Prompt

```
You are a boss AI controller with access to these building blocks:

Projectiles: basic_shot, demon_fire, homing_missile, explosive_shot, ...
Patterns: single, triple_spread, eight_ring, spiral_burst, ...
Behaviors: Chase, Orbit, Shoot, Swirl, ...
Effects: poison, burn, slow, stun, ...

Current situation:
- Boss HP: 45%
- Player HP: 60%
- Phase: 2
- Time in phase: 15s

Generate a new attack that:
1. Is challenging but fair
2. Uses 2-3 building blocks
3. Follows the pattern format

Output format:
{
  "attack": {
    "projectile": "<type>",
    "pattern": "<pattern>",
    "behavior": "<behavior>",
    "cooldown": <number>
  }
}
```

---

## Complete Flow Example

### Creating "Shadow Assassin" Enemy

**1. Design**
- Fast melee enemy
- Teleports behind player
- Throws poison daggers

**2. Projectile** (`config/projectiles.js`)
```javascript
shadow_dagger: {
  id: 'shadow_dagger',
  speed: 28,
  damage: 15,
  effects: ['poison_weak', 'bleed'],
  trail: true,
  color: '#4B0082'
}
```

**3. Enemy Stats** (`public/assets/entities/enemies.json`)
```json
{
  "id": "shadow_assassin",
  "name": "Shadow Assassin",
  "hp": 120,
  "speed": 18,
  "attack": {
    "bulletId": "shadow_dagger",
    "cooldown": 1500,
    "count": 2,
    "spread": 20
  },
  "ai": { "behaviorTree": "ShadowAssassinBT" }
}
```

**4. Behavior** (`src/Behaviours/BehaviorSystem.js`)
```javascript
createShadowAssassinBehavior() {
  const stealthState = new BehaviorState('stealth', [
    new Behaviors.Wander(1.2)
  ]);

  const teleportState = new BehaviorState('teleport', [
    new Behaviors.TeleportBehindPlayer(),
    new Behaviors.Shoot(1.5, 2, Math.PI/12)
  ]);

  const chaseState = new BehaviorState('chase', [
    new Behaviors.Chase(1.8, 80),
    new Behaviors.Shoot(1.5, 2, Math.PI/12)
  ]);

  stealthState.addTransition(
    new Transitions.PlayerWithinRange(150, teleportState)
  );
  teleportState.addTransition(
    new Transitions.TimedTransition(3.0, chaseState)
  );
  chaseState.addTransition(
    new Transitions.NoPlayerWithinRange(200, stealthState)
  );

  return stealthState;
}
```

**5. Spawn** (`config/world-spawns.js`)
```javascript
overworld: {
  spawns: [
    { id: 'shadow_assassin', x: 80, y: 60, comment: 'Dark forest ambush' }
  ]
}
```

**6. Test!**
- Restart server
- Navigate to coordinates
- Debug with enemy logs

---

## Quick Reference

### File Structure
```
config/
├── projectiles.js       # ALL projectile definitions
├── world-spawns.js      # Enemy spawn positions
└── README.md            # Spawn system docs

public/assets/entities/
└── enemies.json         # Enemy stats & attacks

src/Behaviours/
├── BehaviorSystem.js    # Behavior state machines
├── Behaviors.js         # Movement & combat behaviors
└── Transitions.js       # State transition conditions

src/assets/
└── items.json          # Weapons & items

public/editor/
├── behavior-designer.html  # Visual behavior editor
└── enemyEditor.html        # Enemy stats editor
```

### Commands
```bash
# Restart server
npm start

# Test enemy
# Navigate to spawn coordinates in-game
# Check server logs for:
# - [SPAWNS] Loaded X enemies
# - 🤖 [ENEMY STATE] transitions
# - 🔫 [ENEMY SHOOT] firing bullets
```

### Debug Logs
- **Server**: Every 3s enemy positions/states
- **Client**: Every 5s client enemy positions
- **State transitions**: Real-time
- **Shooting**: Every shot logged

---

## Next Steps

1. ✅ **Projectiles defined** - All building blocks ready
2. 🔄 **Update enemies.json** - Use new projectile IDs
3. 🔄 **Connect to behavior editor** - Load projectile types
4. 🔄 **Player equipment system** - Drag-drop weapons
5. 🔄 **LLM integration** - Use building blocks for dynamic attacks

Ready to make this game INFINITELY expandable! 🚀
