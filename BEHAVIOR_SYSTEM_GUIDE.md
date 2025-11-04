# Complete Behavior Designer Guide

## Overview
The Behavior Designer is a visual state machine editor for creating enemy AI behaviors. Enemies move through **states**, and each state defines what the enemy does.

---

## Left Panel: Inspector - State Editor

When you click a state node in the center graph, the Inspector shows all its settings.

### State Name & Settings

```
STATE NAME: combat
Duration: 2000 (milliseconds)
Next State: retreat
```

- **State Name**: Identifier for this state (e.g., "idle", "chase", "attack")
- **Duration**: How long before automatically transitioning (empty = stay forever)
- **Next State**: Which state to go to after duration expires (empty = stay in this state)

---

## The 5 Behavior Categories

### 🟢 MOVEMENT (Runs Continuously)
**What it does**: Controls how the enemy moves around the map

**All movement behaviors in this section run AT THE SAME TIME**

**Available Blocks**:
- **Move Toward**: Chase toward target (player/position)
  - `target`: "ToPlayer" or coordinates
  - `speed`: How fast (0.5 = slow, 2.0 = fast)

- **Move Away**: Run away from target
  - `target`: What to flee from
  - `speed`: Movement speed
  - `minDistance`: Stop fleeing when this far away

- **Orbit**: Circle around target
  - `target`: What to orbit around
  - `radius`: Distance from center (pixels)
  - `speed`: How fast to orbit
  - `clockwise`: true/false direction

- **Zig Zag**: Move in zigzag pattern toward target
  - `target`: Direction to zigzag toward
  - `amplitude`: How wide the zigzags are
  - `frequency`: How often to change direction

- **Circle**: Move in circular pattern (no target)
  - `radius`: Circle size
  - `speed`: How fast

- **Charge**: Rush toward target at high speed
  - `target`: What to charge at
  - `speed`: Rush speed (usually high like 3.0)
  - `duration`: How long to charge

- **Retreat**: Back away while facing target
  - `target`: What to back away from
  - `speed`: Retreat speed

- **Wander**: Random wandering movement
  - `radius`: Area to wander in
  - `speed`: Walk speed

- **Stop**: Halt all movement (stationary)

### 🔴 ATTACK (Runs Continuously)
**What it does**: Controls shooting and offensive actions

**All attack behaviors in this section run AT THE SAME TIME**

**Available Blocks**:
- **Shoot**: Fire single bullet toward target
  - `target`: "ToPlayer" or direction
  - `cooldown`: Delay between shots (ms)
  - `bulletId`: Type of bullet to fire
  - `speed`: Bullet speed
  - `count`: Usually 1 for single shot

- **Shoot Spread**: Fire multiple bullets in arc
  - `target`: Direction to shoot
  - `count`: Number of bullets (3, 5, 8, etc.)
  - `spread`: Arc angle (30 = narrow, 180 = semicircle)
  - `cooldown`: Time between volleys
  - `bulletId`: Bullet type

- **Shoot Circle**: Fire bullets in all directions (360°)
  - `count`: Number of bullets in circle (8 = every 45°)
  - `cooldown`: Time between volleys
  - `bulletId`: Bullet type

- **Shoot Spiral**: Rotating spiral pattern
  - `count`: Bullets per wave
  - `rotationSpeed`: How fast spiral rotates
  - `cooldown`: Time between waves

### 🔵 UTILITY (Runs Continuously)
**What it does**: Buffs, effects, healing, shields - special abilities

**All utility behaviors run AT THE SAME TIME as movement and attack**

**Available Blocks**:
- **Heal Over Time**: Regenerate HP
  - `healPerSecond`: HP restored per second

- **Shield**: Temporary damage reduction
  - `reduction`: Percent damage blocked (0.5 = 50%)
  - `duration`: How long shield lasts

- **Speed Boost**: Movement speed multiplier
  - `multiplier`: Speed increase (1.5 = 50% faster)
  - `duration`: How long boost lasts

- **Invulnerable**: Cannot take damage
  - `duration`: How long invulnerability lasts

### ⚡ ON ENTER (Runs Once)
**What it does**: Executes ONE TIME when entering this state

**Use for**: Initial setup, starting animations, spawn effects

**Example**:
- Play roar sound effect when boss enters "rage" state
- Teleport to specific location when entering "ambush" state
- Fire initial burst when entering "attack" state

### ⚡ ON EXIT (Runs Once)
**What it does**: Executes ONE TIME when leaving this state

**Use for**: Cleanup, ending effects, transition setup

**Example**:
- Remove speed boost when leaving "charge" state
- Play death animation when leaving any state due to death
- Reset position when exiting "stunned" state

---

## How States Work

### State Execution Flow

```
1. ENTER STATE
   └─> Run all ON ENTER behaviors (once)

2. STAY IN STATE
   └─> Run MOVEMENT behaviors (every frame)
   └─> Run ATTACK behaviors (every frame)
   └─> Run UTILITY behaviors (every frame)
   └─> Check if duration expired or transition condition met

3. EXIT STATE
   └─> Run all ON EXIT behaviors (once)
   └─> Go to NEXT state (or loop if none specified)
```

### Self-Looping vs Transitions

**Self-Looping** (no "Next State" set):
```
State: "combat"
Duration: (empty)
Next State: (empty)
→ Stays in "combat" forever, continuously running all behaviors
```

**Timed Transition**:
```
State: "charge"
Duration: 3000
Next State: "rest"
→ Charges for 3 seconds, then switches to "rest" state
```

**Chain of States**:
```
"approach" (5s) → "attack" (3s) → "retreat" (2s) → "approach" (loop)
```

---

## Real Examples

### Example 1: Simple Chaser
**Goal**: Chase player and shoot

```
State: "chase_and_shoot"
├─ 🟢 MOVEMENT
│  └─ Move Toward (target: ToPlayer, speed: 1.0)
├─ 🔴 ATTACK
│  └─ Shoot (target: ToPlayer, cooldown: 2000)
└─ Next State: (none - stays in this state forever)
```

**Result**: Enemy chases player while constantly shooting every 2 seconds

---

### Example 2: Orbiting Boss
**Goal**: Circle around player while firing spread shots

```
State: "combat"
├─ 🟢 MOVEMENT
│  └─ Orbit (target: ToPlayer, radius: 250, speed: 0.8, clockwise: true)
├─ 🔴 ATTACK
│  └─ Shoot Spread (count: 5, spread: 45, cooldown: 1000)
└─ Next State: (none - infinite loop)
```

**Result**: Boss circles player at 250 pixel distance, shooting 5-bullet spread every second

---

### Example 3: Charge Attack Pattern
**Goal**: Charge at player, attack, then retreat

```
State 1: "charging"
├─ 🟢 MOVEMENT
│  └─ Charge (target: ToPlayer, speed: 3.0)
├─ 🔴 ATTACK
│  └─ Shoot (cooldown: 500) - shoots while charging!
├─ Duration: 2000 (charge for 2 seconds)
└─ Next State: "melee_attack"

State 2: "melee_attack"
├─ 🟢 MOVEMENT
│  └─ Stop
├─ 🔴 ATTACK
│  └─ Shoot Circle (count: 8) - explosion when stopping!
├─ Duration: 500
└─ Next State: "retreat"

State 3: "retreat"
├─ 🟢 MOVEMENT
│  └─ Retreat (target: ToPlayer, speed: 1.5)
├─ Duration: 3000
└─ Next State: "charging" (loops back)
```

**Result**:
1. Charges at player for 2s while shooting
2. Stops and fires circular burst
3. Backs away for 3s
4. Repeats cycle

---

### Example 4: Enraged Boss (State Conditions)
**Goal**: Normal behavior until low HP, then go crazy

```
State 1: "normal_combat"
├─ 🟢 MOVEMENT
│  └─ Orbit (radius: 300, speed: 0.6)
├─ 🔴 ATTACK
│  └─ Shoot Spread (count: 3, cooldown: 1500)
└─ Next State: (none - but game code can force transition to "rage" at 30% HP)

State 2: "rage"
├─ ⚡ ON ENTER
│  └─ Speed Boost (multiplier: 1.5, duration: 9999)
├─ 🟢 MOVEMENT
│  └─ Orbit (radius: 200, speed: 1.2) - faster, closer orbit
├─ 🔴 ATTACK
│  ├─ Shoot Spread (count: 8, cooldown: 800) - more bullets, faster
│  └─ Shoot Circle (count: 12, cooldown: 3000) - periodic explosion
└─ Next State: (none - stays enraged until death)
```

---

## Key Concepts

### Simultaneous Behaviors
**ALL behaviors in Movement/Attack/Utility run at the same time!**

```
State: "dancing_death"
├─ 🟢 MOVEMENT
│  ├─ Orbit (player, radius: 200) ← BOTH HAPPEN
│  └─ Zig Zag (amplitude: 50) ← AT THE SAME TIME
├─ 🔴 ATTACK
│  ├─ Shoot (cooldown: 500) ← BOTH HAPPEN
│  └─ Shoot Circle (count: 8, cooldown: 2000) ← AT THE SAME TIME
```

This enemy orbits in a zigzag pattern while shooting single shots AND periodic circles!

---

### State Duration vs Behavior Duration

**State Duration**: When to switch states
**Behavior Cooldown**: How often that behavior fires

```
State: "burst_phase"
Duration: 5000 ← State lasts 5 seconds total
├─ Shoot (cooldown: 500) ← Fires 10 times (5000ms / 500ms)
└─ Next State: "rest"
```

---

### Empty "Next State" = Loop Forever

If you DON'T set a next state:
- State runs indefinitely
- Behaviors keep executing continuously
- Only way out is death or game code forcing transition

If you DO set a next state:
- Transitions after duration expires
- Can chain multiple states together
- Creates phases/patterns

---

## Building Blocks Palette

Located at the bottom of each category section.

**Movement Tab**: All locomotion blocks
**Shooting Tab**: All attack blocks
**State Tab**: Control flow and utility blocks

Click a block to add it to the current category section.

---

## Center Panel: State Machine Graph

### Visual Elements

**Green Node**: Initial/starting state (first in the list)
**Blue Node**: Currently selected state (shows in Inspector)
**Gray Nodes**: Other states

**Green Arrows**: Transitions between states (based on "Next State")

**State Info Displayed**:
```
┌─────────────┐
│ chase       │ ← State name
│ 🟢1 🔴1 🔵0 │ ← Behavior counts (movement/attack/utility)
│ 3000ms      │ ← Duration (if set)
└─────────────┘
```

### Graph Interactions

**Double-click empty space**: Create new state
**Click state**: Select it (shows in Inspector)
**Click & drag state**: Move it around canvas
**Shift+click two states**: Create transition arrow between them

---

## Right Panel: Test Arena

### What It Does
Simulates your behavior in a small 8x8 test environment

**Red square**: Test enemy executing your behaviors
**Blue square**: Simulated player
**Gray squares**: Walls for collision testing

### Controls
**Click to move player**: Test how enemy reacts to player movement
**Start button**: Begin behavior execution
**Stop button**: Pause execution
**Reset button**: Reset to initial positions

Use this to verify your behaviors work before spawning in the real game!

---

## Workflow: Creating an Enemy

### Step 1: Load or Create
- **Load existing**: Select from "Select Enemy..." dropdown (e.g., ChargingShooter)
- **Create new**: Click "+ New Enemy" button

### Step 2: Design States
1. Double-click graph to add a state
2. Click state to select it
3. In Inspector, set state name and duration
4. Add behaviors from Building Blocks palette:
   - Drag movement patterns to 🟢 MOVEMENT section
   - Drag attacks to 🔴 ATTACK section
   - Add any buffs to 🔵 UTILITY section

### Step 3: Configure Transitions
1. Set "Next State" field if you want timed transitions
2. Or Shift+click states to create visual connections

### Step 4: Test
1. Click "Start" in Test Arena
2. Watch the red enemy execute your behaviors
3. Move the blue player square to test reactions
4. Iterate and adjust parameters

### Step 5: Save
1. Click "💾 Save Enemy to Backend"
2. Enemy is saved to `/public/assets/entities/enemies.json`

### Step 6: Spawn in Game
1. Edit `/config/world-spawns.js`
2. Add your enemy:
   ```javascript
   { id: 'my_enemy', x: 30, y: 30 }
   ```
3. Restart server
4. Play game and fight your creation!

---

## Tips & Tricks

### Creating Difficulty Tiers

**Easy Enemy**:
- Slow movement (speed: 0.5)
- Long cooldowns (cooldown: 3000)
- Single shot attacks

**Hard Enemy**:
- Fast movement (speed: 1.5)
- Short cooldowns (cooldown: 500)
- Spread/Circle attacks with high bullet count

### Common Patterns

**Hit-and-Run**:
```
charge (3s) → attack (1s) → retreat (2s) → loop
```

**Stationary Turret**:
```
Movement: Stop
Attack: Shoot Circle (count: 16, cooldown: 2000)
```

**Aggressive Chaser**:
```
Movement: Move Toward (speed: 2.0)
Attack: Shoot (cooldown: 800)
```

**Defensive Kiter**:
```
Movement: Move Away (minDistance: 300)
Attack: Shoot Spread (count: 5)
```

### Combining Behaviors for Complexity

Don't be afraid to add multiple behaviors to one category!

```
State: "chaos_mode"
├─ 🟢 MOVEMENT
│  ├─ Orbit (radius: 200)
│  └─ Zig Zag (amplitude: 30) ← Makes orbit wobble
├─ 🔴 ATTACK
│  ├─ Shoot (cooldown: 300) ← Constant stream
│  └─ Shoot Circle (cooldown: 2000) ← Periodic burst
│  └─ Shoot Spread (cooldown: 1500) ← Another periodic pattern
```

This creates incredibly complex behavior from simple building blocks!

---

## Troubleshooting

**"Script error" on load**:
- Refresh the page
- Check browser console for details

**Enemy doesn't move**:
- Make sure you have behaviors in 🟢 MOVEMENT section
- Check that speed parameter isn't 0

**Enemy doesn't shoot**:
- Verify behaviors in 🔴 ATTACK section
- Check cooldown isn't too long
- Make sure bulletId exists in enemies.json

**State doesn't transition**:
- Set Duration field (can't be empty for transitions)
- Set Next State field to target state name
- State names are case-sensitive!

**Behaviors not saving**:
- Click "💾 Save Enemy to Backend" button
- Check server is running (port 3001)
- Look for success alert message

---

## Summary

**The behavior system lets enemies do multiple things simultaneously:**

- **🟢 MOVEMENT**: How they move (chase, orbit, zigzag)
- **🔴 ATTACK**: How they shoot (single, spread, circle)
- **🔵 UTILITY**: Buffs and special effects
- **⚡ ON ENTER/EXIT**: One-time effects when changing states

**States can loop forever (no next state) or transition after a duration.**

**All behaviors in a category run at the same time, creating complex emergent patterns from simple building blocks.**

Now go create some epic boss fights!
