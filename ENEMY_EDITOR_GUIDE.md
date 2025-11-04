# Enemy Editor - Complete Guide

## What Is It?

The Enemy Editor is an integrated tool for creating complete enemies with stats, sprites, and behaviors all in one place. Everything saves to `/public/assets/entities/enemies.json` - the same place where Goblin, Orc, and Red Demon are stored.

---

## How to Access

**URL**: http://localhost:3001/editor/behavior-designer.html

The top bar now says **"Enemy Editor"** instead of "Behavior Designer" because you're editing complete enemies, not just behaviors.

---

## Creating a New Enemy (Complete Workflow)

### Step 1: Click "+ New Enemy"

In the top bar, click the blue **"+ New Enemy"** button.

This creates a blank enemy with default values:
```javascript
{
  id: "custom_enemy_123456789",  // Auto-generated unique ID
  name: "New Enemy",
  sprite: "chars:goblin",         // Default sprite
  hp: 100,
  speed: 10,
  attack: {
    bulletId: "arrow",
    cooldown: 1000,
    speed: 25,
    lifetime: 2000,
    count: 1,
    spread: 0
  }
}
```

The left panel (Enemy Stats & Attacks) will open automatically.

---

### Step 2: Configure Enemy Stats (Left Panel)

#### **Enemy Sprite Section**

1. **Preview Box**: Shows 👾 icon (placeholder for actual sprite)
2. **Current Sprite**: Displays `chars:goblin` or whatever you've selected
3. **🎨 Choose Sprite Button**: Click to open visual sprite picker
4. **Manual Input**: Or type sprite name directly (format: `chars:name`)

**Using the Sprite Picker**:
- Click **"🎨 Choose Sprite"**
- A popup appears with all 30+ available sprites
- Sprites shown in grid: `goblin`, `orc`, `red_demon`, `skeleton`, `Medusa`, `Jinn`, etc.
- Click any sprite to select it
- Selected sprite is highlighted in green
- Picker closes automatically after selection

**Available Sprites** (partial list):
- **Basic Enemies**: goblin, orc, skeleton, spider, scorpion, robber
- **Demons**: red_demon, grey_demon, red_imp
- **Dragons**: green_dragon, blue_ancient_dragon, red_ancient_dragon
- **Bosses**: Medusa, Jinn, cyclops_god, beholder, dark_lord
- **Knights**: Black_Knight, red_knight, silver_knight
- **Gods**: Skeleton_God, Ent_God, Lizard_God, Minotaur_God, Cerebral_God, Filth_God, Spectre_God, flayer_god

#### **Basic Info Section**

**Enemy ID** (IMPORTANT!):
- This is what you use in `/config/world-spawns.js`
- Must be unique and lowercase
- Use underscores, no spaces
- Example: `fire_wizard`, `ice_knight`, `shadow_demon`
- ⚠️ **This is the spawn identifier!**

**Display Name**:
- What players see in-game
- Can have spaces and capitals
- Example: `Fire Wizard`, `Ice Knight`, `Shadow Demon`

**HP**:
- Enemy health points
- Range: 10 (weak) to 50000+ (super boss)
- Examples: 30 (goblin), 350 (red demon), 1000 (strong boss)

**Speed**:
- Movement speed
- Range: 0 (stationary) to 20+ (very fast)
- Examples: 10 (normal), 15 (fast), 6 (slow boss)

#### **Attack Configuration Section**

**Bullet ID**:
- Identifier for the projectile sprite
- Common values: `arrow`, `dart`, `fire_ball`, `red_demon_bullet`
- Must match a sprite or projectile definition

**Cooldown (ms)**:
- Time between shots in milliseconds
- 500ms = fast (2 shots/second)
- 1000ms = normal (1 shot/second)
- 2500ms = slow (1 shot every 2.5s)

**Speed**:
- Bullet travel speed
- Range: 10 (slow) to 40+ (very fast)
- Examples: 25 (normal), 15 (slow boss bullets), 35 (sniper)

**Lifetime (ms)**:
- How long bullets exist before disappearing
- 1000ms = short range
- 2000ms = medium range
- 3000ms+ = long range

**Count**:
- Number of bullets fired at once
- 1 = single shot
- 3 = triple shot
- 5 = spread attack
- 8+ = circle/burst

**Spread (degrees)**:
- Angle between bullets when count > 1
- 0 = all bullets go same direction (shotgun)
- 30 = narrow spread
- 45 = medium spread (red demon uses this)
- 180 = semicircle
- 360 / count = perfect circle

**Inaccuracy (degrees)** (optional):
- Random deviation per bullet
- 0 = perfect accuracy
- 5 = slight randomness
- 15+ = very inaccurate

---

### Step 3: Design Behaviors (Center Panel - State Machine)

Now design how your enemy moves and attacks!

**Double-click empty space** to add a new state.

Click the state node, then use the **Inspector (left side when state selected)** to add behaviors:

#### **Movement Section (🟢)**
Click movement blocks to add:
- **Move Toward**: Chase player
- **Orbit**: Circle around player (great for bosses!)
- **Charge**: Rush at player
- **Retreat**: Back away
- **Wander**: Random movement
- **Stop**: Stand still (turret mode)

#### **Attack Section (🔴)**
Click attack blocks to add:
- **Shoot**: Single shot toward player
- **Shoot Spread**: Multiple bullets in arc (use with count > 1)
- **Shoot Circle**: 360° burst (use with count = 8 or 12)

#### **Utility Section (🔵)**
- **Heal Over Time**: Regenerates HP
- **Speed Boost**: Temporary speed increase
- **Shield**: Damage reduction

**Remember**: All behaviors in a category run simultaneously!

Example:
```
State: "combat"
🟢 Movement: Orbit (radius: 200, speed: 0.9)
🔴 Attack: Shoot Spread (count: 5, spread: 45, cooldown: 1000)
→ Enemy circles player while shooting 5-bullet spreads every second
```

---

### Step 4: Save Enemy

Click **"💾 Save Enemy to Backend"** button at the bottom of the left panel (or top bar).

**What Happens**:
1. Enemy data is validated
2. Sent to `/api/enemy-editor/save`
3. Saved to `/public/assets/entities/enemies.json`
4. Added alongside Goblin, Orc, Red Demon, etc.
5. Success alert appears: `"New Enemy" saved to backend with state machine!`

**Saved Format**:
```json
{
  "id": "fire_wizard",
  "name": "Fire Wizard",
  "sprite": "chars:red_imp",
  "hp": 200,
  "speed": 12,
  "width": 1,
  "height": 1,
  "renderScale": 2,
  "attack": {
    "bulletId": "fire_ball",
    "cooldown": 800,
    "speed": 30,
    "lifetime": 2500,
    "count": 3,
    "spread": 30
  },
  "behavior": {
    "states": [
      {
        "name": "combat",
        "movement": [
          { "id": "Orbit", "params": { "radius": 200, "speed": 0.9 } }
        ],
        "attack": [
          { "id": "ShootSpread", "params": { "count": 3, "spread": 30 } }
        ],
        "utility": []
      }
    ]
  }
}
```

---

### Step 5: Spawn Enemy in Game

Edit `/config/world-spawns.js`:

```javascript
overworld: {
  description: "Main procedural overworld - test area for custom enemies",
  spawns: [
    // Use the ID from Step 2!
    { id: 'fire_wizard', x: 30, y: 30, comment: 'My custom boss!' },
    { id: 'fire_wizard', x: 40, y: 40, comment: 'Another one' },
  ]
},
```

**Restart the server**:
```bash
# Kill current server (Ctrl+C or pkill node)
node Server.js
```

**Play the game**: http://localhost:3001/game.html

Your enemy will spawn at the coordinates with the sprite, stats, and behaviors you designed!

---

## Editing Existing Enemies

### Load Enemy from Dropdown

Click the green **"Select Enemy..."** dropdown at the top.

Choose any enemy:
- **Goblin**
- **Orc**
- **Red Demon**
- **ChargingShooter**
- **Your Custom Enemies**

The editor loads:
1. All enemy stats in left panel
2. Existing behavior states in center graph (or generates default if none)
3. You can modify anything!

### Modify and Save

1. Change sprite (click "🎨 Choose Sprite")
2. Adjust HP, speed, attack stats
3. Edit behavior states (add/remove behaviors)
4. Click **"💾 Save Enemy to Backend"**
5. Changes are saved to `enemies.json`
6. Restart server to see changes in-game

---

## Complete Example: Creating "Shadow Assassin"

### Goal
Fast, stealthy enemy that teleports around and shoots rapid daggers.

### Step-by-Step

**1. Create Enemy**
- Click **"+ New Enemy"**

**2. Set Stats**
- **ID**: `shadow_assassin`
- **Name**: `Shadow Assassin`
- **Sprite**: Click "🎨 Choose Sprite" → select `skeleton` (or `dark_lord` for cooler look)
- **HP**: `150`
- **Speed**: `18` (very fast!)
- **Bullet ID**: `dart`
- **Cooldown**: `300` (rapid fire!)
- **Speed**: `35` (fast bullets)
- **Lifetime**: `1500`
- **Count**: `1`
- **Spread**: `0`

**3. Design Behavior**
- Double-click graph to add state
- Name it: `stealth_attack`
- **Movement**:
  - Add **"Charge"** (speed: 2.5, duration: 2000)
- **Attack**:
  - Add **"Shoot"** (cooldown: 300)
- **Duration**: `3000`
- **Next State**: `retreat`

- Double-click graph again for second state
- Name it: `retreat`
- **Movement**:
  - Add **"Retreat"** (speed: 2.0, minDistance: 300)
- **Duration**: `2000`
- **Next State**: `stealth_attack` (loops back!)

**Result**: Enemy charges at player while rapid-firing for 3 seconds, then retreats for 2 seconds, then repeats!

**4. Save**
- Click **"💾 Save Enemy to Backend"**
- See success message

**5. Spawn**
Edit `/config/world-spawns.js`:
```javascript
{ id: 'shadow_assassin', x: 35, y: 35, comment: 'Fast assassin boss!' }
```

**6. Test**
- Restart server
- Play game
- Fight your creation!

---

## Pro Tips

### Sprite Selection
- Use **"🎨 Choose Sprite"** for visual browsing
- Or type directly if you know the name
- Format is always: `chars:sprite_name`

### Enemy Balancing

**Weak Enemies** (Trash Mobs):
- HP: 20-50
- Speed: 8-12
- Cooldown: 2000-3000ms
- Count: 1
- Simple chase + shoot behavior

**Medium Enemies** (Minions):
- HP: 80-150
- Speed: 10-15
- Cooldown: 1000-1500ms
- Count: 1-3
- More complex movement (orbit, zigzag)

**Boss Enemies**:
- HP: 300-1000
- Speed: 6-10 (slower but tough)
- Cooldown: 500-1000ms
- Count: 5-8 (spread/circle attacks)
- Multiple states with transitions
- Orbit + spread attacks common pattern

**Super Bosses**:
- HP: 5000-50000
- Speed: 0-5 (often stationary)
- Cooldown: 300-800ms
- Count: 8-16 (massive bursts)
- Complex multi-phase patterns

### Behavior Patterns

**Aggressive Chaser**:
```
Movement: Move Toward (speed: 1.5)
Attack: Shoot (cooldown: 800)
```

**Circling Shooter** (Red Demon style):
```
Movement: Orbit (radius: 250, speed: 0.8)
Attack: Shoot Spread (count: 5, spread: 45)
```

**Hit-and-Run**:
```
State 1: "charge" (3s)
  Movement: Charge (speed: 3.0)
  Attack: Shoot (cooldown: 500)
  Next: "retreat"

State 2: "retreat" (2s)
  Movement: Retreat (speed: 1.5)
  Next: "charge"
```

**Stationary Turret**:
```
Movement: Stop
Attack: Shoot Circle (count: 12, cooldown: 2000)
```

---

## Troubleshooting

**"Enemy not showing in dropdown"**:
- Refresh the page
- Check `/public/assets/entities/enemies.json` was saved correctly

**"Enemy not spawning in game"**:
- Check `/config/world-spawns.js` uses correct enemy `id`
- Restart server after editing spawns
- Check server logs for spawn count: `[SPAWNS] Loaded X enemy spawns`

**"Sprite picker not opening"**:
- Click directly on "🎨 Choose Sprite" button
- Check browser console for errors

**"Enemy appears as white square"**:
- Sprite name might be wrong
- Verify format: `chars:sprite_name`
- Check sprite exists in list

**"State machine not saving"**:
- Make sure you clicked "💾 Save Enemy to Backend"
- Check for success alert message
- Verify in `enemies.json` that `behavior.states` field exists

---

## Summary

**The Enemy Editor is a complete tool for creating game-ready enemies:**

1. **Stats**: HP, speed, attack configuration
2. **Sprite**: Visual appearance with picker or manual input
3. **Behavior**: State machine with movement, attack, utility behaviors
4. **Save**: Everything goes to `enemies.json` in one file
5. **Spawn**: Use enemy `id` in `world-spawns.js` to place in game

**No code editing required!** Everything is visual and saves automatically to the backend database.

Now go create some epic enemies! 🎮
