# Enemy System Reference

## Implementation Status

**Current State:** Basic enemy definitions complete. All 4 enemy types are defined in `/public/assets/entities/enemies.json`.

**Implemented:**
- ✅ Enemy stats and definitions
- ✅ Basic chase and shoot behavior
- ✅ Range-based shooting
- ✅ Bullet definitions

**To Implement:**
- ⏳ Red Demon Cavalry charge/idle cycle (needs RedDemonCavalry behavior in BehaviorSystem.js)
- ⏳ Range-based AI (enemies maintaining optimal distance)
- ⏳ Advanced behavior trees
- ⏳ State machine execution

**Next Steps:**
1. Implement RedDemonCavalry behavior with charge → shoot → idle cycle
2. Test all 4 enemies in game
3. Work on player improvements

## Available Behaviors

### Movement Behaviors
- **MoveToward** - Move directly toward target
  - Params: `{ target: 'ToPlayer', speed: 1.0 }`
- **MoveAway** - Move away from target
  - Params: `{ target: 'ToPlayer', distance: 300 }`
- **Orbit** - Circle around target
  - Params: `{ target: 'ToPlayer', radius: 200, speed: 1.0, clockwise: true }`
- **ZigZag** - Zigzag pattern toward target
  - Params: `{ target: 'ToPlayer', amplitude: 100, frequency: 2 }`
- **Circle** - Move in a circle
  - Params: `{ radius: 150, speed: 1.0 }`
- **Charge** - Fast direct charge at target
  - Params: `{ target: 'ToPlayer', speed: 2.0, duration: 1000 }`
- **Retreat** - Move away to specific distance
  - Params: `{ target: 'ToPlayer', distance: 400, speed: 1.5 }`
- **Wander** - Random wandering movement
  - Params: `{ changeInterval: 1000 }`
- **Stop** - Stop all movement
  - Params: `{}`

### Attack Behaviors
- **Shoot** - Fire single bullet
  - Params: `{ angle: 'GetAngle()', speed: 25, bulletType: 'arrow' }`
- **ShootPattern** - Fire predefined pattern
  - Params: `{ pattern: 'triple_forward', angle: 'GetAngle()', speed: 25 }`
- **ShootSpread** - Fire bullets at multiple angles
  - Params: `{ angles: [-30, 0, 30], speed: 25 }`
- **ShootSpiral** - Rotating spiral pattern
  - Params: `{ count: 8, rotationSpeed: 90, bulletSpeed: 25 }`
- **ShootRing** - Fire bullets in all directions
  - Params: `{ count: 12, speed: 25, bulletType: 'arrow' }`

### Utility Behaviors
- **SetSpeed** - Change movement speed
  - Params: `{ value: 1.0 }`
- **SetAngle** - Set facing angle
  - Params: `{ value: 'ToPlayer' }`
- **SetVar** - Set variable
  - Params: `{ name: 'myVar', value: 0 }`

## Current Enemies (4 Unit Types)

### 1. Goblin Infantry
**Role:** Short-range heavy infantry
- **HP:** 120 (high)
- **Speed:** 12 (slow)
- **Damage:** 5 (low)
- **Range:** 150 (short)
- **Attack:** Single dart
- **Cooldown:** 2000ms
- **Behavior:** Chase and shoot at close range
- **Sprite:** `chars:goblin`
- **Tactical Role:** Tank/Frontline

### 2. Archer
**Role:** Long-range support
- **HP:** 60 (low)
- **Speed:** 8 (very slow)
- **Damage:** 15 (medium)
- **Range:** 600 (very long)
- **Attack:** Fast arrow
- **Cooldown:** 2500ms
- **Behavior:** Maintains distance, shoots from afar
- **Sprite:** `chars:orc`
- **Tactical Role:** Backline DPS

### 3. Red Demon Cavalry
**Role:** Heavy cavalry charger
- **HP:** 500 (very high)
- **Speed:** 8 (fast when charging)
- **Damage:** 25 (high)
- **Range:** 250 (medium-short)
- **Attack:** 3-bullet spread
- **Cooldown:** 600ms (rapid fire while charging)
- **Behavior:** Charges at high speed, shoots while charging, then idles for 10 seconds
- **Sprite:** `chars:red_demon`
- **Tactical Role:** Hit-and-run cavalry
- **Special:** Charge → Shoot → Idle cycle

### 4. Heavy Knight
**Role:** Slow tank
- **HP:** 800 (extremely high)
- **Speed:** 4 (very slow)
- **Damage:** 8 (low)
- **Range:** 200 (short)
- **Attack:** Single heavy projectile
- **Cooldown:** 3000ms (very slow)
- **Behavior:** Slow approach and heavy attacks
- **Sprite:** `chars:red_demon`
- **Tactical Role:** Ultra-tank/Wall

## Enemy Definition Format

```json
{
  "id": "enemy_id",
  "name": "Display Name",
  "sprite": "atlas:sprite_name",
  "hp": 100,
  "speed": 10,
  "width": 1,
  "height": 1,
  "renderScale": 2,
  "attack": {
    "bulletId": "bullet_id",
    "cooldown": 1000,
    "speed": 25,
    "lifetime": 2000,
    "count": 1,
    "spread": 0,
    "inaccuracy": 0,
    "range": 300
  },
  "ai": {
    "behaviorTree": "BasicChaseAndShoot"
  }
}
```

## Available Behavior Trees

### BasicChaseAndShoot
Simple chase and attack behavior
- Moves toward player
- Shoots when in range
- No special patterns

### RedDemonBT
Charging cavalry behavior
- Charges toward player at high speed
- Shoots spread pattern while charging
- Enters cooldown state after charge

### StaticBoss
Stationary boss behavior
- Does not move
- Continuously shoots patterns at player
- High HP for long fights

### ChargingShooter
Charge and shoot behavior
- Charges toward player
- Rapid fire while charging
- Short cooldown between charges

## Spawning Enemies

Edit `/config/world-spawns.js`:

```javascript
export const worldSpawns = {
  overworld: {
    description: "Main overworld",
    spawns: [
      { id: 'goblin', x: 30, y: 30 },
      { id: 'red_demon', x: 50, y: 50 }
    ]
  }
};
```

## Creating New Enemies

1. Add enemy definition to `/public/assets/entities/enemies.json`
2. Create bullet definition if needed
3. Assign existing behavior tree or create new one
4. Add to spawn configuration
5. Test in game

## Behavior State Machine

Enemies use a state machine with concurrent behaviors:

```javascript
{
  movement: [
    { id: 'MoveToward', params: { target: 'ToPlayer', speed: 1.0 } }
  ],
  attack: [
    { id: 'Shoot', params: { angle: 'GetAngle()', speed: 25 } }
  ],
  utility: [],
  onEnter: [],  // Run once when entering state
  onExit: []    // Run once when exiting state
}
```

Movement and attack behaviors run **simultaneously**, not sequentially.

## Bullet Definitions

Bullets are defined as enemies without AI:

```json
{
  "id": "my_bullet",
  "sprite": "chars:arrow",
  "speed": 25,
  "lifetime": 2000
}
```

## Sprite References

Available sprite atlases:
- `chars:` - Character sprites
- `chars2:` - Additional characters
- `Mixed_Units:` - Unit sprites
- `lofi_obj:` - Objects
- `lofi_environment:` - Environment
- `tiles:` - Tile sprites

View all sprites in behavior designer sprite picker.
