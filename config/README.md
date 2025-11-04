# World Enemy Spawn Configuration

This directory contains the configuration system for managing enemy spawns across all worlds in the game.

## Quick Start

### 1. Edit Spawns

Open `world-spawns.js` and edit the spawn arrays for each world:

```javascript
overworld: {
  description: "Main procedural overworld",
  spawns: [
    { id: 'goblin', x: 25, y: 30, comment: 'Tutorial enemy near spawn' },
    { id: 'orc', x: 40, y: 35, comment: 'Forest patrol' },
    // Add more spawns...
  ]
}
```

### 2. Restart Server

After editing, restart the server to apply changes:
```bash
npm start
```

### 3. Verify In-Game

Check the server logs for spawn confirmation:
```
[SPAWNS] Loaded 16 enemy spawns for overworld from config
```

## Available Enemy Types

| ID | Name | HP | Behavior |
|----|------|-----|----------|
| `goblin` | Goblin | 30 | Basic chase and single shot |
| `orc` | Orc | 50 | Tankier, retreats when low HP |
| `red_demon` | Red Demon | 350 | Boss with 5-shot spread, enters rage mode |
| `hyper_demon` | Hyper Demon | 50000 | Super boss with 8-shot spread |
| `charging_shooter` | Charging Shooter | 80 | Charges while shooting rapidly |

## World IDs

| World ID | Map Name | Description |
|----------|----------|-------------|
| `overworld` | Overworld (map_1) | Main procedural world |
| `map_2` | RiverBridge | Bridge encounter dungeon |
| `map_3` | SampleBossRoom | Major boss encounter |
| `map_4` | SampleNexus | Safe trading hub (no enemies) |
| `map_5` | TestDungeon | Enemy variety testing |
| `map_6` | test | Small test arena |

## Spawn Object Format

```javascript
{
  id: 'goblin',           // Enemy type ID (required)
  x: 25,                  // X coordinate in tiles (required)
  y: 30,                  // Y coordinate in tiles (required)
  comment: 'Optional note' // Optional description (optional)
}
```

## Design Tips

### Spawn Placement Strategy

1. **Near Spawn (20-30 tiles)**: Place tutorial enemies (goblins)
2. **Mid-Range (30-50 tiles)**: Place tankier enemies (orcs)
3. **Far Areas (50+ tiles)**: Place bosses and mini-bosses
4. **Safe Zones**: Keep areas like spawn point and Nexus enemy-free

### Grouping Patterns

```javascript
// Boss with minions
{ id: 'red_demon', x: 50, y: 40, comment: 'Boss center' },
{ id: 'charging_shooter', x: 45, y: 40, comment: 'Boss minion L' },
{ id: 'charging_shooter', x: 55, y: 40, comment: 'Boss minion R' },

// Patrol group
{ id: 'orc', x: 40, y: 35, comment: 'Patrol leader' },
{ id: 'goblin', x: 38, y: 35, comment: 'Patrol grunt' },
{ id: 'goblin', x: 42, y: 35, comment: 'Patrol grunt' },

// Scattered wanderers
{ id: 'goblin', x: 30, y: 60 },
{ id: 'goblin', x: 35, y: 62 },
{ id: 'goblin', x: 32, y: 65 },
```

### Difficulty Progression

1. **Starting Area**: 3-5 goblins
2. **Early Game**: 5-10 mixed goblins/orcs
3. **Mid Game**: 10-15 with charging shooters
4. **Late Game**: 15+ with red demons
5. **Boss Rooms**: 1 boss + 4-8 minions

## Visual Editor

Open `http://localhost:3000/spawn-editor.html` for a visual editor interface:

- Select worlds from dropdown
- Add/edit/delete spawns visually
- Export generated JavaScript code
- Copy to `world-spawns.js`

## File Structure

```
config/
├── README.md           # This file
└── world-spawns.js     # Main spawn configuration
```

## Debugging

### Server-Side Logs

Enemy spawns are logged on server start:
```
[SPAWNS] Loaded 16 enemy spawns for overworld from config
[ENEMIES] Spawned 16 enemies for map map_1
```

### In-Game Verification

Server logs show:
- Enemy state transitions: `🤖 [ENEMY STATE] Index 0 transitioned: idle → chase`
- Enemy shooting: `🔫 [ENEMY SHOOT] Index 5 at (50.00, 40.00) firing 5 bullet(s)`
- Enemy positions (every 3s): `📍 [ENEMY POSITIONS] Active enemies: 16`

Client logs show:
- Client enemy positions (every 5s): `🎮 [CLIENT ENEMIES] Active: 16`

## Example Configurations

### Starter Zone
```javascript
spawns: [
  { id: 'goblin', x: 25, y: 30, comment: 'Tutorial 1' },
  { id: 'goblin', x: 30, y: 30, comment: 'Tutorial 2' },
  { id: 'goblin', x: 35, y: 30, comment: 'Tutorial 3' },
]
```

### Boss Arena
```javascript
spawns: [
  // Boss in center
  { id: 'red_demon', x: 16, y: 16, comment: 'Main boss' },

  // Corner minions
  { id: 'charging_shooter', x: 10, y: 10, comment: 'NW minion' },
  { id: 'charging_shooter', x: 22, y: 10, comment: 'NE minion' },
  { id: 'charging_shooter', x: 10, y: 22, comment: 'SW minion' },
  { id: 'charging_shooter', x: 22, y: 22, comment: 'SE minion' },

  // Inner ring guards
  { id: 'orc', x: 13, y: 13 },
  { id: 'orc', x: 19, y: 13 },
  { id: 'orc', x: 13, y: 19 },
  { id: 'orc', x: 19, y: 19 },
]
```

### Safe Zone
```javascript
map_4: {
  description: "Nexus - Safe trading hub",
  spawns: []  // No enemies!
}
```

## Advanced: Dynamic Spawns

To add dynamic spawning (future feature):

```javascript
// In world-spawns.js
export function getWorldSpawns(worldId) {
  const base = worldSpawns[worldId]?.spawns || [];

  // Add time-based spawns
  const hour = new Date().getHours();
  if (hour >= 20 && hour <= 23) {
    // Night time: add nocturnal enemies
    return [...base, { id: 'red_demon', x: 60, y: 60, comment: 'Night spawn' }];
  }

  return base;
}
```

## Troubleshooting

### Problem: Spawns not appearing

**Solution**: Check server logs for:
```
[SPAWNS] Loaded X enemy spawns for worldId from config
[ENEMIES] Spawned X enemies for map worldId
```

### Problem: Wrong enemy types

**Solution**: Verify enemy IDs match exactly:
- ✅ `'goblin'`
- ❌ `'Goblin'` (wrong case)
- ❌ `'goblin_basic'` (wrong ID)

### Problem: Enemies spawning in walls

**Solution**: Adjust X/Y coordinates to open areas. Use map editor to find valid tiles.

## Contributing

When adding new enemy types:

1. Add to `public/assets/entities/enemies.json`
2. Add behavior in `src/Behaviours/BehaviorSystem.js`
3. Update `availableEnemies` array in `world-spawns.js`
4. Update this README with new enemy info

## Related Files

- `src/entities/EnemyManager.js` - Server-side enemy manager
- `public/src/game/ClientEnemyManager.js` - Client-side enemy manager
- `src/Behaviours/BehaviorSystem.js` - Enemy AI behaviors
- `public/assets/entities/enemies.json` - Enemy stats database
- `Server.js` - Server initialization and spawn loading
