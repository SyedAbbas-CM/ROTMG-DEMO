# Terminal Overworld Game - Enhanced Commands

## How to Play
```bash
# Option 1: Use the launcher script
./play.sh

# Option 2: Run directly with Node.js
node terminal-overworld.js
```

## Movement Commands (WASD + Arrow Keys)
- `w` or `n` or `north` - Move north one region
- `s` or `south` - Move south one region
- `a` or `west` - Move west one region  
- `d` or `e` or `east` - Move east one region
- `move <x> <y>` - Move by offset (e.g., `move 2 -1`)
- `tp <x> <y>` - Teleport to specific region coordinates

## Information Commands
- `info` or `i` - Show detailed region and player info
- `conn` - Show connections from current region
- `path <x> <y>` - Find path to target region
- `stats` - Show exploration statistics

## View Commands
- `zoom <level>` - Change zoom level (1-10) with different emoji sets
- `debug` - Toggle debug mode (shows loading info & memory usage)
- `clear` or `cls` - Clear screen and redraw
- `help` or `h` - Show full command help

## General Commands
- `quit` or `q` - Exit the game

## Hierarchical Map System
The game uses different emoji sets based on zoom level:

### **Zoom Level 1-2: Continental View** 🏔️
- 🏔️ **Plains** - Mountain ranges
- ⛰️ **Fortress** - Mountain peaks
- 🗻 **Resource** - Major peaks
- 🏔️ **Crossing** - Mountain passes
- 🌋 **Wasteland** - Volcanoes
- 🏔️ **Capital** - Mountain ranges
- 🗻 **Portal** - Sacred peaks

### **Zoom Level 3-4: Provincial View** 🌲
- 🌲 **Plains** - Pine forests
- 🌳 **Fortress** - Oak forests
- 🌴 **Resource** - Palm groves
- 🎄 **Crossing** - Evergreens
- 🥔 **Wasteland** - Dead trees
- 🌲 **Capital** - Pine forests
- 🌳 **Portal** - Ancient groves

### **Zoom Level 5+: Regional View** (Procedural Variety)
- **Plains**: 🌾🌱🌿🍃 (crops, grass, herbs)
- **Fortress**: 🏰⚔️🛡️🗡️ (castles, weapons, shields)
- **Resource**: ⛏️💎⚡🔧 (tools, gems, energy, machinery)
- **Crossing**: 🌉🛤️🚪🔗 (bridges, roads, doors, chains)
- **Wasteland**: 💀☠️🔥💥 (skulls, poison, fire, explosions)
- **Capital**: 👑🏛️🎭💫 (crowns, temples, masks, stars)
- **Portal**: 🌀✨🌠🔮 (swirls, sparkles, shooting stars, orbs)

## Example Session
```
> tp 0 0          # Teleport to origin
> conn            # Check connections
> path 10 10      # Find path to region (10,10)
> n               # Move north
> info            # Get detailed info
> stats           # Check exploration stats
> zoom 5          # Increase map view
> quit            # Exit game
```

## Features
- **100k x 100k procedural world** with deterministic generation
- **On-demand loading** - Only loads regions when needed
- **Regional connectivity** - Regions connected for strategic movement
- **Pathfinding** - Find routes between distant regions
- **Memory management** - Automatic cleanup of unused areas
- **Exploration tracking** - Keep track of visited regions

Enjoy exploring the vast overworld!