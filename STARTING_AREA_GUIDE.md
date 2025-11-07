# Starting Area Customization Guide

## Overview

Your game now has a **custom starting area** system! Players will always spawn in this designated safe zone instead of random locations in the procedural world.

## Current Setup

- **Starting Area Map**: `/public/maps/StartingArea.json`
- **Size**: 32x32 tiles
- **Default Spawn Points**: 5 spawn locations (center and cardinal directions around center)
- **Current Design**: Simple 6x6 grass platform at the center (rows 13-18, columns 13-18)

## How to Customize Your Starting Area

### Using the Map Editor

1. **Start your server** (if not already running):
   ```bash
   node Server.js
   ```

2. **Open the Map Editor** in your browser:
   ```
   http://localhost:3000/tools/map-editor.html
   ```

3. **Load the Starting Area map**:
   - In the editor sidebar, find the "Existing Maps" dropdown
   - Select `StartingArea.json`
   - Click "Load Selected Map"

4. **Design your starting area**:
   - **Ground Layer**: Paint floor tiles (grass, stone, sand, etc.)
   - **Objects Layer**: Add decorations, walls, obstacles
   - Use the tile palette on the left to select tiles
   - Paint with left-click, erase with right-click
   - Use Shift+Click+Drag to paint rectangles

5. **Set Spawn Points**:
   - In the sidebar, select "Paint Mode" → "Spawn Point"
   - Click where you want players to spawn
   - You can have multiple spawn points - one will be chosen randomly
   - Spawn points appear as green diamonds

6. **Save your changes**:
   - Press `Ctrl+S` or click "Save Map"
   - Save as `StartingArea.json` (overwrite existing)

### Map Editor Controls

- **Left Click**: Paint/place
- **Right Click**: Erase
- **Shift + Click + Drag**: Draw rectangle
- **Middle Mouse + Drag**: Pan view
- **Mouse Wheel**: Zoom in/out
- **R**: Rotate selected tile
- **Space**: Eyedropper tool (pick existing tile)
- **Ctrl+Z**: Undo
- **Ctrl+Y**: Redo
- **1-5**: Brush size shortcuts

### Design Tips

1. **Safe Zone**: Make the starting area free of enemies and hazards
2. **Size**: 32x32 is a good size - not too small, not too large
3. **Multiple Spawns**: Add several spawn points so players don't overlap
4. **Visual Clarity**: Use distinct tiles to mark the starting area (like a platform or courtyard)
5. **Transition**: Consider adding roads or paths leading out to the procedural world

## How It Works

### Server Behavior

When a player joins or respawns, the server:
1. Loads spawn points from `StartingArea.json`
2. Picks a random spawn point from the `entryPoints` array
3. Places the player at that location

### Current Spawn Points

The default `StartingArea.json` has 5 spawn points:
- Center: (16, 16)
- North: (16, 15)
- South: (16, 17)
- West: (15, 16)
- East: (17, 16)

### Modifying Spawn Points Manually

If you prefer to edit the JSON directly, open `/public/maps/StartingArea.json` and find the `entryPoints` array:

```json
"entryPoints": [
  {"x": 16, "y": 16},
  {"x": 15, "y": 16},
  {"x": 17, "y": 16},
  {"x": 16, "y": 15},
  {"x": 16, "y": 17}
]
```

Add or remove points as needed. Coordinates are in tiles (0-31 for a 32x32 map).

## Integration with Procedural World

The starting area exists at the **center of the procedural world**:
- World size: 2560x2560 tiles (4 regions of 640x640 each)
- Starting area: 32x32 tiles at the world center
- Players spawn in the starting area
- They can explore the infinite procedural biomes from there

## Testing Your Changes

1. Save your starting area design
2. Restart the server (Ctrl+C then `node Server.js`)
3. Refresh the game in your browser
4. New players and respawning players will use your new starting area!

## Troubleshooting

**Problem**: Players spawn at random locations instead of starting area
- **Solution**: Make sure `StartingArea.json` exists in `/public/maps/`
- **Solution**: Check that `entryPoints` array is not empty
- **Solution**: Restart the server to reload the spawn points

**Problem**: Map editor shows errors loading the map
- **Solution**: Verify JSON syntax is valid (no trailing commas, proper quotes)
- **Solution**: Ensure all arrays have the correct dimensions (32 rows of 32 columns)

**Problem**: Players spawn inside walls/obstacles
- **Solution**: Move spawn points to open floor areas
- **Solution**: Check the coordinates in `entryPoints` are correct

## Advanced: Creating Multiple Starting Areas

You can create different starting areas for different game modes or regions:

1. Create new map files: `StartingArea_Easy.json`, `StartingArea_Hard.json`, etc.
2. Modify `Server.js` to load different starting areas based on game state
3. Update the `loadStartingAreaSpawnPoints()` function to accept a parameter

## Example Starting Area Designs

### Simple Platform (Current Default)
- 6x6 grass platform
- 5 spawn points in cross pattern
- Minimal obstacles

### Town Square (Suggested)
- Stone floor tiles forming a square
- Decorative objects (benches, fountain)
- Multiple spawn points scattered throughout
- Walls/buildings around edges

### Forest Clearing
- Mix of grass and dirt tiles
- Trees around the perimeter
- Open center for spawning
- Path leading outward

## Need Help?

- Check the existing maps in `/public/maps/` for examples
- Use the map editor's "Load Map" feature to see how other maps are structured
- The editor automatically handles tile rotation and layering

Enjoy designing your perfect starting area!
