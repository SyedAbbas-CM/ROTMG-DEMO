# Quick Fixes Needed

## Issue 1: Water Detection Inefficient
- Currently checking `getTile()` every frame for every character
- Solution: Cache water status on character object, only check when tile position changes

## Issue 2: Water Submersion Not Working
- `tile.type === 3` check failing
- Solution: Check `tile.def.spriteName.includes('water')` instead

## Issue 3: No New Biomes
- Map was already generated with old biome system
- Solution: Need to delete world save or regenerate with new seed

## Issue 4: 26,415 Objects Accumulating
- Objects not being removed from `window.currentObjects`
- Solution: The periodic cleanup is set up but need to verify it's running

## Issue 5: Black Outlines Needed
- Add black stroke around players, enemies, bullets for visibility
- Solution: Use `ctx.strokeStyle` and `ctx.lineWidth` before drawing sprites

## Files to Fix:
1. `/Users/developer-cloudprimero/Desktop/ROTMG-DEMO/public/src/render/render.js` - Water + outlines
2. `/Users/developer-cloudprimero/Desktop/ROTMG-DEMO/public/src/render/renderTopDown.js` - Enemy/bullet outlines
3. Delete or regenerate world to see new biomes
