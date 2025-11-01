# Portal and Collision System Analysis

## Overview
The game has a partially implemented portal system for world/dungeon switching, and a tile-based collision system. However, portal interaction needs to be implemented on both client and server.

## Current Portal Infrastructure

### 1. Server-Side Portal Routes (`src/routes/portalRoutes.js`)
Portal management API endpoints:

- **GET `/api/portals/list?mapId=...`** - List all portals for a map or all maps
- **POST `/api/portals/add`** - Add a portal at (x, y) linking mapId → destMap
- **POST `/api/portals/remove`** - Remove portal at position
- **POST `/api/portals/link-both`** - Create bidirectional portal link between two maps

Portal Object Structure:
```javascript
{
  id: 'portal_<timestamp>_<random>',
  type: 'portal',
  sprite: 'portal',  // or custom sprite
  x: number,  // tile x coordinate
  y: number,  // tile y coordinate
  destMap: 'map_2'  // destination map ID
}
```

### 2. Server Bootstrap (`Server.js` lines 778-820)
Temporary portal setup for testing:
```javascript
// Creates portal at (5,5) linking map_1 → map_2
// Portal loaded from /public/maps/test.json
const portalObj = {
  id: 'portal_tmp_main',
  type: 'portal',
  sprite: 'portal',
  x: 5,
  y: 5,
  destMap: handmadeId
};
```

### 3. Protocol Definition (`common/protocol.js`)
Portal-related message types:
```javascript
MSG_TYPES: {
  PORTAL_ENTER: 54,   // Client → Server: Player entered portal
  WORLD_SWITCH: 55,   // Server → Client: Switch to new world/map
  WORLD_UPDATE: 60,   // Server → Client: World state update
  // ...
}
```

## What's Missing

### Client-Side Portal Interaction ❌
**Issue**: No client code detects when player walks over a portal tile

**Need to Implement**:
1. **Portal Detection** - Check if player position overlaps portal object
2. **Interaction Trigger** - Send `PORTAL_ENTER` message when player walks on portal
3. **World Switch Handler** - Handle `WORLD_SWITCH` response from server

**Location**: `public/src/game/gameManager.js` or `public/src/game/input.js`

```javascript
// Pseudo-code for what's needed:
function checkPortalCollision(player, portals) {
  for (const portal of portals) {
    const dx = player.x - portal.x;
    const dy = player.y - portal.y;
    const distance = Math.sqrt(dx*dx + dy*dy);

    if (distance < PORTAL_INTERACTION_RADIUS) {
      // Send PORTAL_ENTER message to server
      networkManager.send({
        type: MSG_TYPES.PORTAL_ENTER,
        portalId: portal.id,
        destMap: portal.destMap
      });
      return true;
    }
  }
  return false;
}
```

### Server-Side Portal Handler ❌
**Issue**: Server doesn't have handler for `PORTAL_ENTER` messages

**Need to Implement** in `Server.js`:
```javascript
case MSG_TYPES.PORTAL_ENTER: {
  const { portalId, destMap } = msg;

  // Validate portal exists and player is near it
  const portal = findPortal(player.mapId, portalId);
  if (!portal) break;

  const distance = Math.hypot(player.x - portal.x, player.y - portal.y);
  if (distance > 2) break;  // Too far from portal

  // Switch player to new map
  player.mapId = destMap;
  player.x = 2;  // Spawn position in new map
  player.y = 2;

  // Send confirmation to client
  ws.send(JSON.stringify({
    type: MSG_TYPES.WORLD_SWITCH,
    mapId: destMap,
    spawnX: player.x,
    spawnY: player.y
  }));

  // Notify other players
  broadcastToMap(player.mapId, {
    type: MSG_TYPES.PLAYER_LEFT,
    playerId: player.id
  });

  break;
}
```

## Collision System

### Current Implementation

**Server-Side** (`src/world/MapManager.js`):
- Tiles have `isWalkable()` method
- Collision checked during tile generation
- Obstacles (trees, boulders) set as non-walkable

**Client-Side** (`public/src/map/tile.js`):
```javascript
class Tile {
  isWalkable() {
    return !!this.properties.isWalkable;
  }

  isBlockingMovement() {
    return !this.isWalkable();
  }
}
```

**Movement Validation**:
- Client checks collision before sending move
- Server validates movement server-side
- Prevents walking through walls/obstacles

### Known Issues

1. **Portal Collision** - Portals should be walkable (to enter them) but need special handling
2. **Layer Collision** - Decorations (layer 3) should be walkable, obstacles (layer 2) should not
3. **Entity Collision** - No player-to-player collision implemented

## Implementation Priority

### High Priority
1. ✅ **Biome Rendering** - Fixed (case sensitivity bug)
2. ⏸️ **Portal Interaction** - Client detection + server handler
3. ⏸️ **Tile Rendering Layers** - Ground, obstacles, decor, entities
4. ⏸️ **World Switching** - Handle map transitions cleanly

### Medium Priority
5. **Dungeon Generation** - Procedural dungeon maps
6. **Portal Sprite Rendering** - Visual portal indication
7. **Transition Effects** - Fade in/out when switching worlds

### Low Priority
8. **Multiple Dungeon Types** - Different dungeon layouts
9. **Portal Cooldown** - Prevent rapid portal spam
10. **Portal Permissions** - Some portals require keys/conditions

## Files to Modify

### For Portal Interaction:
1. `public/src/game/gameManager.js` - Add portal collision detection
2. `public/src/network/ClientNetworkManager.js` - Add PORTAL_ENTER sender
3. `Server.js` - Add PORTAL_ENTER message handler
4. `public/src/render/renderTopDown.js` - Render portal sprites

### For Collision Improvements:
1. `public/src/map/tile.js` - Add layer-based collision
2. `src/world/MapManager.js` - Mark portals as special walkable tiles
3. `public/src/game/input.js` - Improve movement collision checks

## Testing Checklist

- [ ] Portal renders at (5,5) on map_1
- [ ] Player can walk to portal
- [ ] Walking on portal triggers PORTAL_ENTER
- [ ] Server receives PORTAL_ENTER and validates
- [ ] Server sends WORLD_SWITCH with new map data
- [ ] Client switches to map_2 seamlessly
- [ ] Player spawns at correct position in map_2
- [ ] Return portal from map_2 → map_1 works
- [ ] Other players see player disappear/reappear correctly
- [ ] Collision works correctly after world switch

## Next Steps

1. Implement client-side portal detection in `gameManager.js`
2. Implement server-side PORTAL_ENTER handler in `Server.js`
3. Test basic portal functionality
4. Add visual feedback (portal glow/animation)
5. Implement proper world state management
6. Add dungeon generation system
