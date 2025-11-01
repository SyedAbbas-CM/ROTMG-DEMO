# Shooting System Debug Analysis

## Issue
Player shooting is not working - bullets are not visible/firing in game.

## Complete Message Flow Analysis

### 1. Client Shoots (User Clicks)

**File:** `public/src/game/game.js:1044-1055`

```javascript
function handleShoot(dx, dy, playerX, playerY) {
    const angle = Math.atan2(vy, vx);
    const speed = Math.sqrt(vx * vx + vy * vy);

    networkManager.sendShoot({
        x: playerX,
        y: playerY,
        angle,
        speed,
        damage: 10
    });
}
```

### 2. Client Sends BULLET_CREATE Message

**File:** `public/src/network/ClientNetworkManager.js:1125-1128`

```javascript
sendShoot(bulletData) {
    return this.send(MessageType.BULLET_CREATE, bulletData);
}
```

**Message Format:**
```json
{
  "type": 30,  // BULLET_CREATE
  "data": {
    "x": 10.5,
    "y": 15.2,
    "angle": 1.57,
    "speed": 10,
    "damage": 10
  }
}
```

### 3. Server Receives & Processes

**File:** `Server.js:1069-1070`

```javascript
} else if(packet.type === MessageType.BULLET_CREATE){
  handlePlayerShoot(clientId, packet.data);
}
```

**File:** `Server.js:413-448`

```javascript
function handlePlayerShoot(clientId, bulletData) {
  const client = clients.get(clientId);
  if (!client) return;

  const { x, y, angle, speed, damage } = bulletData;
  const mapId = client.mapId;

  const ctx = getWorldCtx(mapId);
  if (!ctx || !ctx.bulletMgr) {
    console.error(`[SHOOT] No bullet manager for map ${mapId}`);
    return;
  }

  // Create bullet with velocity calculated from angle/speed
  const bullet = {
    id: `bullet_${Date.now()}_${clientId}_${Math.random()}`,
    x: x || client.player.x,
    y: y || client.player.y,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    damage: damage || 10,
    owner: clientId,
    ownerType: 'player',
    worldId: mapId,
    createdAt: Date.now(),
    lifetime: 5000 // 5 seconds (in milliseconds!)
  };

  ctx.bulletMgr.addBullet(bullet);

  if (DEBUG.bulletEvents) {
    console.log(`[SHOOT] Player ${clientId} fired bullet at angle ${angle.toFixed(2)}`);
  }
}
```

### 4. Server Updates Bullets Every Frame

**File:** `Server.js:1119`

```javascript
function updateGame() {
  worldContexts.forEach((ctx, mapId) => {
    ctx.bulletMgr.update(deltaTime);  // Updates all bullets
    ctx.collMgr.checkCollisions();    // Checks collisions
  });

  broadcastWorldUpdates();  // Sends to clients
}
```

### 5. Server Broadcasts WORLD_UPDATE

**File:** `Server.js:1150-1184`

```javascript
function broadcastWorldUpdates() {
  clientsByMap.forEach((idSet, mapId) => {
    const ctx = getWorldCtx(mapId);
    const bullets = ctx.bulletMgr.getBulletsData(mapId);

    idSet.forEach(clientId => {
      sendToClient(client.socket, MessageType.WORLD_UPDATE, {
        players: playersObj,
        enemies: enemiesClamped,
        bullets: bulletsClamped,  // <-- Bullets sent here
        units: unitsClamped
      });
    });
  });
}
```

**File:** `src/entities/BulletManager.js:224-246` (Server-side)

```javascript
getBulletsData(filterWorldId = null) {
  const bullets = [];

  for (let i = 0; i < this.bulletCount; i++) {
    if (filterWorldId && this.worldId[i] !== filterWorldId) continue;
    bullets.push({
      id: this.id[i],
      x: this.x[i],
      y: this.y[i],
      vx: this.vx[i],
      vy: this.vy[i],
      width: this.width[i],
      height: this.height[i],
      life: this.life[i],
      damage: this.damage[i],
      ownerId: this.ownerId[i],
      spriteName: this.spriteName[i],  // Usually null!
      worldId: this.worldId[i]
    });
  }

  return bullets;
}
```

### 6. Client Receives WORLD_UPDATE

**File:** `public/src/network/ClientNetworkManager.js:645-662`

```javascript
this.handlers[MessageType.WORLD_UPDATE] = (data) => {
    if (this.game.updateWorld) {
        this.game.updateWorld(data.enemies, data.bullets, players, data.objects, data.units);
    }
};
```

**File:** `public/src/game/gameManager.js:259-268`

```javascript
updateWorld(enemies, bullets, players, objects = null, bags = null) {
    // Update bullets
    if (bullets && bullets.length > 0) {
      this.bulletManager.updateBullets(bullets);
    }
}
```

### 7. Client Updates Bullet Manager

**File:** `public/src/game/ClientBulletManager.js:243-325`

```javascript
updateBullets(bullets) {
    // Process each bullet from server
    for (const bullet of bullets) {
      const index = this.findIndexById(bullet.id);

      if (index !== -1) {
        // Update existing bullet
        this.x[index] = bullet.x;
        this.y[index] = bullet.y;
        // ... etc
      } else {
        // Add new bullet
        this.addBullet(bullet);
      }
    }
}
```

### 8. Client Renders Bullets

**File:** `public/src/render/render.js:420-500`

```javascript
export function renderBullets() {
  const bm = gameState.bulletManager;
  if (!bm || bm.bulletCount === 0) return;

  for (let i = 0; i < bm.bulletCount; i++) {
    // Convert world position to screen position
    const { x: sx, y: sy } = gameState.camera.worldToScreen(
      bm.x[i], bm.y[i], W, H, TILE_SIZE
    );

    // Try to render sprite (if available)
    if (spriteManager && bm.sprite && bm.sprite[i]) {
      spriteManager.drawSprite(ctx, sprite.spriteSheet, ...);
      continue;
    }

    // Fallback: Render as glowing circle
    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, drawW);
    grad.addColorStop(0, 'rgb(255,255,120)');  // Yellow center
    grad.addColorStop(0.7, 'rgb(255,160,0)');  // Orange
    grad.addColorStop(1, 'rgba(255,100,0,0)'); // Transparent edge

    ctx.fillStyle = grad;
    ctx.arc(sx, sy, drawW, 0, Math.PI*2);
    ctx.fill();
  }
}
```

---

## Critical Issues Identified

### Issue #1: Lifetime Unit Mismatch ⚠️

**Server creates bullets with lifetime in MILLISECONDS:**
```javascript
// Server.js:439
lifetime: 5000  // 5 seconds = 5000 milliseconds
```

**But BulletManager expects lifetime in SECONDS:**
```javascript
// src/entities/BulletManager.js:72
this.life[index] = bulletData.lifetime || 3.0; // Default 3 seconds
```

**Result:** Bullets get a life of 5000 seconds instead of 5 seconds!

**However**, the update loop decrements by deltaTime (0.033s per frame):
```javascript
// BulletManager.js:105
this.life[i] -= deltaTime;  // Decrements by ~0.033 per frame
```

So a 5000-second bullet would take **41 hours** to expire! This isn't the problem.

### Issue #2: Missing ownerId in Bullet Creation ⚠️

**Server creates bullets with `owner` property:**
```javascript
// Server.js:435
owner: clientId,
```

**But getBulletsData sends `ownerId`:**
```javascript
// BulletManager.js:240
ownerId: this.ownerId[i],
```

**But BulletManager stores it in `ownerId` array:**
```javascript
// BulletManager.js:76
this.ownerId[index] = bulletData.ownerId || null;
```

**Problem:** Server sets `bullet.owner` but reads from `bullet.ownerId` - they don't match!

### Issue #3: Bullet Never Added to Manager 🔴 ROOT CAUSE

**When handlePlayerShoot creates a bullet with these properties:**
```javascript
{
  owner: clientId,        // ❌ Wrong property name
  ownerType: 'player',    // ❌ Not used by BulletManager
  worldId: mapId,         // ✅ Correct
  lifetime: 5000          // ⚠️ Wrong units (ms instead of seconds)
}
```

**BulletManager.addBullet expects:**
```javascript
{
  ownerId: clientId,      // Looks for 'ownerId', not 'owner'
  worldId: mapId,         // ✅ Required!
  lifetime: 5.0           // Should be in seconds
}
```

**What happens:**
1. Bullet is created with `owner` property
2. BulletManager reads `bulletData.ownerId` → gets `undefined`
3. Sets `this.ownerId[index] = undefined` (line 76)
4. Bullet IS added to manager (no rejection)
5. Bullet IS broadcasted in WORLD_UPDATE
6. Client receives bullet and should render it

**So why isn't it rendering?**

### Issue #4: Bullet Rendering Requires bulletManager in gameState 🔴

**Render function checks:**
```javascript
// render.js:421-424
const bm = gameState.bulletManager;
if (!bm) {
    return;  // No bullet manager = no rendering
}
```

**Need to verify:** Is `gameState.bulletManager` properly set?

---

## Debugging Steps

### Step 1: Enable Debug Logging ✅ DONE
```javascript
// Server.js:47
bulletEvents: true  // Now enabled
```

### Step 2: Check if Bullets Are Created on Server

When you shoot, you should see:
```
[SHOOT] Player <clientId> fired bullet at angle 1.57
```

### Step 3: Check if Bullets Are in WORLD_UPDATE

Add temporary logging to server:
```javascript
// Server.js after line 1174
console.log(`[BROADCAST] Sending ${bullets.length} bullets to client`);
```

### Step 4: Check if Client Receives Bullets

Add logging to client:
```javascript
// gameManager.js:266
if (bullets && bullets.length > 0) {
  console.log(`[CLIENT] Received ${bullets.length} bullets`);
  this.bulletManager.updateBullets(bullets);
}
```

### Step 5: Check if Bullets Are in Bullet Manager

Add logging to ClientBulletManager:
```javascript
// ClientBulletManager.js:223
console.log(`Set ${bullets.length} bullets from server, total bullets: ${this.bulletCount}`);
```

### Step 6: Check if Rendering is Called

Add logging to render.js:
```javascript
// render.js:428
if (bm.bulletCount === 0) {
  console.log('[RENDER] No bullets to render');
  return;
}
console.log(`[RENDER] Rendering ${bm.bulletCount} bullets`);
```

---

## Fixes Required

### Fix #1: Correct Property Names in handlePlayerShoot

```javascript
// Server.js:428-440
const bullet = {
  id: `bullet_${Date.now()}_${clientId}_${Math.random()}`,
  x: x || client.player.x,
  y: y || client.player.y,
  vx: Math.cos(angle) * speed,
  vy: Math.sin(angle) * speed,
  damage: damage || 10,
  ownerId: clientId,              // Changed from 'owner' to 'ownerId'
  worldId: mapId,                 // Keep worldId
  createdAt: Date.now(),
  lifetime: 5.0                   // Changed from 5000 to 5.0 (seconds)
};
```

### Fix #2: Add width/height to Bullet Creation

```javascript
const bullet = {
  // ... existing properties ...
  width: 8,   // Add explicit width
  height: 8,  // Add explicit height
  lifetime: 5.0
};
```

### Fix #3: Verify gameState.bulletManager Exists

Check in browser console:
```javascript
window.gameState.bulletManager
// Should return ClientBulletManager instance
```

If null, find where it should be initialized.

---

## Expected Behavior After Fixes

1. ✅ Player clicks to shoot
2. ✅ Client sends BULLET_CREATE to server
3. ✅ Server logs: `[SHOOT] Player X fired bullet at angle Y`
4. ✅ Server adds bullet to bulletMgr with correct properties
5. ✅ Server broadcasts bullet in WORLD_UPDATE
6. ✅ Client receives bullet and adds to ClientBulletManager
7. ✅ Client logs: `Set 1 bullets from server, total bullets: 1`
8. ✅ Render function draws bullet as glowing yellow/orange circle
9. ✅ Bullet moves according to vx/vy
10. ✅ Bullet expires after 5 seconds

---

## Testing Checklist

- [ ] Server logs bullet creation when player shoots
- [ ] Server broadcasts bullets in WORLD_UPDATE
- [ ] Client receives bullets in updateWorld
- [ ] Client bulletManager.bulletCount > 0
- [ ] Render function is called with bullets
- [ ] Bullets appear as glowing circles on screen
- [ ] Bullets move in correct direction
- [ ] Bullets collide with enemies
- [ ] Bullets expire after lifetime
