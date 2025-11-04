# Bullet System Analysis - ROTMG-DEMO

## Executive Summary

The game has a functional bullet system with proper ownership tracking, but there's a critical issue where **enemy bullets colliding with each other and not hitting players** suggests the collision logic is checking bullet ownership incorrectly.

---

## 1. BULLET OWNERSHIP TRACKING

### How Bullets Know Who Fired Them

All bullets store an `ownerId` field that identifies the source:

**Server-Side Storage** (`/src/entities/BulletManager.js`, lines 34):
```javascript
this.ownerId = new Array(maxBullets);   // ID of entity that created this bullet
```

**Client-Side Storage** (`/public/src/game/ClientBulletManager.js`, lines 23):
```javascript
this.ownerId = new Array(maxBullets);   // Who fired this bullet
```

### Owner ID Format

- **Player bullets**: `ownerId` = `character.id` (unique player identifier)
- **Enemy bullets**: `ownerId` = `enemyManager.id[index]` (enemy ID, prefixed with 'enemy_')

The system uses string prefix checking:
```javascript
const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
```

---

## 2. BULLET SPRITE/COLOR ASSIGNMENT

### Where Sprites are Set

**Server-side bullet creation** (`/src/Behaviours/Behaviors.js`):

#### Lines 617-630 - Single projectile (player bullet):
```javascript
bulletManager.addBullet({
  x: spawnX,
  y: spawnY,
  vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
  vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
  ownerId: enemyManager.id[index],              // ENEMY ID
  damage: enemyManager.damage[index],
  lifetime: 3.0,
  width: 0.4,
  height: 0.4,
  isEnemy: true,                                 // FLAG indicating enemy bullet
  spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
  worldId: enemyManager.worldId[index]
});
```

### Client-Side Rendering Colors

**File**: `/public/src/render/render.js` (lines 534-633)

The `renderBullets()` function determines bullet colors based on ownership:

```javascript
// Line 611: Check if bullet belongs to local player
const isLocal = bm.ownerId[i] === gameState.character?.id;

// Lines 612-621: Create color gradient based on ownership
const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, drawW);
if (isLocal) {
  // PLAYER BULLETS - Yellow/Orange
  grad.addColorStop(0, 'rgb(255,255,120)');      // Yellow center
  grad.addColorStop(0.7, 'rgb(255,160,0)');      // Orange middle
  grad.addColorStop(1, 'rgba(255,100,0,0)');     // Orange fade
} else {
  // ENEMY/OTHER BULLETS - Red/Purple
  grad.addColorStop(0, 'rgb(255,100,255)');      // Purple center
  grad.addColorStop(0.7, 'rgb(255,0,100)');      // Red/Pink middle
  grad.addColorStop(1, 'rgba(200,0,0,0)');       // Red fade
}
```

**Color Summary**:
- **Player bullets**: Yellow (255,255,120) → Orange (255,160,0)
- **Enemy bullets**: Purple (255,100,255) → Red (255,0,100)

---

## 3. COLLISION DETECTION LOGIC

### The Critical Problem: Enemy Bullet Behavior

**File**: `/src/entities/CollisionManager.js` (server-side)

#### Lines 198-202: Enemy bullets skip enemy collisions
```javascript
// Skip enemy bullets hitting other enemies (they should only hit players)
const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
if (isEnemyBullet) {
  continue; // Check players instead (handled separately below)
}
```

This code correctly prevents enemy-to-enemy bullet collisions.

#### Lines 227-274: Enemy bullets SHOULD hit players
```javascript
const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
if (isEnemyBullet && players && players.length > 0) {
  for (const player of players) {
    // Check collision with player...
    if (this.checkAABBCollision(...)) {
      // Apply damage to player
      player.takeDamage(damage);
      this.bulletManager.markForRemoval(bi);
    }
  }
}
```

### Client-Side Collision (Optimistic)

**File**: `/public/src/collision/ClientCollisionManager.js`

#### Lines 799-896: Enemy bullet check against local player
```javascript
checkEnemyBulletsHitPlayer() {
  // ...
  for (let i = 0; i < this.bulletManager.bulletCount; i++) {
    const ownerId = this.bulletManager.ownerId[i];
    
    // Check if it's an enemy bullet
    const isEnemyBullet = typeof ownerId === 'string' && ownerId.startsWith('enemy_');
    if (!isEnemyBullet) continue;
    
    // Check collision with player hitbox
    const hit = (
      bx < player.x + pw &&
      bx + bw > player.x &&
      by < player.y + ph &&
      by + bh > player.y
    );
    
    if (hit) {
      // Apply damage and remove bullet
      player.takeDamage(dmg);
      this.bulletManager.markForRemoval(i);
    }
  }
}
```

#### Lines 118-141: Player bullets skip enemies with `isEnemyBullet` check
```javascript
for (let i = 0; i < this.bulletManager.bulletCount; i++) {
  const ownerId = this.bulletManager.ownerId[i];
  const isEnemyBullet = ownerId && typeof ownerId === 'string' && ownerId.startsWith('enemy_');

  if (isEnemyBullet) {
    continue;  // SKIP enemy bullets - don't check them against enemies
  }
  
  // Only player bullets reach here for enemy collision checks
```

---

## 4. FACTION/TEAM SYSTEM

Currently, the system uses a **simple binary faction approach**:
- **Faction 1**: Players and their bullets
- **Faction 2**: Enemies and their bullets

There is **NO explicit team/faction/layer enum system** - it's determined by string prefix checking:

```javascript
// The ONLY faction indicator is the 'enemy_' prefix
const isEnemyBullet = bulletOwnerId && typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
```

---

## 5. BULLET CREATION FLOW

### Player Shooting

**File**: `/public/src/game/input.js` (lines 11-70)

1. Player clicks or presses spacebar
2. `handleShoot(targetX, targetY)` is called
3. Bullet data is sent to server via `networkManager.sendShoot()`

```javascript
function handleShoot(targetX, targetY) {
  const bulletSpeed = 10;
  const vx = (dx / distance) * bulletSpeed;
  const vy = (dy / distance) * bulletSpeed;
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

### Server receives and creates player bullet

The server then creates the actual bullet with the player's ID as `ownerId`.

### Enemy Shooting

**File**: `/src/Behaviours/Behaviors.js` (lines 558-661)

The `Shoot` behavior creates bullets:

```javascript
bulletManager.addBullet({
  x: spawnX,
  y: spawnY,
  vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
  vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
  ownerId: enemyManager.id[index],           // ENEMY ID (e.g., "enemy_123")
  damage: enemyManager.damage[index],
  lifetime: 3.0,
  width: 0.4,
  height: 0.4,
  isEnemy: true,
  spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
  worldId: enemyManager.worldId[index]
});
```

---

## 6. IDENTIFIED ISSUES & ROOT CAUSE

### Problem Report Summary
1. ✓ Enemy bullets collide with each other (WRONG)
2. ✗ Enemy bullets don't hit players (WRONG)
3. ✓ Player bullets are yellow/orange (CORRECT)
4. ✓ Enemy bullets are red/purple (CORRECT)
5. ? Enemy bullets treated as player bullets (UNCLEAR)

### Root Cause Analysis

The collision logic appears **correct on the server** (lines 198-274 in `CollisionManager.js`):

```javascript
// Enemy bullets SKIP enemy collision checks
const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
if (isEnemyBullet) {
  continue; // Skip to player check
}

// Then check enemy bullets against PLAYERS
if (isEnemyBullet && players && players.length > 0) {
  // Check collision with player...
  if (this.checkAABBCollision(...)) {
    player.takeDamage(damage);  // SHOULD WORK
    this.bulletManager.markForRemoval(bi);
  }
}
```

**Possible causes**:
1. **Missing `players` array**: The `checkCollisions(deltaTime, players)` method might not be receiving the players array
2. **Incorrect ownerId format**: Enemy IDs might not actually start with 'enemy_'
3. **Server-side damage not syncing**: Damage is applied server-side but client might not show it
4. **worldId mismatch**: Enemy and player might be in different worlds/realms
5. **Collision boxes too small**: Enemy bullet width/height (0.4 tiles) vs player hitbox size mismatch

---

## 7. FILE STRUCTURE & KEY LOCATIONS

### Server-Side Bullet System
- **BulletManager** (SoA data structure): `/src/entities/BulletManager.js`
- **CollisionManager** (Hit detection): `/src/entities/CollisionManager.js`
- **Behaviors** (Enemy shooting): `/src/Behaviours/Behaviors.js` (lines 558-661)
- **Enemy shooting config**: `/src/Behaviours/EnemyBehaviors.js` (line 240)

### Client-Side Bullet System
- **ClientBulletManager**: `/public/src/game/ClientBulletManager.js`
- **ClientCollisionManager**: `/public/src/collision/ClientCollisionManager.js`
- **Bullet Rendering**: `/public/src/render/render.js` (lines 534-633)
- **Player Input/Shooting**: `/public/src/game/input.js` (lines 11-70)

### Network
- **Server Main**: `/Server.js`
- **NetworkManager** (client): `/public/src/network/ClientNetworkManager.js`

---

## 8. KEY DATA STRUCTURES

### Bullet Data Format (when created)
```javascript
{
  id: "bullet_123",
  x: 50.5,                              // World X in tiles
  y: 100.3,                             // World Y in tiles
  vx: 5.0,                              // Velocity X in tiles/sec
  vy: 3.2,                              // Velocity Y in tiles/sec
  width: 0.6,                           // Collision width in tiles
  height: 0.6,                          // Collision height in tiles
  damage: 10,                           // Damage value
  lifetime: 3.0,                        // Lifetime in seconds
  ownerId: "enemy_42" or "player_1",   // WHO FIRED THIS
  spriteName: "projectile_basic",      // Sprite for rendering
  worldId: "world_1",                  // Which world/realm
  isEnemy: true                        // Optional flag
}
```

---

## 9. RECOMMENDATIONS FOR FIXES

### Priority 1: Verify Enemy Bullet Hitting
1. Add logging in `CollisionManager.checkCollisions()` to confirm `players` array is passed
2. Log every enemy bullet-player collision check
3. Verify enemy bullet `ownerId` actually has 'enemy_' prefix

### Priority 2: Improve Faction System
1. Add explicit `Faction` enum instead of string prefix:
   ```javascript
   const Faction = {
     PLAYER: 0,
     ENEMY: 1,
     NPC: 2
   };
   ```
2. Store faction in bullets and use for collision checks

### Priority 3: Enhanced Debugging
1. Add visual debug lines for bullet hitboxes
2. Log all collision checks with positions
3. Add server-side validation of enemy bullet hits

