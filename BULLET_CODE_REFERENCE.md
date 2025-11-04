# Bullet System - Code Reference Guide

Quick reference for all bullet-related code locations and implementations.

---

## Bullet Ownership & Identification

### Enemy ID Prefix Check
```javascript
// Used throughout the codebase to identify enemy bullets
const isEnemyBullet = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
```

**Locations where this pattern is used**:
- `/src/entities/CollisionManager.js:199` - Server-side collision check
- `/src/entities/CollisionManager.js:227` - Server-side player hit check
- `/public/src/collision/ClientCollisionManager.js:121` - Client QuadTree insertion
- `/public/src/collision/ClientCollisionManager.js:286` - Client SpatialGrid insertion
- `/public/src/collision/ClientCollisionManager.js:818` - Client player hit check

---

## Bullet Sprite & Rendering

### Sprite Name Assignment (Server-Side Creation)

**File**: `/src/Behaviours/Behaviors.js`

Line 628 (single projectile):
```javascript
spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
```

Line 656 (multiple projectiles):
```javascript
spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
```

### Color Rendering (Client-Side)

**File**: `/public/src/render/render.js` (lines 534-633)

```javascript
export function renderBullets() {
  // ... setup code ...
  
  for (let i = 0; i < bm.bulletCount; i++) {
    // ... position calculation ...
    
    // Line 611: Determine if bullet is from local player
    const isLocal = bm.ownerId[i] === gameState.character?.id;
    
    // Lines 612-621: Set gradient based on owner
    const grad = ctx.createRadialGradient(sx, sy, 0, sx, sy, drawW);
    if (isLocal) {
      // PLAYER BULLETS - Yellow/Orange gradient
      grad.addColorStop(0, 'rgb(255,255,120)');      // Yellow center
      grad.addColorStop(0.7, 'rgb(255,160,0)');      // Orange mid
      grad.addColorStop(1, 'rgba(255,100,0,0)');     // Orange fade
    } else {
      // ENEMY/OTHER BULLETS - Purple/Red gradient
      grad.addColorStop(0, 'rgb(255,100,255)');      // Purple center
      grad.addColorStop(0.7, 'rgb(255,0,100)');      // Red/Pink mid
      grad.addColorStop(1, 'rgba(200,0,0,0)');       // Red fade
    }
    
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(sx, sy, drawW, 0, Math.PI*2);
    ctx.fill();
  }
}
```

---

## Collision Detection

### Server-Side Collision (Enemy Bullets vs Players)

**File**: `/src/entities/CollisionManager.js` (lines 174-275)

```javascript
checkCollisions(deltaTime = 0.033, players = []) {
  for (let bi = 0; bi < this.bulletManager.bulletCount; bi++) {
    // Skip expired bullets
    if (this.bulletManager.life[bi] <= 0) continue;
    
    const bulletOwnerId = this.bulletManager.ownerId[bi];
    
    // ===== PLAYER BULLET vs ENEMY COLLISION =====
    for (let ei = 0; ei < this.enemyManager.enemyCount; ei++) {
      // Skip self-collision (enemy's own bullet)
      if (bulletOwnerId === enemyId) continue;
      
      // CRITICAL: Skip enemy bullets from hitting enemies
      const isEnemyBullet = typeof bulletOwnerId === 'string' && 
                            bulletOwnerId.startsWith('enemy_');
      if (isEnemyBullet) {
        continue; // Don't check enemies - check players instead
      }
      
      // Check collision for player bullets only
      if (this.checkAABBCollision(...)) {
        this.processCollision(bi, ei, bulletOwnerId);
      }
    }
    
    // ===== ENEMY BULLET vs PLAYER COLLISION =====
    const isEnemyBullet = typeof bulletOwnerId === 'string' && 
                          bulletOwnerId.startsWith('enemy_');
    if (isEnemyBullet && players && players.length > 0) {
      for (const player of players) {
        if (!player || player.health <= 0) continue;
        
        // Check collision
        if (this.checkAABBCollision(
          bulletX, bulletY, bulletWidth, bulletHeight,
          playerX, playerY, playerWidth, playerHeight
        )) {
          // Apply damage to player
          const damage = this.bulletManager.damage[bi] || 10;
          console.log(`[SERVER] Enemy bullet hit player for ${damage} damage`);
          
          if (typeof player.takeDamage === 'function') {
            player.takeDamage(damage);
          } else {
            player.health = Math.max(0, player.health - damage);
          }
          
          // Remove bullet
          this.bulletManager.markForRemoval(bi);
          break; // Move to next bullet
        }
      }
    }
  }
}
```

### Client-Side Collision (Optimistic Player Hit Check)

**File**: `/public/src/collision/ClientCollisionManager.js` (lines 799-896)

```javascript
checkEnemyBulletsHitPlayer() {
  if (!this.bulletManager || !window.gameState?.character) return;
  
  const player = window.gameState.character;
  
  for (let i = 0; i < this.bulletManager.bulletCount; i++) {
    if (this.bulletManager.life[i] <= 0) continue;
    
    const ownerId = this.bulletManager.ownerId[i];
    
    // Check if it's an enemy bullet
    const isEnemyBullet = typeof ownerId === 'string' && 
                          ownerId.startsWith('enemy_');
    if (!isEnemyBullet) continue;
    
    // Check worldId match
    if (this.bulletManager.worldId[i] !== player.worldId) continue;
    
    // Get bullet and player hitboxes
    const bx = this.bulletManager.x[i];
    const by = this.bulletManager.y[i];
    const bw = this.bulletManager.width[i];
    const bh = this.bulletManager.height[i];
    const pw = player.collisionWidth || 1;
    const ph = player.collisionHeight || 1;
    
    // Simple AABB collision check
    const hit = (
      bx < player.x + pw &&
      bx + bw > player.x &&
      by < player.y + ph &&
      by + bh > player.y
    );
    
    if (hit) {
      const dmg = this.bulletManager.damage[i] || 10;
      console.error(`[PLAYER HIT] Enemy bullet ${this.bulletManager.id[i]} hit player for ${dmg} damage!`);
      
      // Apply damage
      if (typeof player.takeDamage === 'function') {
        player.takeDamage(dmg);
      } else {
        player.health = Math.max(0, (player.health || 100) - dmg);
      }
      
      // Remove bullet locally
      this.bulletManager.markForRemoval(i);
      
      // Update UI
      if (window.gameUI?.updateHealth) {
        window.gameUI.updateHealth(player.health, player.maxHealth || 100);
      }
    }
  }
}
```

### Client-Side Player Bullets vs Enemies

**File**: `/public/src/collision/ClientCollisionManager.js` (lines 278-359)

```javascript
_updateWithSpatialGrid(deltaTime) {
  // ... grid setup ...
  
  // Get potential collision pairs
  const potentialPairs = this.grid.getPotentialCollisionPairs();
  
  for (const [bulletIndex, enemyIndex] of potentialPairs) {
    // Verify indices are valid
    if (bulletIndex >= this.bulletManager.bulletCount || 
        enemyIndex >= this.enemyManager.enemyCount) continue;
    
    // CRITICAL: Skip enemy bullets entirely
    const ownerId = this.bulletManager.ownerId[bulletIndex];
    const isEnemyBullet = ownerId && typeof ownerId === 'string' && 
                          ownerId.startsWith('enemy_');
    
    if (isEnemyBullet) {
      continue; // Skip enemy bullets - only check player bullets
    }
    
    // Skip dead enemies
    if (this.enemyManager.health[enemyIndex] <= 0) continue;
    
    // Check actual collision
    if (this.checkAABBCollision(
      this.bulletManager.x[bulletIndex],
      this.bulletManager.y[bulletIndex],
      this.bulletManager.width[bulletIndex],
      this.bulletManager.height[bulletIndex],
      this.enemyManager.x[enemyIndex],
      this.enemyManager.y[enemyIndex],
      this.enemyManager.width[enemyIndex],
      this.enemyManager.height[enemyIndex]
    )) {
      this.handleCollision(bulletIndex, enemyIndex);
    }
  }
}
```

---

## Bullet Creation

### Enemy Bullet Creation (Shoot Behavior)

**File**: `/src/Behaviours/Behaviors.js` (lines 605-660)

```javascript
fireProjectiles(index, enemyManager, bulletManager, baseAngle) {
  // Calculate spawn offset from enemy center
  const spawnOffset = (enemyManager.width[index] * 0.5) + 0.3;
  
  // Single or multiple projectile case
  for (let i = 0; i < this.projectileCount; i++) {
    const angle = baseAngle + 
                  (this.spread * i) + 
                  (Math.random() * 2 - 1) * this.inaccuracy;
    
    const spawnX = enemyManager.x[index] + Math.cos(angle) * spawnOffset;
    const spawnY = enemyManager.y[index] + Math.sin(angle) * spawnOffset;
    
    // CREATE BULLET WITH ENEMY AS OWNER
    bulletManager.addBullet({
      x: spawnX,
      y: spawnY,
      vx: Math.cos(angle) * enemyManager.bulletSpeed[index],
      vy: Math.sin(angle) * enemyManager.bulletSpeed[index],
      ownerId: enemyManager.id[index],            // THIS IS THE ENEMY'S ID!
      damage: enemyManager.damage[index],
      lifetime: 3.0,
      width: 0.4,
      height: 0.4,
      isEnemy: true,
      spriteName: enemyManager.bulletSpriteName[index] || 'projectile_basic',
      worldId: enemyManager.worldId[index]
    });
  }
}
```

**Called by**:
- `Shoot.execute()` at line 602

### Enemy Grenade Creation (GrenadeThrow Behavior)

**File**: `/src/Behaviours/Behaviors.js` (lines 531-552)

```javascript
throwGrenade(index, enemyManager, bulletManager, targetX, targetY) {
  bulletManager.addBullet({
    x: enemyManager.x[index],
    y: enemyManager.y[index],
    vx: (targetX - enemyManager.x[index]) / 1.5,
    vy: (targetY - enemyManager.y[index]) / 1.5,
    ownerId: enemyManager.id[index],             // ENEMY OWNER
    damage: 0,
    lifetime: 1.5,
    width: 0.6,
    height: 0.6,
    isEnemy: true,
    isGrenade: true,
    explosionRadius: this.radius,
    explosionDamage: this.damage,
    explosionEffect: this.effect,
    explosionEffectDuration: this.effectDuration,
    spriteName: 'grenade',
    worldId: enemyManager.worldId[index]
  });
}
```

### Player Bullet Creation (Input Handler)

**File**: `/public/src/game/input.js` (lines 11-70)

```javascript
function handleShoot(targetX, targetY) {
  const networkManager = window.networkManager || window.gameState?.networkManager;
  
  if (!gameState.character || !networkManager) return;
  
  // Calculate direction from player to target
  const playerX = gameState.character.x;
  const playerY = gameState.character.y;
  const dx = targetX - playerX;
  const dy = targetY - playerY;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  if (distance === 0) return;
  
  // Calculate velocity
  const bulletSpeed = 10;
  const vx = (dx / distance) * bulletSpeed;
  const vy = (dy / distance) * bulletSpeed;
  const angle = Math.atan2(vy, vx);
  const speed = Math.sqrt(vx * vx + vy * vy);
  
  // Send to server
  if (typeof networkManager.sendShoot === 'function') {
    networkManager.sendShoot({
      x: playerX,
      y: playerY,
      angle,
      speed,
      damage: 10
    });
  }
}
```

---

## Data Structures

### BulletManager (Server-Side)

**File**: `/src/entities/BulletManager.js` (lines 7-49)

Structure of Arrays (SoA) layout for performance:
```javascript
constructor(maxBullets = 10000) {
  this.bulletCount = 0;
  this.nextBulletId = 1;
  
  // SoA arrays
  this.id = new Array(maxBullets);
  this.x = new Float32Array(maxBullets);      // TILE UNITS
  this.y = new Float32Array(maxBullets);      // TILE UNITS
  this.vx = new Float32Array(maxBullets);     // TILES/SECOND
  this.vy = new Float32Array(maxBullets);     // TILES/SECOND
  this.life = new Float32Array(maxBullets);   // Seconds remaining
  this.width = new Float32Array(maxBullets);  // TILE UNITS (0.4-0.6)
  this.height = new Float32Array(maxBullets); // TILE UNITS (0.4-0.6)
  this.damage = new Float32Array(maxBullets);
  this.ownerId = new Array(maxBullets);       // WHO FIRED THIS
  this.spriteName = new Array(maxBullets);    // For rendering
  this.worldId = new Array(maxBullets);       // World/realm ID
}
```

### ClientBulletManager (Client-Side)

**File**: `/public/src/game/ClientBulletManager.js` (lines 10-43)

Similar SoA structure for client prediction:
```javascript
constructor(maxBullets = 10000) {
  this.bulletCount = 0;
  
  // Core arrays (same as server)
  this.id = new Array(maxBullets);
  this.x = new Float32Array(maxBullets);
  this.y = new Float32Array(maxBullets);
  this.vx = new Float32Array(maxBullets);
  this.vy = new Float32Array(maxBullets);
  this.life = new Float32Array(maxBullets);
  this.width = new Float32Array(maxBullets);
  this.height = new Float32Array(maxBullets);
  this.ownerId = new Array(maxBullets);      // WHO FIRED THIS
  this.damage = new Float32Array(maxBullets);
  this.worldId = new Array(maxBullets);
  
  // Sprite info
  this.sprite = new Array(maxBullets);       // Legacy sprite sheet
  this.spriteName = new Array(maxBullets);   // New system
  
  // Tracking
  this.idToIndex = new Map();                // Fast ID lookup
  this.localBullets = new Set();             // Local predictions
}
```

---

## AABB Collision Detection

### Server-Side AABB Check

**File**: `/src/entities/CollisionManager.js` (lines 514-532)

```javascript
checkAABBCollision(ax, ay, awidth, aheight, bx, by, bwidth, bheight) {
  // Convert center positions to min/max extents
  const aMinX = ax - awidth / 2;
  const aMaxX = ax + awidth / 2;
  const aMinY = ay - aheight / 2;
  const aMaxY = ay + aheight / 2;
  
  const bMinX = bx - bwidth / 2;
  const bMaxX = bx + bwidth / 2;
  const bMinY = by - bheight / 2;
  const bMaxY = by + bheight / 2;
  
  return (
    aMinX < bMaxX &&
    aMaxX > bMinX &&
    aMinY < bMaxY &&
    aMaxY > bMinY
  );
}
```

### Client-Side AABB Check

**File**: `/public/src/collision/ClientCollisionManager.js` (lines 491-545)

Same logic but with additional coordinate debugging.

---

## Key Constants & Sizes

| Property | Value | Units | Notes |
|----------|-------|-------|-------|
| Bullet collision width | 0.4-0.6 | tiles | 40-60% of a tile |
| Bullet collision height | 0.4-0.6 | tiles | 40-60% of a tile |
| Bullet lifetime | 3.0 | seconds | Default max flight time |
| Bullet speed (player) | 10 | tiles/sec | Normalized direction * 10 |
| Bullet speed (enemy) | varies | tiles/sec | enemyManager.bulletSpeed[i] |
| TILE_SIZE | 12 | pixels | Conversion factor world→screen |

---

## Common Issues & Debugging

### Issue: Enemy bullets not hitting players

**Check these in order**:
1. Verify `checkCollisions()` receives `players` array as parameter
2. Log enemy bullet `ownerId` to confirm 'enemy_' prefix
3. Check `worldId` matches between bullet and player
4. Verify collision box sizes (width/height not 0)
5. Check player health application code

### Issue: Bullets wrong color

**Likely causes**:
1. `ownerId` not set correctly when creating bullet
2. Client's character ID doesn't match server's ID
3. Rendering code using wrong comparison operator

### Issue: Enemy bullets colliding with each other

**Root cause**: Early code didn't have `isEnemyBullet` check at line 199. Fixed in current version but verify it's in your code.

---

## Testing Commands

```javascript
// In browser console to test collision
window.COLLISION_STATS = { 
  entityCollisions: 0,
  lastEntityCollisions: []
};

// Monitor in real-time
setInterval(() => {
  console.log('Collisions:', window.COLLISION_STATS);
}, 1000);
```

