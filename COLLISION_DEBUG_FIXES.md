# Collision Debugging & Enemy Editor Improvements

**Date**: 2025-11-02
**Issues Fixed**:
1. Added comprehensive diagnostic logging for player-enemy projectile collisions
2. Added diagnostic logging for enemy self-collision detection
3. Added sprite preview to enemy editor

---

## Changes Made

### 1. Client-Side Player Collision Diagnostics

**File**: `public/src/collision/ClientCollisionManager.js`

**Changes**:
- Added detailed logging at function entry to check if required objects exist
- Added enemy bullet counting diagnostics
- Enhanced collision detection logging to show:
  - Which bullets are enemy bullets vs player bullets
  - WorldId mismatches (when bullets from different worlds are skipped)
  - Near-miss logging (bullets that pass close to player)
  - **Bright red error logs** when player is actually hit

**Key Additions**:
```javascript
// Line ~810: Count enemy bullets
let enemyBulletCount = 0;
for (let i = 0; i < this.bulletManager.bulletCount; i++) {
    const ownerId = this.bulletManager.ownerId[i];
    if (typeof ownerId === 'string' && ownerId.startsWith('enemy_')) {
        enemyBulletCount++;
    }
}
```

```javascript
// Line ~843: Always log player hits
console.error(`🎯 [PLAYER HIT] Enemy bullet ${bulletId} from ${ownerId} hit player for ${dmg} damage!`);
console.error(`   Bullet pos: (${bx}, ${by}), size: ${bw}x${bh}`);
console.error(`   Player pos: (${player.x}, ${player.y}), size: ${pw}x${ph}`);
```

**What to Look For**:
- If you see `[ENEMY BULLET CHECK] Found N enemy bullets` → System is detecting enemy bullets
- If you DON'T see `🎯 [PLAYER HIT]` when enemy shoots you → Collision detection failing
- Check for `Skipping bullet - wrong worldId` → WorldId mismatch issue

---

### 2. Server-Side Enemy Self-Collision Diagnostics

**File**: `src/entities/CollisionManager.js`

**Changes**:
- Added logging when enemies skip their own bullets (self-collision prevention)
- Added logging for enemy-to-enemy bullet collisions

**Key Additions**:
```javascript
// Line ~191: Log self-collision skip
if (bulletOwnerId === enemyId) {
  if (Math.random() < 0.01) {
    console.log(`[COLLISION] Skipping self-collision: Enemy ${enemyId} won't collide with own bullet`);
  }
  continue;
}

// Line ~198: Log enemy bullet vs enemy checks
const isEnemyShootingEnemy = typeof bulletOwnerId === 'string' && bulletOwnerId.startsWith('enemy_');
if (isEnemyShootingEnemy && Math.random() < 0.05) {
  console.log(`[COLLISION] Checking enemy bullet ${bulletId} (owner: ${bulletOwnerId}) vs enemy ${enemyId}`);
}
```

**What to Look For**:
- If you see `Skipping self-collision: Enemy enemy_X won't collide with own bullet` → Self-collision prevention working
- If you see enemies taking damage from their own bullets → ownerId mismatch issue
- Check console for which enemy IDs are shooting which enemies

---

### 3. Enemy Editor Sprite Preview

**File**: `public/editor/enemyEditor.js`

**Changes**:
- Added `SpritePreview` React component
- Component tries multiple sprite locations:
  1. `/assets/sprites/{spriteName}.png`
  2. `/assets/images/{spriteName}.png`
  3. `/assets/enemies/{spriteName}.png`
  4. `/assets/lofi_char.png` (fallback)
- Shows live preview with configurable scale
- Shows loading state and error messages
- Comprehensive console logging for debugging

**Features**:
- ✅ Pixel-perfect rendering (no blurring)
- ✅ Checkerboard background (shows transparency)
- ✅ Live updates when sprite name or scale changes
- ✅ Error handling with visual feedback
- ✅ Console logs show exactly what's being loaded

**Console Output**:
```
[SPRITE PREVIEW] Attempting to load sprite: skeleton
[SPRITE PREVIEW] Trying path: /assets/sprites/skeleton.png
[SPRITE PREVIEW] Failed to load from: /assets/sprites/skeleton.png
[SPRITE PREVIEW] Trying path: /assets/images/skeleton.png
[SPRITE PREVIEW] Successfully loaded sprite from: /assets/images/skeleton.png
[SPRITE PREVIEW] Image dimensions: 64x64
```

---

## How to Use the Diagnostics

### Testing Player Collision with Enemy Bullets

1. **Start the game** and open browser console (F12)
2. **Spawn an enemy** that shoots projectiles
3. **Look for these logs**:
   ```
   [ENEMY BULLET CHECK] Found 3 enemy bullets to check against player at (10.50, 15.25)
   ```
   This means the system detects enemy bullets.

4. **Stand in front of enemy and let it shoot you**
5. **You should see**:
   ```
   🎯 [PLAYER HIT] Enemy bullet bullet_123 from enemy_5 hit player for 10 damage!
      Bullet pos: (10.30, 15.20), size: 0.40x0.40
      Player pos: (10.50, 15.25), size: 1.00x1.00
      Player health: 100 → 90
   ```

6. **If you DON'T see the hit log**:
   - Check if bullet owner is correct: `[ENEMY BULLET CHECK] Bullet X is NOT enemy bullet (owner: Y)`
   - Check worldId: `Skipping bullet - wrong worldId`
   - Check near-misses: `Near miss - bullet at (...), player at (...), distance: 2.5`

---

### Testing Enemy Self-Collision

1. **Start server** and check console
2. **Spawn 2+ enemies that shoot**
3. **Look for these logs**:
   ```
   [COLLISION] Skipping self-collision: Enemy enemy_5 won't collide with own bullet bullet_123
   ```
   This means enemies are correctly ignoring their own bullets.

4. **If enemies ARE hitting themselves**:
   - Check if you see the skip log above
   - If NOT, check bullet `ownerId` format: should be `enemy_X` matching enemy ID
   - Look for collision processing logs to see which bullets hit which enemies

5. **Enemy-to-enemy friendly fire**:
   - You SHOULD see: `[COLLISION] Checking enemy bullet bullet_123 (owner: enemy_5) vs enemy enemy_6`
   - This is expected! Enemies CAN hit other enemies (just not themselves)

---

### Testing Enemy Editor Sprite Preview

1. **Open enemy editor**: `/enemy-attack-editor.html` or `/editor/enemyEditor.html`
2. **Open browser console** (F12)
3. **Type a sprite name** in the "Sprite" field (e.g., "skeleton", "goblin", "default")
4. **Watch console logs**:
   ```
   [SPRITE PREVIEW] Attempting to load sprite: skeleton
   [SPRITE PREVIEW] Trying path: /assets/sprites/skeleton.png
   [SPRITE PREVIEW] Failed to load from: /assets/sprites/skeleton.png
   [SPRITE PREVIEW] Trying path: /assets/images/skeleton.png
   [SPRITE PREVIEW] Successfully loaded sprite from: /assets/images/skeleton.png
   [SPRITE PREVIEW] Image dimensions: 64x64
   ```

5. **If sprite doesn't show**:
   - Check console for all 4 paths tried
   - Verify sprite file exists in one of those locations
   - Check browser Network tab to see actual HTTP requests
   - Look for error message in the preview box

6. **Change Render Scale** slider to see sprite at different sizes

---

## Common Issues & Solutions

### Issue 1: Player Not Taking Damage from Enemy Bullets

**Symptoms**:
- Enemy shoots but player health doesn't decrease
- No `🎯 [PLAYER HIT]` logs in console

**Debugging**:
1. Check if `[ENEMY BULLET CHECK] Found N enemy bullets` appears
   - If NO: Bullets not being created or not marked as enemy bullets
   - Check bullet `ownerId` in server logs when bullets are created

2. Check for worldId mismatches:
   - Look for: `Skipping bullet - wrong worldId`
   - Verify player and bullets are in same world

3. Check collision detection:
   - Look for near-misses: `Near miss - bullet at (...), distance: X`
   - If distance is small (< 1.0), collision should trigger
   - Check player collision size: `collisionWidth` and `collisionHeight`

**Fix**:
- Ensure bullets have `ownerId: "enemy_X"` format
- Ensure bullets have correct `worldId`
- Verify bullet `isEnemy: true` property

---

### Issue 2: Enemies Hitting Themselves

**Symptoms**:
- Enemy shoots and immediately takes damage
- Enemy health decreases when it fires

**Debugging**:
1. Check server console for: `Skipping self-collision: Enemy enemy_X won't collide with own bullet`
   - If NOT appearing: Self-collision check failing

2. Check bullet creation logs to see ownerId:
   ```
   [SERVER BULLET CREATE] ID: bullet_123, Owner: enemy_5
   ```

3. Check collision logs:
   ```
   [COLLISION] Checking enemy bullet bullet_123 (owner: enemy_5) vs enemy enemy_5
   ```
   Should be skipped!

**Fix**:
- Verify `bulletOwnerId === enemyId` check on line ~189 of CollisionManager.js
- Ensure enemy IDs and bullet ownerIds use same format: `enemy_X`
- Check that `bulletManager.ownerId[i]` is being set correctly

---

### Issue 3: Sprite Preview Not Working

**Symptoms**:
- Red "?" appears in preview box
- Error message: "Sprite 'X' not found in any location"

**Debugging**:
1. Check console logs to see which paths were tried:
   ```
   [SPRITE PREVIEW] Trying path: /assets/sprites/skeleton.png
   [SPRITE PREVIEW] Failed to load from: ...
   ```

2. Open browser Network tab (F12 → Network)
   - Filter by "Img"
   - See which requests returned 404

3. Check actual sprite file locations:
   - Look in `/public/assets/sprites/`
   - Look in `/public/assets/images/`
   - Look in `/public/assets/enemies/`

**Fix**:
- Add sprite files to one of the checked locations
- OR update `possiblePaths` array in SpritePreview component (line ~30)
- OR use sprite sheet coordinates instead of individual files

---

## Next Steps (If Issues Persist)

### If Player Still Not Colliding with Enemy Bullets:

1. **Check if function is being called**:
   ```javascript
   // Add to checkEnemyBulletsHitPlayer() at line ~798
   console.log('[ENEMY BULLET CHECK] Function called at', new Date().toISOString());
   ```

2. **Check bullet manager state**:
   ```javascript
   // In checkEnemyBulletsHitPlayer()
   console.log('[ENEMY BULLET CHECK] BulletManager state:', {
       bulletCount: this.bulletManager.bulletCount,
       totalBullets: this.bulletManager.id.length,
       sampleBullet: {
           id: this.bulletManager.id[0],
           ownerId: this.bulletManager.ownerId[0],
           x: this.bulletManager.x[0],
           y: this.bulletManager.y[0]
       }
   });
   ```

3. **Verify collision math**:
   Add this before the `if (hit)` check:
   ```javascript
   const overlap = {
       left: bx < player.x + pw,
       right: bx + bw > player.x,
       top: by < player.y + ph,
       bottom: by + bh > player.y
   };
   console.log(`[COLLISION MATH]`, overlap, `all: ${Object.values(overlap).every(v => v)}`);
   ```

### If Enemies Still Hitting Themselves:

1. **Add assertion at collision point**:
   ```javascript
   // In CollisionManager.js before the AABB check
   if (bulletOwnerId === enemyId) {
       console.error(`❌ BUG: Self-collision not skipped! Bullet: ${bulletId}, Enemy: ${enemyId}`);
       continue;
   }
   ```

2. **Log all bullet-enemy pairs being checked**:
   ```javascript
   // At start of enemy loop (line ~174)
   console.log(`[COLLISION] Checking ${bulletId} (owner: ${bulletOwnerId}) against ${enemyManager.enemyCount} enemies`);
   ```

### If Sprite Preview Still Broken:

1. **Check if React hooks are working**:
   ```javascript
   // In SpritePreview useEffect
   console.log('[SPRITE PREVIEW] useEffect triggered', { spriteName, scale, canvas: !!canvasRef.current });
   ```

2. **Verify canvas context**:
   ```javascript
   // After getting ctx
   if (!ctx) {
       console.error('[SPRITE PREVIEW] Failed to get 2D context!');
       return;
   }
   console.log('[SPRITE PREVIEW] Canvas context obtained', ctx);
   ```

3. **Test with absolute URL**:
   ```javascript
   // Replace relative paths with absolute
   const possiblePaths = [
       `http://localhost:3000/assets/sprites/${spriteName}.png`,
       // ...
   ];
   ```

---

## Summary

✅ **Added comprehensive logging** for all collision scenarios
✅ **Enemy editor now has sprite preview** with error handling
✅ **Self-collision prevention** is being logged
✅ **Player-enemy bullet collision** detection is verbose

**Next**: Run the game, check the console logs, and report what you see!

The logs will tell us exactly what's happening (or not happening) with collisions.
