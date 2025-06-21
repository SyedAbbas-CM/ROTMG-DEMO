/* DebugTools.js
 * Convenience helpers exposed on window.DebugTools for quick testing in the browser console.
 */

export function setupDebugTools({ spriteDatabase, enemyManager, bulletManager, mapManager, playerManager }) {
  const DebugTools = {
    spriteDB: spriteDatabase,
    enemyManager,
    bulletManager,
    mapManager,
    playerManager,

    // Reload a list of atlases
    async reloadAtlases(pathsArray) {
      if (!Array.isArray(pathsArray)) {
        console.error('reloadAtlases expects an array of paths');
        return;
      }
      console.log('[DebugTools] Reloading atlases:', pathsArray);
      await spriteDatabase.loadAtlases(pathsArray);
      console.log('[DebugTools] Atlases reloaded. Stats:', spriteDatabase.getStats());
    },

    // List sprites (optionally filter by prefix) and limit output
    listSprites(prefix = '', limit = 100) {
      const all = spriteDatabase.getAllSpriteNames();
      const filtered = prefix ? all.filter(n => n.startsWith(prefix)) : all;
      const slice = filtered.slice(0, limit);
      console.log(`[DebugTools] ${slice.length}/${filtered.length} sprites${prefix ? ' with prefix ' + prefix : ''}:`, slice);
      return slice;
    },

    // Render a sprite into a floating debug canvas for visual inspection
    showSprite(spriteName, size = 64) {
      const sprite = spriteDatabase.getSprite(spriteName);
      if (!sprite) {
        console.warn(`[DebugTools] Sprite not found: ${spriteName}`);
        return null;
      }
      let canvas = document.getElementById('debugSpriteCanvas');
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'debugSpriteCanvas';
        canvas.style.position = 'fixed';
        canvas.style.bottom = '10px';
        canvas.style.right = '10px';
        canvas.style.border = '2px solid #fff';
        canvas.style.background = '#000';
        canvas.width = size;
        canvas.height = size;
        canvas.style.zIndex = 9999;
        document.body.appendChild(canvas);
      }
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      spriteDatabase.drawSprite(ctx, spriteName, 0, 0, size, size);
      console.log(`[DebugTools] Displaying sprite '${spriteName}' at ${size}px`);
      return sprite;
    },

    // Dump current enemy types mapping
    dumpEnemyTypes() {
      if (!enemyManager) {
        console.warn('[DebugTools] enemyManager not available');
        return;
      }
      console.table(enemyManager.enemyTypes.map((t, i) => ({ index: i, name: t.name, sprite: t.spriteName, x: t.spriteX, y: t.spriteY })));
    },

    // Quick stats printout
    stats() {
      console.log('[DebugTools] SpriteDB stats:', spriteDatabase.getStats());
      if (bulletManager) {
        console.log('[DebugTools] BulletManager:', { count: bulletManager.bulletCount });
      }
      if (enemyManager) {
        console.log('[DebugTools] EnemyManager:', { count: enemyManager.enemyCount });
      }
    }
  };

  window.DebugTools = DebugTools;
  console.log('%cDebugTools ready. Try `DebugTools.listSprites()`', 'color: #0f0');
} 