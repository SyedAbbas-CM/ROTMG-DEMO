import { spriteManager } from './assets/spriteManager.js';
import { spriteDatabase } from './assets/SpriteDatabase.js';
import { ClientNetworkManager } from './network/ClientNetworkManager.js';
import { entityDatabase } from './assets/EntityDatabase.js';
import { tileDatabase } from './assets/TileDatabase.js';

// Initialize sprite manager
async function initializeSpriteManager() {
  try {
    console.log('🎨 Initializing sprite database system...');

    // Ask the server for the list of available atlases
    let atlasPaths = [];
    try {
      const res = await fetch('/api/assets/atlases');
      if (res.ok) {
        const data = await res.json();
        atlasPaths = Array.isArray(data.atlases) ? data.atlases : [];
      } else {
        console.warn(`⚠️ Could not retrieve atlas list from server (HTTP ${res.status}).`);
      }
    } catch (err) {
      console.warn('⚠️ Failed to contact server for atlas list:', err);
    }

    // Ensure core atlases present; prepend critical ones to guarantee unique names resolve correctly
    const essentialAtlases = [
      'assets/atlases/chars.json',   // main character/enemy sheet (includes red_demon)
      'assets/atlases/chars2.json'  // variant sheet
    ];

    // If server list was empty, use essentials; otherwise merge, keeping order (essentials first)
    if (atlasPaths.length === 0) {
      console.warn('⚠️ No atlases reported by server. Falling back to essential list.');
      atlasPaths = [...essentialAtlases];
    } else {
      // Prepend any missing essential atlases to the list (maintain unique paths)
      essentialAtlases.reverse().forEach(p => {
        if (!atlasPaths.includes(p)) atlasPaths.unshift(p);
      });
    }

    // Load the atlases (SpriteDatabase now tolerates failures)
    await spriteDatabase.loadAtlases(atlasPaths);

    // Load entities
    await entityDatabase.load();
    tileDatabase.merge(entityDatabase.getAll('tiles'));
    await spriteDatabase.loadEntities();

    // Developer diagnostics
    console.log('✅ Sprite database ready. Stats:', spriteDatabase.getStats());
    console.log('✅ Entity database ready – enemy count:', entityDatabase.getAll('enemies').length);

    // Expose globally for console experimentation
    window.spriteDatabase = spriteDatabase;
    window.entityDatabase = entityDatabase;

  } catch (error) {
    console.error('❌ Failed to initialize sprite database:', error);
    console.error('The game will attempt to continue, but rendering issues may occur.');
  }
}

export { initializeSpriteManager }; 