/**
 * public/src/world/ClientWorld.js
 * ---------------------------------------------------------------
 * A lightweight container for all client-side state that belongs
 * to a specific Realm/World.  It owns its own ClientMapManager,
 * ClientEnemyManager and ClientBulletManager instances so that we
 * can dispose them cleanly when the player switches through a portal.
 *
 * NOTE:  The game still keeps a single global Enemy/Bullet manager
 *        reference on purpose (debugging & "intentional leaks"), but
 *        those globals will simply point at the current world's
 *        managers.  When you dispose() a world the old managers live
 *        on for inspection but are no longer ticked or rendered.
 */

import { ClientMapManager, ClientBulletManager, ClientEnemyManager } from '../managers.js';

export class ClientWorld {
  /**
   * @param {Object}   opts
   * @param {Object}   opts.mapData        Metadata payload received from the server (MAP_INFO or WORLD_SWITCH)
   * @param {Object}   opts.networkManager Reference to the live ClientNetworkManager (used by the map manager)
   * @param {number}   [opts.maxEnemies]   Optional cap for the enemy manager
   * @param {number}   [opts.maxBullets]   Optional cap for the bullet manager
   */
  constructor({ mapData, networkManager, maxEnemies = 1000, maxBullets = 10000 }) {
    // ---------------------------------------------------------------------
    // Managers
    // ---------------------------------------------------------------------
    this.mapManager    = new ClientMapManager({ networkManager });
    this.enemyManager  = new ClientEnemyManager(maxEnemies);
    this.bulletManager = new ClientBulletManager(maxBullets);

    // Tell the new map manager about its first MAP_INFO straight away
    if (mapData) {
      this.mapManager.initMap(mapData);
    }

    // Make sure sprite data is re-checked once the sheets are in cache
    if (typeof this.enemyManager.reinitializeSprites === 'function') {
      setTimeout(() => this.enemyManager.reinitializeSprites(), 50);
    }
  }

  /**
   * Dispose of all heavy resources held by this world.  Call this BEFORE
   * dropping references so that render caches, arrays and intervals are
   * freed.
   */
  dispose() {
    // 1) Clear renderer tile caches so old tiles vanish immediately
    if (window.clearStrategicCache) window.clearStrategicCache();
    if (window.clearTopDownCache)   window.clearTopDownCache();

    // 2) Empty entity arrays so lingering sprites disappear
    if (this.enemyManager?.setEnemies)  this.enemyManager.setEnemies([]);
    if (this.bulletManager?.setBullets) this.bulletManager.setBullets([]);

    // 3) TODO: remove THREE.js meshes, particles, sounds tied to this world
    // For now the garbage collector will take care of detached arrays.
  }
}

// Convenience helper so devs can poke the active world from the console
if (typeof window !== 'undefined') {
  window.ClientWorld = ClientWorld;
} 