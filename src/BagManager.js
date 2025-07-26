// src/BagManager.js
// Simple server-side loot bag handler.  Each Bag is an entity placed in the
// world that owns an array of item instance IDs (managed by ItemManager).
// Bags auto-expire after `lifetime` seconds.  For visual / drag-and-drop on the
// client we emit bag metadata along with the usual world entity sync.
//
// IMPORTANT:   This first pass focuses on backend logic.  The UI side (displaying
//              bags, drag-and-drop loot windows) will be implemented separately
//              in the public/ folder and integrated via websocket messages.

export default class BagManager {
  /**
   * @param {number} maxBags – hard cap to prevent runaway entity counts
   */
  constructor(maxBags = 500) {
    this.maxBags = maxBags;
    this.bagCount = 0;

    // SoA layout for performance (mirrors EnemyManager/BulletManager pattern)
    this.id = new Array(maxBags);
    this.x = new Float32Array(maxBags);
    this.y = new Float32Array(maxBags);
    this.creationTime = new Float32Array(maxBags);
    this.lifetime = new Float32Array(maxBags);
    this.itemSlots = new Array(maxBags); // each entry is Array<itemInstanceId>
    this.bagType = new Uint8Array(maxBags);
    this.owners = new Array(maxBags); // array of clientIds allowed to see
    this.worldId = new Array(maxBags);

    this.nextBagId = 1;
  }

  /**
   * Spawn a new loot bag containing the supplied item instance IDs.
   * @param {number} x
   * @param {number} y
   * @param {Array<number>} itemIds – references from ItemManager
   * @param {string} worldId
   * @param {number} ttl – seconds before despawn (default 300 = 5 min)
   */
  spawnBag(x, y, itemIds = [], worldId = 'default', ttl = 300, bagType = 0, owners = null) {
    if (this.bagCount >= this.maxBags) {
      console.warn('[BagManager] Max bag capacity reached');
      return null;
    }
    const idx = this.bagCount++;
    const bagId = `bag_${this.nextBagId++}`;

    this.id[idx] = bagId;
    this.x[idx] = x;
    this.y[idx] = y;
    this.creationTime[idx] = Date.now() / 1000; // seconds
    this.lifetime[idx] = ttl;
    this.itemSlots[idx] = itemIds.slice(0, 8);
    this.bagType[idx] = bagType;
    this.owners[idx] = owners; // null → public
    this.worldId[idx] = worldId;
    return bagId;
  }

  /**
   * Per-tick cleanup – remove bags past their lifetime.
   * @param {number} nowSec – current time in seconds
   */
  update(nowSec) {
    for (let i = 0; i < this.bagCount; i++) {
      if (nowSec - this.creationTime[i] >= this.lifetime[i]) {
        this._swapRemove(i);
        i--; // re-check swapped index
      }
    }
  }

  /**
   * Called when a player takes an item from the bag.  Returns true if the bag
   * became empty and was removed.
   */
  removeItemFromBag(bagId, itemInstanceId) {
    const idx = this._findIndexById(bagId);
    if (idx === -1) return false;
    const arr = this.itemSlots[idx];
    const pos = arr.indexOf(itemInstanceId);
    if (pos !== -1) arr.splice(pos, 1);
    if (arr.length === 0) {
      this._swapRemove(idx);
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------
  _findIndexById(bagId) {
    for (let i = 0; i < this.bagCount; i++) {
      if (this.id[i] === bagId) return i;
    }
    return -1;
  }

  _swapRemove(idx) {
    const last = this.bagCount - 1;
    if (idx !== last) {
      this.id[idx] = this.id[last];
      this.x[idx] = this.x[last];
      this.y[idx] = this.y[last];
      this.creationTime[idx] = this.creationTime[last];
      this.lifetime[idx] = this.lifetime[last];
      this.itemSlots[idx] = this.itemSlots[last];
      this.worldId[idx] = this.worldId[last];
    }
    this.bagCount--;
  }

  // Utility for network sync – returns array of bag DTOs
  getBagsData(filterWorldId = null, viewerId = null) {
    const out = [];
    for (let i = 0; i < this.bagCount; i++) {
      if (filterWorldId && this.worldId[i] !== filterWorldId) continue;
      // visibility check: if owners defined ensure viewer included
      if(this.owners[i] && viewerId && !this.owners[i].includes(viewerId)) continue;
      out.push({
        id: this.id[i],
        x: this.x[i],
        y: this.y[i],
        bagType: this.bagType[i],
        items: this.itemSlots[i],
      });
    }
    return out;
  }
} 