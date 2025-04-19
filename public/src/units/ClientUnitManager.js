/**
 *  Lightweight SoA mirror of the server‑side UnitManager.
 *  No AI – just interpolates positions for rendering.
 */

export default class ClientUnitManager {
    constructor(maxUnits = 10000) {
      this.max = maxUnits;
      this.count = 0;
  
      this.id = new Array(maxUnits);
      this.typeIdx = new Uint8Array(maxUnits);
      this.x = new Float32Array(maxUnits);
      this.y = new Float32Array(maxUnits);
      this.vx = new Float32Array(maxUnits);
      this.vy = new Float32Array(maxUnits);
      this.hp = new Float32Array(maxUnits);
  
      this.id2index = new Map();
    }
  
    _ensure(id) {
      let idx = this.id2index.get(id);
      if (idx !== undefined) return idx;
      idx = this.count++;
      this.id[idx] = id;
      this.id2index.set(id, idx);
      return idx;
    }
  
    /** Apply UNIT_CREATE (array of units) */
    spawnMany(arr) {
      arr.forEach(u => this.applyUpdate(u));
    }
  
    /** Apply UNIT_UPDATE diff object */
    applyUpdate(u) {
      const i = this._ensure(u.id);
      this.typeIdx[i] = u.type;
      this.x[i] = u.x;  this.y[i] = u.y;
      this.vx[i] = u.vx; this.vy[i] = u.vy;
      this.hp[i] = u.hp;
    }
  
    /** Apply UNIT_REMOVE (array of ids) */
    remove(ids) {
      ids.forEach(id => {
        const idx = this.id2index.get(id);
        if (idx === undefined) return;
        const last = this.count - 1;
        if (idx !== last) {
          this.id[idx]      = this.id[last];
          this.typeIdx[idx] = this.typeIdx[last];
          this.x[idx]       = this.x[last];
          this.y[idx]       = this.y[last];
          this.vx[idx]      = this.vx[last];
          this.vy[idx]      = this.vy[last];
          this.hp[idx]      = this.hp[last];
          this.id2index.set(this.id[idx], idx);
        }
        this.count--;
        this.id2index.delete(id);
      });
    }
  }
  