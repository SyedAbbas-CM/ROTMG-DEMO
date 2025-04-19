// =============================================
// FILE: src/Managers/SoldierManager.js
// =============================================

/**
 * SoldierManager
 * -------------------------------------------------
 *  • Maintains every *individual* combatant (aka “unit” / “soldier”).
 *  • No aggregation logic here – groups / cohorts will be handled by
 *    a separate GroupManager that operates on soldier indices.
 *
 *  Data‑layout: Structure‑of‑Arrays (SoA) for cache efficiency and
 *  effortless transfer to WASM / web worker later.
 */

export default class SoldierManager {
    /**
     * @param {number} maxSoldiers – hard cap for allocation (default = 10 000)
     */
    constructor(maxSoldiers = 10_000) {
      this.max = maxSoldiers;
      this.count = 0;
      this.nextId = 1;
  
      /* ---------- SoA buffers ---------- */
  
      this.id          = new Array(maxSoldiers);      // string | number
      this.type        = new Uint8Array(maxSoldiers); // soldier type enum
  
      // position / motion
      this.x           = new Float32Array(maxSoldiers);
      this.y           = new Float32Array(maxSoldiers);
      this.vx          = new Float32Array(maxSoldiers);
      this.vy          = new Float32Array(maxSoldiers);
      this.ax          = new Float32Array(maxSoldiers);
      this.ay          = new Float32Array(maxSoldiers);
  
      // core stats
      this.health      = new Float32Array(maxSoldiers);
      this.maxHealth   = new Float32Array(maxSoldiers);
      this.armor       = new Float32Array(maxSoldiers);
      this.damage      = new Float32Array(maxSoldiers);
      this.range       = new Float32Array(maxSoldiers);
  
      // physics / morale
      this.mass        = new Float32Array(maxSoldiers); // kg equivalent
      this.morale      = new Float32Array(maxSoldiers); // 0‑100
      this.stability   = new Float32Array(maxSoldiers); // resistance to knock‑back
      this.state       = new Uint8Array(maxSoldiers);   // 0=idle,1=move,2=attack,3=rout,…
  
      // book‑keeping map (id → index)
      this.idToIndex   = new Map();
  
      /* ---------- soldier type catalogue ---------- */
  
      this.SOLDIER_TYPES = {
        0: { name: 'Infantry',       health: 100, armor:  5, speed: 20, damage: 15, range:  2,
             mass: 80,  stability: 50 },
  
        1: { name: 'HeavyInfantry',  health: 140, armor: 15, speed: 15, damage: 18, range:  2,
             mass: 90,  stability: 70 },
  
        2: { name: 'LightCavalry',   health: 110, armor:  5, speed: 35, damage: 20, range:  2,
             mass: 300, stability: 40 },
  
        3: { name: 'HeavyCavalry',   health: 160, armor: 20, speed: 28, damage: 26, range:  2,
             mass: 450, stability: 60 },
  
        4: { name: 'Archer',         health:  80, armor:  2, speed: 22, damage: 12, range: 60,
             mass: 70,  stability: 30 },
  
        5: { name: 'Crossbow',       health:  85, armor:  4, speed: 20, damage: 18, range: 55,
             mass: 75,  stability: 30 },
      };
    }
  
    /* ======================================================================
       SPAWN / REMOVE
       ====================================================================== */
  
    /**
     * Spawn a new soldier.
     * @param {number} type – enum key defined in SOLDIER_TYPES
     * @param {number} x,y – starting coordinates
     * @param {object} [overrides] – optional per‑instance stat tweaks
     * @return {string} soldierId or null if capacity reached
     */
    spawn(type, x, y, overrides = {}) {
      if (this.count >= this.max) {
        console.warn('[SoldierManager] capacity reached');
        return null;
      }
  
      const t = this.SOLDIER_TYPES[type] ? type : 0; // fall back to infantry
      const cfg = this.SOLDIER_TYPES[t];
  
      const idx = this.count++;
      const id  = overrides.id || `soldier_${this.nextId++}`;
  
      /* write to buffers */
      this.id[idx]        = id;
      this.type[idx]      = t;
  
      this.x[idx]         = x;
      this.y[idx]         = y;
      this.vx[idx]        = 0;
      this.vy[idx]        = 0;
      this.ax[idx]        = 0;
      this.ay[idx]        = 0;
  
      this.health[idx]    = overrides.health    ?? cfg.health;
      this.maxHealth[idx] = cfg.health;
      this.armor[idx]     = overrides.armor     ?? cfg.armor;
      this.damage[idx]    = overrides.damage    ?? cfg.damage;
      this.range[idx]     = overrides.range     ?? cfg.range;
  
      this.mass[idx]      = overrides.mass      ?? cfg.mass;
      this.morale[idx]    = 100;                // fresh morale
      this.stability[idx] = overrides.stability ?? cfg.stability;
      this.state[idx]     = 0;                  // idle
  
      this.idToIndex.set(id, idx);
      return id;
    }
  
    /**
     * Remove soldier by index (swap‑and‑pop).
     */
    _removeAt(index) {
      const last = this.count - 1;
      const removedId = this.id[index];
  
      // overwrite with last
      if (index !== last) {
        for (const key of ['id','type','x','y','vx','vy','ax','ay','health','maxHealth',
                           'armor','damage','range','mass','morale','stability','state']) {
          this[key][index] = this[key][last];
        }
        this.idToIndex.set(this.id[index], index);
      }
  
      this.idToIndex.delete(removedId);
      this.count--;
    }
  
    /**
     * Remove soldier by ID.
     */
    removeById(id) {
      const idx = this.idToIndex.get(id);
      if (idx !== undefined) this._removeAt(idx);
    }
  
    /* ======================================================================
       UPDATE LOOP
       ====================================================================== */
  
    /**
     * Advance physics & simple AI.
     * @param {number} dt – delta time (seconds)
     * @param {object} [params] – { ordersById, terrainFn }
     */
    update(dt, params = {}) {
      const g = 0; // no gravity in top‑down
  
      for (let i = 0; i < this.count; i++) {
        /* 1. process orders → acceleration */
        // TODO: read params.ordersById[this.id[i]] to set ax/ay or state
  
        /* 2. basic kinematics */
        this.vx[i] += this.ax[i] * dt;
        this.vy[i] += this.ay[i] * dt;
  
        const speedCap = this.SOLDIER_TYPES[this.type[i]].speed;
        const vMag = Math.hypot(this.vx[i], this.vy[i]);
        if (vMag > speedCap) {
          const s = speedCap / vMag;
          this.vx[i] *= s;
          this.vy[i] *= s;
        }
  
        this.x[i] += this.vx[i] * dt;
        this.y[i] += this.vy[i] * dt;
  
        /* 3. morale regeneration / decay */
        if (this.state[i] !== 3 /*rout*/) {
          this.morale[i] = Math.min(100, this.morale[i] + 5 * dt);
        }
      }
    }
  
    /* ======================================================================
       COMBAT HELPERS
       ====================================================================== */
  
    /**
     * Apply damage taking armor into account.
     * Also knocks down morale proportionally.
     */
    applyDamage(idx, rawDmg, momentum = 0) {
      const effective = Math.max(1, rawDmg - this.armor[idx] * 0.4);
      this.health[idx] -= effective;
  
      // morale shock: proportional to % health lost + momentum factor
      const shock = (effective / this.maxHealth[idx]) * 100 + momentum * 0.05;
      this.morale[idx] = Math.max(0, this.morale[idx] - shock);
  
      if (this.health[idx] <= 0 || this.morale[idx] === 0) {
        // soldier is killed or routs
        this._removeAt(idx);
        return true;
      }
      return false;
    }
  
    /* ======================================================================
       DATA FOR NETWORK / RENDER
       ====================================================================== */
  
    /**
     * Return light‑weight array for transmission.
     */
    getSoldiersData() {
      const arr = [];
      for (let i = 0; i < this.count; i++) {
        arr.push({
          id:        this.id[i],
          type:      this.type[i],
          x:         this.x[i],
          y:         this.y[i],
          vx:        this.vx[i],
          vy:        this.vy[i],
          health:    this.health[i],
          morale:    this.morale[i],
          state:     this.state[i],
        });
      }
      return arr;
    }
  
    findIndexById(id) {
      const idx = this.idToIndex.get(id);
      return idx !== undefined ? idx : -1;
    }
  
    getActiveCount() { return this.count; }
  
    cleanup() {
      this.count = 0;
      this.idToIndex.clear();
    }
  }
  