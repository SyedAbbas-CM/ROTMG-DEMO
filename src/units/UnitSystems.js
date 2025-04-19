// server/src/units/UnitSystems.js
import { UnitTypes } from './UnitTypes.js';
import SpatialGrid   from '../../shared/spatialGrid.js';

export default class UnitSystems {
  constructor(unitManager, mapManager) {
    this.u   = unitManager;
    this.map = mapManager;
    this.grid = new SpatialGrid(32, 2048, 2048);
  }

  /** called once per server tick */
  update(dt) {
    const {u} = this;
    this.grid.clear();

    // pass 1: physics & command handling
    for (let i=0;i<u.count;i++) {
      const def = UnitTypes[UnitTypes.__keys[u.typeIdx[i]]];

      /* command → acceleration */
      if (u.cmdKind[i] === 1) {            // move
        const dx = u.cmdTX[i]-u.x[i];
        const dy = u.cmdTY[i]-u.y[i];
        const d2 = dx*dx+dy*dy;
        if (d2 > 1) {
          const inv = 1/Math.sqrt(d2);
          u.vx[i] += dx*inv*def.accel*dt;
          u.vy[i] += dy*inv*def.accel*dt;
        } else { u.cmdKind[i]=0; }         // reached
      }

      /* integrate */
      u.x[i] += u.vx[i]*dt;
      u.y[i] += u.vy[i]*dt;

      /* clamp */
      const s = Math.hypot(u.vx[i],u.vy[i]);
      if (s>def.speed) {
        const k = def.speed/s;
        u.vx[i]*=k; u.vy[i]*=k;
      }

      /* wall bounce */
      if (this.map.isWallOrOutOfBounds(u.x[i], u.y[i])) {
        u.x[i]-=u.vx[i]*dt;
        u.y[i]-=u.vy[i]*dt;
        u.vx[i]=u.vy[i]=0;
      }

      /* drop into grid */
      this.grid.insertEnemy(i, u.x[i], u.y[i], 16, 16);

      /* cooldown */
      if (u.cool[i]>0) u.cool[i]-=dt;
    }

    // pass 2: cheap melee combat (O(n²) local via grid)
    const kills = [];
    this.grid.getPotentialCollisionPairs().forEach(([ai,bi])=>{
      if (this.u.owner[ai] === this.u.owner[bi]) return;  // friendly
      const now = Date.now();

      // simple exchange
      this._meleeHit(ai, bi, kills);
      this._meleeHit(bi, ai, kills);
    });

    // remove dead
    kills.forEach(idx=>this.u.removeIndex(idx));
  }

  _meleeHit(a,b,kills){
    const atkDef = UnitTypes[UnitTypes.__keys[this.u.typeIdx[a]]];
    if (this.u.cool[a]>0) return;
    this.u.cool[a] = 0.8;

    this.u.hp[b] -= 10;                   // TODO: use atkDef.damage
    if (this.u.hp[b]<=0 && !kills.includes(b)) kills.push(b);
  }
}
