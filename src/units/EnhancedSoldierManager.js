// =============================================
// FILE: src/units/EnhancedSoldierManager.js
// =============================================

/**
 * EnhancedSoldierManager - Updated version with comprehensive unit definitions
 * Uses the new UnitDefinitions system with sprites, detailed stats, and tactical data
 */

import { UnitDefinitions, getUnitDefinition } from './UnitDefinitions.js';

export default class EnhancedSoldierManager {
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
  
      // core stats (from UnitDefinitions)
      this.health      = new Float32Array(maxSoldiers);
      this.maxHealth   = new Float32Array(maxSoldiers);
      this.armor       = new Float32Array(maxSoldiers);
      this.damage      = new Float32Array(maxSoldiers);
      this.range       = new Float32Array(maxSoldiers);
      this.attackSpeed = new Float32Array(maxSoldiers);
      this.accuracy    = new Float32Array(maxSoldiers);
  
      // physics / morale
      this.mass        = new Float32Array(maxSoldiers);
      this.morale      = new Float32Array(maxSoldiers); // 0‑100
      this.stability   = new Float32Array(maxSoldiers);
      this.state       = new Uint8Array(maxSoldiers);   // 0=idle,1=move,2=attack,3=rout,…
      
      // Enhanced properties
      this.team        = new Array(maxSoldiers);        // Team/faction identifier
      this.experience  = new Float32Array(maxSoldiers); // XP for leveling
      this.level       = new Uint8Array(maxSoldiers);   // Unit level
      this.lastAttack  = new Float32Array(maxSoldiers); // Attack cooldown tracking
  
      // book‑keeping map (id → index)
      this.idToIndex   = new Map();
      
      // Unit definitions reference
      this.unitDefs = UnitDefinitions;
      
      console.log(`[EnhancedSoldierManager] Initialized with ${Object.keys(UnitDefinitions).length} unit types`);
    }
    
    /**
     * Get unit definition for a type
     */
    getUnitDefinition(type) {
      return getUnitDefinition(type);
    }
    
    /**
     * Get all available unit types
     */
    getAvailableUnitTypes() {
      return Object.values(UnitDefinitions);
    }
  
    /**
     * Spawn a new soldier with enhanced properties
     */
    spawn(type, x, y, overrides = {}) {
      if (this.count >= this.max) {
        console.warn('[EnhancedSoldierManager] capacity reached');
        return null;
      }
  
      const unitDef = getUnitDefinition(type);
      if (!unitDef) {
        console.warn(`[EnhancedSoldierManager] Unknown unit type: ${type}`);
        return null;
      }
  
      const idx = this.count++;
      const id  = overrides.id || `soldier_${this.nextId++}`;
  
      /* write to buffers using UnitDefinition stats */
      this.id[idx]        = id;
      this.type[idx]      = type;
  
      this.x[idx]         = x;
      this.y[idx]         = y;
      this.vx[idx]        = 0;
      this.vy[idx]        = 0;
      this.ax[idx]        = 0;
      this.ay[idx]        = 0;
  
      // Use stats from unit definition
      this.health[idx]    = overrides.health    ?? unitDef.stats.maxHealth;
      this.maxHealth[idx] = unitDef.stats.maxHealth;
      this.armor[idx]     = overrides.armor     ?? unitDef.stats.armor;
      this.damage[idx]    = overrides.damage    ?? unitDef.combat.attackDamage;
      this.range[idx]     = overrides.range     ?? unitDef.combat.attackRange;
      this.attackSpeed[idx] = 1.0 / unitDef.combat.attackCooldown;
      this.accuracy[idx]  = unitDef.combat.accuracy;
  
      this.mass[idx]      = overrides.mass      ?? unitDef.stats.mass;
      this.morale[idx]    = unitDef.morale.baseValue;
      this.stability[idx] = overrides.stability ?? unitDef.morale.stability;
      this.state[idx]     = 0;                  // idle
      
      // Enhanced properties
      this.team[idx]      = overrides.team || 'neutral';
      this.experience[idx] = overrides.experience || 0;
      this.level[idx]     = overrides.level || 1;
      this.lastAttack[idx] = 0;
  
      this.idToIndex.set(id, idx);
      
      console.log(`[EnhancedSoldierManager] Spawned ${unitDef.displayName} at (${x}, ${y}) for team ${this.team[idx]}`);
      return id;
    }
    
    /**
     * Get comprehensive soldier data for networking
     */
    getSoldiersData() {
      const arr = [];
      for (let i = 0; i < this.count; i++) {
        const unitDef = getUnitDefinition(this.type[i]);
        arr.push({
          id:        this.id[i],
          type:      this.type[i],
          typeName:  unitDef?.name || `Type${this.type[i]}`,
          displayName: unitDef?.displayName || `Unit ${this.type[i]}`,
          category:  unitDef?.category || 'unknown',
          x:         this.x[i],
          y:         this.y[i],
          vx:        this.vx[i],
          vy:        this.vy[i],
          health:    this.health[i],
          maxHealth: this.maxHealth[i],
          morale:    this.morale[i],
          state:     this.state[i],
          team:      this.team[i],
          level:     this.level[i],
          // Sprite information for client rendering
          sprite:    unitDef?.sprite || null
        });
      }
      return arr;
    }
    
    /**
     * Apply damage to a unit with enhanced combat mechanics
     */
    applyDamage(idx, rawDmg, attacker = null, momentum = 0) {
      if (idx < 0 || idx >= this.count) return false;
      
      const unitDef = getUnitDefinition(this.type[idx]);
      const armorReduction = this.armor[idx] * 0.1; // 10% damage reduction per armor point
      const effective = Math.max(1, rawDmg - armorReduction);
      
      this.health[idx] -= effective;
  
      // Enhanced morale system
      const healthLossRatio = effective / this.maxHealth[idx];
      const moraleShock = (healthLossRatio * 50) + (momentum * 0.1);
      this.morale[idx] = Math.max(0, this.morale[idx] - moraleShock);
      
      // Experience gain for attacker
      if (attacker && this.health[idx] <= 0) {
        const attackerIdx = this.findIndexById(attacker);
        if (attackerIdx !== -1) {
          this.experience[attackerIdx] += 10;
          this.checkLevelUp(attackerIdx);
        }
      }
  
      if (this.health[idx] <= 0) {
        console.log(`[EnhancedSoldierManager] ${unitDef?.displayName} killed by ${attacker || 'unknown'}`);
        this._removeAt(idx);
        return true;
      }
      
      return false;
    }
    
    /**
     * Check if unit should level up
     */
    checkLevelUp(idx) {
      const xpNeeded = this.level[idx] * 100; // Simple leveling system
      if (this.experience[idx] >= xpNeeded && this.level[idx] < 10) {
        this.level[idx]++;
        // Level up bonuses
        this.maxHealth[idx] *= 1.1;
        this.health[idx] = this.maxHealth[idx]; // Full heal on level up
        this.damage[idx] *= 1.05;
        console.log(`[EnhancedSoldierManager] Unit ${this.id[idx]} leveled up to ${this.level[idx]}`);
      }
    }
    
    /**
     * Update soldier with enhanced AI considerations
     */
    update(dt, params = {}) {
      for (let i = 0; i < this.count; i++) {
        const unitDef = getUnitDefinition(this.type[i]);
        
        /* 1. Update cooldowns */
        if (this.lastAttack[i] > 0) {
          this.lastAttack[i] -= dt;
        }
        
        /* 2. Morale effects */
        if (this.morale[i] < unitDef.morale.retreatThreshold) {
          this.state[i] = 3; // Retreat state
        }
        
        /* 3. Enhanced movement with unit-specific speeds */
        this.vx[i] += this.ax[i] * dt;
        this.vy[i] += this.ay[i] * dt;
  
        const speedCap = unitDef.stats.speed;
        const vMag = Math.hypot(this.vx[i], this.vy[i]);
        if (vMag > speedCap) {
          const s = speedCap / vMag;
          this.vx[i] *= s;
          this.vy[i] *= s;
        }
  
        this.x[i] += this.vx[i] * dt;
        this.y[i] += this.vy[i] * dt;
  
        /* 4. Gradual morale recovery */
        if (this.state[i] !== 3) {
          const recoveryRate = unitDef.morale.discipline * 10;
          this.morale[i] = Math.min(unitDef.morale.baseValue, this.morale[i] + recoveryRate * dt);
        }
      }
    }
    
    /**
     * Get units by team for tactical coordination
     */
    getUnitsByTeam(team) {
      const units = [];
      for (let i = 0; i < this.count; i++) {
        if (this.team[i] === team) {
          units.push({
            index: i,
            id: this.id[i],
            type: this.type[i],
            x: this.x[i],
            y: this.y[i],
            health: this.health[i],
            morale: this.morale[i],
            state: this.state[i]
          });
        }
      }
      return units;
    }
    
    /**
     * Get unit statistics for UI/debugging
     */
    getStatistics() {
      const stats = {
        total: this.count,
        byType: {},
        byTeam: {},
        byState: { idle: 0, moving: 0, attacking: 0, retreating: 0 },
        avgHealth: 0,
        avgMorale: 0
      };
      
      let totalHealth = 0;
      let totalMorale = 0;
      
      for (let i = 0; i < this.count; i++) {
        const unitDef = getUnitDefinition(this.type[i]);
        const typeName = unitDef?.displayName || `Type${this.type[i]}`;
        const team = this.team[i] || 'unknown';
        
        stats.byType[typeName] = (stats.byType[typeName] || 0) + 1;
        stats.byTeam[team] = (stats.byTeam[team] || 0) + 1;
        
        const stateNames = ['idle', 'moving', 'attacking', 'retreating'];
        const stateName = stateNames[this.state[i]] || 'idle';
        stats.byState[stateName]++;
        
        totalHealth += (this.health[i] / this.maxHealth[i]);
        totalMorale += this.morale[i];
      }
      
      if (this.count > 0) {
        stats.avgHealth = (totalHealth / this.count) * 100;
        stats.avgMorale = totalMorale / this.count;
      }
      
      return stats;
    }

    // Existing methods adapted for compatibility
    _removeAt(index) {
      const last = this.count - 1;
      const removedId = this.id[index];
  
      if (index !== last) {
        // Copy all arrays
        for (const key of ['id','type','x','y','vx','vy','ax','ay','health','maxHealth',
                           'armor','damage','range','attackSpeed','accuracy','mass','morale',
                           'stability','state','team','experience','level','lastAttack']) {
          if (this[key]) this[key][index] = this[key][last];
        }
        this.idToIndex.set(this.id[index], index);
      }
  
      this.idToIndex.delete(removedId);
      this.count--;
    }

    removeById(id) {
      const idx = this.idToIndex.get(id);
      if (idx !== undefined) this._removeAt(idx);
    }

    findIndexById(id) {
      const idx = this.idToIndex.get(id);
      return idx !== undefined ? idx : -1;
    }

    getActiveCount() { 
      return this.count; 
    }

    cleanup() {
      this.count = 0;
      this.idToIndex.clear();
    }
}