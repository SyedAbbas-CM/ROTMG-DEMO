// server/src/units/UnitSystems.js
import { UnitTypes } from './UnitTypes.js';
import SpatialGrid   from '../../public/src/shared/spatialGrid.js';
import { TILE_PROPERTIES } from '../../public/src/constants/constants.js';

export default class UnitSystems {
  constructor(unitManager, mapManager) {
    this.u   = unitManager;
    this.map = mapManager;
    this.grid = new SpatialGrid(32, 2048, 2048);
    
    // Advanced AI state management
    this.formations = new Map(); // formationId -> formation data
    this.squads = new Map(); // squadId -> squad data
    this.threatMap = new Map(); // position -> threat level
    this.lastThreatUpdate = 0;
    this.tacticalDecisions = new Map(); // unitId -> tactical state
    
    // Performance optimization
    this.updateCounter = 0;
    this.aiUpdateFrequency = 10; // Update AI every 10 ticks for performance
  }

  /** called once per server tick */
  update(dt) {
    const {u} = this;
    this.grid.clear();
    this.updateCounter++;
    
    // Update threat map periodically
    if (Date.now() - this.lastThreatUpdate > 1000) {
      this.updateThreatMap();
      this.lastThreatUpdate = Date.now();
    }

    // pass 1: physics & advanced command handling
    for (let i=0;i<u.count;i++) {
      const def = UnitTypes[UnitTypes.__keys[u.typeIdx[i]]];
      
      // Advanced AI decision making (throttled for performance)
      if (this.updateCounter % this.aiUpdateFrequency === i % this.aiUpdateFrequency) {
        this.processAdvancedAI(i, dt);
      }

      /* command → acceleration */
      this.processCommand(i, dt, def);

      /* integrate with tile-based movement cost (water slow, etc.) */
      let movementMultiplier = 1.0;
      const tileX = Math.floor(u.x[i]);
      const tileY = Math.floor(u.y[i]);
      const tile = this.map.getTile(tileX, tileY);

      if (tile && TILE_PROPERTIES[tile.type]) {
        const movementCost = TILE_PROPERTIES[tile.type].movementCost || 1.0;
        movementMultiplier = 1.0 / movementCost; // Higher cost = slower (0.5x for water)
      }

      u.x[i] += u.vx[i] * dt * movementMultiplier;
      u.y[i] += u.vy[i] * dt * movementMultiplier;

      /* clamp velocity */
      const s = Math.hypot(u.vx[i],u.vy[i]);
      if (s>def.speed) {
        const k = def.speed/s;
        u.vx[i]*=k; u.vy[i]*=k;
      }

      /* wall collision with pathfinding */
      if (this.map.isWallOrOutOfBounds(u.x[i], u.y[i])) {
        u.x[i]-=u.vx[i]*dt;
        u.y[i]-=u.vy[i]*dt;
        u.vx[i]=u.vy[i]=0;
        
        // Try to find alternate path
        this.findAlternatePath(i);
      }

      /* drop into grid */
      this.grid.insertEnemy(i, u.x[i], u.y[i], 16, 16);

      /* cooldown */
      if (u.cool[i]>0) u.cool[i]-=dt;
      
      // Update morale and stability
      this.updateMoraleAndStability(i, dt);
    }

    // pass 2: advanced combat system
    const kills = [];
    this.grid.getPotentialCollisionPairs().forEach(([ai,bi])=>{
      if (this.u.owner[ai] === this.u.owner[bi]) return;  // friendly
      
      // Check unit types for combat interactions
      this.processCombatInteraction(ai, bi, kills, dt);
    });

    // pass 3: formation maintenance
    this.maintainFormations(dt);
    
    // pass 4: squad coordination
    this.coordinateSquads(dt);

    // remove dead units
    kills.forEach(idx=>{
      this.handleUnitDeath(idx);
      this.u.removeIndex(idx);
    });
  }

  processCommand(unitIndex, dt, def) {
    const u = this.u;
    
    switch (u.cmdKind[unitIndex]) {
      case 1: // move
        this.processMove(unitIndex, dt, def);
        break;
      case 2: // attack-move
        this.processAttackMove(unitIndex, dt, def);
        break;
      case 3: // guard
        this.processGuard(unitIndex, dt, def);
        break;
      case 4: // patrol
        this.processPatrol(unitIndex, dt, def);
        break;
      case 5: // formation move
        this.processFormationMove(unitIndex, dt, def);
        break;
      default:
        this.processIdle(unitIndex, dt, def);
    }
  }

  processMove(unitIndex, dt, def) {
    const u = this.u;
    const dx = u.cmdTX[unitIndex] - u.x[unitIndex];
    const dy = u.cmdTY[unitIndex] - u.y[unitIndex];
    const d2 = dx*dx + dy*dy;
    
    if (d2 > 1) {
      const inv = 1/Math.sqrt(d2);
      u.vx[unitIndex] += dx*inv*def.accel*dt;
      u.vy[unitIndex] += dy*inv*def.accel*dt;
    } else {
      u.cmdKind[unitIndex] = 0; // reached destination
    }
  }

  processAttackMove(unitIndex, dt, def) {
    const u = this.u;
    
    // First, look for enemies in range
    const enemy = this.findNearestEnemy(unitIndex, def.attackRange || 50);
    
    if (enemy !== -1) {
      // Attack the enemy
      this.engageEnemy(unitIndex, enemy, dt, def);
    } else {
      // Move toward destination
      this.processMove(unitIndex, dt, def);
    }
  }

  processGuard(unitIndex, dt, def) {
    const u = this.u;
    
    // Look for enemies near guard position
    const guardX = u.cmdTX[unitIndex];
    const guardY = u.cmdTY[unitIndex];
    const enemy = this.findEnemyNearPosition(guardX, guardY, def.attackRange || 50);
    
    if (enemy !== -1) {
      this.engageEnemy(unitIndex, enemy, dt, def);
    } else {
      // Return to guard position if too far
      const dx = guardX - u.x[unitIndex];
      const dy = guardY - u.y[unitIndex];
      const d2 = dx*dx + dy*dy;
      
      if (d2 > 25) { // 5 tiles away
        const inv = 1/Math.sqrt(d2);
        u.vx[unitIndex] += dx*inv*def.accel*dt*0.5; // Slower return
        u.vy[unitIndex] += dy*inv*def.accel*dt*0.5;
      }
    }
  }

  processPatrol(unitIndex, dt, def) {
    const u = this.u;
    
    // Get patrol state
    if (!this.tacticalDecisions.has(unitIndex)) {
      this.tacticalDecisions.set(unitIndex, {
        patrolWaypoints: [],
        currentWaypoint: 0,
        patrolDirection: 1
      });
    }
    
    const state = this.tacticalDecisions.get(unitIndex);
    
    // Look for enemies while patrolling
    const enemy = this.findNearestEnemy(unitIndex, def.attackRange || 50);
    if (enemy !== -1) {
      this.engageEnemy(unitIndex, enemy, dt, def);
      return;
    }
    
    // Continue patrol
    if (state.patrolWaypoints.length > 0) {
      const waypoint = state.patrolWaypoints[state.currentWaypoint];
      const dx = waypoint.x - u.x[unitIndex];
      const dy = waypoint.y - u.y[unitIndex];
      const d2 = dx*dx + dy*dy;
      
      if (d2 > 1) {
        const inv = 1/Math.sqrt(d2);
        u.vx[unitIndex] += dx*inv*def.accel*dt;
        u.vy[unitIndex] += dy*inv*def.accel*dt;
      } else {
        // Reached waypoint, move to next
        state.currentWaypoint += state.patrolDirection;
        if (state.currentWaypoint >= state.patrolWaypoints.length) {
          state.currentWaypoint = state.patrolWaypoints.length - 1;
          state.patrolDirection = -1;
        } else if (state.currentWaypoint < 0) {
          state.currentWaypoint = 0;
          state.patrolDirection = 1;
        }
      }
    }
  }

  processFormationMove(unitIndex, dt, def) {
    const u = this.u;
    const formationId = u.formation[unitIndex];
    
    if (!formationId || !this.formations.has(formationId)) {
      // Fall back to normal move
      this.processMove(unitIndex, dt, def);
      return;
    }
    
    const formation = this.formations.get(formationId);
    const formationPos = this.calculateFormationPosition(unitIndex, formation);
    
    // Move toward formation position
    const dx = formationPos.x - u.x[unitIndex];
    const dy = formationPos.y - u.y[unitIndex];
    const d2 = dx*dx + dy*dy;
    
    if (d2 > 1) {
      const inv = 1/Math.sqrt(d2);
      u.vx[unitIndex] += dx*inv*def.accel*dt;
      u.vy[unitIndex] += dy*inv*def.accel*dt;
    }
  }

  processIdle(unitIndex, dt, def) {
    const u = this.u;
    
    // Look for nearby enemies to engage
    const enemy = this.findNearestEnemy(unitIndex, def.attackRange || 50);
    if (enemy !== -1) {
      this.engageEnemy(unitIndex, enemy, dt, def);
      return;
    }
    
    // Apply friction when idle
    u.vx[unitIndex] *= 0.9;
    u.vy[unitIndex] *= 0.9;
  }

  processAdvancedAI(unitIndex, dt) {
    const u = this.u;
    const def = UnitTypes[UnitTypes.__keys[u.typeIdx[unitIndex]]];
    
    // Get or create tactical state
    if (!this.tacticalDecisions.has(unitIndex)) {
      this.tacticalDecisions.set(unitIndex, {
        targetEnemy: -1,
        lastTargetUpdate: 0,
        flankingManeuver: false,
        retreating: false,
        supportingAlly: -1,
        tacticalRole: this.determineTacticalRole(unitIndex, def)
      });
    }
    
    const state = this.tacticalDecisions.get(unitIndex);
    const now = Date.now();
    
    // Update target periodically
    if (now - state.lastTargetUpdate > 500) {
      state.targetEnemy = this.selectOptimalTarget(unitIndex, def);
      state.lastTargetUpdate = now;
    }
    
    // Execute tactical behavior based on unit type and situation
    this.executeTacticalBehavior(unitIndex, state, def, dt);
  }

  determineTacticalRole(unitIndex, def) {
    // Determine role based on unit type
    switch (def.type) {
      case 'Infantry': return 'frontline';
      case 'Heavy Infantry': return 'tank';
      case 'Light Cavalry': return 'flanker';
      case 'Heavy Cavalry': return 'charger';
      case 'Archer': return 'ranged';
      case 'Crossbow': return 'ranged';
      default: return 'support';
    }
  }

  selectOptimalTarget(unitIndex, def) {
    const u = this.u;
    const enemies = this.findAllEnemiesInRange(unitIndex, def.attackRange * 1.5);
    
    if (enemies.length === 0) return -1;
    
    let bestTarget = -1;
    let bestScore = -1;
    
    for (const enemyIndex of enemies) {
      const score = this.calculateTargetPriority(unitIndex, enemyIndex, def);
      if (score > bestScore) {
        bestScore = score;
        bestTarget = enemyIndex;
      }
    }
    
    return bestTarget;
  }

  calculateTargetPriority(unitIndex, enemyIndex, def) {
    const u = this.u;
    const enemyDef = UnitTypes[UnitTypes.__keys[u.typeIdx[enemyIndex]]];
    
    let score = 0;
    
    // Distance factor (closer = higher priority)
    const dx = u.x[enemyIndex] - u.x[unitIndex];
    const dy = u.y[enemyIndex] - u.y[unitIndex];
    const distance = Math.sqrt(dx*dx + dy*dy);
    score += Math.max(0, 100 - distance);
    
    // Health factor (weaker enemies = higher priority)
    const healthPercent = u.hp[enemyIndex] / (enemyDef.health || 100);
    score += (1 - healthPercent) * 50;
    
    // Type effectiveness
    score += this.getTypeEffectiveness(def.type, enemyDef.type) * 30;
    
    // Threat level
    score += this.getUnitThreatLevel(enemyIndex) * 20;
    
    return score;
  }

  getTypeEffectiveness(attackerType, defenderType) {
    // Rock-paper-scissors style effectiveness
    const effectiveness = {
      'Infantry': { 'Archer': 1.5, 'Crossbow': 1.5, 'Light Cavalry': 0.7 },
      'Heavy Infantry': { 'Heavy Cavalry': 1.3, 'Infantry': 1.2 },
      'Light Cavalry': { 'Archer': 1.8, 'Crossbow': 1.8, 'Infantry': 0.8 },
      'Heavy Cavalry': { 'Archer': 2.0, 'Heavy Infantry': 0.7 },
      'Archer': { 'Heavy Cavalry': 0.5, 'Light Cavalry': 0.6, 'Infantry': 1.2 },
      'Crossbow': { 'Heavy Infantry': 1.4, 'Heavy Cavalry': 0.7 }
    };
    
    return effectiveness[attackerType]?.[defenderType] || 1.0;
  }

  executeTacticalBehavior(unitIndex, state, def, dt) {
    switch (state.tacticalRole) {
      case 'frontline':
        this.executeFrontlineBehavior(unitIndex, state, def, dt);
        break;
      case 'tank':
        this.executeTankBehavior(unitIndex, state, def, dt);
        break;
      case 'flanker':
        this.executeFlankingBehavior(unitIndex, state, def, dt);
        break;
      case 'charger':
        this.executeChargeBehavior(unitIndex, state, def, dt);
        break;
      case 'ranged':
        this.executeRangedBehavior(unitIndex, state, def, dt);
        break;
      default:
        this.executeSupportBehavior(unitIndex, state, def, dt);
    }
  }

  executeFrontlineBehavior(unitIndex, state, def, dt) {
    // Engage nearest enemy, maintain formation
    if (state.targetEnemy !== -1) {
      this.engageEnemy(unitIndex, state.targetEnemy, dt, def);
    }
  }

  executeTankBehavior(unitIndex, state, def, dt) {
    // Protect weaker allies, absorb damage
    const weakAlly = this.findNearestWoundedAlly(unitIndex);
    if (weakAlly !== -1) {
      this.protectAlly(unitIndex, weakAlly, dt, def);
    } else if (state.targetEnemy !== -1) {
      this.engageEnemy(unitIndex, state.targetEnemy, dt, def);
    }
  }

  executeFlankingBehavior(unitIndex, state, def, dt) {
    // Try to attack enemies from the side or rear
    if (state.targetEnemy !== -1) {
      const flankPosition = this.calculateFlankingPosition(unitIndex, state.targetEnemy);
      this.moveToPosition(unitIndex, flankPosition.x, flankPosition.y, dt, def);
    }
  }

  executeChargeBehavior(unitIndex, state, def, dt) {
    // Build up momentum and charge at enemies
    if (state.targetEnemy !== -1) {
      const distance = this.getDistanceToUnit(unitIndex, state.targetEnemy);
      if (distance > 30) {
        // Build momentum
        this.chargeAtEnemy(unitIndex, state.targetEnemy, dt, def);
      } else {
        // Close combat
        this.engageEnemy(unitIndex, state.targetEnemy, dt, def);
      }
    }
  }

  executeRangedBehavior(unitIndex, state, def, dt) {
    // Maintain distance and provide ranged support
    if (state.targetEnemy !== -1) {
      const distance = this.getDistanceToUnit(unitIndex, state.targetEnemy);
      const optimalRange = (def.attackRange || 50) * 0.8;
      
      if (distance < optimalRange * 0.6) {
        // Too close, retreat
        this.retreatFromEnemy(unitIndex, state.targetEnemy, dt, def);
      } else if (distance > optimalRange * 1.2) {
        // Too far, advance
        this.advanceTowardsEnemy(unitIndex, state.targetEnemy, dt, def);
      } else {
        // Good range, attack
        this.rangedAttack(unitIndex, state.targetEnemy, dt, def);
      }
    }
  }

  // Helper methods for tactical behaviors
  findNearestEnemy(unitIndex, range) {
    const u = this.u;
    const enemies = this.findAllEnemiesInRange(unitIndex, range);
    
    if (enemies.length === 0) return -1;
    
    let nearest = -1;
    let nearestDistance = Infinity;
    
    for (const enemyIndex of enemies) {
      const distance = this.getDistanceToUnit(unitIndex, enemyIndex);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = enemyIndex;
      }
    }
    
    return nearest;
  }

  findAllEnemiesInRange(unitIndex, range) {
    const u = this.u;
    const enemies = [];
    
    for (let i = 0; i < u.count; i++) {
      if (i === unitIndex) continue;
      if (u.owner[i] === u.owner[unitIndex]) continue; // Same team
      if (u.hp[i] <= 0) continue; // Dead
      
      const distance = this.getDistanceToUnit(unitIndex, i);
      if (distance <= range) {
        enemies.push(i);
      }
    }
    
    return enemies;
  }

  getDistanceToUnit(unitA, unitB) {
    const u = this.u;
    const dx = u.x[unitB] - u.x[unitA];
    const dy = u.y[unitB] - u.y[unitA];
    return Math.sqrt(dx*dx + dy*dy);
  }

  engageEnemy(unitIndex, enemyIndex, dt, def) {
    // Implementation for engaging enemy
    const distance = this.getDistanceToUnit(unitIndex, enemyIndex);
    const attackRange = def.attackRange || 20;
    
    if (distance <= attackRange) {
      // In range, attack
      this.performAttack(unitIndex, enemyIndex, dt, def);
    } else {
      // Move closer
      this.moveTowardsUnit(unitIndex, enemyIndex, dt, def);
    }
  }

  performAttack(unitIndex, enemyIndex, dt, def) {
    const u = this.u;
    
    if (u.cool[unitIndex] > 0) return; // On cooldown
    
    // Calculate damage
    const damage = def.damage || 10;
    const effectiveness = this.getTypeEffectiveness(def.type, UnitTypes[UnitTypes.__keys[u.typeIdx[enemyIndex]]].type);
    const finalDamage = damage * effectiveness;
    
    // Apply damage
    u.hp[enemyIndex] -= finalDamage;
    u.cool[unitIndex] = def.attackCooldown || 1.0;
    
    // Reduce morale of attacked unit
    if (u.morale) {
      u.morale[enemyIndex] = Math.max(0, u.morale[enemyIndex] - 5);
    }
  }

  moveTowardsUnit(unitIndex, targetIndex, dt, def) {
    const u = this.u;
    const dx = u.x[targetIndex] - u.x[unitIndex];
    const dy = u.y[targetIndex] - u.y[unitIndex];
    const distance = Math.sqrt(dx*dx + dy*dy);
    
    if (distance > 0) {
      const inv = 1/distance;
      u.vx[unitIndex] += dx*inv*def.accel*dt;
      u.vy[unitIndex] += dy*inv*def.accel*dt;
    }
  }

  updateThreatMap() {
    // Implementation for threat assessment
    // This would analyze enemy positions and update threat levels
  }

  maintainFormations(dt) {
    // Implementation for formation maintenance
    // This would ensure units stay in their assigned formations
  }

  coordinateSquads(dt) {
    // Implementation for squad coordination
    // This would handle inter-unit communication and coordination
  }

  updateMoraleAndStability(unitIndex, dt) {
    const u = this.u;
    
    // Initialize morale and stability if not present
    if (!u.morale) u.morale = new Float32Array(u.maxCount);
    if (!u.stability) u.stability = new Float32Array(u.maxCount);
    
    // Gradually restore morale when not in combat
    const nearestEnemy = this.findNearestEnemy(unitIndex, 50);
    if (nearestEnemy === -1) {
      u.morale[unitIndex] = Math.min(100, u.morale[unitIndex] + 10 * dt);
    }
    
    // Morale affects combat effectiveness
    const moraleEffect = u.morale[unitIndex] / 100;
    // This could be used to modify damage, speed, etc.
  }

  handleUnitDeath(unitIndex) {
    // Remove from formations and squads
    const u = this.u;
    const formationId = u.formation?.[unitIndex];
    const squadId = u.squad?.[unitIndex];
    
    if (formationId && this.formations.has(formationId)) {
      const formation = this.formations.get(formationId);
      formation.units = formation.units.filter(id => id !== unitIndex);
    }
    
    if (squadId && this.squads.has(squadId)) {
      const squad = this.squads.get(squadId);
      squad.units = squad.units.filter(id => id !== unitIndex);
    }
    
    // Remove tactical decisions
    this.tacticalDecisions.delete(unitIndex);
    
    // Affect morale of nearby allies
    const allies = this.findAlliesInRange(unitIndex, 30);
    for (const allyIndex of allies) {
      if (u.morale) {
        u.morale[allyIndex] = Math.max(0, u.morale[allyIndex] - 15);
      }
    }
  }

  findAlliesInRange(unitIndex, range) {
    const u = this.u;
    const allies = [];
    
    for (let i = 0; i < u.count; i++) {
      if (i === unitIndex) continue;
      if (u.owner[i] !== u.owner[unitIndex]) continue; // Different team
      if (u.hp[i] <= 0) continue; // Dead
      
      const distance = this.getDistanceToUnit(unitIndex, i);
      if (distance <= range) {
        allies.push(i);
      }
    }
    
    return allies;
  }

  // Additional helper methods would go here...
  findAlternatePath(unitIndex) {
    // Simple pathfinding - try 45-degree angles
    const u = this.u;
    const angles = [-Math.PI/4, Math.PI/4, -Math.PI/2, Math.PI/2];
    
    for (const angle of angles) {
      const testX = u.x[unitIndex] + Math.cos(angle) * 10;
      const testY = u.y[unitIndex] + Math.sin(angle) * 10;
      
      if (!this.map.isWallOrOutOfBounds(testX, testY)) {
        u.vx[unitIndex] = Math.cos(angle) * 5;
        u.vy[unitIndex] = Math.sin(angle) * 5;
        break;
      }
    }
  }

  processCombatInteraction(ai, bi, kills, dt) {
    if (this.u.owner[ai] === this.u.owner[bi]) return;  // friendly
    
    // More sophisticated combat
    this._meleeHit(ai, bi, kills);
    this._meleeHit(bi, ai, kills);
  }

  _meleeHit(a,b,kills){
    const atkDef = UnitTypes[UnitTypes.__keys[this.u.typeIdx[a]]];
    if (this.u.cool[a]>0) return;
    this.u.cool[a] = atkDef.attackCooldown || 0.8;

    const damage = atkDef.damage || 10;
    const defDef = UnitTypes[UnitTypes.__keys[this.u.typeIdx[b]]];
    const effectiveness = this.getTypeEffectiveness(atkDef.type, defDef.type);
    const finalDamage = damage * effectiveness;
    
    this.u.hp[b] -= finalDamage;
    if (this.u.hp[b]<=0 && !kills.includes(b)) kills.push(b);
  }
}
