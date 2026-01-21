// server/src/units/UnitSystems.js
// Uses SoldierManager's SOLDIER_TYPES instead of separate UnitTypes
import SpatialGrid   from '../../public/src/shared/spatialGrid.js';
import { TILE_PROPERTIES } from '../../public/src/constants/constants.js';

export default class UnitSystems {
  constructor(unitManager, mapManager) {
    this.u   = unitManager;
    this.map = mapManager;
    this.grid = new SpatialGrid(32, 2048, 2048);

    // Cooldown tracking (SoldierManager doesn't have this)
    this.cooldowns = new Float32Array(unitManager.max);

    // Advanced AI state management
    this.formations = new Map(); // formationId -> formation data
    this.squads = new Map(); // squadId -> squad data
    this.threatMap = new Map(); // position -> threat level
    this.lastThreatUpdate = 0;
    this.tacticalDecisions = new Map(); // unitId -> tactical state

    // Command system (SoldierManager doesn't have this)
    this.cmdKind = new Uint8Array(unitManager.max); // 0=idle,1=move,2=attack-move,3=guard,4=patrol,5=formation
    this.cmdTX = new Float32Array(unitManager.max);
    this.cmdTY = new Float32Array(unitManager.max);

    // Performance optimization
    this.updateCounter = 0;
    this.aiUpdateFrequency = 10; // Update AI every 10 ticks for performance
  }

  // Get unit definition from SoldierManager's SOLDIER_TYPES
  getUnitDef(unitIndex) {
    if (!this.u || !this.u.SOLDIER_TYPES || !this.u.type) {
      return { speed: 50, damage: 10, range: 25, mass: 75, stability: 50 }; // fallback
    }
    const typeId = this.u.type[unitIndex] ?? 0;
    return this.u.SOLDIER_TYPES[typeId] || this.u.SOLDIER_TYPES[0];
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
      const def = this.getUnitDef(i);

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
      if (this.cooldowns[i]>0) this.cooldowns[i]-=dt;

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

    // remove dead units (iterate in reverse to avoid index shifting issues)
    kills.sort((a, b) => b - a).forEach(idx=>{
      this.handleUnitDeath(idx);
      this.u._removeAt(idx);
    });
  }

  processCommand(unitIndex, dt, def) {
    switch (this.cmdKind[unitIndex]) {
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
    const dx = this.cmdTX[unitIndex] - u.x[unitIndex];
    const dy = this.cmdTY[unitIndex] - u.y[unitIndex];
    const d2 = dx*dx + dy*dy;
    // Use speed as acceleration base (SoldierManager has speed, not accel)
    const accel = def.speed * 4;

    if (d2 > 1) {
      const inv = 1/Math.sqrt(d2);
      u.vx[unitIndex] += dx*inv*accel*dt;
      u.vy[unitIndex] += dy*inv*accel*dt;
    } else {
      this.cmdKind[unitIndex] = 0; // reached destination
    }
  }

  processAttackMove(unitIndex, dt, def) {
    // First, look for enemies in range
    const enemy = this.findNearestEnemy(unitIndex, def.range || 50);

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
    const accel = def.speed * 4;

    // Look for enemies near guard position
    const guardX = this.cmdTX[unitIndex];
    const guardY = this.cmdTY[unitIndex];
    const enemy = this.findEnemyNearPosition(guardX, guardY, def.range || 50);

    if (enemy !== -1) {
      this.engageEnemy(unitIndex, enemy, dt, def);
    } else {
      // Return to guard position if too far
      const dx = guardX - u.x[unitIndex];
      const dy = guardY - u.y[unitIndex];
      const d2 = dx*dx + dy*dy;

      if (d2 > 25) { // 5 tiles away
        const inv = 1/Math.sqrt(d2);
        u.vx[unitIndex] += dx*inv*accel*dt*0.5; // Slower return
        u.vy[unitIndex] += dy*inv*accel*dt*0.5;
      }
    }
  }

  processPatrol(unitIndex, dt, def) {
    const u = this.u;
    const accel = def.speed * 4;

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
    const enemy = this.findNearestEnemy(unitIndex, def.range || 50);
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
        u.vx[unitIndex] += dx*inv*accel*dt;
        u.vy[unitIndex] += dy*inv*accel*dt;
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
    const accel = def.speed * 4;
    // Get formation from tactical decisions if assigned
    const state = this.tacticalDecisions.get(unitIndex);
    const formationId = state?.formationId;

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
      u.vx[unitIndex] += dx*inv*accel*dt;
      u.vy[unitIndex] += dy*inv*accel*dt;
    }
  }

  processIdle(unitIndex, dt, def) {
    const u = this.u;

    // Look for nearby enemies to engage
    const enemy = this.findNearestEnemy(unitIndex, def.range || 50);
    if (enemy !== -1) {
      this.engageEnemy(unitIndex, enemy, dt, def);
      return;
    }

    // Apply friction when idle
    u.vx[unitIndex] *= 0.9;
    u.vy[unitIndex] *= 0.9;
  }

  processAdvancedAI(unitIndex, dt) {
    const def = this.getUnitDef(unitIndex);

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
    // Determine role based on unit category (from SOLDIER_TYPES)
    const category = def.category || 'infantry';
    const name = def.name || '';

    if (category === 'ranged') return 'ranged';
    if (category === 'cavalry') {
      return name.includes('Heavy') ? 'charger' : 'flanker';
    }
    if (category === 'infantry') {
      return name.includes('Heavy') ? 'tank' : 'frontline';
    }
    return 'support';
  }

  selectOptimalTarget(unitIndex, def) {
    const enemies = this.findAllEnemiesInRange(unitIndex, (def.range || 50) * 1.5);

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
    const enemyDef = this.getUnitDef(enemyIndex);

    let score = 0;

    // Distance factor (closer = higher priority)
    const dx = u.x[enemyIndex] - u.x[unitIndex];
    const dy = u.y[enemyIndex] - u.y[unitIndex];
    const distance = Math.sqrt(dx*dx + dy*dy);
    score += Math.max(0, 100 - distance);

    // Health factor (weaker enemies = higher priority)
    const healthPercent = u.health[enemyIndex] / (enemyDef.health || 100);
    score += (1 - healthPercent) * 50;

    // Type effectiveness
    score += this.getTypeEffectiveness(def.category, enemyDef.category) * 30;

    // Threat level
    score += this.getUnitThreatLevel(enemyIndex) * 20;

    return score;
  }

  getTypeEffectiveness(attackerCategory, defenderCategory) {
    // Rock-paper-scissors style effectiveness based on category
    const effectiveness = {
      'infantry': { 'ranged': 1.5, 'cavalry': 0.8 },
      'cavalry': { 'ranged': 1.8, 'infantry': 1.2 },
      'ranged': { 'cavalry': 0.6, 'infantry': 1.1 },
      'boss': { 'infantry': 1.5, 'cavalry': 1.5, 'ranged': 1.5 }
    };

    return effectiveness[attackerCategory]?.[defenderCategory] || 1.0;
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
    const attackRange = def.range || 20;

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

    if (this.cooldowns[unitIndex] > 0) return; // On cooldown

    // Calculate damage
    const damage = def.damage || 10;
    const enemyDef = this.getUnitDef(enemyIndex);
    const effectiveness = this.getTypeEffectiveness(def.category, enemyDef.category);
    const finalDamage = damage * effectiveness;

    // Apply damage
    u.health[enemyIndex] -= finalDamage;
    this.cooldowns[unitIndex] = 1.0; // 1 second cooldown

    // Reduce morale of attacked unit
    u.morale[enemyIndex] = Math.max(0, u.morale[enemyIndex] - 5);
  }

  moveTowardsUnit(unitIndex, targetIndex, dt, def) {
    const u = this.u;
    const accel = def.speed * 4;
    const dx = u.x[targetIndex] - u.x[unitIndex];
    const dy = u.y[targetIndex] - u.y[unitIndex];
    const distance = Math.sqrt(dx*dx + dy*dy);

    if (distance > 0) {
      const inv = 1/distance;
      u.vx[unitIndex] += dx*inv*accel*dt;
      u.vy[unitIndex] += dy*inv*accel*dt;
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

    // SoldierManager already has morale and stability arrays

    // Gradually restore morale when not in combat
    const nearestEnemy = this.findNearestEnemy(unitIndex, 50);
    if (nearestEnemy === -1) {
      u.morale[unitIndex] = Math.min(100, u.morale[unitIndex] + 10 * dt);
    }

    // Morale affects combat effectiveness
    // const moraleEffect = u.morale[unitIndex] / 100;
    // This could be used to modify damage, speed, etc.
  }

  handleUnitDeath(unitIndex) {
    // Remove from formations and squads (using tactical decisions map)
    const u = this.u;
    const state = this.tacticalDecisions.get(unitIndex);
    const formationId = state?.formationId;
    const squadId = state?.squadId;

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
      u.morale[allyIndex] = Math.max(0, u.morale[allyIndex] - 15);
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

  // Additional helper methods

  findEnemyNearPosition(x, y, range) {
    const u = this.u;
    let nearest = -1;
    let nearestDist = Infinity;

    for (let i = 0; i < u.count; i++) {
      const dx = u.x[i] - x;
      const dy = u.y[i] - y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < range && dist < nearestDist) {
        nearestDist = dist;
        nearest = i;
      }
    }
    return nearest;
  }

  calculateFormationPosition(unitIndex, formation) {
    // Simple formation: return target position for now
    return { x: this.cmdTX[unitIndex], y: this.cmdTY[unitIndex] };
  }

  getUnitThreatLevel(unitIndex) {
    const def = this.getUnitDef(unitIndex);
    // Higher damage and health = higher threat
    return ((def.damage || 10) + (def.health || 100) / 10) / 20;
  }

  findNearestWoundedAlly(unitIndex) {
    const u = this.u;
    let nearest = -1;
    let nearestDist = Infinity;

    for (let i = 0; i < u.count; i++) {
      if (i === unitIndex) continue;
      if (u.owner[i] !== u.owner[unitIndex]) continue;
      // Check if wounded (health < 50%)
      const def = this.getUnitDef(i);
      if (u.health[i] < (def.health || 100) * 0.5) {
        const dist = this.getDistanceToUnit(unitIndex, i);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = i;
        }
      }
    }
    return nearest;
  }

  protectAlly(unitIndex, allyIndex, dt, def) {
    // Move between ally and nearest enemy
    const enemy = this.findNearestEnemy(allyIndex, 100);
    if (enemy !== -1) {
      // Position between ally and enemy
      const u = this.u;
      const midX = (u.x[allyIndex] + u.x[enemy]) / 2;
      const midY = (u.y[allyIndex] + u.y[enemy]) / 2;
      this.moveToPosition(unitIndex, midX, midY, dt, def);
    }
  }

  calculateFlankingPosition(unitIndex, enemyIndex) {
    const u = this.u;
    // Calculate position 90 degrees to the side of enemy
    const dx = u.x[unitIndex] - u.x[enemyIndex];
    const dy = u.y[unitIndex] - u.y[enemyIndex];
    const angle = Math.atan2(dy, dx) + Math.PI / 2;
    const flankDist = 20;
    return {
      x: u.x[enemyIndex] + Math.cos(angle) * flankDist,
      y: u.y[enemyIndex] + Math.sin(angle) * flankDist
    };
  }

  moveToPosition(unitIndex, x, y, dt, def) {
    const u = this.u;
    const accel = def.speed * 4;
    const dx = x - u.x[unitIndex];
    const dy = y - u.y[unitIndex];
    const dist = Math.sqrt(dx*dx + dy*dy);

    if (dist > 1) {
      const inv = 1/dist;
      u.vx[unitIndex] += dx*inv*accel*dt;
      u.vy[unitIndex] += dy*inv*accel*dt;
    }
  }

  chargeAtEnemy(unitIndex, enemyIndex, dt, def) {
    // Charge with bonus speed
    const u = this.u;
    const accel = def.speed * 6; // Extra acceleration for charge
    const dx = u.x[enemyIndex] - u.x[unitIndex];
    const dy = u.y[enemyIndex] - u.y[unitIndex];
    const dist = Math.sqrt(dx*dx + dy*dy);

    if (dist > 0) {
      const inv = 1/dist;
      u.vx[unitIndex] += dx*inv*accel*dt;
      u.vy[unitIndex] += dy*inv*accel*dt;
    }
  }

  retreatFromEnemy(unitIndex, enemyIndex, dt, def) {
    const u = this.u;
    const accel = def.speed * 4;
    const dx = u.x[unitIndex] - u.x[enemyIndex];
    const dy = u.y[unitIndex] - u.y[enemyIndex];
    const dist = Math.sqrt(dx*dx + dy*dy);

    if (dist > 0) {
      const inv = 1/dist;
      u.vx[unitIndex] += dx*inv*accel*dt;
      u.vy[unitIndex] += dy*inv*accel*dt;
    }
  }

  advanceTowardsEnemy(unitIndex, enemyIndex, dt, def) {
    this.moveTowardsUnit(unitIndex, enemyIndex, dt, def);
  }

  rangedAttack(unitIndex, enemyIndex, dt, def) {
    // Perform ranged attack if in range and off cooldown
    if (this.cooldowns[unitIndex] > 0) return;

    const distance = this.getDistanceToUnit(unitIndex, enemyIndex);
    if (distance <= (def.range || 50)) {
      this.performAttack(unitIndex, enemyIndex, dt, def);
    }
  }

  executeSupportBehavior(unitIndex, state, def, dt) {
    // Support units stay back and help wounded allies
    const wounded = this.findNearestWoundedAlly(unitIndex);
    if (wounded !== -1) {
      this.moveTowardsUnit(unitIndex, wounded, dt, def);
    } else {
      this.processIdle(unitIndex, dt, def);
    }
  }

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
    const atkDef = this.getUnitDef(a);
    if (this.cooldowns[a]>0) return;
    this.cooldowns[a] = 0.8;

    const damage = atkDef.damage || 10;
    const defDef = this.getUnitDef(b);
    const effectiveness = this.getTypeEffectiveness(atkDef.category, defDef.category);
    const finalDamage = damage * effectiveness;

    this.u.health[b] -= finalDamage;
    if (this.u.health[b]<=0 && !kills.includes(b)) kills.push(b);
  }
}
