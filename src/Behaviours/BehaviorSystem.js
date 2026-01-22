/**
 * BehaviorSystem.js - Core behavior system for enemies
 * Inspired by RotMG's behavior system but optimized for JavaScript and SoA pattern
 */

import { BehaviorState } from './BehaviorState.js';
import * as Behaviors from './BehaviorsCore.js';
import * as Transitions from './Transitions.js';

/**
 * Manages behavior templates and enemy behavior state
 */
export default class BehaviorSystem {
  constructor() {
    // Registry of behavior templates by type
    this.behaviorTemplates = new Map();

    // SoA data layout for current behavior states
    this.maxEnemies = 1000; // Match EnemyManager's default
    this.currentState = new Array(this.maxEnemies);
    this.stateData = new Array(this.maxEnemies); // Fallback for complex behaviors
    this.behaviorTimers = new Float32Array(this.maxEnemies);
    this.transitionChecks = new Array(this.maxEnemies);

    // ========================================
    // SoA Packed Arrays for Common Behaviors
    // ========================================

    // Wander behavior state
    this.wanderDirX = new Float32Array(this.maxEnemies);
    this.wanderDirY = new Float32Array(this.maxEnemies);
    this.wanderTimer = new Float32Array(this.maxEnemies);

    // CavalryCharge behavior state
    // Phase: 0=IDLE, 1=CHARGING, 2=DECELERATING, 3=STATIONARY
    this.chargePhase = new Uint8Array(this.maxEnemies);
    this.chargeSpeed = new Float32Array(this.maxEnemies);
    this.chargeTimer = new Float32Array(this.maxEnemies);
    this.chargeDirX = new Float32Array(this.maxEnemies);
    this.chargeDirY = new Float32Array(this.maxEnemies);

    // Orbit behavior state
    this.orbitAngle = new Float32Array(this.maxEnemies);
    this.orbitCenterX = new Float32Array(this.maxEnemies);
    this.orbitCenterY = new Float32Array(this.maxEnemies);

    // Swirl behavior state
    this.swirlAngle = new Float32Array(this.maxEnemies);
    this.swirlCenterX = new Float32Array(this.maxEnemies);
    this.swirlCenterY = new Float32Array(this.maxEnemies);

    // Follow behavior state
    // State: 0=DontKnowWhere, 1=Acquired, 2=Resting
    this.followState = new Uint8Array(this.maxEnemies);
    this.followLastX = new Float32Array(this.maxEnemies);
    this.followLastY = new Float32Array(this.maxEnemies);
    this.followRestTimer = new Float32Array(this.maxEnemies);

    // Simple charge behavior (Charge class)
    // State: 0=ready, 1=charging, 2=cooldown
    this.simpleChargeState = new Uint8Array(this.maxEnemies);
    this.simpleChargeTimer = new Float32Array(this.maxEnemies);
    this.simpleChargeCooldown = new Float32Array(this.maxEnemies);
    this.simpleChargeDirX = new Float32Array(this.maxEnemies);
    this.simpleChargeDirY = new Float32Array(this.maxEnemies);

    // Flash behavior state
    // State: 0=idle, 1=flashing, 2=cooldown
    this.flashState = new Uint8Array(this.maxEnemies);
    this.flashTimer = new Float32Array(this.maxEnemies);
    this.flashCooldown = new Float32Array(this.maxEnemies);

    // Grenade behavior cooldown
    this.grenadeCooldown = new Float32Array(this.maxEnemies);

    // Spawn position tracking (for StayCloseToSpawn, ReturnToSpawn)
    this.spawnPosX = new Float32Array(this.maxEnemies);
    this.spawnPosY = new Float32Array(this.maxEnemies);
    this.spawnPosSet = new Uint8Array(this.maxEnemies); // 0=not set, 1=set

    // MoveTo behavior
    this.moveToCompleted = new Uint8Array(this.maxEnemies);

    // Generic cooldown slots for heal/spawn behaviors (4 slots per enemy)
    this.cooldownSlots = new Float32Array(this.maxEnemies * 4);

    // Buzz behavior state
    this.buzzPhase = new Uint8Array(this.maxEnemies); // 0=waiting, 1=buzzing
    this.buzzTimer = new Float32Array(this.maxEnemies);
    this.buzzAngle = new Float32Array(this.maxEnemies);

    // BackAndForth behavior state
    this.backForthDir = new Int8Array(this.maxEnemies); // 1 or -1
    this.backForthTimer = new Float32Array(this.maxEnemies);
    this.backForthSteps = new Uint8Array(this.maxEnemies);

    // Invisibility state
    this.invisiVisible = new Uint8Array(this.maxEnemies); // 0=invisible, 1=visible
    this.invisiTimer = new Float32Array(this.maxEnemies);

    // Initialize the default behavior templates
    this.initDefaultBehaviors();
  }

  /**
   * Reset all SoA state for an enemy index (called on spawn/init)
   * @param {number} index - Enemy index
   */
  resetSoAState(index) {
    // Wander
    this.wanderDirX[index] = 0;
    this.wanderDirY[index] = 0;
    this.wanderTimer[index] = 0;

    // CavalryCharge
    this.chargePhase[index] = 0;
    this.chargeSpeed[index] = 0;
    this.chargeTimer[index] = 0;
    this.chargeDirX[index] = 0;
    this.chargeDirY[index] = 0;

    // Orbit
    this.orbitAngle[index] = 0;
    this.orbitCenterX[index] = 0;
    this.orbitCenterY[index] = 0;

    // Swirl
    this.swirlAngle[index] = 0;
    this.swirlCenterX[index] = 0;
    this.swirlCenterY[index] = 0;

    // Follow
    this.followState[index] = 0;
    this.followLastX[index] = 0;
    this.followLastY[index] = 0;
    this.followRestTimer[index] = 0;

    // Simple charge
    this.simpleChargeState[index] = 0;
    this.simpleChargeTimer[index] = 0;
    this.simpleChargeCooldown[index] = 0;
    this.simpleChargeDirX[index] = 0;
    this.simpleChargeDirY[index] = 0;

    // Flash
    this.flashState[index] = 0;
    this.flashTimer[index] = 0;
    this.flashCooldown[index] = 0;

    // Grenade
    this.grenadeCooldown[index] = 0;

    // Spawn position
    this.spawnPosX[index] = 0;
    this.spawnPosY[index] = 0;
    this.spawnPosSet[index] = 0;

    // MoveTo
    this.moveToCompleted[index] = 0;

    // Cooldown slots
    const slotBase = index * 4;
    this.cooldownSlots[slotBase] = 0;
    this.cooldownSlots[slotBase + 1] = 0;
    this.cooldownSlots[slotBase + 2] = 0;
    this.cooldownSlots[slotBase + 3] = 0;

    // Buzz
    this.buzzPhase[index] = 0;
    this.buzzTimer[index] = 0;
    this.buzzAngle[index] = 0;

    // BackAndForth
    this.backForthDir[index] = 1;
    this.backForthTimer[index] = 0;
    this.backForthSteps[index] = 0;

    // Invisibility
    this.invisiVisible[index] = 1;
    this.invisiTimer[index] = 0;
  }
  
  /**
   * Initialize default behavior templates for standard enemy types
   * NOTE: These are ENEMY behaviors, not unit behaviors. Units use UnitSystems.
   */
  initDefaultBehaviors() {
    // Type 0: BasicChaser (imp) - chases and shoots single bullets
    this.registerBehaviorTemplate(0, this.createBasicEnemyBehavior());

    // Type 1: Sniper (skeleton) - long-range shooter that maintains 18+ tile distance
    this.registerBehaviorTemplate(1, this.createSniperBehavior());

    // Type 2: Charger (beholder) - charges with velocity phases, directional shooting
    this.registerBehaviorTemplate(2, this.createChargerBehavior());

    // Type 3: HeavyCharger (red_demon) - slower but tankier charger
    this.registerBehaviorTemplate(3, this.createHeavyChargerBehavior());

    // Type 4: Defender (green_dragon) - defensive, chases when close, rages when damaged
    this.registerBehaviorTemplate(4, this.createDefenderBehavior());

    // Type 5: BossWander (boss_enemy) - wander only, attacks handled by AIPatternBoss
    this.registerBehaviorTemplate(5, this.createBossWanderBehavior());

    // Type 6: RushAttacker - charges while shooting rapidly
    this.registerBehaviorTemplate(6, this.createRushAttackerBehavior());

    // Type 7: Defender variant - defensive, only chases when player is close
    this.registerBehaviorTemplate(7, this.createDefenderBehavior());
  }
  
  /**
   * Register a behavior template for an enemy type
   * @param {number} type - Enemy type
   * @param {BehaviorState} rootState - Root behavior state
   */
  registerBehaviorTemplate(type, rootState) {
    this.behaviorTemplates.set(type, rootState);
  }
  
  /**
   * Initialize behavior for a new enemy
   * @param {number} index - Enemy index in the manager
   * @param {number} type - Enemy type
   */
  initBehavior(index, type) {
    const template = this.behaviorTemplates.get(type);

    if (!template) {
      console.warn(`No behavior template for enemy type ${type}`);
      return;
    }

    // Reset all SoA state arrays for this index
    this.resetSoAState(index);

    // Clone the behavior state for this instance
    this.currentState[index] = template;
    this.stateData[index] = {}; // Fallback for complex behaviors that need object storage
    this.behaviorTimers[index] = 0;
    this.transitionChecks[index] = [...template.transitions]; // Copy transitions

    // Initialize state
    this.onStateEntry(index);
  }
  
  /**
   * Process state entry
   * @param {number} index - Enemy index
   */
  onStateEntry(index) {
    const state = this.currentState[index];
    if (!state) return;
    
    if (state.onEntry) {
      state.onEntry(index, this.stateData[index]);
    }

    // Initialise each behaviour (once per state entry)
    if (state.behaviors) {
      for (const behavior of state.behaviors) {
        if (typeof behavior.init === 'function') {
          behavior.init(this.stateData[index]);
        }
      }
    }
  }
  
  /**
   * Process state exit
   * @param {number} index - Enemy index
   */
  onStateExit(index) {
    const state = this.currentState[index];
    if (!state) return;
    
    if (state.onExit) {
      state.onExit(index, this.stateData[index]);
    }
  }
  
  /**
   * Switch to a different state
   * @param {number} index - Enemy index
   * @param {BehaviorState} newState - State to switch to
   */
  switchState(index, newState) {
    if (!newState) return;

    // Exit current state
    const oldState = this.currentState[index];
    this.onStateExit(index);

    // Change to new state
    this.currentState[index] = newState;
    this.stateData[index] = {}; // Reset state data
    this.behaviorTimers[index] = 0;
    this.transitionChecks[index] = [...newState.transitions]; // Copy transitions

    // Enter new state
    this.onStateEntry(index);

    // Log state transition
    console.log(`🤖 [ENEMY STATE] Index ${index} transitioned: ${oldState?.name || 'unknown'} → ${newState.name}`);
  }
  
  /**
   * Update behavior for an enemy
   * @param {number} index - Enemy index
   * @param {Object} enemyManager - Reference to enemy manager
   * @param {Object} bulletManager - Reference to bullet manager
   * @param {Object} target - Target for behaviors (usually player)
   * @param {number} deltaTime - Time elapsed in seconds
   */
  updateBehavior(index, enemyManager, bulletManager, target, deltaTime) {
    const state = this.currentState[index];
    if (!state) return;

    // Update timer
    this.behaviorTimers[index] += deltaTime;

    // Check transitions first
    if (this.transitionChecks[index]) {
      for (const transition of this.transitionChecks[index]) {
        if (transition.check(index, enemyManager, target, deltaTime, this.stateData[index])) {
          this.switchState(index, transition.targetState);
          return; // Exit after transition
        }
      }
    }

    // Execute behaviors
    // Pass both stateData (for complex behaviors) and behaviorSystem (for SoA access)
    if (state.behaviors) {
      for (const behavior of state.behaviors) {
        behavior.execute(index, enemyManager, bulletManager, target, deltaTime, this.stateData[index], this);
      }
    }
  }

  /**
   * Swap behavior data when EnemyManager swaps enemies during removal
   * This is critical for maintaining correct behavior when using swap-and-pop pattern
   * @param {number} index - Index being filled (dead enemy position)
   * @param {number} lastIndex - Index being moved (last enemy position)
   */
  swapBehaviorData(index, lastIndex) {
    // Swap core state data
    this.currentState[index] = this.currentState[lastIndex];
    this.stateData[index] = this.stateData[lastIndex];
    this.behaviorTimers[index] = this.behaviorTimers[lastIndex];
    this.transitionChecks[index] = this.transitionChecks[lastIndex];

    // Swap SoA arrays - Wander
    this.wanderDirX[index] = this.wanderDirX[lastIndex];
    this.wanderDirY[index] = this.wanderDirY[lastIndex];
    this.wanderTimer[index] = this.wanderTimer[lastIndex];

    // Swap SoA arrays - CavalryCharge
    this.chargePhase[index] = this.chargePhase[lastIndex];
    this.chargeSpeed[index] = this.chargeSpeed[lastIndex];
    this.chargeTimer[index] = this.chargeTimer[lastIndex];
    this.chargeDirX[index] = this.chargeDirX[lastIndex];
    this.chargeDirY[index] = this.chargeDirY[lastIndex];

    // Swap SoA arrays - Orbit
    this.orbitAngle[index] = this.orbitAngle[lastIndex];
    this.orbitCenterX[index] = this.orbitCenterX[lastIndex];
    this.orbitCenterY[index] = this.orbitCenterY[lastIndex];

    // Swap SoA arrays - Swirl
    this.swirlAngle[index] = this.swirlAngle[lastIndex];
    this.swirlCenterX[index] = this.swirlCenterX[lastIndex];
    this.swirlCenterY[index] = this.swirlCenterY[lastIndex];

    // Swap SoA arrays - Follow
    this.followState[index] = this.followState[lastIndex];
    this.followLastX[index] = this.followLastX[lastIndex];
    this.followLastY[index] = this.followLastY[lastIndex];
    this.followRestTimer[index] = this.followRestTimer[lastIndex];

    // Swap SoA arrays - Simple charge
    this.simpleChargeState[index] = this.simpleChargeState[lastIndex];
    this.simpleChargeTimer[index] = this.simpleChargeTimer[lastIndex];
    this.simpleChargeCooldown[index] = this.simpleChargeCooldown[lastIndex];
    this.simpleChargeDirX[index] = this.simpleChargeDirX[lastIndex];
    this.simpleChargeDirY[index] = this.simpleChargeDirY[lastIndex];

    // Swap SoA arrays - Flash
    this.flashState[index] = this.flashState[lastIndex];
    this.flashTimer[index] = this.flashTimer[lastIndex];
    this.flashCooldown[index] = this.flashCooldown[lastIndex];

    // Swap SoA arrays - Grenade
    this.grenadeCooldown[index] = this.grenadeCooldown[lastIndex];

    // Swap SoA arrays - Spawn position
    this.spawnPosX[index] = this.spawnPosX[lastIndex];
    this.spawnPosY[index] = this.spawnPosY[lastIndex];
    this.spawnPosSet[index] = this.spawnPosSet[lastIndex];

    // Swap SoA arrays - MoveTo
    this.moveToCompleted[index] = this.moveToCompleted[lastIndex];

    // Swap SoA arrays - Cooldown slots
    const srcBase = lastIndex * 4;
    const dstBase = index * 4;
    this.cooldownSlots[dstBase] = this.cooldownSlots[srcBase];
    this.cooldownSlots[dstBase + 1] = this.cooldownSlots[srcBase + 1];
    this.cooldownSlots[dstBase + 2] = this.cooldownSlots[srcBase + 2];
    this.cooldownSlots[dstBase + 3] = this.cooldownSlots[srcBase + 3];

    // Swap SoA arrays - Buzz
    this.buzzPhase[index] = this.buzzPhase[lastIndex];
    this.buzzTimer[index] = this.buzzTimer[lastIndex];
    this.buzzAngle[index] = this.buzzAngle[lastIndex];

    // Swap SoA arrays - BackAndForth
    this.backForthDir[index] = this.backForthDir[lastIndex];
    this.backForthTimer[index] = this.backForthTimer[lastIndex];
    this.backForthSteps[index] = this.backForthSteps[lastIndex];

    // Swap SoA arrays - Invisibility
    this.invisiVisible[index] = this.invisiVisible[lastIndex];
    this.invisiTimer[index] = this.invisiTimer[lastIndex];

    // Clear the last index data
    this.currentState[lastIndex] = null;
    this.stateData[lastIndex] = null;
    this.behaviorTimers[lastIndex] = 0;
    this.transitionChecks[lastIndex] = null;
    // Note: SoA arrays don't need clearing - they'll be overwritten on next use
  }
  
  /* === Behavior Template Factories === */
  
  /**
   * Create behavior template for Basic enemy (Type 0) - Light Infantry
   */
  createBasicEnemyBehavior() {
    // Idle state when no players nearby
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.5) // Slow wandering
    ]);

    // Chase state when players are detected
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(1.0, 6), // Chase at normal speed, stop at 6 tiles
      new Behaviors.Shoot(1.0, 1, 0) // Single shots
    ]);

    // Transitions - Engagement range: 25 tiles (matches cavalry shootRange)
    idleState.addTransition(new Transitions.PlayerWithinRange(25, chaseState)); // Engage within 25 tiles
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(28, idleState)); // Disengage at 28 tiles
    chaseState.addTransition(new Transitions.TimedTransition(10.0, idleState)); // Give up after 10 seconds

    return idleState; // Return the root state
  }
  
  /**
   * Create behavior template for Fast enemy (Type 1)
   */
  createFastEnemyBehavior() {
    // Idle state when no players nearby
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.8) // Faster wandering
    ]);
    
    // Chase state when players are detected
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(1.5, 1), // Fast chase
      new Behaviors.Shoot(0.5, 1, 0) // Quick single shots
    ]);
    
    // Retreat state when low health
    const retreatState = new BehaviorState('retreat', [
      new Behaviors.RunAway(2.0, 300) // Run away quickly
    ]);
    
    // Transitions
    idleState.addTransition(new Transitions.PlayerWithinRange(300, chaseState));
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(350, idleState));
    chaseState.addTransition(new Transitions.HealthBelow(0.3, retreatState)); // Retreat when below 30% health
    retreatState.addTransition(new Transitions.NoPlayerWithinRange(400, idleState));
    retreatState.addTransition(new Transitions.HealthAbove(0.5, chaseState)); // Return to chase if health recovers
    
    return idleState;
  }

  /**
   * Create behavior template for Sniper enemy (skeleton - Type 1)
   * Long-range shooter that maintains distance and retreats when threatened
   */
  createSniperBehavior() {
    // Idle state when no players nearby
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.4) // Slow wandering
    ]);

    // Sniper state - maintains long range distance
    const sniperState = new BehaviorState('sniper', [
      new Behaviors.Chase(1.0, 22), // Stop at 22 tiles (long range)
      new Behaviors.Shoot(2.5, 1, 0) // Single precise shots with longer cooldown
    ]);

    // Retreat state - runs away when player gets too close, NO SHOOTING
    const retreatState = new BehaviorState('retreat', [
      new Behaviors.RunAway(1.5, 18) // Run away until 18 tiles away
    ]);

    // Transitions - Engagement range: 30 tiles (long range)
    idleState.addTransition(new Transitions.PlayerWithinRange(30, sniperState)); // Engage within 30 tiles
    sniperState.addTransition(new Transitions.NoPlayerWithinRange(32, idleState)); // Disengage at 32 tiles
    sniperState.addTransition(new Transitions.PlayerWithinRange(8, retreatState)); // Retreat if player gets within 8 tiles
    retreatState.addTransition(new Transitions.NoPlayerWithinRange(12, sniperState)); // Return to sniping when safe

    return idleState;
  }

  /**
   * Create behavior template for Heavy enemy (Type 2)
   */
  createHeavyEnemyBehavior() {
    // Idle state
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.3) // Slow wandering
    ]);
    
    // Chase state
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(0.6, 1), // Slow chase
      new Behaviors.Shoot(2.0, 3, Math.PI/8) // Triple shot with spread
    ]);
    
    // Rage state when low health
    const rageState = new BehaviorState('rage', [
      new Behaviors.Chase(0.8, 1), // Slightly faster chase
      new Behaviors.Shoot(1.0, 5, Math.PI/6) // Five shots with wide spread
    ]);
    
    // Transitions
    idleState.addTransition(new Transitions.PlayerWithinRange(350, chaseState));
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(400, idleState));
    chaseState.addTransition(new Transitions.HealthBelow(0.4, rageState));
    rageState.addTransition(new Transitions.NoPlayerWithinRange(500, chaseState));
    
    return idleState;
  }
  
  /**
   * Create behavior template for Turret enemy (Type 3) - Archer
   * Shoots from distance, retreats if player gets too close
   */
  createTurretBehavior() {
    // Idle state - wander slowly until player in range
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.3) // Slow wandering
    ]);

    // Shoot state - stationary shooting from safe distance
    const shootState = new BehaviorState('shoot', [
      new Behaviors.Shoot(1.5, 1, 0) // Single shots with medium cooldown
    ]);

    // Retreat state - run away when player gets too close
    const retreatState = new BehaviorState('retreat', [
      new Behaviors.RunAway(2.2, 20), // Run away at 2.2x speed for up to 20 tiles (faster retreat)
      new Behaviors.Shoot(2.0, 1, 0) // Slower shooting while retreating
    ]);

    // Rage state - faster shooting at low health
    const rageState = new BehaviorState('rage', [
      new Behaviors.Shoot(0.75, 2, Math.PI/12) // Double shots with shorter cooldown
    ]);

    // Transitions
    idleState.addTransition(new Transitions.PlayerWithinRange(18, shootState)); // Detect player at 18 tiles (matches range)
    shootState.addTransition(new Transitions.PlayerWithinRange(5, retreatState)); // Retreat if player within 5 tiles
    shootState.addTransition(new Transitions.NoPlayerWithinRange(22, idleState)); // Return to idle if player too far
    shootState.addTransition(new Transitions.HealthBelow(0.5, rageState)); // Rage at low health
    retreatState.addTransition(new Transitions.NoPlayerWithinRange(8, shootState)); // Return to shooting when safe
    rageState.addTransition(new Transitions.PlayerWithinRange(5, retreatState)); // Can still retreat in rage
    rageState.addTransition(new Transitions.NoPlayerWithinRange(22, idleState)); // Return to idle if player escapes

    return idleState;
  }
  
  /**
   * Create behavior template for Melee enemy (Type 4)
   */
  createMeleeEnemyBehavior() {
    // Idle state
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.7) // Medium speed wandering
    ]);
    
    // Chase state - very aggressive
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(1.2, 1) // Fast chase with close range
    ]);
    
    // Circle state - circle around target when very close
    const circleState = new BehaviorState('circle', [
      new Behaviors.Orbit(1.0, 40) // Orbit around target
    ]);
    
    // Transitions
    idleState.addTransition(new Transitions.PlayerWithinRange(200, chaseState));
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(250, idleState));
    chaseState.addTransition(new Transitions.PlayerWithinRange(60, circleState));
    circleState.addTransition(new Transitions.NoPlayerWithinRange(80, chaseState));
    
    return idleState;
  }

  /**
   * Create behavior template for Charger enemy (beholder - Type 2)
   * Charges with velocity phases - fast charging with limited turning, slow mode for shooting
   */
  createChargerBehavior() {
    // Idle state - wander until enemy detected
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.5) // Medium speed wandering
    ]);

    // Slow mode - can turn and shoot (only forward-facing)
    const slowState = new BehaviorState('slow', [
      new Behaviors.CavalryCharge(2.5, 6.0, 2.5, 8.0, 0.6, 8.0), // chargeSpeed: 2.5x, accel: 6.0, duration: 2.5s, range: 8 tiles, idle: 0.6x, decel: 8.0 (fast stop)
      new Behaviors.DirectionalShoot(1.0, Math.PI/3) // Shoots from JSON config, allows 60° cone
    ]);

    // Transitions - Engagement range: 30 tiles (matches cavalry range)
    idleState.addTransition(new Transitions.PlayerWithinRange(30, slowState)); // Engage within 30 tiles
    slowState.addTransition(new Transitions.NoPlayerWithinRange(35, idleState)); // Disengage at 35 tiles

    return idleState;
  }

  /**
   * Create behavior template for HeavyCharger enemy (red_demon - Type 3)
   * Slower base speed but charges to high speed, tankier than regular charger
   */
  createHeavyChargerBehavior() {
    // Idle state - wander until enemy detected
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.4) // Slower wandering than light cavalry
    ]);

    // Slow mode - can turn and shoot (only forward-facing)
    const slowState = new BehaviorState('slow', [
      new Behaviors.CavalryCharge(3.5, 5.0, 4.0, 8.0, 0.3, 7.0), // chargeSpeed: 3.5x (very high), accel: 5.0 (heavy buildup), duration: 4.0s (longest), range: 8 tiles, idle: 0.3x (slowest), decel: 7.0 (heavy stop)
      new Behaviors.DirectionalShoot(1.0, Math.PI/3) // Shoots from JSON config, allows 60° cone
    ]);

    // Transitions - Engagement range: 30 tiles
    idleState.addTransition(new Transitions.PlayerWithinRange(30, slowState)); // Engage within 30 tiles
    slowState.addTransition(new Transitions.NoPlayerWithinRange(35, idleState)); // Disengage at 35 tiles

    return idleState;
  }

  /**
   * Create behavior template for Defender enemy (green_dragon - Type 4)
   * Only chases when player is very close, otherwise stays defensive
   */
  createDefenderBehavior() {
    // Idle/defensive state - wander slowly
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.2) // Very slow wandering
    ]);

    // Chase state - only when player is close
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(0.8, 6), // Chase at slower speed, stop at 6 tiles
      new Behaviors.Shoot(2.0, 2, Math.PI/12) // Double shots with 15° spread
    ]);

    // Rage state - faster when damaged
    const rageState = new BehaviorState('rage', [
      new Behaviors.Chase(1.2, 6), // Chase faster when enraged, stop at 6 tiles
      new Behaviors.Shoot(1.5, 2, Math.PI/12) // Double shots with 15° spread
    ]);

    // Transitions - Engagement range: 15 tiles (defensive unit)
    idleState.addTransition(new Transitions.PlayerWithinRange(15, chaseState)); // Engage within 15 tiles
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(18, idleState)); // Disengage at 18 tiles
    chaseState.addTransition(new Transitions.HealthBelow(0.4, rageState)); // Rage when health below 40%
    rageState.addTransition(new Transitions.NoPlayerWithinRange(20, idleState)); // Disengage from rage at 20 tiles

    return idleState;
  }

  /**
   * Create a simple behavior template that only chases (optional) and shoots with custom parameters
   * @param {Object} opts
   *   - chaseSpeed {number} default 1.0
   *   - chaseRadius {number} default 200
   *   - projectileCount {number}
   *   - spread {number}
   *   - shootCooldown {number}
   *   - inaccuracy {number}
   *   - wanderSpeed {number}
   */
  createCustomShootBehavior(opts={}) {
    const {
      chaseSpeed = 1.0,
      chaseRadius = 200,
      projectileCount = 1,
      spread = 0,
      shootCooldown = 2.0,
      inaccuracy = 0,
      wanderSpeed = 0.4
    } = opts;

    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(wanderSpeed)
    ]);

    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(chaseSpeed, chaseRadius),
      // cooldownMultiplier is relative to base enemy cooldown; we just pass 1 here and rely on enemy-specific cooldown field
      new Behaviors.Shoot(1.0, projectileCount, spread, inaccuracy)
    ]);

    idleState.addTransition(new Transitions.PlayerWithinRange(chaseRadius, chaseState));
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(chaseRadius*1.2, idleState));

    // Override the Shoot behavior's default cooldown via prototype to respect shootCooldown
    chaseState.behaviors.forEach(b=>{
      if (b instanceof Behaviors.Shoot) {
        b.cooldownMultiplier = shootCooldown / 1.0; // assume enemyManager.cooldown default already holds shootCooldown; multiplier keeps ratio
      }
    });

    return idleState;
  }

  /**
   * Create behavior template for BossWander (boss_enemy - Type 5)
   * No attacks - attacks are handled by AIPatternBoss system
   */
  createBossWanderBehavior() {
    // Single state that just wanders, no transitions, no attacks
    const wanderState = new BehaviorState('wander', [
      new Behaviors.Wander(1.0) // Normal wandering speed
    ]);

    return wanderState;
  }

  /**
   * Create behavior template for Advanced enemy (Type 5)
   */
  createAdvancedEnemyBehavior() {
    // Idle state wandering
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.6)
    ]);

    // Orbit shoot state around player
    const orbitState = new BehaviorState('orbitShoot', [
      new Behaviors.Orbit(1.0, 6),
      new Behaviors.Shoot(1.2, 3, Math.PI / 10)
    ]);

    // Teleport away when player too close
    const teleportState = new BehaviorState('teleport', [
      new Behaviors.RunAway(2.5, 400),
      new Behaviors.Shoot(0.8, 5, Math.PI / 6)
    ]);

    // Swirl rage state when low health
    const swirlState = new BehaviorState('swirl', [
      new Behaviors.Swirl(1.5, 5, 8, true),
      new Behaviors.Shoot(0.4, 8, Math.PI / 8)
    ]);

    // Transitions
    idleState.addTransition(new Transitions.PlayerWithinRange(300, orbitState));
    orbitState.addTransition(new Transitions.NoPlayerWithinRange(350, idleState));
    orbitState.addTransition(new Transitions.PlayerWithinRange(80, teleportState));
    teleportState.addTransition(new Transitions.NoPlayerWithinRange(120, orbitState));
    teleportState.addTransition(new Transitions.TimedTransition(5.0, orbitState));
    orbitState.addTransition(new Transitions.HealthBelow(0.5, swirlState));
    swirlState.addTransition(new Transitions.HealthAbove(0.7, orbitState));

    return idleState;
  }

  /**
   * Create behavior template for RushAttacker enemy (Type 6)
   * Charges aggressively while shooting rapidly
   */
  createRushAttackerBehavior() {
    // Charge state - aggressively chase and shoot rapidly
    const chargeState = new BehaviorState('charge', [
      new Behaviors.Chase(2.0, 50),    // Fast aggressive chase
      new Behaviors.Shoot(0.5, 1, 0)   // Rapid single shots
    ]);

    // Wander state - move around slowly between charges
    const wanderState = new BehaviorState('wander', [
      new Behaviors.Wander(0.5)         // Slow wandering
    ]);

    // Transitions - cycle between charge and wander
    chargeState.addTransition(new Transitions.TimedTransition(2.0, wanderState));
    wanderState.addTransition(new Transitions.TimedTransition(1.5, chargeState));

    return chargeState; // Start in charge state
  }
} 