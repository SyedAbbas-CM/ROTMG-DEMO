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
    this.stateData = new Array(this.maxEnemies);
    this.behaviorTimers = new Float32Array(this.maxEnemies);
    this.transitionChecks = new Array(this.maxEnemies);
    
    // Initialize the default behavior templates
    this.initDefaultBehaviors();
  }
  
  /**
   * Initialize default behavior templates for standard enemy types
   */
  initDefaultBehaviors() {
    // Type 0: Light Infantry - chases and shoots single bullets
    this.registerBehaviorTemplate(0, this.createBasicEnemyBehavior());

    // Type 1: Archer - long-range sniper that maintains 18+ tile distance
    this.registerBehaviorTemplate(1, this.createArcherBehavior());

    // Type 2: Light Cavalry - charges with velocity phases, directional shooting
    this.registerBehaviorTemplate(2, this.createCavalryBehavior());

    // Type 3: Heavy Cavalry - slower but tankier cavalry, slightly slower max speed
    this.registerBehaviorTemplate(3, this.createHeavyCavalryBehavior());

    // Type 4: Heavy Infantry - defensive, chases when close, rages when damaged
    this.registerBehaviorTemplate(4, this.createDefensiveHeavyBehavior());

    // Type 5: Cavalry - charges with velocity phases, limited turning when fast
    this.registerBehaviorTemplate(5, this.createCavalryBehavior());

    // Type 6: Charging Shooter - charges while shooting rapidly
    this.registerBehaviorTemplate(6, this.createChargingShooterBehavior());

    // Type 7: Heavy Infantry - defensive, only chases when player is close
    this.registerBehaviorTemplate(7, this.createDefensiveHeavyBehavior());
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
    
    // Clone the behavior state for this instance
    this.currentState[index] = template;
    this.stateData[index] = {}; // State-specific data
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
    if (state.behaviors) {
      for (const behavior of state.behaviors) {
        behavior.execute(index, enemyManager, bulletManager, target, deltaTime, this.stateData[index]);
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
    // Swap all behavior state data from lastIndex to index
    this.currentState[index] = this.currentState[lastIndex];
    this.stateData[index] = this.stateData[lastIndex];
    this.behaviorTimers[index] = this.behaviorTimers[lastIndex];
    this.transitionChecks[index] = this.transitionChecks[lastIndex];

    // Clear the last index data (optional, but good practice)
    this.currentState[lastIndex] = null;
    this.stateData[lastIndex] = null;
    this.behaviorTimers[lastIndex] = 0;
    this.transitionChecks[lastIndex] = null;
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
   * Create behavior template for Archer (Type 1)
   * Long-range sniper that maintains distance and retreats when threatened
   */
  createArcherBehavior() {
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
   * Create behavior template for Cavalry (Type 5)
   * Charges with velocity phases - fast charging with limited turning, slow mode for shooting
   */
  createCavalryBehavior() {
    // Idle state - wander until enemy detected
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.5) // Medium speed wandering
    ]);

    // Slow mode - can turn and shoot (only forward-facing)
    const slowState = new BehaviorState('slow', [
      new Behaviors.CavalryCharge(0.6, 2.5, 6.0, 2.5, 1.2), // Charge at 2.5x max speed, faster accel (6.0), better turn
      new Behaviors.DirectionalShoot(1.0, Math.PI/3) // Shoots from JSON config, allows 60° cone
    ]);

    // Transitions - Engagement range: 30 tiles (matches cavalry range)
    idleState.addTransition(new Transitions.PlayerWithinRange(30, slowState)); // Engage within 30 tiles
    slowState.addTransition(new Transitions.NoPlayerWithinRange(35, idleState)); // Disengage at 35 tiles

    return idleState;
  }

  /**
   * Create behavior template for Heavy Cavalry (Type 3)
   * Slower base speed but charges to high speed, tankier than light cavalry
   */
  createHeavyCavalryBehavior() {
    // Idle state - wander until enemy detected
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.4) // Slower wandering than light cavalry
    ]);

    // Slow mode - can turn and shoot (only forward-facing)
    const slowState = new BehaviorState('slow', [
      new Behaviors.CavalryCharge(0.4, 2.8, 3.0, 3.0, 1.5), // Slowest start (0.4), high max (2.8x), slow accel (3.0), longer charge
      new Behaviors.DirectionalShoot(1.0, Math.PI/3) // Shoots from JSON config, allows 60° cone
    ]);

    // Transitions - Engagement range: 30 tiles
    idleState.addTransition(new Transitions.PlayerWithinRange(30, slowState)); // Engage within 30 tiles
    slowState.addTransition(new Transitions.NoPlayerWithinRange(35, idleState)); // Disengage at 35 tiles

    return idleState;
  }

  /**
   * Create behavior template for Defensive Heavy Infantry (Type 4)
   * Only chases when player is very close, otherwise stays defensive
   */
  createDefensiveHeavyBehavior() {
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
   * Create behavior template for Charging Shooter (Type 6)
   */
  createChargingShooterBehavior() {
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