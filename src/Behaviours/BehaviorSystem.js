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
    // Type 0: Basic enemy - chases and shoots single bullets
    this.registerBehaviorTemplate(0, this.createBasicEnemyBehavior());
    
    // Type 1: Fast enemy - quickly chases and fires rapid shots
    this.registerBehaviorTemplate(1, this.createFastEnemyBehavior());
    
    // Type 2: Heavy enemy - slow movement, multiple projectile shots
    this.registerBehaviorTemplate(2, this.createHeavyEnemyBehavior());
    
    // Type 3: Stationary turret - doesn't move, shoots from fixed position
    this.registerBehaviorTemplate(3, this.createTurretBehavior());
    
    // Type 4: Melee enemy - aggressively chases but doesn't shoot
    this.registerBehaviorTemplate(4, this.createMeleeEnemyBehavior());
    
    // Type 5: Advanced enemy with orbit and teleport tactics
    this.registerBehaviorTemplate(5, this.createAdvancedEnemyBehavior());
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
    this.onStateExit(index);
    
    // Change to new state
    this.currentState[index] = newState;
    this.stateData[index] = {}; // Reset state data
    this.behaviorTimers[index] = 0;
    this.transitionChecks[index] = [...newState.transitions]; // Copy transitions
    
    // Enter new state
    this.onStateEntry(index);
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
  
  /* === Behavior Template Factories === */
  
  /**
   * Create behavior template for Basic enemy (Type 0)
   */
  createBasicEnemyBehavior() {
    // Idle state when no players nearby
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.5) // Slow wandering
    ]);
    
    // Chase state when players are detected
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(1.0, 200), // Chase at normal speed
      new Behaviors.Shoot(1.0, 1, 0) // Single shots
    ]);
    
    // Transitions
    idleState.addTransition(new Transitions.PlayerWithinRange(250, chaseState));
    chaseState.addTransition(new Transitions.NoPlayerWithinRange(300, idleState));
    
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
      new Behaviors.Chase(1.5, 150), // Fast chase
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
   * Create behavior template for Heavy enemy (Type 2)
   */
  createHeavyEnemyBehavior() {
    // Idle state
    const idleState = new BehaviorState('idle', [
      new Behaviors.Wander(0.3) // Slow wandering
    ]);
    
    // Chase state
    const chaseState = new BehaviorState('chase', [
      new Behaviors.Chase(0.6, 250), // Slow chase
      new Behaviors.Shoot(2.0, 3, Math.PI/8) // Triple shot with spread
    ]);
    
    // Rage state when low health
    const rageState = new BehaviorState('rage', [
      new Behaviors.Chase(0.8, 200), // Slightly faster chase
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
   * Create behavior template for Turret enemy (Type 3)
   */
  createTurretBehavior() {
    // Only one state - stationary shooting
    const shootState = new BehaviorState('shoot', [
      new Behaviors.Shoot(1.5, 1, 0) // Single shots with medium cooldown
    ]);
    
    // Add a self-transition for health triggers
    const rageState = new BehaviorState('rage', [
      new Behaviors.Shoot(0.75, 2, Math.PI/12) // Double shots with shorter cooldown
    ]);
    
    // Transitions
    shootState.addTransition(new Transitions.HealthBelow(0.5, rageState));
    
    return shootState;
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
      new Behaviors.Chase(1.2, 50) // Fast chase with close range
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
} 