/**
 * BehaviorState.js - Implements a state machine for enemy behaviors
 * Inspired by RotMG's state pattern but optimized for JavaScript
 */

/**
 * Represents a state in a behavior state machine
 */
export class BehaviorState {
  /**
   * Create a new behavior state
   * @param {string} name - Name of this state for debugging
   * @param {Array} behaviors - Array of behavior components to execute in this state
   */
  constructor(name, behaviors = []) {
    this.name = name;
    this.behaviors = behaviors || [];
    this.transitions = [];
    this.onEntry = null;
    this.onExit = null;
  }
  
  /**
   * Add a transition to this state
   * @param {Transition} transition - Transition to add
   */
  addTransition(transition) {
    this.transitions.push(transition);
    return this;
  }
  
  /**
   * Add a behavior to this state
   * @param {Behavior} behavior - Behavior to add
   */
  addBehavior(behavior) {
    this.behaviors.push(behavior);
    return this;
  }
  
  /**
   * Set a callback to execute when entering this state
   * @param {Function} callback - Function to call on state entry
   */
  setOnEntry(callback) {
    this.onEntry = callback;
    return this;
  }
  
  /**
   * Set a callback to execute when exiting this state
   * @param {Function} callback - Function to call on state exit
   */
  setOnExit(callback) {
    this.onExit = callback;
    return this;
  }
  
  /**
   * Clone this state for a new enemy instance
   * @returns {BehaviorState} A new copy of this state
   */
  clone() {
    const newState = new BehaviorState(this.name, [...this.behaviors]);
    newState.transitions = [...this.transitions];
    newState.onEntry = this.onEntry;
    newState.onExit = this.onExit;
    return newState;
  }
}

/**
 * Base class for all behavior transitions
 */
export class Transition {
  /**
   * Create a new transition
   * @param {BehaviorState} targetState - State to transition to when condition is met
   */
  constructor(targetState) {
    this.targetState = targetState;
  }
  
  /**
   * Check if this transition should be taken
   * @param {number} index - Enemy index
   * @param {Object} enemyManager - Reference to enemy manager
   * @param {Object} target - Target (usually the player)
   * @param {number} deltaTime - Time elapsed in seconds
   * @param {Object} stateData - State-specific data for this enemy
   * @returns {boolean} True if transition should be taken
   */
  check(index, enemyManager, target, deltaTime, stateData) {
    // Base class always returns false - override in subclasses
    return false;
  }
}

/**
 * Base class for all behavior components
 */
export class Behavior {
  constructor() {
    // Each behaviour instance gets its own symbol key for per-enemy storage
    this._storeKey = Symbol('behaviourStore');
  }

  /**
   * Optional initialisation that runs once when the parent state becomes active.
   * Sub-classes can override to set up timers or cached data.
   * @param {Object} stateData – per-enemy state bag supplied by BehaviourSystem.
   */
  init(stateData) { /* default no-op */ }

  /**
   * Fetch (and lazily create) this behaviour's private storage object.
   * Mirrors the C# ref object pattern used in RotMG server code.
   */
  _getStore(stateData) {
    // eslint-disable-next-line no-prototype-builtins
    if (!stateData.hasOwnProperty(this._storeKey)) {
      stateData[this._storeKey] = {};
    }
    return stateData[this._storeKey];
  }
  /**
   * Execute this behavior
   * @param {number} index - Enemy index
   * @param {Object} enemyManager - Reference to enemy manager
   * @param {Object} bulletManager - Reference to bullet manager
   * @param {Object} target - Target for behavior (usually player)
   * @param {number} deltaTime - Time elapsed in seconds
   * @param {Object} stateData - State-specific data for this enemy (fallback for complex behaviors)
   * @param {Object} behaviorSystem - Reference to BehaviorSystem for SoA array access
   */
  execute(index, enemyManager, bulletManager, target, deltaTime, stateData, behaviorSystem) {
    // Base class does nothing - override in subclasses
  }
} 