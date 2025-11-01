/**
 * DSL Compiler - Converts Behavior DSL AST to SoA-optimized behavior executors
 *
 * Takes the Abstract Syntax Tree from BehaviorDSL and compiles it into
 * efficient, Structure-of-Arrays compatible behavior functions that can
 * run in the EnemyManager's hot loop.
 *
 * Design Goals:
 * - Zero allocation in hot paths
 * - Pre-computed angles, trajectories, patterns
 * - Batch operations where possible
 * - Minimal branching
 */

export class DSLCompiler {
  constructor(enemyManager, bulletManager) {
    this.enemyManager = enemyManager;
    this.bulletManager = bulletManager;

    // Compiled behavior cache
    this.compiledBehaviors = new Map();

    // Runtime state per enemy (indexed by enemy array index)
    this.stateTimers = new Float32Array(10000);
    this.currentStates = new Array(10000);
    this.stateData = new Array(10000); // Per-state custom data
  }

  /**
   * Compile a complete behavior tree from DSL AST
   * @param {object} ast - AST from BehaviorDSL.parse()
   * @param {string} behaviorId - Unique identifier for this behavior
   * @returns {object} Compiled behavior executor
   */
  compile(ast, behaviorId) {
    if (this.compiledBehaviors.has(behaviorId)) {
      return this.compiledBehaviors.get(behaviorId);
    }

    const compiled = {
      id: behaviorId,
      states: new Map(),
      initialState: ast.states.length > 0 ? ast.states[0].name : 'idle',
      variables: this.compileVariables(ast.variables)
    };

    // Compile each state
    for (const state of ast.states) {
      compiled.states.set(state.name, this.compileState(state));
    }

    this.compiledBehaviors.set(behaviorId, compiled);
    return compiled;
  }

  /**
   * Compile a single state into executable functions
   * @param {object} state - State AST node
   * @returns {object} Compiled state with onEnter, onTick, onExit functions
   */
  compileState(state) {
    return {
      name: state.name,
      onEnter: this.compileActions(state.onEnter),
      onTick: this.compileActions(state.onTick),
      onExit: this.compileActions(state.onExit),
      duration: state.duration,
      next: state.next,
      conditions: state.conditions.map(c => ({
        check: this.compileExpression(c.condition),
        action: this.compileExpression(c.action)
      }))
    };
  }

  /**
   * Compile an array of action expressions
   * @param {Array} actions - Array of action AST nodes
   * @returns {Function} Compiled function that executes all actions
   */
  compileActions(actions) {
    const compiledActions = actions.map(action => this.compileExpression(action));

    return (enemyIndex, enemyManager, bulletManager, target, deltaTime, stateData) => {
      for (const action of compiledActions) {
        action(enemyIndex, enemyManager, bulletManager, target, deltaTime, stateData);
      }
    };
  }

  /**
   * Compile a single expression into executable function
   * @param {object} expr - Expression AST node
   * @returns {Function} Compiled function
   */
  compileExpression(expr) {
    if (!expr || !expr.type) {
      return () => {};
    }

    switch (expr.type) {
      case 'CALL':
        return this.compileCall(expr);
      case 'VAR':
        return this.compileVarRef(expr);
      default:
        return () => {};
    }
  }

  /**
   * Compile a function call
   * @param {object} call - Call AST node
   * @returns {Function} Compiled function
   */
  compileCall(call) {
    const funcName = call.function;
    const args = call.args;
    const kwargs = call.kwargs;

    // Movement behaviors
    if (funcName === 'MoveToward') {
      return this.compileMoveToward(args, kwargs);
    }
    if (funcName === 'MoveAway') {
      return this.compileMoveAway(args, kwargs);
    }
    if (funcName === 'Orbit') {
      return this.compileOrbit(args, kwargs);
    }
    if (funcName === 'ZigZag') {
      return this.compileZigZag(args, kwargs);
    }
    if (funcName === 'Circle') {
      return this.compileCircle(args, kwargs);
    }
    if (funcName === 'Charge') {
      return this.compileCharge(args, kwargs);
    }
    if (funcName === 'Retreat') {
      return this.compileRetreat(args, kwargs);
    }
    if (funcName === 'Wander') {
      return this.compileWander(args, kwargs);
    }
    if (funcName === 'Stop') {
      return this.compileStop(args, kwargs);
    }

    // Shooting behaviors
    if (funcName === 'Shoot') {
      return this.compileShoot(args, kwargs);
    }
    if (funcName === 'ShootPattern') {
      return this.compileShootPattern(args, kwargs);
    }
    if (funcName === 'ShootSpread') {
      return this.compileShootSpread(args, kwargs);
    }
    if (funcName === 'ShootSpiral') {
      return this.compileShootSpiral(args, kwargs);
    }
    if (funcName === 'ShootRing') {
      return this.compileShootRing(args, kwargs);
    }

    // State management
    if (funcName === 'SetSpeed') {
      return this.compileSetSpeed(args, kwargs);
    }
    if (funcName === 'SetAngle') {
      return this.compileSetAngle(args, kwargs);
    }
    if (funcName === 'SetVar') {
      return this.compileSetVar(args, kwargs);
    }

    console.warn(`[DSLCompiler] Unknown function: ${funcName}`);
    return () => {};
  }

  /**
   * Compile variables into runtime storage
   */
  compileVariables(variables) {
    const compiled = {};
    for (const [name, value] of Object.entries(variables)) {
      compiled[name] = this.evaluateConstant(value);
    }
    return compiled;
  }

  /**
   * Evaluate constant expressions at compile time
   */
  evaluateConstant(expr) {
    if (typeof expr === 'number' || typeof expr === 'string' || typeof expr === 'boolean') {
      return expr;
    }
    if (expr && expr.type === 'CALL') {
      // Handle special compile-time functions like Random()
      if (expr.function === 'Random') {
        const min = expr.args[0] || 0;
        const max = expr.args[1] || 1;
        return Math.random() * (max - min) + min;
      }
    }
    return 0;
  }

  // ========== MOVEMENT COMPILERS ==========

  compileMoveToward(args, kwargs) {
    const target = args[0] || kwargs.target;
    const speed = args[1] || kwargs.speed || 1.0;

    return (enemyIndex, enemyManager, bulletManager, targetObj) => {
      if (!targetObj) return;

      const ex = enemyManager.x[enemyIndex];
      const ey = enemyManager.y[enemyIndex];
      const tx = targetObj.x;
      const ty = targetObj.y;

      const dx = tx - ex;
      const dy = ty - ey;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 0) {
        const moveSpeed = enemyManager.speed[enemyIndex] * speed;
        enemyManager.vx[enemyIndex] = (dx / dist) * moveSpeed;
        enemyManager.vy[enemyIndex] = (dy / dist) * moveSpeed;
      }
    };
  }

  compileMoveAway(args, kwargs) {
    const target = args[0] || kwargs.target;
    const distance = args[1] || kwargs.distance || 300;

    return (enemyIndex, enemyManager, bulletManager, targetObj) => {
      if (!targetObj) return;

      const ex = enemyManager.x[enemyIndex];
      const ey = enemyManager.y[enemyIndex];
      const tx = targetObj.x;
      const ty = targetObj.y;

      const dx = ex - tx;
      const dy = ey - ty;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < distance && dist > 0) {
        const moveSpeed = enemyManager.speed[enemyIndex];
        enemyManager.vx[enemyIndex] = (dx / dist) * moveSpeed;
        enemyManager.vy[enemyIndex] = (dy / dist) * moveSpeed;
      } else {
        enemyManager.vx[enemyIndex] = 0;
        enemyManager.vy[enemyIndex] = 0;
      }
    };
  }

  compileOrbit(args, kwargs) {
    const target = args[0] || kwargs.target;
    const radius = args[1] || kwargs.radius || 200;
    const speed = args[2] || kwargs.speed || 1.0;
    const clockwise = args[3] !== undefined ? args[3] : (kwargs.clockwise !== undefined ? kwargs.clockwise : true);

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (!targetObj) return;

      // Initialize orbit angle if first time
      if (!stateData.orbitAngle) {
        const dx = enemyManager.x[enemyIndex] - targetObj.x;
        const dy = enemyManager.y[enemyIndex] - targetObj.y;
        stateData.orbitAngle = Math.atan2(dy, dx);
      }

      // Update angle
      const direction = clockwise ? -1 : 1;
      stateData.orbitAngle += direction * speed * deltaTime * 0.001;

      // Calculate target position on orbit
      const targetX = targetObj.x + Math.cos(stateData.orbitAngle) * radius;
      const targetY = targetObj.y + Math.sin(stateData.orbitAngle) * radius;

      // Move toward orbit position
      const ex = enemyManager.x[enemyIndex];
      const ey = enemyManager.y[enemyIndex];
      const dx = targetX - ex;
      const dy = targetY - ey;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 0) {
        const moveSpeed = enemyManager.speed[enemyIndex];
        enemyManager.vx[enemyIndex] = (dx / dist) * moveSpeed;
        enemyManager.vy[enemyIndex] = (dy / dist) * moveSpeed;
      }
    };
  }

  compileZigZag(args, kwargs) {
    const target = args[0] || kwargs.target;
    const amplitude = args[1] || kwargs.amplitude || 100;
    const frequency = args[2] || kwargs.frequency || 2;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (!targetObj) return;

      // Initialize zigzag state
      if (!stateData.zigzagTime) {
        stateData.zigzagTime = 0;
        const dx = targetObj.x - enemyManager.x[enemyIndex];
        const dy = targetObj.y - enemyManager.y[enemyIndex];
        stateData.baseAngle = Math.atan2(dy, dx);
      }

      stateData.zigzagTime += deltaTime * 0.001;

      // Calculate zigzag offset
      const offset = Math.sin(stateData.zigzagTime * frequency * Math.PI * 2) * amplitude;

      // Calculate perpendicular direction
      const perpAngle = stateData.baseAngle + Math.PI / 2;
      const offsetX = Math.cos(perpAngle) * offset;
      const offsetY = Math.sin(perpAngle) * offset;

      // Target position with zigzag
      const targetX = targetObj.x + offsetX;
      const targetY = targetObj.y + offsetY;

      const ex = enemyManager.x[enemyIndex];
      const ey = enemyManager.y[enemyIndex];
      const dx = targetX - ex;
      const dy = targetY - ey;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 0) {
        const moveSpeed = enemyManager.speed[enemyIndex];
        enemyManager.vx[enemyIndex] = (dx / dist) * moveSpeed;
        enemyManager.vy[enemyIndex] = (dy / dist) * moveSpeed;
      }
    };
  }

  compileCircle(args, kwargs) {
    const radius = args[0] || kwargs.radius || 150;
    const speed = args[1] || kwargs.speed || 1.0;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      // Initialize circle state
      if (!stateData.circleAngle) {
        stateData.circleAngle = 0;
        stateData.centerX = enemyManager.x[enemyIndex];
        stateData.centerY = enemyManager.y[enemyIndex];
      }

      stateData.circleAngle += speed * deltaTime * 0.001;

      const targetX = stateData.centerX + Math.cos(stateData.circleAngle) * radius;
      const targetY = stateData.centerY + Math.sin(stateData.circleAngle) * radius;

      const ex = enemyManager.x[enemyIndex];
      const ey = enemyManager.y[enemyIndex];
      const dx = targetX - ex;
      const dy = targetY - ey;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > 0) {
        const moveSpeed = enemyManager.speed[enemyIndex];
        enemyManager.vx[enemyIndex] = (dx / dist) * moveSpeed;
        enemyManager.vy[enemyIndex] = (dy / dist) * moveSpeed;
      }
    };
  }

  compileCharge(args, kwargs) {
    const target = args[0] || kwargs.target;
    const speed = args[1] || kwargs.speed || 2.0;
    const duration = args[2] || kwargs.duration || 1000;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (!targetObj) return;

      // Initialize charge
      if (!stateData.chargeStarted) {
        const dx = targetObj.x - enemyManager.x[enemyIndex];
        const dy = targetObj.y - enemyManager.y[enemyIndex];
        const dist = Math.sqrt(dx * dx + dy * dy);

        stateData.chargeStarted = true;
        stateData.chargeDx = dist > 0 ? dx / dist : 0;
        stateData.chargeDy = dist > 0 ? dy / dist : 0;
      }

      const moveSpeed = enemyManager.speed[enemyIndex] * speed;
      enemyManager.vx[enemyIndex] = stateData.chargeDx * moveSpeed;
      enemyManager.vy[enemyIndex] = stateData.chargeDy * moveSpeed;
    };
  }

  compileRetreat(args, kwargs) {
    const target = args[0] || kwargs.target;
    const distance = args[1] || kwargs.distance || 400;
    const speed = args[2] || kwargs.speed || 1.5;

    return (enemyIndex, enemyManager, bulletManager, targetObj) => {
      if (!targetObj) return;

      const ex = enemyManager.x[enemyIndex];
      const ey = enemyManager.y[enemyIndex];
      const tx = targetObj.x;
      const ty = targetObj.y;

      const dx = ex - tx;
      const dy = ey - ty;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < distance && dist > 0) {
        const moveSpeed = enemyManager.speed[enemyIndex] * speed;
        enemyManager.vx[enemyIndex] = (dx / dist) * moveSpeed;
        enemyManager.vy[enemyIndex] = (dy / dist) * moveSpeed;
      }
    };
  }

  compileWander(args, kwargs) {
    const changeInterval = args[0] || kwargs.changeInterval || 1000;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (!stateData.wanderTime) {
        stateData.wanderTime = 0;
        stateData.wanderAngle = Math.random() * Math.PI * 2;
      }

      stateData.wanderTime += deltaTime;

      if (stateData.wanderTime >= changeInterval) {
        stateData.wanderTime = 0;
        stateData.wanderAngle = Math.random() * Math.PI * 2;
      }

      const moveSpeed = enemyManager.speed[enemyIndex];
      enemyManager.vx[enemyIndex] = Math.cos(stateData.wanderAngle) * moveSpeed;
      enemyManager.vy[enemyIndex] = Math.sin(stateData.wanderAngle) * moveSpeed;
    };
  }

  compileStop(args, kwargs) {
    return (enemyIndex, enemyManager) => {
      enemyManager.vx[enemyIndex] = 0;
      enemyManager.vy[enemyIndex] = 0;
    };
  }

  // ========== SHOOTING COMPILERS ==========

  compileShoot(args, kwargs) {
    const angle = args[0] || kwargs.angle;
    const speed = args[1] || kwargs.speed || 25;
    const bulletType = args[2] || kwargs.bulletType || 'arrow';

    return (enemyIndex, enemyManager, bulletManager, targetObj) => {
      const shootAngle = typeof angle === 'number' ? angle : this.getAngleToPlayer(enemyIndex, enemyManager, targetObj);

      bulletManager.create(
        enemyManager.x[enemyIndex],
        enemyManager.y[enemyIndex],
        shootAngle,
        speed,
        bulletType,
        'enemy'
      );
    };
  }

  compileShootPattern(args, kwargs) {
    const patternName = args[0] || kwargs.pattern || 'aimed';
    const params = kwargs;

    return (enemyIndex, enemyManager, bulletManager, targetObj) => {
      const pattern = this.getPattern(patternName);
      const baseAngle = params.angle || this.getAngleToPlayer(enemyIndex, enemyManager, targetObj);
      const speed = params.speed || 25;
      const bulletType = params.bulletType || 'arrow';

      if (pattern.type === 'spread') {
        for (const angleOffset of pattern.angles) {
          const shootAngle = baseAngle + (angleOffset * Math.PI / 180);
          bulletManager.create(
            enemyManager.x[enemyIndex],
            enemyManager.y[enemyIndex],
            shootAngle,
            speed,
            bulletType,
            'enemy'
          );
        }
      } else if (pattern.type === 'ring') {
        const count = pattern.count || 12;
        for (let i = 0; i < count; i++) {
          const shootAngle = (Math.PI * 2 * i) / count;
          bulletManager.create(
            enemyManager.x[enemyIndex],
            enemyManager.y[enemyIndex],
            shootAngle,
            speed,
            bulletType,
            'enemy'
          );
        }
      } else if (pattern.type === 'spiral') {
        const count = pattern.count || 8;
        const rotation = pattern.rotation || 0;
        for (let i = 0; i < count; i++) {
          const shootAngle = baseAngle + ((Math.PI * 2 * i) / count) + (rotation * Math.PI / 180);
          bulletManager.create(
            enemyManager.x[enemyIndex],
            enemyManager.y[enemyIndex],
            shootAngle,
            speed,
            bulletType,
            'enemy'
          );
        }
      }
    };
  }

  compileShootSpread(args, kwargs) {
    const angles = args[0] || kwargs.angles || [-30, 0, 30];
    const speed = args[1] || kwargs.speed || 25;
    const bulletType = args[2] || kwargs.bulletType || 'arrow';

    return (enemyIndex, enemyManager, bulletManager, targetObj) => {
      const baseAngle = this.getAngleToPlayer(enemyIndex, enemyManager, targetObj);

      for (const angleOffset of angles) {
        const shootAngle = baseAngle + (angleOffset * Math.PI / 180);
        bulletManager.create(
          enemyManager.x[enemyIndex],
          enemyManager.y[enemyIndex],
          shootAngle,
          speed,
          bulletType,
          'enemy'
        );
      }
    };
  }

  compileShootSpiral(args, kwargs) {
    const count = args[0] || kwargs.count || 8;
    const rotationSpeed = args[1] || kwargs.rotationSpeed || 90;
    const bulletSpeed = args[2] || kwargs.bulletSpeed || 25;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (!stateData.spiralAngle) {
        stateData.spiralAngle = 0;
      }

      stateData.spiralAngle += rotationSpeed * deltaTime * 0.001 * Math.PI / 180;

      for (let i = 0; i < count; i++) {
        const shootAngle = stateData.spiralAngle + ((Math.PI * 2 * i) / count);
        bulletManager.create(
          enemyManager.x[enemyIndex],
          enemyManager.y[enemyIndex],
          shootAngle,
          bulletSpeed,
          'arrow',
          'enemy'
        );
      }
    };
  }

  compileShootRing(args, kwargs) {
    const count = args[0] || kwargs.count || 12;
    const speed = args[1] || kwargs.speed || 25;
    const bulletType = args[2] || kwargs.bulletType || 'arrow';

    return (enemyIndex, enemyManager, bulletManager) => {
      for (let i = 0; i < count; i++) {
        const shootAngle = (Math.PI * 2 * i) / count;
        bulletManager.create(
          enemyManager.x[enemyIndex],
          enemyManager.y[enemyIndex],
          shootAngle,
          speed,
          bulletType,
          'enemy'
        );
      }
    };
  }

  // ========== STATE MANAGEMENT COMPILERS ==========

  compileSetSpeed(args, kwargs) {
    const value = args[0] || kwargs.value || 1.0;

    return (enemyIndex, enemyManager) => {
      enemyManager.speed[enemyIndex] = value;
    };
  }

  compileSetAngle(args, kwargs) {
    const value = args[0] || kwargs.value;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (typeof value === 'number') {
        stateData.angle = value;
      } else if (value && value.type === 'TARGET' && value.target === 'player') {
        stateData.angle = this.getAngleToPlayer(enemyIndex, enemyManager, targetObj);
      }
    };
  }

  compileSetVar(args, kwargs) {
    const name = args[0] || kwargs.name;
    const value = args[1] || kwargs.value;

    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (!stateData.vars) stateData.vars = {};
      stateData.vars[name] = value;
    };
  }

  compileVarRef(expr) {
    const varName = expr.name;
    return (enemyIndex, enemyManager, bulletManager, targetObj, deltaTime, stateData) => {
      if (stateData.vars && stateData.vars[varName] !== undefined) {
        return stateData.vars[varName];
      }
      return 0;
    };
  }

  // ========== HELPER METHODS ==========

  getAngleToPlayer(enemyIndex, enemyManager, target) {
    if (!target) return 0;
    const dx = target.x - enemyManager.x[enemyIndex];
    const dy = target.y - enemyManager.y[enemyIndex];
    return Math.atan2(dy, dx);
  }

  getPattern(patternName) {
    const patterns = {
      'triple_forward': { type: 'spread', angles: [-30, 0, 30] },
      'five_spread': { type: 'spread', angles: [-60, -30, 0, 30, 60] },
      'ring': { type: 'ring', count: 12 },
      'spiral': { type: 'spiral', count: 8, rotation: 90 },
      'cross': { type: 'spread', angles: [0, 90, 180, 270] },
      'x_pattern': { type: 'spread', angles: [45, 135, 225, 315] },
      'aimed': { type: 'spread', angles: [0] }
    };
    return patterns[patternName] || patterns['aimed'];
  }

  /**
   * Main tick function - runs all active enemies' current states
   * Call this from EnemyManager.update()
   */
  tick(enemyIndex, behaviorId, enemyManager, bulletManager, target, deltaTime) {
    const compiled = this.compiledBehaviors.get(behaviorId);
    if (!compiled) {
      console.warn(`[DSLCompiler] Behavior ${behaviorId} not found`);
      return;
    }

    // Initialize state if first tick
    if (!this.currentStates[enemyIndex]) {
      this.currentStates[enemyIndex] = compiled.initialState;
      this.stateTimers[enemyIndex] = 0;
      this.stateData[enemyIndex] = {};

      // Run onEnter
      const state = compiled.states.get(this.currentStates[enemyIndex]);
      if (state && state.onEnter) {
        state.onEnter(enemyIndex, enemyManager, bulletManager, target, deltaTime, this.stateData[enemyIndex]);
      }
    }

    const currentStateName = this.currentStates[enemyIndex];
    const state = compiled.states.get(currentStateName);
    if (!state) return;

    // Run tick
    if (state.onTick) {
      state.onTick(enemyIndex, enemyManager, bulletManager, target, deltaTime, this.stateData[enemyIndex]);
    }

    // Check conditions
    for (const condition of state.conditions) {
      const result = condition.check(enemyIndex, enemyManager, bulletManager, target, deltaTime, this.stateData[enemyIndex]);
      if (result) {
        condition.action(enemyIndex, enemyManager, bulletManager, target, deltaTime, this.stateData[enemyIndex]);
      }
    }

    // Update timer and check transitions
    this.stateTimers[enemyIndex] += deltaTime;

    if (state.duration && this.stateTimers[enemyIndex] >= state.duration) {
      // Duration expired, transition to next state
      if (state.next) {
        this.transitionState(enemyIndex, state.next, compiled, enemyManager, bulletManager, target, deltaTime);
      }
    }
  }

  /**
   * Transition to a new state
   */
  transitionState(enemyIndex, newStateName, compiled, enemyManager, bulletManager, target, deltaTime) {
    const oldState = compiled.states.get(this.currentStates[enemyIndex]);
    const newState = compiled.states.get(newStateName);

    if (!newState) {
      console.warn(`[DSLCompiler] State ${newStateName} not found`);
      return;
    }

    // Run old state's onExit
    if (oldState && oldState.onExit) {
      oldState.onExit(enemyIndex, enemyManager, bulletManager, target, deltaTime, this.stateData[enemyIndex]);
    }

    // Transition
    this.currentStates[enemyIndex] = newStateName;
    this.stateTimers[enemyIndex] = 0;
    this.stateData[enemyIndex] = {}; // Clear state data

    // Run new state's onEnter
    if (newState.onEnter) {
      newState.onEnter(enemyIndex, enemyManager, bulletManager, target, deltaTime, this.stateData[enemyIndex]);
    }
  }

  /**
   * Reset enemy state (when enemy spawns or respawns)
   */
  resetEnemy(enemyIndex) {
    this.currentStates[enemyIndex] = null;
    this.stateTimers[enemyIndex] = 0;
    this.stateData[enemyIndex] = {};
  }
}

export default DSLCompiler;
