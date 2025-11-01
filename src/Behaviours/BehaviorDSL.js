/**
 * Behavior DSL (Domain-Specific Language) for Enemy AI
 *
 * A simple, visual-friendly scripting language that compiles to SoA-optimized operations.
 * Designed to be human-readable, LLM-friendly, and efficient at runtime.
 *
 * Syntax Example:
 * ```
 * STATE shoot {
 *   ON_ENTER: SetAngle(ToPlayer)
 *   TICK: ShootPattern("triple_forward", angle=GetAngle(), spread=30)
 *   DURATION: 2s
 *   NEXT: retreat
 * }
 *
 * STATE retreat {
 *   ON_ENTER: SetSpeed(1.5)
 *   TICK: MoveAway(FromPlayer, distance=300)
 *   DURATION: 3s
 *   NEXT: shoot
 * }
 * ```
 */

export class BehaviorDSL {
  constructor() {
    // Built-in functions that map to SoA operations
    this.builtins = {
      // Movement
      'MoveToward': this.buildMoveToward,
      'MoveAway': this.buildMoveAway,
      'Orbit': this.buildOrbit,
      'ZigZag': this.buildZigZag,
      'Circle': this.buildCircle,
      'Charge': this.buildCharge,
      'Retreat': this.buildRetreat,
      'Wander': this.buildWander,
      'Stop': this.buildStop,

      // Shooting
      'Shoot': this.buildShoot,
      'ShootPattern': this.buildShootPattern,
      'ShootSpread': this.buildShootSpread,
      'ShootSpiral': this.buildShootSpiral,
      'ShootRing': this.buildShootRing,
      'ShootFrom': this.buildShootFrom,

      // State management
      'SetSpeed': this.buildSetSpeed,
      'SetAngle': this.buildSetAngle,
      'SetHP': this.buildSetHP,
      'SetVar': this.buildSetVar,
      'GetVar': this.buildGetVar,

      // Utility
      'ToPlayer': () => ({ type: 'TARGET', target: 'player' }),
      'FromPlayer': () => ({ type: 'TARGET', target: 'player', invert: true }),
      'ToSpawn': () => ({ type: 'TARGET', target: 'spawn' }),
      'GetAngle': () => ({ type: 'GET_ANGLE' }),
      'GetDistance': () => ({ type: 'GET_DISTANCE' }),
      'Random': (min, max) => ({ type: 'RANDOM', min, max }),

      // Math
      'Sin': (v) => ({ type: 'SIN', value: v }),
      'Cos': (v) => ({ type: 'COS', value: v }),
      'Abs': (v) => ({ type: 'ABS', value: v }),
    };

    // Pattern library
    this.patterns = {
      'triple_forward': { type: 'spread', angles: [-30, 0, 30] },
      'five_spread': { type: 'spread', angles: [-60, -30, 0, 30, 60] },
      'ring': { type: 'ring', count: 12 },
      'spiral': { type: 'spiral', count: 8, rotation: 90 },
      'cross': { type: 'spread', angles: [0, 90, 180, 270] },
      'x_pattern': { type: 'spread', angles: [45, 135, 225, 315] },
      'aimed': { type: 'aimed', count: 1 },
    };
  }

  /**
   * Parse DSL code into an Abstract Syntax Tree (AST)
   * @param {string} code - The DSL code to parse
   * @returns {object} AST representation
   */
  parse(code) {
    const ast = {
      states: [],
      variables: {},
      metadata: {}
    };

    // Simple tokenizer
    const lines = code.split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('//'));

    let currentState = null;
    let braceDepth = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // STATE declaration
      if (line.startsWith('STATE ')) {
        const match = line.match(/STATE\s+(\w+)\s*\{/);
        if (match) {
          currentState = {
            name: match[1],
            onEnter: [],
            onTick: [],
            onExit: [],
            duration: null,
            next: null,
            conditions: []
          };
          ast.states.push(currentState);
          braceDepth++;
        }
      }
      // Closing brace
      else if (line === '}') {
        braceDepth--;
        if (braceDepth === 0) {
          currentState = null;
        }
      }
      // Inside a state
      else if (currentState && braceDepth > 0) {
        // ON_ENTER
        if (line.startsWith('ON_ENTER:')) {
          const code = line.substring('ON_ENTER:'.length).trim();
          currentState.onEnter.push(this.parseExpression(code));
        }
        // TICK
        else if (line.startsWith('TICK:')) {
          const code = line.substring('TICK:'.length).trim();
          currentState.onTick.push(this.parseExpression(code));
        }
        // ON_EXIT
        else if (line.startsWith('ON_EXIT:')) {
          const code = line.substring('ON_EXIT:'.length).trim();
          currentState.onExit.push(this.parseExpression(code));
        }
        // DURATION
        else if (line.startsWith('DURATION:')) {
          const duration = line.substring('DURATION:'.length).trim();
          currentState.duration = this.parseDuration(duration);
        }
        // NEXT
        else if (line.startsWith('NEXT:')) {
          currentState.next = line.substring('NEXT:'.length).trim();
        }
        // IF condition
        else if (line.startsWith('IF ')) {
          const match = line.match(/IF\s+(.+)\s+THEN\s+(.+)/);
          if (match) {
            currentState.conditions.push({
              condition: this.parseExpression(match[1]),
              action: this.parseExpression(match[2])
            });
          }
        }
      }
      // VAR declaration
      else if (line.startsWith('VAR ')) {
        const match = line.match(/VAR\s+(\w+)\s*=\s*(.+)/);
        if (match) {
          ast.variables[match[1]] = this.parseExpression(match[2]);
        }
      }
    }

    return ast;
  }

  /**
   * Parse a single expression (function call or value)
   * @param {string} expr - Expression to parse
   * @returns {object} Expression AST node
   */
  parseExpression(expr) {
    expr = expr.trim();

    // Function call: FunctionName(arg1, arg2, key=value)
    const funcMatch = expr.match(/^(\w+)\((.*)\)$/);
    if (funcMatch) {
      const funcName = funcMatch[1];
      const argsStr = funcMatch[2];

      const args = [];
      const kwargs = {};

      if (argsStr) {
        // Parse arguments
        const argList = this.parseArguments(argsStr);
        for (const arg of argList) {
          if (arg.includes('=')) {
            const [key, value] = arg.split('=').map(s => s.trim());
            kwargs[key] = this.parseValue(value);
          } else {
            args.push(this.parseValue(arg));
          }
        }
      }

      return {
        type: 'CALL',
        function: funcName,
        args,
        kwargs
      };
    }

    // Direct value
    return this.parseValue(expr);
  }

  /**
   * Parse function arguments, respecting nested parentheses
   * @param {string} argsStr - Arguments string
   * @returns {Array} List of argument strings
   */
  parseArguments(argsStr) {
    const args = [];
    let current = '';
    let depth = 0;

    for (let i = 0; i < argsStr.length; i++) {
      const char = argsStr[i];
      if (char === '(' || char === '[') depth++;
      else if (char === ')' || char === ']') depth--;
      else if (char === ',' && depth === 0) {
        args.push(current.trim());
        current = '';
        continue;
      }
      current += char;
    }

    if (current.trim()) {
      args.push(current.trim());
    }

    return args;
  }

  /**
   * Parse a value (number, string, or nested expression)
   * @param {string} value - Value to parse
   * @returns {any} Parsed value
   */
  parseValue(value) {
    value = value.trim();

    // String literal
    if (value.startsWith('"') && value.endsWith('"')) {
      return value.slice(1, -1);
    }

    // Number
    if (/^-?\d+(\.\d+)?$/.test(value)) {
      return parseFloat(value);
    }

    // Boolean
    if (value === 'true') return true;
    if (value === 'false') return false;

    // Nested function call
    if (value.includes('(')) {
      return this.parseExpression(value);
    }

    // Variable reference
    return { type: 'VAR', name: value };
  }

  /**
   * Parse duration string (e.g., "2s", "500ms", "1.5s")
   * @param {string} duration - Duration string
   * @returns {number} Duration in milliseconds
   */
  parseDuration(duration) {
    if (duration.endsWith('ms')) {
      return parseFloat(duration);
    }
    if (duration.endsWith('s')) {
      return parseFloat(duration) * 1000;
    }
    return parseFloat(duration) * 1000; // Default to seconds
  }

  // ============= BUILT-IN FUNCTION BUILDERS =============

  buildMoveToward(target, speed = 1.0) {
    return {
      type: 'MOVE',
      direction: 'toward',
      target,
      speed
    };
  }

  buildMoveAway(target, distance) {
    return {
      type: 'MOVE',
      direction: 'away',
      target,
      distance
    };
  }

  buildOrbit(target, radius, speed = 1.0, clockwise = true) {
    return {
      type: 'ORBIT',
      target,
      radius,
      speed,
      clockwise
    };
  }

  buildZigZag(target, amplitude, frequency) {
    return {
      type: 'ZIGZAG',
      target,
      amplitude,
      frequency
    };
  }

  buildCircle(radius, speed) {
    return {
      type: 'CIRCLE',
      radius,
      speed
    };
  }

  buildCharge(target, speed, duration) {
    return {
      type: 'CHARGE',
      target,
      speed,
      duration
    };
  }

  buildRetreat(target, distance, speed) {
    return {
      type: 'RETREAT',
      target,
      distance,
      speed
    };
  }

  buildWander(changeInterval = 1000) {
    return {
      type: 'WANDER',
      changeInterval
    };
  }

  buildStop() {
    return {
      type: 'STOP'
    };
  }

  buildShoot(angle, speed = 25, bulletType = 'arrow') {
    return {
      type: 'SHOOT',
      angle,
      speed,
      bulletType
    };
  }

  buildShootPattern(patternName, kwargs) {
    return {
      type: 'SHOOT_PATTERN',
      pattern: patternName,
      params: kwargs
    };
  }

  buildShootSpread(angles, speed, bulletType) {
    return {
      type: 'SHOOT_SPREAD',
      angles,
      speed,
      bulletType
    };
  }

  buildShootSpiral(count, rotationSpeed, bulletSpeed) {
    return {
      type: 'SHOOT_SPIRAL',
      count,
      rotationSpeed,
      bulletSpeed
    };
  }

  buildShootRing(count, speed, bulletType) {
    return {
      type: 'SHOOT_RING',
      count,
      speed,
      bulletType
    };
  }

  buildShootFrom(origins, pattern, kwargs) {
    return {
      type: 'SHOOT_FROM',
      origins,
      pattern,
      params: kwargs
    };
  }

  buildSetSpeed(value) {
    return {
      type: 'SET_SPEED',
      value
    };
  }

  buildSetAngle(value) {
    return {
      type: 'SET_ANGLE',
      value
    };
  }

  buildSetHP(value) {
    return {
      type: 'SET_HP',
      value
    };
  }

  buildSetVar(name, value) {
    return {
      type: 'SET_VAR',
      name,
      value
    };
  }

  buildGetVar(name) {
    return {
      type: 'GET_VAR',
      name
    };
  }
}

export default BehaviorDSL;
