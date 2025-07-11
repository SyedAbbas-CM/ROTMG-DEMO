// src/mutators/dynamic_movement.js
// Ultra-flexible movement patterns that can be chained and combined

/**
 * Dynamic movement with multiple patterns and smooth transitions
 * @param {object} state - per-action scratch object
 * @param {{pattern:string, speed:number, duration:number, radius:number, 
 *          amplitude:number, frequency:number, target:string, smooth:boolean,
 *          chain:array, loop:boolean, reverse:boolean, easing:string}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function dynamic_movement(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  const idx = 0;
  
  if (!state.init) {
    state.init = true;
    state.pattern = args.pattern ?? 'circle';
    state.speed = args.speed ?? 5;
    state.duration = args.duration ?? 3.0;
    state.radius = args.radius ?? 4;
    state.amplitude = args.amplitude ?? 2;
    state.frequency = args.frequency ?? 1;
    state.target = args.target ?? 'center';
    state.smooth = args.smooth ?? true;
    state.chain = args.chain ?? [];
    state.loop = args.loop ?? false;
    state.reverse = args.reverse ?? false;
    state.easing = args.easing ?? 'linear';
    
    state.time = 0;
    state.chainIndex = 0;
    state.centerX = bossMgr.x[idx];
    state.centerY = bossMgr.y[idx];
    state.startX = bossMgr.x[idx];
    state.startY = bossMgr.y[idx];
    state.phase = 0;
  }
  
  state.time += dt;
  const progress = Math.min(1, state.time / state.duration);
  const easedProgress = applyEasing(progress, state.easing);
  
  // Update center based on target
  updateMovementCenter(state, bossMgr, idx);
  
  // Calculate new position based on pattern
  const newPos = calculatePatternPosition(state, easedProgress);
  
  // Apply smooth movement
  if (state.smooth) {
    const moveSpeed = state.speed;
    const dx = newPos.x - bossMgr.x[idx];
    const dy = newPos.y - bossMgr.y[idx];
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > 0) {
      const maxMove = moveSpeed * dt;
      if (distance <= maxMove) {
        bossMgr.x[idx] = newPos.x;
        bossMgr.y[idx] = newPos.y;
      } else {
        bossMgr.x[idx] += (dx / distance) * maxMove;
        bossMgr.y[idx] += (dy / distance) * maxMove;
      }
    }
  } else {
    bossMgr.x[idx] = newPos.x;
    bossMgr.y[idx] = newPos.y;
  }
  
  // Check if pattern is complete
  if (progress >= 1.0) {
    // Handle chaining
    if (state.chain.length > 0 && state.chainIndex < state.chain.length) {
      const nextPattern = state.chain[state.chainIndex];
      Object.assign(state, nextPattern);
      state.time = 0;
      state.chainIndex++;
      state.startX = bossMgr.x[idx];
      state.startY = bossMgr.y[idx];
      return false; // Continue with next pattern
    }
    
    // Handle looping
    if (state.loop) {
      state.time = 0;
      state.chainIndex = 0;
      state.startX = bossMgr.x[idx];
      state.startY = bossMgr.y[idx];
      return false; // Continue looping
    }
    
    return true; // Finished
  }
  
  return false;
}

function calculatePatternPosition(state, progress) {
  const t = state.reverse ? 1 - progress : progress;
  
  switch (state.pattern) {
    case 'circle':
      return getCirclePosition(state, t);
    case 'figure8':
      return getFigure8Position(state, t);
    case 'spiral':
      return getSpiralPosition(state, t);
    case 'zigzag':
      return getZigzagPosition(state, t);
    case 'sine_wave':
      return getSineWavePosition(state, t);
    case 'square':
      return getSquarePosition(state, t);
    case 'triangle':
      return getTrianglePosition(state, t);
    case 'star':
      return getStarPosition(state, t);
    case 'random_walk':
      return getRandomWalkPosition(state, t);
    case 'pulse':
      return getPulsePosition(state, t);
    case 'orbit_player':
      return getOrbitPlayerPosition(state, t);
    case 'linear':
      return getLinearPosition(state, t);
    default:
      return { x: state.centerX, y: state.centerY };
  }
}

function getCirclePosition(state, t) {
  const angle = t * Math.PI * 2;
  return {
    x: state.centerX + Math.cos(angle) * state.radius,
    y: state.centerY + Math.sin(angle) * state.radius
  };
}

function getFigure8Position(state, t) {
  const angle = t * Math.PI * 2;
  const scale = Math.sin(angle);
  return {
    x: state.centerX + Math.cos(angle) * state.radius * scale,
    y: state.centerY + Math.sin(angle * 2) * state.radius * 0.5
  };
}

function getSpiralPosition(state, t) {
  const angle = t * Math.PI * 4;
  const radius = state.radius * t;
  return {
    x: state.centerX + Math.cos(angle) * radius,
    y: state.centerY + Math.sin(angle) * radius
  };
}

function getZigzagPosition(state, t) {
  const segments = 4;
  const segmentProgress = (t * segments) % 1;
  const segmentIndex = Math.floor(t * segments);
  const direction = segmentIndex % 2 === 0 ? 1 : -1;
  
  return {
    x: state.centerX + (t - 0.5) * state.radius * 2,
    y: state.centerY + direction * segmentProgress * state.amplitude
  };
}

function getSineWavePosition(state, t) {
  const wavePhase = t * Math.PI * 2 * state.frequency;
  return {
    x: state.centerX + (t - 0.5) * state.radius * 2,
    y: state.centerY + Math.sin(wavePhase) * state.amplitude
  };
}

function getSquarePosition(state, t) {
  const side = Math.floor(t * 4);
  const sideProgress = (t * 4) % 1;
  const halfRadius = state.radius;
  
  switch (side) {
    case 0: // Top
      return { x: state.centerX - halfRadius + sideProgress * state.radius * 2, y: state.centerY - halfRadius };
    case 1: // Right
      return { x: state.centerX + halfRadius, y: state.centerY - halfRadius + sideProgress * state.radius * 2 };
    case 2: // Bottom
      return { x: state.centerX + halfRadius - sideProgress * state.radius * 2, y: state.centerY + halfRadius };
    case 3: // Left
      return { x: state.centerX - halfRadius, y: state.centerY + halfRadius - sideProgress * state.radius * 2 };
    default:
      return { x: state.centerX, y: state.centerY };
  }
}

function getTrianglePosition(state, t) {
  const side = Math.floor(t * 3);
  const sideProgress = (t * 3) % 1;
  const r = state.radius;
  
  const points = [
    { x: state.centerX, y: state.centerY - r },
    { x: state.centerX + r * 0.866, y: state.centerY + r * 0.5 },
    { x: state.centerX - r * 0.866, y: state.centerY + r * 0.5 }
  ];
  
  const start = points[side];
  const end = points[(side + 1) % 3];
  
  return {
    x: start.x + (end.x - start.x) * sideProgress,
    y: start.y + (end.y - start.y) * sideProgress
  };
}

function getStarPosition(state, t) {
  const points = 5;
  const angle = t * Math.PI * 2;
  const starAngle = (Math.floor(t * points * 2) % (points * 2)) * Math.PI / points;
  const isOuter = Math.floor(t * points * 2) % 2 === 0;
  const radius = isOuter ? state.radius : state.radius * 0.5;
  
  return {
    x: state.centerX + Math.cos(starAngle) * radius,
    y: state.centerY + Math.sin(starAngle) * radius
  };
}

function getRandomWalkPosition(state, t) {
  // Seed-based random walk for consistency
  const seed = Math.floor(t * 20);
  const random1 = (Math.sin(seed * 12.9898) * 43758.5453) % 1;
  const random2 = (Math.sin(seed * 78.233) * 43758.5453) % 1;
  
  return {
    x: state.centerX + (random1 - 0.5) * state.radius * 2,
    y: state.centerY + (random2 - 0.5) * state.radius * 2
  };
}

function getPulsePosition(state, t) {
  const pulseScale = 1 + Math.sin(t * Math.PI * 2 * state.frequency) * 0.5;
  return {
    x: state.centerX,
    y: state.centerY
  };
}

function getOrbitPlayerPosition(state, t) {
  // Mock player position - would get actual player
  const playerX = state.centerX + 10;
  const playerY = state.centerY + 5;
  const angle = t * Math.PI * 2;
  
  return {
    x: playerX + Math.cos(angle) * state.radius,
    y: playerY + Math.sin(angle) * state.radius
  };
}

function getLinearPosition(state, t) {
  return {
    x: state.startX + (state.centerX - state.startX) * t,
    y: state.startY + (state.centerY - state.startY) * t
  };
}

function updateMovementCenter(state, bossMgr, idx) {
  switch (state.target) {
    case 'center':
      // Keep original center
      break;
    case 'player':
      // Mock player tracking
      state.centerX = bossMgr.x[idx] + Math.sin(state.time) * 5;
      state.centerY = bossMgr.y[idx] + Math.cos(state.time) * 5;
      break;
    case 'spawn':
      // Return to spawn point
      break;
  }
}

function applyEasing(t, easing) {
  switch (easing) {
    case 'linear': return t;
    case 'ease_in': return t * t;
    case 'ease_out': return 1 - (1 - t) * (1 - t);
    case 'ease_in_out': return t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
    case 'bounce': return 1 - Math.abs(Math.sin(t * Math.PI));
    case 'elastic': return Math.sin(t * Math.PI * 2) * Math.pow(2, -10 * t);
    default: return t;
  }
}