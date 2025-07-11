// src/mutators/pattern_shoot.js
// Ultra-flexible shooting patterns for complex behaviors

/**
 * Advanced pattern shooting with multiple modes and parameters
 * @param {object} state - per-action scratch object
 * @param {{pattern:string, count:number, speed:number, damage:number, spread:number, 
 *          waves:number, waveDelay:number, tracking:boolean, spiral:boolean, 
 *          spiralSpeed:number, burst:boolean, burstCount:number, bounces:number,
 *          homing:boolean, penetrating:boolean, explosive:boolean, chain:boolean}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function pattern_shoot(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  const idx = 0;
  
  if (!state.init) {
    state.init = true;
    state.pattern = args.pattern ?? 'radial';
    state.count = args.count ?? 8;
    state.speed = args.speed ?? 6;
    state.damage = args.damage ?? 80;
    state.spread = args.spread ?? Math.PI * 2;
    state.waves = args.waves ?? 1;
    state.waveDelay = args.waveDelay ?? 0.2;
    state.tracking = args.tracking ?? false;
    state.spiral = args.spiral ?? false;
    state.spiralSpeed = args.spiralSpeed ?? 2;
    state.burst = args.burst ?? false;
    state.burstCount = args.burstCount ?? 3;
    state.bounces = args.bounces ?? 0;
    state.homing = args.homing ?? false;
    state.penetrating = args.penetrating ?? false;
    state.explosive = args.explosive ?? false;
    state.chain = args.chain ?? false;
    
    state.currentWave = 0;
    state.waveTimer = 0;
    state.spiralAngle = 0;
    state.burstIndex = 0;
    state.burstTimer = 0;
  }
  
  // Handle wave timing
  if (state.currentWave >= state.waves) return true;
  
  state.waveTimer += dt;
  if (state.waveTimer < state.waveDelay && state.currentWave > 0) return false;
  
  // Fire current wave
  if (state.waveTimer >= state.waveDelay || state.currentWave === 0) {
    firePattern(state, bossMgr, bulletMgr, idx);
    state.currentWave++;
    state.waveTimer = 0;
  }
  
  // Update spiral angle
  if (state.spiral) {
    state.spiralAngle += state.spiralSpeed * dt;
  }
  
  return state.currentWave >= state.waves;
}

function firePattern(state, bossMgr, bulletMgr, idx) {
  const cx = bossMgr.x[idx];
  const cy = bossMgr.y[idx];
  
  switch (state.pattern) {
    case 'radial':
      fireRadialPattern(state, cx, cy, bulletMgr);
      break;
    case 'shotgun':
      fireShotgunPattern(state, cx, cy, bulletMgr);
      break;
    case 'spiral':
      fireSpiralPattern(state, cx, cy, bulletMgr);
      break;
    case 'cross':
      fireCrossPattern(state, cx, cy, bulletMgr);
      break;
    case 'star':
      fireStarPattern(state, cx, cy, bulletMgr);
      break;
    case 'wave':
      fireWavePattern(state, cx, cy, bulletMgr);
      break;
    case 'random':
      fireRandomPattern(state, cx, cy, bulletMgr);
      break;
    case 'targeted':
      fireTargetedPattern(state, cx, cy, bulletMgr);
      break;
  }
}

function fireRadialPattern(state, cx, cy, bulletMgr) {
  const angleStep = state.spread / state.count;
  const startAngle = state.spiral ? state.spiralAngle : 0;
  
  for (let i = 0; i < state.count; i++) {
    const angle = startAngle + i * angleStep;
    fireBullet(state, cx, cy, angle, bulletMgr);
  }
}

function fireShotgunPattern(state, cx, cy, bulletMgr) {
  const baseAngle = state.tracking ? getPlayerAngle(cx, cy) : 0;
  const spreadHalf = state.spread / 2;
  
  for (let i = 0; i < state.count; i++) {
    const angle = baseAngle + (Math.random() - 0.5) * spreadHalf;
    fireBullet(state, cx, cy, angle, bulletMgr);
  }
}

function fireSpiralPattern(state, cx, cy, bulletMgr) {
  const angleStep = (Math.PI * 2) / state.count;
  
  for (let i = 0; i < state.count; i++) {
    const angle = state.spiralAngle + i * angleStep;
    fireBullet(state, cx, cy, angle, bulletMgr);
  }
}

function fireCrossPattern(state, cx, cy, bulletMgr) {
  const angles = [0, Math.PI/2, Math.PI, Math.PI*1.5];
  const offset = state.spiral ? state.spiralAngle : 0;
  
  for (const baseAngle of angles) {
    for (let i = 0; i < state.count / 4; i++) {
      const spread = (i - (state.count/8)) * 0.2;
      fireBullet(state, cx, cy, baseAngle + offset + spread, bulletMgr);
    }
  }
}

function fireStarPattern(state, cx, cy, bulletMgr) {
  const points = 5;
  const angleStep = (Math.PI * 2) / points;
  
  for (let point = 0; point < points; point++) {
    const baseAngle = point * angleStep + (state.spiral ? state.spiralAngle : 0);
    for (let i = 0; i < state.count / points; i++) {
      const angle = baseAngle + (i - state.count/(points*2)) * 0.1;
      fireBullet(state, cx, cy, angle, bulletMgr);
    }
  }
}

function fireWavePattern(state, cx, cy, bulletMgr) {
  const waveWidth = Math.PI / 2;
  const baseAngle = state.tracking ? getPlayerAngle(cx, cy) : 0;
  
  for (let i = 0; i < state.count; i++) {
    const t = i / (state.count - 1);
    const waveOffset = Math.sin(t * Math.PI * 3) * 0.3;
    const angle = baseAngle + (t - 0.5) * waveWidth + waveOffset;
    fireBullet(state, cx, cy, angle, bulletMgr);
  }
}

function fireRandomPattern(state, cx, cy, bulletMgr) {
  for (let i = 0; i < state.count; i++) {
    const angle = Math.random() * Math.PI * 2;
    fireBullet(state, cx, cy, angle, bulletMgr);
  }
}

function fireTargetedPattern(state, cx, cy, bulletMgr) {
  const playerAngle = getPlayerAngle(cx, cy);
  const spreadHalf = state.spread / 2;
  
  for (let i = 0; i < state.count; i++) {
    const spread = (i / (state.count - 1) - 0.5) * spreadHalf;
    fireBullet(state, cx, cy, playerAngle + spread, bulletMgr);
  }
}

function fireBullet(state, cx, cy, angle, bulletMgr) {
  const vx = Math.cos(angle) * state.speed;
  const vy = Math.sin(angle) * state.speed;
  
  bulletMgr.addBullet({
    x: cx,
    y: cy,
    vx: vx,
    vy: vy,
    damage: state.damage,
    lifetime: 8.0,
    width: 0.4,
    height: 0.4,
    isEnemy: true,
    owner: 'boss',
    bounces: state.bounces,
    homing: state.homing,
    penetrating: state.penetrating,
    explosive: state.explosive,
    chain: state.chain,
    spriteName: getBulletSprite(state),
    worldId: 'default'
  });
}

function getBulletSprite(state) {
  if (state.explosive) return 'bullet_explosive';
  if (state.homing) return 'bullet_homing';
  if (state.penetrating) return 'bullet_piercing';
  if (state.chain) return 'bullet_chain';
  return 'bullet_basic';
}

function getPlayerAngle(cx, cy) {
  // Mock implementation - would get actual nearest player
  return Math.random() * Math.PI * 2;
}