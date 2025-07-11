// src/mutators/spawn_formation.js
// Advanced formation spawning with tactical positioning

/**
 * Spawn enemies in tactical formations around the boss
 * @param {object} state - per-action scratch object
 * @param {{formation:string, count:number, radius:number, type:number, spacing:number}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function spawn_formation(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  if (state.done) return true;

  const idx = 0;
  const cx = bossMgr.x[idx];
  const cy = bossMgr.y[idx];
  const formation = args.formation ?? 'circle';
  const count = Math.max(1, Math.min(32, args.count ?? 6));
  const radius = args.radius ?? 3;
  const type = args.type ?? 0;
  const spacing = args.spacing ?? 1;

  let positions = [];

  switch (formation) {
    case 'circle':
      positions = getCircleFormation(cx, cy, radius, count);
      break;
    case 'line':
      positions = getLineFormation(cx, cy, radius, count, args.angle ?? 0);
      break;
    case 'wedge':
      positions = getWedgeFormation(cx, cy, radius, count, args.angle ?? 0);
      break;
    case 'diamond':
      positions = getDiamondFormation(cx, cy, radius, count);
      break;
    case 'star':
      positions = getStarFormation(cx, cy, radius, count);
      break;
    case 'random':
      positions = getRandomFormation(cx, cy, radius, count);
      break;
    default:
      positions = getCircleFormation(cx, cy, radius, count);
  }

  // Spawn enemies at calculated positions
  for (let i = 0; i < positions.length; i++) {
    const pos = positions[i];
    if (typeof enemyMgr.spawnEnemy === 'function') {
      enemyMgr.spawnEnemy(type, pos.x, pos.y, bossMgr.worldId?.[idx] ?? 'default');
    } else if (typeof enemyMgr.addEnemy === 'function') {
      enemyMgr.addEnemy(pos.x, pos.y, type);
    }
  }

  state.done = true;
  return true;
}

function getCircleFormation(cx, cy, radius, count) {
  const positions = [];
  const twoPi = Math.PI * 2;
  for (let i = 0; i < count; i++) {
    const theta = (i / count) * twoPi;
    positions.push({
      x: cx + Math.cos(theta) * radius,
      y: cy + Math.sin(theta) * radius
    });
  }
  return positions;
}

function getLineFormation(cx, cy, length, count, angle) {
  const positions = [];
  const spacing = length / (count - 1);
  const startX = cx - Math.cos(angle) * length / 2;
  const startY = cy - Math.sin(angle) * length / 2;
  
  for (let i = 0; i < count; i++) {
    positions.push({
      x: startX + Math.cos(angle) * spacing * i,
      y: startY + Math.sin(angle) * spacing * i
    });
  }
  return positions;
}

function getWedgeFormation(cx, cy, radius, count, angle) {
  const positions = [];
  const wedgeAngle = Math.PI / 3; // 60 degrees
  
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const rowRadius = radius + row * 0.8;
    const colAngle = angle + (col - 1) * wedgeAngle / 3;
    
    positions.push({
      x: cx + Math.cos(colAngle) * rowRadius,
      y: cy + Math.sin(colAngle) * rowRadius
    });
  }
  return positions;
}

function getDiamondFormation(cx, cy, radius, count) {
  const positions = [];
  const corners = [
    { x: cx, y: cy - radius },      // top
    { x: cx + radius, y: cy },      // right
    { x: cx, y: cy + radius },      // bottom
    { x: cx - radius, y: cy }       // left
  ];
  
  for (let i = 0; i < count; i++) {
    const corner = corners[i % 4];
    const offset = Math.floor(i / 4) * 0.5;
    positions.push({
      x: corner.x + (Math.random() - 0.5) * offset,
      y: corner.y + (Math.random() - 0.5) * offset
    });
  }
  return positions;
}

function getStarFormation(cx, cy, radius, count) {
  const positions = [];
  const points = 5;
  const twoPi = Math.PI * 2;
  
  for (let i = 0; i < count; i++) {
    const point = i % points;
    const layer = Math.floor(i / points);
    const theta = (point / points) * twoPi;
    const r = radius + layer * 0.6;
    
    positions.push({
      x: cx + Math.cos(theta) * r,
      y: cy + Math.sin(theta) * r
    });
  }
  return positions;
}

function getRandomFormation(cx, cy, radius, count) {
  const positions = [];
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * radius;
    positions.push({
      x: cx + Math.cos(angle) * r,
      y: cy + Math.sin(angle) * r
    });
  }
  return positions;
}