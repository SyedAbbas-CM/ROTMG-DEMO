// src/mutators/summon_orbitals.js
// Summon orbital projectiles that circle the boss

/**
 * Summon orbital projectiles that rotate around boss
 * @param {object} state - per-action scratch object
 * @param {{count:number, radius:number, speed:number, damage:number, duration:number}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function summon_orbitals(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {\n  const idx = 0;\n  \n  if (!state.init) {\n    state.init = true;\n    state.count = args.count ?? 6;\n    state.radius = args.radius ?? 3;\n    state.speed = args.speed ?? 2; // radians per second\n    state.damage = args.damage ?? 80;\n    state.duration = args.duration ?? 5.0;\n    state.time = 0;\n    state.angle = 0;\n    state.orbitals = [];\n    \n    // Create orbital projectiles\n    const twoPi = Math.PI * 2;\n    for (let i = 0; i < state.count; i++) {\n      const startAngle = (i / state.count) * twoPi;\n      const orbital = {\n        id: `orbital_${i}`,\n        angle: startAngle,\n        active: true\n      };\n      state.orbitals.push(orbital);\n    }\n  }\n  \n  // Update orbital positions\n  state.angle += state.speed * dt;\n  \n  const cx = bossMgr.x[idx];\n  const cy = bossMgr.y[idx];\n  \n  for (let i = 0; i < state.orbitals.length; i++) {\n    const orbital = state.orbitals[i];\n    if (!orbital.active) continue;\n    \n    const currentAngle = orbital.angle + state.angle;\n    const x = cx + Math.cos(currentAngle) * state.radius;\n    const y = cy + Math.sin(currentAngle) * state.radius;\n    \n    // Create orbital bullet\n    bulletMgr.addBullet({\n      x: x,\n      y: y,\n      vx: 0, // Stationary relative to boss\n      vy: 0,\n      damage: state.damage,\n      lifetime: 0.1, // Very short lifetime, continuously spawned\n      width: 0.5,\n      height: 0.5,\n      isEnemy: true,\n      owner: 'boss',\n      type: 'orbital',\n      spriteName: 'orbital_projectile',\n      worldId: bossMgr.worldId?.[idx] ?? 'default'\n    });\n  }\n  \n  // Add orbital trail effect\n  if (bulletMgr.addEffect) {\n    bulletMgr.addEffect({\n      type: 'orbital_trail',\n      x: cx,\n      y: cy,\n      radius: state.radius,\n      angle: state.angle,\n      count: state.count,\n      duration: 0.2\n    });\n  }\n  \n  state.time += dt;\n  return state.time >= state.duration;\n}