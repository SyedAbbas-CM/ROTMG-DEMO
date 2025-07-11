// src/mutators/heal_self.js
// Boss heals itself over time or instantly

/**
 * Heal boss HP over time or instantly
 * @param {object} state - per-action scratch object
 * @param {{amount:number, duration:number, instant:boolean, effect:boolean}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function heal_self(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {\n  const idx = 0;\n  \n  if (!state.init) {\n    state.init = true;\n    state.amount = args.amount ?? 200;\n    state.duration = args.duration ?? 2.0;\n    state.instant = args.instant ?? false;\n    state.effect = args.effect ?? true;\n    state.time = 0;\n    state.healRate = state.instant ? state.amount : state.amount / state.duration;\n    state.totalHealed = 0;\n  }\n  \n  // Calculate heal amount this frame\n  const healThisFrame = state.instant ? state.amount : state.healRate * dt;\n  \n  // Apply healing (check if boss has HP system)\n  if (bossMgr.hp && bossMgr.maxHp) {\n    const currentHp = bossMgr.hp[idx];\n    const maxHp = bossMgr.maxHp[idx];\n    const newHp = Math.min(maxHp, currentHp + healThisFrame);\n    bossMgr.hp[idx] = newHp;\n    state.totalHealed += newHp - currentHp;\n  }\n  \n  // Add healing effect\n  if (state.effect && bulletMgr.addEffect) {\n    bulletMgr.addEffect({\n      type: 'heal',\n      x: bossMgr.x[idx],\n      y: bossMgr.y[idx],\n      amount: healThisFrame,\n      duration: 0.5,\n      color: 0x00ff00 // Green\n    });\n  }\n  \n  state.time += dt;\n  \n  // Instant heal completes immediately\n  if (state.instant) {\n    return true;\n  }\n  \n  // Duration-based heal completes when time elapsed\n  return state.time >= state.duration;\n}