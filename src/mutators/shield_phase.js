// src/mutators/shield_phase.js
// Boss enters invulnerable shield phase

/**
 * Boss becomes invulnerable for specified duration
 * @param {object} state - per-action scratch object
 * @param {{duration:number, effect:boolean, color:number, pulseRate:number}} args
 * @param {number} dt - deltaTime (seconds)
 * @param {*} bossMgr - BossManager instance
 * @param {*} bulletMgr - BulletManager instance
 * @param {*} mapMgr - MapManager instance
 * @param {*} enemyMgr - EnemyManager instance
 * @returns {boolean} finished?
 */
export function shield_phase(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {\n  const idx = 0;\n  \n  if (!state.init) {\n    state.init = true;\n    state.duration = args.duration ?? 3.0;\n    state.effect = args.effect ?? true;\n    state.color = args.color ?? 0x00ffff; // Cyan\n    state.pulseRate = args.pulseRate ?? 2.0;\n    state.time = 0;\n    state.pulseTimer = 0;\n    \n    // Enable invulnerability\n    if (bossMgr.invulnerable) {\n      bossMgr.invulnerable[idx] = true;\n    }\n    \n    // Visual indicator\n    if (state.effect && bulletMgr.addEffect) {\n      bulletMgr.addEffect({\n        type: 'shield_start',\n        x: bossMgr.x[idx],\n        y: bossMgr.y[idx],\n        color: state.color,\n        duration: 1.0\n      });\n    }\n  }\n  \n  // Update shield visual effect\n  if (state.effect && bulletMgr.addEffect) {\n    state.pulseTimer += dt;\n    if (state.pulseTimer >= 1.0 / state.pulseRate) {\n      state.pulseTimer = 0;\n      \n      bulletMgr.addEffect({\n        type: 'shield_pulse',\n        x: bossMgr.x[idx],\n        y: bossMgr.y[idx],\n        color: state.color,\n        duration: 0.5,\n        intensity: 1.0 - (state.time / state.duration)\n      });\n    }\n  }\n  \n  state.time += dt;\n  \n  // End shield phase\n  if (state.time >= state.duration) {\n    // Disable invulnerability\n    if (bossMgr.invulnerable) {\n      bossMgr.invulnerable[idx] = false;\n    }\n    \n    // End effect\n    if (state.effect && bulletMgr.addEffect) {\n      bulletMgr.addEffect({\n        type: 'shield_end',\n        x: bossMgr.x[idx],\n        y: bossMgr.y[idx],\n        color: state.color,\n        duration: 0.8\n      });\n    }\n    \n    return true;\n  }\n  \n  return false;\n}