// src/mutators/spawn_minions.js
// Deterministic helper that spawns a short ring of minions around the boss.
// NOTE: This is deliberately simple – the intent is to prove the plumbing.
// Complex formation logic will land later in a dedicated capability.

/**
 * Spawn a handful of basic enemies around the boss.
 * @param {object} state – per-action scratch object (auto-created by runner)
 * @param {{count?:number, radius?:number, type?:number}} args
 * @param {number} dt – deltaTime (seconds)
 * @param {*} bossMgr – BossManager instance
 * @param {*} bulletMgr – BulletManager instance (unused)
 * @param {*} mapMgr – MapManager instance (unused)
 * @param {*} enemyMgr – EnemyManager instance
 * @returns {boolean} finished?
 */
export function spawn_minions(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  if (state.done) return true; // one-shot

  const idx = 0; // single-boss support for now
  const cx  = bossMgr.x[idx];
  const cy  = bossMgr.y[idx];
  const count   = Math.max(1, Math.min(32, args.count ?? 4));
  const radius  = args.radius ?? 2;
  const type    = args.type   ?? 0; // default enemy type index
  const twoPi   = Math.PI * 2;
  for (let i = 0; i < count; i++) {
    const theta = (i / count) * twoPi;
    const x = cx + Math.cos(theta) * radius;
    const y = cy + Math.sin(theta) * radius;
    if (typeof enemyMgr.spawnEnemy === 'function') {
      enemyMgr.spawnEnemy(type, x, y, bossMgr.worldId?.[idx] ?? 'default');
    } else if (typeof enemyMgr.addEnemy === 'function') {
      enemyMgr.addEnemy(x, y, type);
    }
  }

  state.done = true;
  return true;
} 