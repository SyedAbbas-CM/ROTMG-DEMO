export function compile(brick) {
  return {
    ability: 'radial_burst',
    args: {
      projectiles: brick.projectiles ?? 12,
      speed: brick.speed ?? 8
    },
    _capType: brick.type,
  };
}

export function invoke(node, state, { dt, bossMgr, bulletMgr, enemyMgr }) {
  if (state.done) return true;
  const idx = 0; // single boss index for now (multi-boss later)
  const eIdx = bossMgr.enemyIndex ? bossMgr.enemyIndex[idx] : -1;
  const ownerId = (eIdx >= 0 && enemyMgr && enemyMgr.id) ? enemyMgr.id[eIdx] : bossMgr.id[idx];

  const projectiles = node.args.projectiles;
  const speed = node.args.speed;
  const damage = node.args.damage ?? 12;
  const width  = node.args.width  ?? 0.6;
  const height = node.args.height ?? 0.6;
  const twoPi = Math.PI * 2;
  for (let i = 0; i < projectiles; i++) {
    const theta = (i / projectiles) * twoPi;
    bulletMgr.addBullet({
      x: bossMgr.x[idx], y: bossMgr.y[idx],
      vx: Math.cos(theta) * speed,
      vy: Math.sin(theta) * speed,
      ownerId,
      damage,
      width,
      height,
      spriteName: null,
      worldId: bossMgr.worldId[idx]
    });
  }
  state.done = true;
  return true;
} 