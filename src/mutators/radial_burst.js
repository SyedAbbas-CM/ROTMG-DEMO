export function radial_burst(state, args, dt, bossMgr, bulletMgr) {
  if (state.done) return true;
  const idx = 0;
  const projectiles = args.projectiles ?? 12;
  const speed = args.speed ?? 8;
  const twoPi = Math.PI * 2;
  for (let i = 0; i < projectiles; i++) {
    const theta = (i / projectiles) * twoPi;
    bulletMgr.addBullet({
      x: bossMgr.x[idx], y: bossMgr.y[idx],
      vx: Math.cos(theta) * speed,
      vy: Math.sin(theta) * speed,
      owner: 'boss', worldId: bossMgr.worldId[idx]
    });
  }
  state.done = true;
  return true;
} 