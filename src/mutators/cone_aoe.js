// src/mutators/cone_aoe.js
// Fires a cone of projectiles in a forward-facing direction.
// The cone parameters are intentionally conservative to keep behaviour predictable.
// Args:
//   count: number of projectiles across the cone (default 5)
//   spreadDeg: total spread angle in degrees (default 60)
//   speed: projectile speed (default 6)
//   facingDeg: central direction (degrees, 0 = +X) – optional; if omitted boss→player later.

export function cone_aoe(state, args, dt, bossMgr, bulletMgr) {
  if (state.done) return true;

  const idx = 0;
  const cx = bossMgr.x[idx];
  const cy = bossMgr.y[idx];

  const count = Math.max(1, Math.min(64, args.count ?? 5));
  const spreadDeg = args.spreadDeg ?? 60;
  const speed = args.speed ?? 6;
  const facingDeg = args.facingDeg ?? 0;

  const facingRad = facingDeg * (Math.PI/180);
  const half = (spreadDeg * (Math.PI/180)) / 2;

  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0 : i / (count - 1); // 0..1 across cone
    const angle = facingRad - half + t * (2*half);
    bulletMgr.addBullet({
      x: cx,
      y: cy,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      owner: 'boss',
      worldId: bossMgr.worldId?.[idx] ?? 'default'
    });
  }

  state.done = true;
  return true;
} 