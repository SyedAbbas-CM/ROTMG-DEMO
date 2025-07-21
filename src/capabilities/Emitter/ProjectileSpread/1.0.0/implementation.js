// capabilities/Emitter/ProjectileSpread/1.0.0/implementation.js

export function compile(brick) {
  return {
    ability: 'projectile_spread',
    args: {
      count: brick.count ?? 8,
      arc: brick.arc ?? 360,
      initialAngle: brick.initialAngle ?? 0,
      speed: brick.speed ?? 8,
      acceleration: brick.acceleration ?? 0,
      damage: brick.damage ?? 10,
      width: brick.width ?? 0.6,
      height: brick.height ?? 0.6
    },
    _capType: brick.type || 'Emitter:ProjectileSpread@1.0.0'
  };
}

// One-shot emitter: spawns projectiles then finishes in same tick
export function invoke(node, state = {}, { bossMgr, bulletMgr }) {
  if (state.done) return true;
  if (!bossMgr || !bulletMgr) return true;

  const bi = 0; // single boss
  const x = bossMgr.x[bi];
  const y = bossMgr.y[bi];
  const {
    count = 8,
    arc = 360,
    initialAngle = 0,
    speed = 8,
    damage = 10,
    width = 0.6,
    height = 0.6
  } = node.args;

  const arcRad = (arc * Math.PI) / 180;
  const startRad = (initialAngle * Math.PI) / 180 - arcRad / 2;
  const step = count > 1 ? arcRad / (count - 1) : 0;

  for (let i = 0; i < count; i++) {
    const theta = startRad + step * i;
    bulletMgr.addBullet({
      x,
      y,
      vx: Math.cos(theta) * speed,
      vy: Math.sin(theta) * speed,
      ownerId: bossMgr.id[bi],
      damage,
      width,
      height,
      spriteName: null,
      worldId: bossMgr.worldId[bi]
    });
  }

  state.done = true;
  return true;
} 