// capabilities/Emitter/RadialBurst/1.0.0/implementation.js
// Converts a validated RadialBurst brick into the engine mutator-node.
// NOTE: keep extremely light; complex behaviour mapping can be replaced later.

export function compile(brick) {
  return {
    ability: 'radial_burst',
    args: {
      projectiles: brick.projectiles ?? 12,
      speed: brick.speed ?? 8,
    },
    _capType: brick.type || 'Emitter:RadialBurst@1.0.0'
  };
}

// Runtime executor – fires the burst once then reports finished
export function invoke(node, state = {}, ctx) {
  const { bossMgr, bulletMgr } = ctx;
  if (!bossMgr || !bulletMgr) return true;
  if (state.done) return true;

  const bi = 0; // single boss for now
  const x = bossMgr.x[bi];
  const y = bossMgr.y[bi];
  const proj = node.args.projectiles || 12;
  const speed = node.args.speed || 8;

  const twoPi = Math.PI * 2;
  for (let i = 0; i < proj; i++) {
    const theta = (i / proj) * twoPi;
    bulletMgr.addBullet({
      x,
      y,
      vx: Math.cos(theta) * speed,
      vy: Math.sin(theta) * speed,
      ownerId: bossMgr.id[bi],
      damage: 10,
      width: 0.6,
      height: 0.6,
      worldId: bossMgr.worldId[bi]
    });
  }
  state.done = true;
  return true;
} 