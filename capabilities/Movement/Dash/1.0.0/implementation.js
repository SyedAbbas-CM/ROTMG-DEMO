// capabilities/Movement/Dash/1.0.0/implementation.js
export function compile(brick) {
  return {
    ability: 'dash',
    args: {
      dx: brick.dx ?? 0,
      dy: brick.dy ?? 0,
      speed: brick.speed ?? 10,
      duration: brick.duration ?? 0.5,
    },
    _capType: brick.type || 'Movement:Dash@1.0.0'
  };
}

export function invoke(node, state = {}, ctx) {
  const { dt, bossMgr } = ctx;
  if (!bossMgr) return true;
  state.elapsed = (state.elapsed || 0) + dt;

  const bi = 0;
  const { dx = 0, dy = 0, speed = 10, duration = 0.5 } = node.args;
  bossMgr.x[bi] += dx * speed * dt;
  bossMgr.y[bi] += dy * speed * dt;

  return state.elapsed >= duration;
} 