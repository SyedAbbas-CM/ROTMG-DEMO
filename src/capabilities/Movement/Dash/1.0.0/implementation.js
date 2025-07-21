export function compile(brick) {
  return {
    ability: 'dash',
    args: {
      dx: brick.dx ?? 0,
      dy: brick.dy ?? 0,
      speed: brick.speed ?? 10,
      duration: brick.duration ?? 0.5
    },
    _capType: brick.type,
  };
}

export function invoke(node, state, { dt, bossMgr }) {
  const idx = 0;
  if (!state.init) {
    state.init = true;
    const speed = node.args.speed;
    const dirX = node.args.dx;
    const dirY = node.args.dy;
    const len = Math.hypot(dirX, dirY) || 1;
    state.vx = (dirX / len) * speed;
    state.vy = (dirY / len) * speed;
    state.time = 0;
    state.maxTime = node.args.duration;
  }
  bossMgr.x[idx] += state.vx * dt;
  bossMgr.y[idx] += state.vy * dt;
  state.time += dt;
  return state.time >= state.maxTime;
} 