// src/mutators/dash.js
export function dash(state, args, dt, bossMgr) {
  const idx = 0;
  if (!state.init) {
    state.init = true;
    const speed = args.speed ?? 10;
    const dirX = args.dx ?? 0;
    const dirY = args.dy ?? 0;
    const len = Math.hypot(dirX, dirY) || 1;
    state.vx = (dirX/len) * speed;
    state.vy = (dirY/len) * speed;
    state.time = 0;
    state.maxTime = args.duration ?? 0.5;
  }
  bossMgr.x[idx] += state.vx * dt;
  bossMgr.y[idx] += state.vy * dt;
  state.time += dt;
  return state.time >= state.maxTime;
} 