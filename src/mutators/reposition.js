// src/mutators/reposition.js
// Smoothly move the boss to an absolute target position over a fixed duration.
// Args: { x:number, y:number, duration?:number }

export function reposition(state, args, dt, bossMgr) {
  const idx = 0;
  const targetX = args.x ?? bossMgr.x[idx];
  const targetY = args.y ?? bossMgr.y[idx];
  const duration = Math.max(0.01, args.duration ?? 1);

  if (!state.init) {
    state.init = true;
    state.elapsed = 0;
    state.startX = bossMgr.x[idx];
    state.startY = bossMgr.y[idx];
    state.dx = targetX - state.startX;
    state.dy = targetY - state.startY;
  }

  state.elapsed += dt;
  const t = Math.min(1, state.elapsed / duration);
  bossMgr.x[idx] = state.startX + state.dx * t;
  bossMgr.y[idx] = state.startY + state.dy * t;

  return t >= 1;
} 