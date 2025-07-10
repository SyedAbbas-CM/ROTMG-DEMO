export function taunt(state, args, dt, bossMgr, bulletMgr, mapMgr, enemyMgr) {
  if (state.done) return true;
  const line = args.line || '...';
  console.log(`[Boss Taunt] ${line}`);
  state.done = true;
  return true; // immediate finish
} 