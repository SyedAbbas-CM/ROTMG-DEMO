export function compile(brick){
  return {
    ability: 'wait',
    args: { duration: brick.duration ?? 1 },
    _capType: brick.type,
  };
}

export function invoke(node, state, { dt }){
  state.elapsed = (state.elapsed || 0) + dt;
  return state.elapsed >= (node.args.duration || 1);
} 