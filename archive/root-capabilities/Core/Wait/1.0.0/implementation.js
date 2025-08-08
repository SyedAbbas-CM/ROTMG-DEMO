// capabilities/Core/Wait/1.0.0/implementation.js
export function compile(brick = {}) {
  return {
    ability: 'wait',
    args: {
      duration: brick.duration ?? brick.args?.duration ?? 1
    },
    _capType: brick.type || 'Core:Wait@1.0.0'
  };
}

export function invoke(node, state = {}, { dt }) {
  state.elapsed = (state.elapsed || 0) + dt;
  const dur = node.args.duration ?? 1;
  return state.elapsed >= dur;
} 