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
  };
} 