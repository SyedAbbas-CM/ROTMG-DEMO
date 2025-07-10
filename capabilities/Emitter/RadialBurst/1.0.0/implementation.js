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
  };
} 