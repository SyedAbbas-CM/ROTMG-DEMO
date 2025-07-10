// src/simulator/HeadlessSimulator.js
// Lightweight, render-less simulation stub used to generate fake KPI metrics.

export function simulate({
  patternJson = [],
  bossSeed = 0,
  frames = 1800,
  dt = 1 / 60,
} = {}) {
  // Placeholder calculations – replace with actual game logic later.
  const bulletsSpawned = patternJson.length * frames;
  const dpsAvg = bulletsSpawned * 0.01; // arbitrary scaling
  const unavoidableDamagePct = 0; // assume none for stub

  return {
    framesSimulated: frames,
    bulletsSpawned,
    dpsAvg,
    unavoidableDamagePct,
    bossSeed,
  };
} 