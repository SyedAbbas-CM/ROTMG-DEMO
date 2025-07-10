// src/critic/DifficultyCritic.js
// Simple rule-based difficulty critic.

const DPS_LIMITS = { easy: 50, mid: 100, hard: 200 };
const UNAVOIDABLE_LIMIT = 0.2;

/**
 * Evaluate KPI metrics and decide if the pattern is acceptable.
 * @param {object} kpi – KPI result from HeadlessSimulator.
 * @param {{tier?:'easy'|'mid'|'hard', ruleset?:string}} opts
 * @returns {{ok:boolean,reasons:string[]}}
 */
export function evaluate(kpi, { tier = 'mid', ruleset = 'default' } = {}) {
  const reasons = [];

  if (typeof kpi.dpsAvg === 'number' && kpi.dpsAvg > DPS_LIMITS[tier]) {
    reasons.push('dpsTooHigh');
  }

  if (
    typeof kpi.unavoidableDamagePct === 'number' &&
    kpi.unavoidableDamagePct > UNAVOIDABLE_LIMIT
  ) {
    reasons.push('tooMuchUnavoidableDamage');
  }

  return { ok: reasons.length === 0, reasons };
} 