// src/boss/addons/AdaptiveFrequency.js
// Calculate adaptive cooldown based on game state

/**
 * Calculate adaptive cooldown period based on game state
 * @param {Object} snapshot - Current game snapshot
 * @param {Object} config - Configuration with minPeriod and maxPeriod
 * @returns {number} Cooldown in seconds
 */
export function getAdaptiveCooldown(snapshot, config) {
  const minPeriod = config.tacticalMinInterval || 10;
  const maxPeriod = config.tacticalMaxInterval || 30;

  if (!snapshot?.boss || !snapshot?.players) {
    return maxPeriod;
  }

  const boss = snapshot.boss;
  const players = snapshot.players;

  // High priority situations (faster response)

  // Boss is low HP - make decisions faster
  const hpPercent = boss.hp / boss.maxHp;
  if (hpPercent < 0.25) {
    return minPeriod;
  }

  // Multiple players - need faster tactical response
  if (players.length > 2) {
    return minPeriod + 5;
  }

  // Players very close - immediate threat
  const closestPlayer = players.reduce((min, p) =>
    p.distance < min ? p.distance : min, Infinity);
  if (closestPlayer < 5) {
    return minPeriod;
  }

  // Low priority situations (slower response to save API calls)

  // No players - conserve API calls
  if (players.length === 0) {
    return maxPeriod * 2;
  }

  // Boss at full health and players far away
  if (hpPercent > 0.9 && closestPlayer > 20) {
    return maxPeriod;
  }

  // Default: middle ground
  return (minPeriod + maxPeriod) / 2;
}
