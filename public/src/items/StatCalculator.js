/**
 * StatCalculator.js - Calculates player stats from base + equipment
 * Used by both client and server
 */

import { getItem } from './ItemDatabase.js';

/**
 * Calculate total stats for a player based on base stats and equipment
 * @param {Object} baseStats - Base stats from class (damage, defense, speed, etc.)
 * @param {Object} equipment - Equipment slots { weapon, armor, ability, ring }
 * @returns {Object} Computed stats
 */
export function calculateStats(baseStats, equipment) {
  // Start with base stats
  const stats = {
    damage: baseStats.damage || 0,
    defense: baseStats.defense || 0,
    speed: baseStats.speed || 0,
    maxHealth: baseStats.maxHealth || 100,
    maxMana: baseStats.maxMana || 100,
    attackSpeed: baseStats.attackSpeed || 0  // Bonus rate of fire
  };

  // Add stats from each equipment slot
  const slots = ['weapon', 'armor', 'ability', 'ring'];
  for (const slot of slots) {
    const itemId = equipment?.[slot]?.id || equipment?.[slot];
    if (!itemId) continue;

    const item = typeof itemId === 'object' ? itemId : getItem(itemId);
    if (!item?.stats) continue;

    // Add each stat modifier
    for (const [stat, value] of Object.entries(item.stats)) {
      if (stats[stat] !== undefined) {
        stats[stat] += value;
      }
    }
  }

  // Ensure minimums
  stats.damage = Math.max(1, stats.damage);
  stats.defense = Math.max(0, stats.defense);
  stats.speed = Math.max(1, stats.speed);
  stats.maxHealth = Math.max(1, stats.maxHealth);
  stats.maxMana = Math.max(0, stats.maxMana);

  return stats;
}

/**
 * Get pattern override from equipped weapon
 * @param {Object} equipment - Equipment slots
 * @returns {Object|null} Pattern override or null
 */
export function getPatternOverride(equipment) {
  const weaponId = equipment?.weapon?.id || equipment?.weapon;
  if (!weaponId) return null;

  const item = typeof weaponId === 'object' ? weaponId : getItem(weaponId);
  return item?.pattern || null;
}

/**
 * Get bullet modifiers from all equipment
 * @param {Object} equipment - Equipment slots
 * @returns {Object} Combined bullet modifiers
 */
export function getBulletModifiers(equipment) {
  const mods = {
    piercing: 0,
    explosive: false,
    explosionRadius: 0,
    lifetimeBonus: 0,
    speedBonus: 0,
    damageBonus: 0
  };

  const slots = ['weapon', 'armor', 'ability', 'ring'];
  for (const slot of slots) {
    const itemId = equipment?.[slot]?.id || equipment?.[slot];
    if (!itemId) continue;

    const item = typeof itemId === 'object' ? itemId : getItem(itemId);
    if (!item?.bulletMods) continue;

    // Combine modifiers
    for (const [mod, value] of Object.entries(item.bulletMods)) {
      if (typeof value === 'boolean') {
        mods[mod] = mods[mod] || value;  // OR for booleans
      } else if (typeof value === 'number') {
        mods[mod] = (mods[mod] || 0) + value;  // Sum for numbers
      }
    }
  }

  return mods;
}

/**
 * Apply stat modifiers to create a modified weapon config
 * @param {Object} baseWeapon - Class weapon from ClassWeapons.js
 * @param {Object} equipment - Equipment slots
 * @returns {Object} Modified weapon config
 */
export function applyEquipmentToWeapon(baseWeapon, equipment) {
  const weapon = { ...baseWeapon };

  // Get pattern override from weapon
  const patternOverride = getPatternOverride(equipment);
  if (patternOverride) {
    weapon.pattern = patternOverride.type;
    weapon.patternConfig = patternOverride.config;
  }

  // Get bullet modifiers
  const bulletMods = getBulletModifiers(equipment);

  // Apply lifetime and speed bonuses
  if (bulletMods.lifetimeBonus) {
    weapon.projectileLifetime = (weapon.projectileLifetime || 0.5) + bulletMods.lifetimeBonus;
  }
  if (bulletMods.speedBonus) {
    weapon.projectileSpeed = (weapon.projectileSpeed || 10) + bulletMods.speedBonus;
  }

  // Store bullet mods for later application
  weapon.bulletMods = bulletMods;

  return weapon;
}

/**
 * Calculate effective rate of fire
 * @param {number} baseRate - Base rate of fire from weapon
 * @param {number} attackSpeedBonus - Attack speed bonus from stats
 * @returns {number} Effective rate of fire
 */
export function calculateRateOfFire(baseRate, attackSpeedBonus) {
  // attackSpeedBonus is additive to rate of fire
  // e.g., baseRate 1.5 + bonus 0.5 = 2.0 attacks/sec
  return Math.max(0.5, baseRate + (attackSpeedBonus || 0));
}
