
/**
 *  unitTypes.js  – authoritative per‑unit templates
 *  Add or tweak values here and the sim will adapt automatically.
 *
 *  Note: keep the prop list in sync with UnitManager’s column order.
 */
export const UnitTypes = {
    INFANTRY_LIGHT: {
      width: 10, height: 10,
      maxHealth: 100, armor: 5,
      moveSpeed: 55,   acceleration: 220,
      attackRange: 30, attackCooldown: 0.9, attackDamage: 12,
      momentum: 1.0, morale: 70,
      projectileSpeed: 0,  // 0 = melee
    },
    INFANTRY_HEAVY: {
      width: 12, height: 12,
      maxHealth: 160, armor: 15,
      moveSpeed: 40,  acceleration: 140,
      attackRange: 32, attackCooldown: 1.1, attackDamage: 18,
      momentum: 1.4, morale: 90,
      projectileSpeed: 0,
    },
    CAVALRY_LIGHT: {
      width: 14, height: 14,
      maxHealth: 120, armor: 8,
      moveSpeed: 85,  acceleration: 320,
      attackRange: 30, attackCooldown: 1.0, attackDamage: 15,
      momentum: 2.0, morale: 80,
      projectileSpeed: 0,
    },
    CAVALRY_HEAVY: {
      width: 16, height: 16,
      maxHealth: 180, armor: 18,
      moveSpeed: 70,  acceleration: 260,
      attackRange: 32, attackCooldown: 1.2, attackDamage: 24,
      momentum: 3.0, morale: 95,
      projectileSpeed: 0,
    },
    ARCHER_LIGHT: {
      width: 10, height: 10,
      maxHealth: 80, armor: 2,
      moveSpeed: 58,  acceleration: 210,
      attackRange: 220, attackCooldown: 1.4, attackDamage: 10,
      momentum: 0.6, morale: 60,
      projectileSpeed: 180,
    },
    ARCHER_HEAVY: {
      width: 12, height: 12,
      maxHealth: 110, armor: 6,
      moveSpeed: 48,  acceleration: 180,
      attackRange: 260, attackCooldown: 1.7, attackDamage: 14,
      momentum: 0.8, morale: 70,
      projectileSpeed: 190,
    }
  };

  // Add __keys array for numeric index lookup (used by UnitSystems)
  UnitTypes.__keys = Object.keys(UnitTypes);

  export const UnitTypeKeys = UnitTypes.__keys;   // handy for random spawn
  