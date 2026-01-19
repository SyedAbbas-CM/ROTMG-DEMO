/**
 * World Enemy Spawn Configuration
 *
 * This file defines enemy spawns for each world/map.
 * Easy to edit - just add enemy objects with { id, x, y } format.
 *
 * NEW Enemy Types (5 tiers):
 * - 'imp'          : Tier 1 - Fast, weak (50 HP), single shots
 * - 'skeleton'     : Tier 2 - Medium (150 HP), 2-bullet spread
 * - 'beholder'     : Tier 3 - Tanky (400 HP), 4-way spread
 * - 'red_demon'    : Tier 4 - Strong (800 HP), 3-shot burst
 * - 'green_dragon' : Tier 5 - Boss (2000 HP), 8-bullet ring
 * - 'boss_enemy'   : Pattern Boss (5000 HP), AI patterns
 */

export const worldSpawns = {

  // ========================================
  // OVERWORLD (Procedural Generated World)
  // ========================================
  overworld: {
    description: "Mixed enemy tiers for testing progression",
    spawns: [
      // Tier 1 - Imps (weak, fast scouts)
      { id: 'imp', x: 15, y: 25, comment: 'Imp Scout 1' },
      { id: 'imp', x: 20, y: 25, comment: 'Imp Scout 2' },
      { id: 'imp', x: 25, y: 25, comment: 'Imp Scout 3' },
      { id: 'imp', x: 30, y: 25, comment: 'Imp Scout 4' },
      { id: 'imp', x: 35, y: 25, comment: 'Imp Scout 5' },

      // Tier 2 - Skeletons (medium fighters)
      { id: 'skeleton', x: 18, y: 32, comment: 'Skeleton Warrior 1' },
      { id: 'skeleton', x: 24, y: 32, comment: 'Skeleton Warrior 2' },
      { id: 'skeleton', x: 30, y: 32, comment: 'Skeleton Warrior 3' },
      { id: 'skeleton', x: 36, y: 32, comment: 'Skeleton Warrior 4' },

      // Tier 3 - Beholders (tough ranged)
      { id: 'beholder', x: 22, y: 40, comment: 'Beholder 1' },
      { id: 'beholder', x: 32, y: 40, comment: 'Beholder 2' },

      // Tier 4 - Red Demon (elite)
      { id: 'red_demon', x: 50, y: 35, comment: 'Red Demon Elite' },

      // Tier 5 - Green Dragon (mini-boss)
      { id: 'green_dragon', x: 70, y: 40, comment: 'Green Dragon' },
    ]
  },

  // ========================================
  // RIVER BRIDGE DUNGEON
  // ========================================
  map_2: {
    description: "RiverBridge.json - Small bridge encounter",
    spawns: [
      { id: 'imp', x: 10, y: 15, comment: 'Bridge Imp 1' },
      { id: 'imp', x: 20, y: 15, comment: 'Bridge Imp 2' },
      { id: 'skeleton', x: 15, y: 20, comment: 'Bridge Guard' },
    ]
  },

  // ========================================
  // BOSS ROOM - Dragon Encounter
  // ========================================
  map_3: {
    description: "SampleBossRoom.json - Dragon boss encounter",
    spawns: [
      // The Dragon Boss - center of the room
      { id: 'green_dragon', x: 25, y: 25, comment: 'Green Dragon Boss' },
      // Guardian beholders
      { id: 'beholder', x: 15, y: 15, comment: 'Beholder Guard Left' },
      { id: 'beholder', x: 35, y: 15, comment: 'Beholder Guard Right' },
      // Skeleton sentinels
      { id: 'skeleton', x: 15, y: 35, comment: 'Skeleton Sentinel Left' },
      { id: 'skeleton', x: 35, y: 35, comment: 'Skeleton Sentinel Right' },
    ]
  },

  // ========================================
  // NEXUS (Safe Zone)
  // ========================================
  map_4: {
    description: "SampleNexus.json - Safe trading hub (no enemies)",
    spawns: [
      // Intentionally empty - safe zone
    ]
  },

  // ========================================
  // STARTING AREA - Easy Enemies
  // ========================================
  map_5: {
    description: "StartingArea.json - New player introduction",
    spawns: [
      // Only tier 1 enemies for new players
      { id: 'imp', x: 15, y: 15, comment: 'Training Imp 1' },
      { id: 'imp', x: 25, y: 15, comment: 'Training Imp 2' },
      { id: 'imp', x: 20, y: 25, comment: 'Training Imp 3' },
      // One skeleton for challenge
      { id: 'skeleton', x: 25, y: 30, comment: 'Skeleton Challenge' },
    ]
  },

  // ========================================
  // TEST MAP - All Enemy Types
  // ========================================
  map_6: {
    description: "test.json - All enemy types showcase",
    spawns: [
      // One of each tier for testing
      { id: 'imp', x: 10, y: 15, comment: 'Tier 1 - Imp' },
      { id: 'skeleton', x: 20, y: 15, comment: 'Tier 2 - Skeleton' },
      { id: 'beholder', x: 30, y: 15, comment: 'Tier 3 - Beholder' },
      { id: 'red_demon', x: 20, y: 25, comment: 'Tier 4 - Red Demon' },
      { id: 'green_dragon', x: 25, y: 35, comment: 'Tier 5 - Dragon' },
    ]
  },

};

/**
 * Helper function to get spawns for a specific world
 * @param {string} worldId - The world/map ID (e.g., 'map_1', 'overworld')
 * @returns {Array} Array of enemy spawn objects
 */
export function getWorldSpawns(worldId) {
  // Handle overworld special case
  if (worldId === 'map_1') {
    return worldSpawns.overworld?.spawns || [];
  }

  // Return spawns for the specified world
  return worldSpawns[worldId]?.spawns || [];
}

/**
 * Enemy Types Reference
 * All enemies are defined in public/assets/entities/enemies.json
 */
export const availableEnemies = [
  // Tier 1 - Trash mobs
  { id: 'imp', name: 'Fire Imp', hp: 50, tier: 1, description: 'Fast, weak, single shot' },

  // Tier 2 - Common enemies
  { id: 'skeleton', name: 'Skeleton Warrior', hp: 150, tier: 2, description: 'Medium HP, 2-bullet spread' },

  // Tier 3 - Strong enemies
  { id: 'beholder', name: 'Beholder', hp: 400, tier: 3, description: 'Tanky, 4-way spread attack' },

  // Tier 4 - Elite enemies
  { id: 'red_demon', name: 'Red Demon', hp: 800, tier: 4, description: 'Elite, 3-shot burst' },

  // Tier 5 - Mini-bosses
  { id: 'green_dragon', name: 'Green Dragon', hp: 2000, tier: 5, description: 'Boss, 8-bullet ring attack' },

  // Boss
  { id: 'boss_enemy', name: 'AI Pattern Boss', hp: 5000, tier: 6, description: 'Pattern boss with ML attacks' },
];

/**
 * Unit Types Reference (separate from enemies - for controllable units)
 * Units are defined in public/assets/entities/units.json
 */
export const availableUnits = [
  { id: 'light_infantry', name: 'Light Infantry', hp: 120, type: 'infantry' },
  { id: 'archer', name: 'Archer', hp: 60, type: 'ranged' },
  { id: 'light_cavalry', name: 'Light Cavalry', hp: 500, type: 'cavalry' },
  { id: 'heavy_cavalry', name: 'Heavy Cavalry', hp: 900, type: 'cavalry' },
  { id: 'heavy_infantry', name: 'Heavy Infantry', hp: 800, type: 'infantry' },
];
