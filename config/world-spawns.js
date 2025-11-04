/**
 * World Enemy Spawn Configuration
 *
 * This file defines enemy spawns for each world/map.
 * Easy to edit - just add enemy objects with { id, x, y } format.
 *
 * Available Enemy Types:
 * - 'goblin'           : Basic chase enemy, 30 HP, single shots
 * - 'charging_shooter' : Charges while shooting rapidly, 80 HP
 *
 * Add your custom enemies after creating them in the behavior designer!
 */

export const worldSpawns = {

  // ========================================
  // OVERWORLD (Procedural Generated World)
  // ========================================
  overworld: {
    description: "Main procedural overworld - test area for 4 enemy types",
    spawns: [
      // === 4 Basic Enemy Types ===
      { id: 'goblin', x: 25, y: 25, comment: 'Infantry - Tank' },
      { id: 'archer', x: 35, y: 25, comment: 'Archer - Long Range' },
      { id: 'red_demon', x: 45, y: 25, comment: 'Cavalry - Charger' },
      { id: 'heavy_knight', x: 55, y: 25, comment: 'Knight - Ultra Tank' }
    ]
  },

  // ========================================
  // RIVER BRIDGE DUNGEON
  // ========================================
  map_2: {
    description: "RiverBridge.json - Small bridge encounter",
    spawns: [
      // Empty for now
    ]
  },

  // ========================================
  // BOSS ROOM
  // ========================================
  map_3: {
    description: "SampleBossRoom.json - Major boss encounter",
    spawns: [
      // Empty for now
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
  // TEST DUNGEON
  // ========================================
  map_5: {
    description: "TestDungeon.json - Enemy variety testing ground",
    spawns: [
      // Empty for now - add your custom enemies here for testing
    ]
  },

  // ========================================
  // TEST MAP
  // ========================================
  map_6: {
    description: "test.json - Small test arena for custom enemies",
    spawns: [
      // Empty for now - perfect place to test your new custom enemy!
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
 * Get all available enemy types
 * Note: Add your custom enemies here after creating them in the behavior designer
 */
export const availableEnemies = [
  { id: 'goblin', name: 'Goblin', hp: 30, description: 'Basic chase enemy' },
  { id: 'charging_shooter', name: 'Charging Shooter', hp: 80, description: 'Charges while shooting' },
  // Add your custom enemies here:
  // { id: 'my_custom_enemy', name: 'My Enemy', hp: 100, description: 'My custom enemy' },
];
