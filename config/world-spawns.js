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
    description: "Battle formation: 5 cavalry, 10 soldiers, 5 archers",
    spawns: [
      // Front line: 5 Light Cavalry (red_demon)
      { id: 'red_demon', x: 20, y: 30, comment: 'Cavalry Left Flank' },
      { id: 'red_demon', x: 25, y: 30, comment: 'Cavalry Left' },
      { id: 'red_demon', x: 30, y: 30, comment: 'Cavalry Center' },
      { id: 'red_demon', x: 35, y: 30, comment: 'Cavalry Right' },
      { id: 'red_demon', x: 40, y: 30, comment: 'Cavalry Right Flank' },

      // Middle line: 10 Infantry (goblin soldiers)
      { id: 'goblin', x: 18, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 22, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 26, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 30, y: 35, comment: 'Infantry Center' },
      { id: 'goblin', x: 34, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 38, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 42, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 46, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 50, y: 35, comment: 'Infantry' },
      { id: 'goblin', x: 54, y: 35, comment: 'Infantry' },

      // Back line: 5 Archers
      { id: 'archer', x: 25, y: 40, comment: 'Archer Left' },
      { id: 'archer', x: 30, y: 40, comment: 'Archer Left Center' },
      { id: 'archer', x: 35, y: 40, comment: 'Archer Center' },
      { id: 'archer', x: 40, y: 40, comment: 'Archer Right Center' },
      { id: 'archer', x: 45, y: 40, comment: 'Archer Right' },

      // NEW CUSTOM ENEMIES (for testing)
      { id: 'necromancer', x: 60, y: 30, comment: 'Necromancer - summons skeletons' },
      { id: 'berserker', x: 65, y: 30, comment: 'Berserker - rages at low HP' },
      { id: 'skeleton_minion', x: 55, y: 35, comment: 'Skeleton 1' },
      { id: 'skeleton_minion', x: 70, y: 35, comment: 'Skeleton 2' },
      { id: 'lich_king', x: 80, y: 40, comment: 'Lich King - 3 phase boss' },
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
  // BOSS ROOM - Lich King Encounter
  // ========================================
  map_3: {
    description: "SampleBossRoom.json - Lich King boss encounter",
    spawns: [
      // The Lich King - center of the room
      { id: 'lich_king', x: 25, y: 25, comment: 'Lich King Boss' },
      // Guardian necromancers
      { id: 'necromancer', x: 15, y: 15, comment: 'Necromancer Guard Left' },
      { id: 'necromancer', x: 35, y: 15, comment: 'Necromancer Guard Right' },
      // Berserker sentinels
      { id: 'berserker', x: 15, y: 35, comment: 'Berserker Sentinel Left' },
      { id: 'berserker', x: 35, y: 35, comment: 'Berserker Sentinel Right' },
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
  // TEST DUNGEON - New Enemy Testing
  // ========================================
  map_5: {
    description: "TestDungeon.json - Enemy variety testing ground",
    spawns: [
      // Necromancers (casters that summon skeletons)
      { id: 'necromancer', x: 20, y: 20, comment: 'Necromancer 1' },
      { id: 'necromancer', x: 30, y: 20, comment: 'Necromancer 2' },
      // Berserkers (melee brutes that rage at low HP)
      { id: 'berserker', x: 25, y: 30, comment: 'Berserker 1' },
      { id: 'berserker', x: 35, y: 30, comment: 'Berserker 2' },
      // Skeleton minions
      { id: 'skeleton_minion', x: 15, y: 25, comment: 'Skeleton 1' },
      { id: 'skeleton_minion', x: 40, y: 25, comment: 'Skeleton 2' },
      { id: 'skeleton_minion', x: 25, y: 15, comment: 'Skeleton 3' },
    ]
  },

  // ========================================
  // TEST MAP - Custom Enemy Testing
  // ========================================
  map_6: {
    description: "test.json - Small test arena for custom enemies",
    spawns: [
      // Test the new custom enemies defined in public/assets/enemies-custom/
      { id: 'fire_mage', x: 20, y: 20, comment: 'Fire Mage - Triple shot caster' },
      { id: 'fire_mage', x: 25, y: 20, comment: 'Fire Mage 2' },
      { id: 'shadow_assassin', x: 30, y: 25, comment: 'Shadow Assassin - Fast dashing attacker' },
      { id: 'crystal_guardian', x: 25, y: 30, comment: 'Crystal Guardian - Multi-phase boss' },
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
 * Get all available enemy types (from public/assets/entities/enemies.json)
 * To add new enemies, edit enemies.json and restart the server.
 */
export const availableEnemies = [
  // Infantry Units
  { id: 'goblin', name: 'Light Infantry', hp: 120, description: 'Basic melee unit, moderate damage' },
  { id: 'heavy_knight', name: 'Heavy Infantry', hp: 800, description: 'Tanky melee unit with axe attacks' },

  // Ranged Units
  { id: 'archer', name: 'Archer', hp: 60, description: 'Long range, high single-shot damage' },

  // Cavalry Units
  { id: 'red_demon', name: 'Light Cavalry', hp: 500, description: 'Fast, multi-projectile attacks' },
  { id: 'heavy_cavalry', name: 'Heavy Cavalry', hp: 900, description: 'Heavily armored, devastating charge' },

  // Boss
  { id: 'enemy_8', name: 'AI Pattern Boss', hp: 5000, description: 'Large boss, uses ML patterns' },

  // Custom Enemies (loaded from public/assets/enemies-custom/*.enemy.json)
  { id: 'fire_mage', name: 'Fire Mage', hp: 200, description: 'Caster with triple-shot attack, retreats when close' },
  { id: 'crystal_guardian', name: 'Crystal Guardian', hp: 1500, description: 'Multi-phase boss with increasing aggression' },
  { id: 'shadow_assassin', name: 'Shadow Assassin', hp: 100, description: 'Fast, teleporting assassin with dash attacks' },

  // New Enemies (Jan 2026)
  { id: 'necromancer', name: 'Necromancer', hp: 150, description: 'Dark caster, summons skeletons, flees when close' },
  { id: 'berserker', name: 'Berserker', hp: 400, description: 'Melee brute, rages at low HP' },
  { id: 'skeleton_minion', name: 'Skeleton Minion', hp: 40, description: 'Summoned minion, weak but numerous' },
  { id: 'lich_king', name: 'Lich King', hp: 3000, description: '3-phase boss: orbit->spiral->rage' },
];
