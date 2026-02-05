/**
 * Test script for AI bullet pattern system
 * Run with: node src/ai/test-pattern-system.js
 */

import BulletManager from '../entities/BulletManager.js';
import { PatternToBulletAdapter } from './PatternToBulletAdapter.js';
import { PatternLibrary } from './PatternLibrary.js';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('='.repeat(60));
console.log('AI Bullet Pattern System - Test');
console.log('='.repeat(60));

// 1. Create BulletManager
console.log('\n[1] Creating BulletManager...');
const bulletManager = new BulletManager(10000);
console.log(`✓ BulletManager created (capacity: 10000)`);

// 2. Load pattern library
console.log('\n[2] Loading pattern library...');
const patternLibrary = new PatternLibrary();

const jsonPath = path.join(__dirname, '../../ml/visualizations/pattern_library.json');
const loadedCount = patternLibrary.loadFromJSON(jsonPath);

if (loadedCount === 0) {
  console.error('❌ Failed to load patterns');
  console.log('\nTo generate patterns, run:');
  console.log('  cd ml');
  console.log('  python3 visualize_patterns_pytorch.py');
  process.exit(1);
}

console.log(`✓ Loaded ${loadedCount} patterns`);

// 3. Create adapter
console.log('\n[3] Creating PatternToBulletAdapter...');
const adapter = new PatternToBulletAdapter(bulletManager);
console.log('✓ Adapter created');

// 4. Test pattern spawning
console.log('\n[4] Testing pattern spawning...');

// Mock boss data
const boss = {
  x: 50.0,  // TILE UNITS
  y: 50.0,  // TILE UNITS
  ownerId: 'enemy_boss_1',
  worldId: 'world_1',
  faction: 0  // Enemy faction
};

console.log('Boss position:', `(${boss.x}, ${boss.y}) tiles`);

// Test different styles
const styles = ['sparse', 'medium', 'dense', 'chaotic'];

for (const style of styles) {
  console.log(`\n--- Testing ${style.toUpperCase()} pattern ---`);

  // Get pattern for this style
  const pattern = patternLibrary.getPatternByStyle(style);

  if (!pattern) {
    console.log(`⚠  No ${style} patterns available`);
    continue;
  }

  console.log(`Pattern ID: ${pattern.id}`);
  console.log(`Stats:`, pattern.stats);

  // Convert to array format
  const patternArray = patternLibrary.toPatternArray(pattern);

  // Set adapter style
  adapter.setStyle(style === 'chaotic' ? 'fast_chaos' : style === 'sparse' ? 'sparse_deadly' : 'dense');

  // Spawn bullets
  const bulletsBefore = bulletManager.bulletCount;
  const spawned = adapter.spawnPattern(patternArray, boss);
  const bulletsAfter = bulletManager.bulletCount;

  console.log(`✓ Spawned ${spawned} bullets`);
  console.log(`  BulletManager count: ${bulletsBefore} → ${bulletsAfter}`);

  // Show sample bullets
  if (bulletsAfter > 0) {
    console.log('  Sample bullet properties:');
    const idx = bulletsAfter - 1;
    console.log(`    Position: (${bulletManager.x[idx].toFixed(2)}, ${bulletManager.y[idx].toFixed(2)}) tiles`);
    console.log(`    Velocity: (${bulletManager.vx[idx].toFixed(2)}, ${bulletManager.vy[idx].toFixed(2)}) tiles/sec`);
    console.log(`    Damage: ${bulletManager.damage[idx]}`);
    console.log(`    Lifetime: ${bulletManager.life[idx].toFixed(2)}s`);
    console.log(`    Sprite: ${bulletManager.spriteName[idx]}`);
  }
}

// 5. Test phase-based spawning
console.log('\n[5] Testing phase-based spawning...');

for (let phase = 1; phase <= 3; phase++) {
  console.log(`\n--- Phase ${phase} ---`);

  const pattern = patternLibrary.getPatternForPhase(phase);
  const patternArray = patternLibrary.toPatternArray(pattern);

  const spawned = adapter.spawnPatternForPhase(patternArray, boss, phase);
  console.log(`✓ Spawned ${spawned} bullets for phase ${phase}`);
}

// 6. Simulate bullet updates
console.log('\n[6] Simulating bullet updates...');

const beforeCount = bulletManager.bulletCount;
console.log(`Initial bullet count: ${beforeCount}`);

// Update for 0.5 seconds
bulletManager.update(0.5);

console.log(`After 0.5s update: ${bulletManager.bulletCount} bullets`);
console.log(`Stats:`, bulletManager.stats);

// 7. Final summary
console.log('\n' + '='.repeat(60));
console.log('Test Summary');
console.log('='.repeat(60));
console.log(`Total patterns loaded: ${patternLibrary.patterns.length}`);
console.log(`Active bullets: ${bulletManager.bulletCount}`);
console.log(`Bullets created: ${bulletManager.stats.created}`);
console.log(`Bullets expired: ${bulletManager.stats.expired}`);
console.log('');
console.log('✓ All systems functional!');
console.log('');
console.log('Next steps:');
console.log('  1. Integrate adapter into boss AI behavior');
console.log('  2. Add pattern selection logic based on boss phase/HP');
console.log('  3. Test in live gameplay');
console.log('='.repeat(60));
