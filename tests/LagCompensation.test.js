// File: tests/LagCompensation.test.js
// Unit tests for LagCompensation system

import { LagCompensation } from '../src/entities/LagCompensation.js';
import { CircularBuffer } from '../src/utils/CircularBuffer.js';

// Test runner helpers
let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
  if (!condition) {
    console.error(`❌ FAILED: ${message}`);
    testsFailed++;
    throw new Error(message);
  } else {
    console.log(`✅ PASSED: ${message}`);
    testsPassed++;
  }
}

function assertApprox(actual, expected, tolerance, message) {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    console.error(`❌ FAILED: ${message} (expected ${expected}, got ${actual}, diff ${diff})`);
    testsFailed++;
    throw new Error(message);
  } else {
    console.log(`✅ PASSED: ${message}`);
    testsPassed++;
  }
}

// Helper to create mock player with position history
function createMockPlayer(id, x, y, rtt = 100) {
  return {
    id,
    x,
    y,
    rtt,
    positionHistory: new CircularBuffer(10)
  };
}

console.log('\n========== LAG COMPENSATION TESTS ==========\n');

// Test 1: Basic initialization
console.log('Test 1: Basic initialization');
{
  const lagComp = new LagCompensation({
    enabled: true,
    maxRewindMs: 200,
    minRTT: 50,
    debug: false
  });

  assert(lagComp.enabled === true, 'Should be enabled');
  assert(lagComp.maxRewindMs === 200, 'Max rewind should be 200ms');
  assert(lagComp.minRTT === 50, 'Min RTT should be 50ms');
}

// Test 2: Calculate rewind amount - below threshold
console.log('\nTest 2: Calculate rewind amount - below threshold');
{
  const lagComp = new LagCompensation({ minRTT: 50 });

  const rewind = lagComp.calculateRewindAmount(30);
  assert(rewind === 0, 'Should not rewind for RTT below threshold');
}

// Test 3: Calculate rewind amount - normal RTT
console.log('\nTest 3: Calculate rewind amount - normal RTT');
{
  const lagComp = new LagCompensation({ minRTT: 50 });

  const rewind = lagComp.calculateRewindAmount(100);
  assertApprox(rewind, 50, 0.001, 'Should rewind by half RTT (50ms)');
}

// Test 4: Calculate rewind amount - high RTT
console.log('\nTest 4: Calculate rewind amount - high RTT');
{
  const lagComp = new LagCompensation({ minRTT: 50 });

  const rewind = lagComp.calculateRewindAmount(300);
  assertApprox(rewind, 150, 0.001, 'Should rewind by half RTT (150ms)');
}

// Test 5: Calculate rewind amount - clamped to max
console.log('\nTest 5: Calculate rewind amount - clamped to max');
{
  const lagComp = new LagCompensation({ minRTT: 50, maxRewindMs: 200 });

  const rewind = lagComp.calculateRewindAmount(1000); // RTT = 1000ms
  assertApprox(rewind, 200, 0.001, 'Should clamp to max rewind (200ms)');
}

// Test 6: Rewind single player position
console.log('\nTest 6: Rewind single player position');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 0 });
  const player = createMockPlayer('player1', 100, 50, 200);

  // Add position history (player was at 90, 50 at t=1000)
  player.positionHistory.add(90, 50, 1000);
  player.positionHistory.add(95, 50, 1050);
  player.positionHistory.add(100, 50, 1100);

  // Rewind by 100ms from t=1100
  const result = lagComp.rewindPlayerPosition(player, 100, 1100);

  assert(result.rewound === true, 'Should be rewound');
  assert(result.found === true, 'Should find position in history');
  assertApprox(result.x, 90, 0.5, 'Rewound X should be approximately 90');
  assertApprox(result.y, 50, 0.5, 'Rewound Y should be 50');
  assertApprox(result.originalX, 100, 0.001, 'Original X should be preserved');
  assertApprox(result.originalY, 50, 0.001, 'Original Y should be preserved');
}

// Test 7: Rewind disabled
console.log('\nTest 7: Rewind when disabled');
{
  const lagComp = new LagCompensation({ enabled: false });
  const player = createMockPlayer('player1', 100, 50);

  const result = lagComp.rewindPlayerPosition(player, 100, 1100);

  assert(result.rewound === false, 'Should not rewind when disabled');
  assertApprox(result.x, 100, 0.001, 'Should return current X');
  assertApprox(result.y, 50, 0.001, 'Should return current Y');
}

// Test 8: Rewind all players
console.log('\nTest 8: Rewind all players');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 0 });

  const player1 = createMockPlayer('p1', 100, 50);
  const player2 = createMockPlayer('p2', 200, 100);

  // Add history for both players
  player1.positionHistory.add(90, 45, 1000);
  player1.positionHistory.add(100, 50, 1100);

  player2.positionHistory.add(180, 90, 1000);
  player2.positionHistory.add(200, 100, 1100);

  const players = [player1, player2];

  // Rewind both by 50ms from t=1100
  const originalPositions = lagComp.rewindAllPlayers(players, 50, 1100);

  assert(originalPositions.size === 2, 'Should store 2 original positions');

  // Check player1 rewound position
  assertApprox(player1.x, 95, 1, 'Player1 X should be rewound to ~95');
  assertApprox(player1.y, 47.5, 1, 'Player1 Y should be rewound to ~47.5');

  // Check player2 rewound position
  assertApprox(player2.x, 190, 1, 'Player2 X should be rewound to ~190');
  assertApprox(player2.y, 95, 1, 'Player2 Y should be rewound to ~95');

  // Check original positions stored
  const p1Original = originalPositions.get('p1');
  assertApprox(p1Original.x, 100, 0.001, 'Player1 original X stored');
  assertApprox(p1Original.y, 50, 0.001, 'Player1 original Y stored');
}

// Test 9: Restore all players
console.log('\nTest 9: Restore all players');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 0 });

  const player1 = createMockPlayer('p1', 100, 50);
  const player2 = createMockPlayer('p2', 200, 100);

  player1.positionHistory.add(90, 45, 1000);
  player1.positionHistory.add(100, 50, 1100);
  player2.positionHistory.add(180, 90, 1000);
  player2.positionHistory.add(200, 100, 1100);

  const players = [player1, player2];

  // Rewind
  const originalPositions = lagComp.rewindAllPlayers(players, 50, 1100);

  // Verify rewound
  assert(player1.x !== 100, 'Player1 should be rewound');

  // Restore
  lagComp.restoreAllPlayers(players, originalPositions);

  // Verify restored
  assertApprox(player1.x, 100, 0.001, 'Player1 X should be restored');
  assertApprox(player1.y, 50, 0.001, 'Player1 Y should be restored');
  assertApprox(player2.x, 200, 0.001, 'Player2 X should be restored');
  assertApprox(player2.y, 100, 0.001, 'Player2 Y should be restored');
}

// Test 10: Statistics tracking
console.log('\nTest 10: Statistics tracking');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 0 });

  const player = createMockPlayer('p1', 100, 50);
  player.positionHistory.add(90, 50, 1000);
  player.positionHistory.add(100, 50, 1100);

  // Perform some rewinds
  lagComp.rewindPlayerPosition(player, 50, 1100);
  lagComp.rewindPlayerPosition(player, 75, 1100);
  lagComp.rewindPlayerPosition(player, 100, 1100);

  const stats = lagComp.getStats();
  assert(stats.rewindsPerformed === 3, 'Should track 3 rewinds');
  assertApprox(stats.avgRewindAmount, 75, 0.1, 'Average rewind should be 75ms');
  assert(stats.maxRewindAmount === 100, 'Max rewind should be 100ms');
}

// Test 11: Empty position history
console.log('\nTest 11: Rewind with empty position history');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 0 });

  const player = createMockPlayer('p1', 100, 50);
  // No history added

  const result = lagComp.rewindPlayerPosition(player, 100, 1100);

  assert(result.found === false, 'Should not find position in empty history');
  // Should still return current position as fallback
  assertApprox(result.x, 100, 0.001, 'Should return current X as fallback');
  assertApprox(result.y, 50, 0.001, 'Should return current Y as fallback');
}

// Test 12: Rewind amount of zero
console.log('\nTest 12: Rewind amount of zero');
{
  const lagComp = new LagCompensation({ enabled: true });

  const player = createMockPlayer('p1', 100, 50);
  player.positionHistory.add(90, 50, 1000);

  const result = lagComp.rewindPlayerPosition(player, 0, 1100);

  assert(result.rewound === false, 'Should not rewind when amount is 0');
  assertApprox(result.x, 100, 0.001, 'Should return current position');
}

// Test 13: Realistic gameplay scenario
console.log('\nTest 13: Realistic gameplay scenario (300ms RTT)');
{
  const lagComp = new LagCompensation({
    enabled: true,
    minRTT: 50,
    maxRewindMs: 200
  });

  const player = createMockPlayer('p1', 100, 50, 300);

  // Simulate player movement over 300ms at 30Hz
  const currentTime = 10000; // Use fixed timestamp for predictability
  const positions = [];

  // Add position history: moving from x=90 to x=99 over 297ms
  for (let i = 0; i < 10; i++) {
    const t = currentTime - (300 - i * 33); // Spread over ~300ms in the past
    const x = 90 + i; // Moving from 90 to 99
    player.positionHistory.add(x, 50, t);
    positions.push({ x, y: 50, t });
  }

  player.x = 100; // Current position (at currentTime)
  player.y = 50;

  // Calculate rewind for 300ms RTT
  const rewindAmount = lagComp.calculateRewindAmount(300);
  assertApprox(rewindAmount, 150, 0.001, 'Should rewind by 150ms (RTT/2)');

  // Rewind the player by 150ms
  const result = lagComp.rewindPlayerPosition(player, rewindAmount, currentTime);

  // Debug: Check history
  const historyDebug = player.positionHistory.getDebugInfo();
  console.log(`  History count: ${historyDebug.count}, oldest=${historyDebug.oldestTimestamp}, newest=${historyDebug.newestTimestamp}`);
  console.log(`  Current time: ${currentTime}, rewind amount: ${rewindAmount}`);
  console.log(`  Target timestamp: ${currentTime - rewindAmount}`);
  console.log(`  Result found: ${result.found}, x: ${result.x.toFixed(2)}, y: ${result.y}`);

  // Result should be interpolated position 150ms ago
  assert(result.x < 100, 'Rewound X should be less than current (100)');
  assert(result.x >= 90, 'Rewound X should be >= start (90)');

  console.log(`  Rewound from (100, 50) to (${result.x.toFixed(2)}, ${result.y.toFixed(2)})`);
}

// Test 14: Reset stats
console.log('\nTest 14: Reset statistics');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 0 });

  const player = createMockPlayer('p1', 100, 50);
  player.positionHistory.add(90, 50, 1000);
  player.positionHistory.add(100, 50, 1100);

  lagComp.rewindPlayerPosition(player, 50, 1100);

  let stats = lagComp.getStats();
  assert(stats.rewindsPerformed === 1, 'Should have 1 rewind before reset');

  lagComp.resetStats();

  stats = lagComp.getStats();
  assert(stats.rewindsPerformed === 0, 'Should have 0 rewinds after reset');
  assert(stats.avgRewindAmount === 0, 'Avg rewind should be 0 after reset');
  assert(stats.maxRewindAmount === 0, 'Max rewind should be 0 after reset');
}

// Test 15: Multiple players with different RTTs
console.log('\nTest 15: Multiple players with different RTTs');
{
  const lagComp = new LagCompensation({ enabled: true, minRTT: 50 });

  const lowPingPlayer = createMockPlayer('low', 100, 50, 60);
  const highPingPlayer = createMockPlayer('high', 200, 100, 400);

  // Add history
  lowPingPlayer.positionHistory.add(95, 50, 1000);
  lowPingPlayer.positionHistory.add(100, 50, 1100);

  highPingPlayer.positionHistory.add(150, 100, 1000);
  highPingPlayer.positionHistory.add(200, 100, 1100);

  // Low ping player - should get small rewind
  const lowPingRewind = lagComp.calculateRewindAmount(60);
  assertApprox(lowPingRewind, 30, 0.001, 'Low ping rewind should be 30ms');

  // High ping player - should get larger rewind
  const highPingRewind = lagComp.calculateRewindAmount(400);
  assertApprox(highPingRewind, 200, 0.001, 'High ping rewind should be clamped to 200ms');
}

// Test 16: Restore with empty originalPositions map
console.log('\nTest 16: Restore with empty original positions');
{
  const lagComp = new LagCompensation({ enabled: true });

  const player = createMockPlayer('p1', 100, 50);
  const players = [player];
  const emptyMap = new Map();

  // Should not crash
  lagComp.restoreAllPlayers(players, emptyMap);

  // Position should be unchanged
  assertApprox(player.x, 100, 0.001, 'Player X should be unchanged');
  assertApprox(player.y, 50, 0.001, 'Player Y should be unchanged');
}

// Summary
console.log('\n========== TEST SUMMARY ==========');
console.log(`✅ Passed: ${testsPassed}`);
console.log(`❌ Failed: ${testsFailed}`);
console.log(`Total: ${testsPassed + testsFailed}`);
console.log('==================================\n');

if (testsFailed > 0) {
  process.exit(1);
}
