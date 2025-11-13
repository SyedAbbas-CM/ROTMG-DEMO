// File: tests/LagCompensation.integration.test.js
// Integration test for lag compensation with collision detection

import { LagCompensation } from '../src/entities/LagCompensation.js';
import { CircularBuffer } from '../src/utils/CircularBuffer.js';

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

// Mock AABB collision check
function checkAABBCollision(x1, y1, w1, h1, x2, y2, w2, h2) {
  return !(x1 + w1 < x2 || x1 > x2 + w2 || y1 + h1 < y2 || y1 > y2 + h2);
}

console.log('\n========== LAG COMPENSATION INTEGRATION TEST ==========\n');

// Scenario 1: Player shoots enemy with 300ms latency
console.log('Scenario 1: High-latency player shoots moving enemy');
{
  const lagComp = new LagCompensation({
    enabled: true,
    minRTT: 50,
    maxRewindMs: 200
  });

  // Create player (high RTT)
  const player = {
    id: 'player1',
    x: 100,
    y: 50,
    rtt: 300, // 300ms RTT
    positionHistory: new CircularBuffer(10)
  };

  // Create enemy (moving target)
  const enemy = {
    id: 'enemy1',
    x: 110, // Current position
    y: 50,
    width: 1,
    height: 1
  };

  // Simulate enemy movement history (moving right)
  const currentTime = 10000;
  for (let i = 0; i < 10; i++) {
    const t = currentTime - (300 - i * 33);
    const x = 100 + i; // Was at x=100, now at x=109
    // Pretend enemy is a player with position history
    if (!enemy.positionHistory) {
      enemy.positionHistory = new CircularBuffer(10);
      enemy.rtt = 100; // Enemy has lower latency
    }
    enemy.positionHistory.add(x, 50, t);
  }

  // Player shoots bullet at where they SEE the enemy (150ms ago due to latency)
  const playerPerceivedEnemyX = 105; // Player sees enemy at older position
  const bullet = {
    x: playerPerceivedEnemyX,
    y: 50,
    width: 0.5,
    height: 0.5
  };

  console.log(`\n  Setup:`);
  console.log(`  - Player RTT: ${player.rtt}ms`);
  console.log(`  - Enemy current position: (${enemy.x}, ${enemy.y})`);
  console.log(`  - Enemy historical positions: 100 → 109`);
  console.log(`  - Bullet position: (${bullet.x}, ${bullet.y})`);
  console.log(`  - Player perceived enemy at: (${playerPerceivedEnemyX}, 50)`);

  // WITHOUT lag compensation - bullet misses
  console.log(`\n  WITHOUT lag compensation:`);
  const hitWithoutLagComp = checkAABBCollision(
    bullet.x, bullet.y, bullet.width, bullet.height,
    enemy.x, enemy.y, enemy.width, enemy.height
  );
  console.log(`  - Bullet hits: ${hitWithoutLagComp} (checking against current enemy pos)`);

  // WITH lag compensation - rewind enemy position
  console.log(`\n  WITH lag compensation:`);
  const rewindAmount = lagComp.calculateRewindAmount(player.rtt);
  console.log(`  - Rewind amount: ${rewindAmount}ms (RTT/2)`);

  const originalPos = { x: enemy.x, y: enemy.y };
  const rewoundPos = lagComp.rewindPlayerPosition(enemy, rewindAmount, currentTime);

  console.log(`  - Enemy rewound from (${originalPos.x}, ${originalPos.y}) to (${rewoundPos.x.toFixed(2)}, ${rewoundPos.y})`);

  // Check collision against rewound position
  const hitWithLagComp = checkAABBCollision(
    bullet.x, bullet.y, bullet.width, bullet.height,
    rewoundPos.x, rewoundPos.y, enemy.width, enemy.height
  );

  console.log(`  - Bullet hits: ${hitWithLagComp} (checking against rewound enemy pos)`);

  // Verify lag compensation makes hit detection fairer
  assert(rewoundPos.x < enemy.x, 'Rewound position should be behind current position');
  assertApprox(rewoundPos.x, playerPerceivedEnemyX, 2, 'Rewound position should match what player saw');

  console.log(`\n  Result: Lag compensation ${hitWithLagComp ? 'HIT' : 'MISS'} vs no compensation ${hitWithoutLagComp ? 'HIT' : 'MISS'}`);
}

// Scenario 2: Multiple players with different latencies
console.log('\n\nScenario 2: Multiple players with varying latencies');
{
  const lagComp = new LagCompensation({
    enabled: true,
    minRTT: 50,
    maxRewindMs: 200
  });

  const players = [
    {
      id: 'low_ping',
      x: 100,
      y: 50,
      rtt: 60,
      positionHistory: new CircularBuffer(10)
    },
    {
      id: 'med_ping',
      x: 200,
      y: 100,
      rtt: 150,
      positionHistory: new CircularBuffer(10)
    },
    {
      id: 'high_ping',
      x: 300,
      y: 150,
      rtt: 350, // Will be clamped to maxRewindMs
      positionHistory: new CircularBuffer(10)
    }
  ];

  // Add position history for all players
  const currentTime = 10000;
  for (const player of players) {
    for (let i = 0; i < 10; i++) {
      const t = currentTime - (300 - i * 33);
      player.positionHistory.add(player.x - 10 + i, player.y, t);
    }
  }

  console.log(`\n  Players:`);
  players.forEach(p => console.log(`  - ${p.id}: RTT=${p.rtt}ms, pos=(${p.x}, ${p.y})`));

  // Rewind all players
  const originalPositions = new Map();
  console.log(`\n  Rewinding...`);

  for (const player of players) {
    originalPositions.set(player.id, { x: player.x, y: player.y });

    const rewindAmount = lagComp.calculateRewindAmount(player.rtt);
    const rewoundPos = lagComp.rewindPlayerPosition(player, rewindAmount, currentTime);

    player.x = rewoundPos.x;
    player.y = rewoundPos.y;

    console.log(`  - ${player.id}: ${rewindAmount}ms rewind, (${originalPositions.get(player.id).x}, ${originalPositions.get(player.id).y}) → (${player.x.toFixed(2)}, ${player.y})`);
  }

  // Verify rewind amounts are correct
  assert(players[0].x > originalPositions.get('low_ping').x - 2, 'Low ping should rewind less');
  assert(players[1].x < originalPositions.get('med_ping').x - 2, 'Med ping should rewind more');
  assert(players[2].x < originalPositions.get('high_ping').x - 5, 'High ping should rewind most (clamped)');

  // Restore all players
  console.log(`\n  Restoring...`);
  lagComp.restoreAllPlayers(players, originalPositions);

  players.forEach(p => {
    const original = originalPositions.get(p.id);
    assertApprox(p.x, original.x, 0.001, `${p.id} X should be restored`);
    assertApprox(p.y, original.y, 0.001, `${p.id} Y should be restored`);
    console.log(`  - ${p.id}: restored to (${p.x}, ${p.y})`);
  });
}

// Scenario 3: Collision detection with try-finally pattern
console.log('\n\nScenario 3: Safe rewind/restore with try-finally');
{
  const lagComp = new LagCompensation({
    enabled: true,
    minRTT: 50,
    maxRewindMs: 200
  });

  const player = {
    id: 'test_player',
    x: 100,
    y: 50,
    rtt: 200,
    positionHistory: new CircularBuffer(10)
  };

  // Add history
  const currentTime = 10000;
  for (let i = 0; i < 10; i++) {
    const t = currentTime - (300 - i * 33);
    player.positionHistory.add(90 + i, 50, t);
  }

  const players = [player];
  let originalPositions = null;

  console.log(`\n  Original position: (${player.x}, ${player.y})`);

  // Simulate collision check with proper error handling
  try {
    // Rewind
    const rewindAmount = lagComp.calculateRewindAmount(200);
    originalPositions = lagComp.rewindAllPlayers(players, rewindAmount, currentTime);

    console.log(`  Rewound position: (${player.x.toFixed(2)}, ${player.y})`);

    // Simulate some collision checks here...
    assert(player.x < 100, 'Position should be rewound');

    // Simulate an error during collision processing
    // (This would normally be a real collision check)

  } finally {
    // Always restore positions
    if (originalPositions) {
      lagComp.restoreAllPlayers(players, originalPositions);
      console.log(`  Restored position: (${player.x}, ${player.y})`);
    }
  }

  assertApprox(player.x, 100, 0.001, 'Position should be restored after try-finally');
}

// Scenario 4: Stress test - many players
console.log('\n\nScenario 4: Stress test with many players');
{
  const lagComp = new LagCompensation({
    enabled: true,
    minRTT: 50,
    maxRewindMs: 200
  });

  const numPlayers = 100;
  const players = [];

  // Create 100 players with random RTTs
  for (let i = 0; i < numPlayers; i++) {
    const player = {
      id: `player_${i}`,
      x: 100 + i,
      y: 50 + (i % 10),
      rtt: 50 + Math.random() * 300, // 50-350ms RTT
      positionHistory: new CircularBuffer(10)
    };

    // Add history
    const currentTime = 10000;
    for (let j = 0; j < 10; j++) {
      const t = currentTime - (300 - j * 33);
      player.positionHistory.add(player.x - 10 + j, player.y, t);
    }

    players.push(player);
  }

  console.log(`\n  Created ${numPlayers} players`);

  // Measure performance
  const startTime = Date.now();

  // Calculate average RTT
  let totalRTT = 0;
  for (const player of players) {
    totalRTT += player.rtt;
  }
  const avgRTT = totalRTT / players.length;
  const rewindAmount = lagComp.calculateRewindAmount(avgRTT);

  // Rewind all
  const originalPositions = lagComp.rewindAllPlayers(players, rewindAmount, 10000);

  // Restore all
  lagComp.restoreAllPlayers(players, originalPositions);

  const elapsed = Date.now() - startTime;

  console.log(`  Average RTT: ${avgRTT.toFixed(2)}ms`);
  console.log(`  Rewind amount: ${rewindAmount.toFixed(2)}ms`);
  console.log(`  Rewind + restore time: ${elapsed}ms for ${numPlayers} players`);
  console.log(`  Average per player: ${(elapsed / numPlayers).toFixed(4)}ms`);

  assert(elapsed < 100, `Should complete in < 100ms (took ${elapsed}ms)`);

  // Verify all positions restored
  for (const player of players) {
    const original = originalPositions.get(player.id);
    assertApprox(player.x, original.x, 0.001, `Player ${player.id} should be restored`);
  }
}

console.log('\n========== INTEGRATION TEST SUMMARY ==========');
console.log(`✅ Passed: ${testsPassed}`);
console.log(`❌ Failed: ${testsFailed}`);
console.log(`Total: ${testsPassed + testsFailed}`);
console.log('============================================\n');

if (testsFailed > 0) {
  process.exit(1);
}
