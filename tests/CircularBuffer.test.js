// File: tests/CircularBuffer.test.js
// Unit tests for CircularBuffer position history

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

console.log('\n========== CIRCULAR BUFFER TESTS ==========\n');

// Test 1: Basic initialization
console.log('Test 1: Basic initialization');
{
  const buffer = new CircularBuffer(5);
  assert(buffer.count === 0, 'Initial count should be 0');
  assert(buffer.maxSamples === 5, 'Max samples should be 5');
  const latest = buffer.getLatest();
  assert(!latest.found, 'getLatest should return found=false when empty');
}

// Test 2: Adding single position
console.log('\nTest 2: Adding single position');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10.5, 20.3, 1000);
  assert(buffer.count === 1, 'Count should be 1 after adding one sample');

  const latest = buffer.getLatest();
  assert(latest.found, 'getLatest should find the sample');
  assertApprox(latest.x, 10.5, 0.001, 'X position should match');
  assertApprox(latest.y, 20.3, 0.001, 'Y position should match');
  assert(latest.timestamp === 1000, 'Timestamp should match');
}

// Test 3: Adding multiple positions
console.log('\nTest 3: Adding multiple positions');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);
  buffer.add(11, 21, 1100);
  buffer.add(12, 22, 1200);

  assert(buffer.count === 3, 'Count should be 3');

  const latest = buffer.getLatest();
  assertApprox(latest.x, 12, 0.001, 'Latest X should be 12');
  assertApprox(latest.y, 22, 0.001, 'Latest Y should be 22');
  assert(latest.timestamp === 1200, 'Latest timestamp should be 1200');
}

// Test 4: Circular buffer overflow
console.log('\nTest 4: Circular buffer overflow');
{
  const buffer = new CircularBuffer(3);
  buffer.add(1, 1, 100);
  buffer.add(2, 2, 200);
  buffer.add(3, 3, 300);
  buffer.add(4, 4, 400); // Should overwrite first entry
  buffer.add(5, 5, 500); // Should overwrite second entry

  assert(buffer.count === 3, 'Count should stay at max (3)');

  const oldest = buffer.getOldestTimestamp();
  assert(oldest === 300, 'Oldest timestamp should be 300 (first two overwritten)');

  const latest = buffer.getLatest();
  assertApprox(latest.x, 5, 0.001, 'Latest X should be 5');
}

// Test 5: Linear interpolation - exact match
console.log('\nTest 5: Linear interpolation - exact match');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);
  buffer.add(20, 30, 2000);
  buffer.add(30, 40, 3000);

  const result = buffer.getPositionAt(2000);
  assert(result.found, 'Should find exact match');
  assertApprox(result.x, 20, 0.001, 'X should be 20 at t=2000');
  assertApprox(result.y, 30, 0.001, 'Y should be 30 at t=2000');
}

// Test 6: Linear interpolation - between samples
console.log('\nTest 6: Linear interpolation - between samples');
{
  const buffer = new CircularBuffer(5);
  buffer.add(0, 0, 1000);   // t=1000: (0, 0)
  buffer.add(10, 20, 2000);  // t=2000: (10, 20)

  // At t=1500 (halfway), should be (5, 10)
  const result = buffer.getPositionAt(1500);
  assert(result.found, 'Should interpolate between samples');
  assertApprox(result.x, 5, 0.001, 'X should be 5 at t=1500');
  assertApprox(result.y, 10, 0.001, 'Y should be 10 at t=1500');
}

// Test 7: Linear interpolation - 25% between samples
console.log('\nTest 7: Linear interpolation - 25% between samples');
{
  const buffer = new CircularBuffer(5);
  buffer.add(0, 0, 1000);    // t=1000: (0, 0)
  buffer.add(100, 200, 2000); // t=2000: (100, 200)

  // At t=1250 (25% of the way), should be (25, 50)
  const result = buffer.getPositionAt(1250);
  assertApprox(result.x, 25, 0.001, 'X should be 25 at t=1250');
  assertApprox(result.y, 50, 0.001, 'Y should be 50 at t=1250');
}

// Test 8: Query before oldest sample (extrapolation)
console.log('\nTest 8: Query before oldest sample');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);
  buffer.add(20, 30, 2000);

  const result = buffer.getPositionAt(500); // Before oldest
  assert(!result.found, 'Should return found=false for extrapolation');
  assertApprox(result.x, 10, 0.001, 'Should return oldest sample X');
  assertApprox(result.y, 20, 0.001, 'Should return oldest sample Y');
}

// Test 9: Query after newest sample (extrapolation)
console.log('\nTest 9: Query after newest sample');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);
  buffer.add(20, 30, 2000);

  const result = buffer.getPositionAt(3000); // After newest
  assert(!result.found, 'Should return found=false for extrapolation');
  assertApprox(result.x, 20, 0.001, 'Should return newest sample X');
  assertApprox(result.y, 30, 0.001, 'Should return newest sample Y');
}

// Test 10: Complex interpolation scenario (realistic movement)
console.log('\nTest 10: Realistic player movement scenario');
{
  const buffer = new CircularBuffer(10);

  // Simulate player moving from (0,0) to (10,0) over 300ms at 30Hz
  const startTime = 1000;
  const tickInterval = 33; // ~30Hz
  const speed = 10 / 0.3; // 10 tiles in 300ms

  for (let i = 0; i <= 9; i++) {
    const t = startTime + i * tickInterval;
    const x = (i / 9) * 10;
    buffer.add(x, 0, t);
  }

  // Query position at 150ms (halfway through movement)
  const result = buffer.getPositionAt(startTime + 150);
  assert(result.found, 'Should find interpolated position');
  assertApprox(result.x, 5, 0.5, 'X should be approximately 5 at halfway point');
  assertApprox(result.y, 0, 0.001, 'Y should remain 0');
}

// Test 11: Clear buffer
console.log('\nTest 11: Clear buffer');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);
  buffer.add(20, 30, 2000);

  assert(buffer.count === 2, 'Should have 2 samples before clear');

  buffer.clear();

  assert(buffer.count === 0, 'Count should be 0 after clear');
  const latest = buffer.getLatest();
  assert(!latest.found, 'Should not find any samples after clear');
}

// Test 12: Debug info
console.log('\nTest 12: Debug info');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);
  buffer.add(20, 30, 2000);

  const info = buffer.getDebugInfo();
  assert(info.count === 2, 'Debug info count should be 2');
  assert(info.maxSamples === 5, 'Debug info maxSamples should be 5');
  assert(info.samples.length === 2, 'Debug info should have 2 samples');
  assert(info.oldestTimestamp === 1000, 'Oldest timestamp should be 1000');
  assert(info.newestTimestamp === 2000, 'Newest timestamp should be 2000');
}

// Test 13: Stress test - many samples
console.log('\nTest 13: Stress test - 100 samples');
{
  const buffer = new CircularBuffer(100);

  for (let i = 0; i < 100; i++) {
    buffer.add(i, i * 2, 1000 + i * 10);
  }

  assert(buffer.count === 100, 'Should store 100 samples');

  const result = buffer.getPositionAt(1500); // Query in middle
  assert(result.found, 'Should find interpolated position in middle');

  const latest = buffer.getLatest();
  assertApprox(latest.x, 99, 0.001, 'Latest X should be 99');
  assertApprox(latest.y, 198, 0.001, 'Latest Y should be 198');
}

// Test 14: Edge case - single sample interpolation
console.log('\nTest 14: Single sample interpolation');
{
  const buffer = new CircularBuffer(5);
  buffer.add(10, 20, 1000);

  const result = buffer.getPositionAt(1500);
  assert(result.found, 'Should return the only sample available');
  assertApprox(result.x, 10, 0.001, 'Should return single sample X');
  assertApprox(result.y, 20, 0.001, 'Should return single sample Y');
}

// Test 15: Performance test
console.log('\nTest 15: Performance test');
{
  const buffer = new CircularBuffer(10);

  // Fill buffer
  for (let i = 0; i < 10; i++) {
    buffer.add(i, i, 1000 + i * 100);
  }

  // Measure interpolation performance
  const iterations = 10000;
  const startTime = Date.now();

  for (let i = 0; i < iterations; i++) {
    buffer.getPositionAt(1450); // Interpolate
  }

  const endTime = Date.now();
  const elapsed = endTime - startTime;
  const avgTime = elapsed / iterations;

  console.log(`  Performance: ${iterations} interpolations in ${elapsed}ms (${avgTime.toFixed(4)}ms avg)`);
  assert(avgTime < 1, `Average interpolation time should be < 1ms (got ${avgTime.toFixed(4)}ms)`);
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
