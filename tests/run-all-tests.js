// File: tests/run-all-tests.js
// Run all lag compensation tests

import { execSync } from 'child_process';

console.log('\n╔══════════════════════════════════════════════════════════╗');
console.log('║  LAG COMPENSATION TEST SUITE                             ║');
console.log('╚══════════════════════════════════════════════════════════╝\n');

const tests = [
  { name: 'CircularBuffer', file: 'tests/CircularBuffer.test.js' },
  { name: 'LagCompensation', file: 'tests/LagCompensation.test.js' },
  { name: 'Integration', file: 'tests/LagCompensation.integration.test.js' }
];

let totalPassed = 0;
let totalFailed = 0;

for (const test of tests) {
  console.log(`\n┌─────────────────────────────────────────────────────────┐`);
  console.log(`│ Running: ${test.name.padEnd(48)} │`);
  console.log(`└─────────────────────────────────────────────────────────┘`);

  try {
    const output = execSync(`node ${test.file}`, {
      encoding: 'utf-8',
      stdio: 'pipe'
    });

    // Parse results from output
    const passedMatch = output.match(/✅ Passed: (\d+)/);
    const failedMatch = output.match(/❌ Failed: (\d+)/);

    if (passedMatch) totalPassed += parseInt(passedMatch[1]);
    if (failedMatch) totalFailed += parseInt(failedMatch[1]);

    // Show summary line
    const summary = output.split('\n').find(line => line.includes('Passed:'));
    if (summary) {
      console.log(`\n${summary}`);
    }

    console.log(`✅ ${test.name} PASSED\n`);

  } catch (error) {
    console.error(`❌ ${test.name} FAILED`);
    console.error(error.stdout || error.message);
    totalFailed++;
  }
}

console.log('\n╔══════════════════════════════════════════════════════════╗');
console.log('║  FINAL RESULTS                                           ║');
console.log('╠══════════════════════════════════════════════════════════╣');
console.log(`║  ✅ Total Passed:  ${String(totalPassed).padEnd(43)} ║`);
console.log(`║  ❌ Total Failed:  ${String(totalFailed).padEnd(43)} ║`);
console.log(`║  📊 Total Tests:   ${String(totalPassed + totalFailed).padEnd(43)} ║`);
console.log('╚══════════════════════════════════════════════════════════╝\n');

if (totalFailed > 0) {
  process.exit(1);
}
