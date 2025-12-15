// Test script for C++ native modules
import { createRequire } from 'module';
import { performance } from 'perf_hooks';
const require = createRequire(import.meta.url);

console.log('Testing C++ Native Modules...\n');

// Test 1: Load protocol module
console.log('=== Test 1: Protocol Module ===');
let protocol;
try {
    protocol = require('./build/Release/protocol.node');
    console.log('✓ Protocol module loaded successfully');
    console.log('  Functions available:', Object.keys(protocol));

    // Test encoding
    const testData = Buffer.from(JSON.stringify({ x: 10, y: 20, health: 100 }));
    const encoded = protocol.encode(12, testData); // Type 12 = PLAYER_UPDATE
    console.log('✓ Encoded message:', encoded.length, 'bytes');
    console.log('  Original data size:', testData.length, 'bytes');
    console.log('  Overhead:', encoded.length - testData.length, 'bytes (header)');

    // Test decoding
    const decoded = protocol.decode(encoded);
    console.log('✓ Decoded message:', decoded);
    console.log('  Type:', decoded.type, '=', protocol.getTypeName(decoded.type));
    console.log('  Payload size:', decoded.payload.length, 'bytes');

    // Verify round-trip
    const originalJson = testData.toString();
    const decodedJson = decoded.payload.toString();
    if (originalJson === decodedJson) {
        console.log('✓ Round-trip successful - data matches!');
    } else {
        console.log('✗ Round-trip failed - data mismatch');
    }

} catch (err) {
    console.log('✗ Protocol module failed:', err.message);
    process.exit(1);
}

console.log('\n=== Test 2: Collision Module ===');
try {
    const collision = require('./build/Release/collision.node');
    console.log('✓ Collision module loaded successfully');
    console.log('  (Collision module is C-only, no JS bindings yet)');
    console.log('  This is normal - it will be used via FFI or future N-API wrapper');

} catch (err) {
    console.log('Note: Collision module is C-only (expected):', err.message);
}

console.log('\n=== Performance Comparison ===');

// Compare JSON vs Binary encoding performance
const { BinaryPacket } = await import('./common/protocol.js');

const testPayload = {
    players: {
        '1': { x: 100.5, y: 200.3, health: 1000, maxHealth: 1000 },
        '2': { x: 150.2, y: 180.7, health: 850, maxHealth: 1000 },
        '3': { x: 120.8, y: 220.1, health: 950, maxHealth: 1000 }
    },
    enemies: Array.from({ length: 50 }, (_, i) => ({
        id: `enemy_${i}`,
        x: Math.random() * 512,
        y: Math.random() * 512,
        health: 100,
        type: Math.floor(Math.random() * 5)
    })),
    bullets: Array.from({ length: 100 }, (_, i) => ({
        id: `bullet_${i}`,
        x: Math.random() * 512,
        y: Math.random() * 512,
        vx: Math.random() * 10,
        vy: Math.random() * 10
    })),
    timestamp: Date.now()
};

// Benchmark JavaScript implementation
const iterations = 1000;

console.log(`Running ${iterations} iterations with realistic game state...`);

const jsStart = performance.now();
for (let i = 0; i < iterations; i++) {
    const encoded = BinaryPacket.encode(60, testPayload); // WORLD_UPDATE
    const decoded = BinaryPacket.decode(encoded);
}
const jsEnd = performance.now();
const jsTime = jsEnd - jsStart;

// Benchmark C++ implementation
const cppStart = performance.now();
for (let i = 0; i < iterations; i++) {
    const jsonBuffer = Buffer.from(JSON.stringify(testPayload));
    const encoded = protocol.encode(60, jsonBuffer);
    const decoded = protocol.decode(encoded);
}
const cppEnd = performance.now();
const cppTime = cppEnd - cppStart;

console.log('\nResults:');
console.log(`  JavaScript: ${jsTime.toFixed(2)}ms (${(jsTime/iterations).toFixed(3)}ms per iteration)`);
console.log(`  C++:        ${cppTime.toFixed(2)}ms (${(cppTime/iterations).toFixed(3)}ms per iteration)`);
console.log(`  Speedup:    ${(jsTime/cppTime).toFixed(2)}x faster`);

// Size comparison
const jsEncoded = BinaryPacket.encode(60, testPayload);
const jsonBuffer = Buffer.from(JSON.stringify(testPayload));
const cppEncoded = protocol.encode(60, jsonBuffer);

console.log('\nSize comparison:');
console.log(`  JSON payload:     ${JSON.stringify(testPayload).length} bytes`);
console.log(`  JS encoded:       ${jsEncoded.byteLength} bytes`);
console.log(`  C++ encoded:      ${cppEncoded.length} bytes`);
console.log(`  Difference:       ${jsEncoded.byteLength - cppEncoded.length} bytes`);

console.log('\n=== All Tests Complete ===');
