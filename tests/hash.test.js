// tests/hash.test.js
import xxhash32 from 'xxhash-wasm';

let hasher;
beforeAll(async () => {
  hasher = await xxhash32();
});

describe('xxhash32 snapshot hashing', () => {
  const SEED = 0xABCD1234;

  test('identical snapshots have identical hashes', () => {
    const snap1 = { a: 1, b: [2, 3] };
    const snap2 = { a: 1, b: [2, 3] }; // deep equal but different ref
    const h1 = hasher.h32(JSON.stringify(snap1), SEED);
    const h2 = hasher.h32(JSON.stringify(snap2), SEED);
    expect(h1).toBe(h2);
  });

  test('different snapshots produce different hashes', () => {
    const snap1 = { a: 1 };
    const snap2 = { a: 2 }; // changed value
    const h1 = hasher.h32(JSON.stringify(snap1), SEED);
    const h2 = hasher.h32(JSON.stringify(snap2), SEED);
    expect(h1).not.toBe(h2);
  });
}); 