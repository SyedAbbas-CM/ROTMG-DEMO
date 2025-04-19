// server/wasm/wasmLoader.js
import fs from 'fs';
import path from 'path';

let bulletWasmExports = null;

/**
 * Loads bulletUpdate.wasm, returning the exports (like _updateBullets).
 */
export async function loadBulletWasm() {
  if (bulletWasmExports) {
    return bulletWasmExports; // already loaded
  }

  // Adjust the path if needed
  const wasmPath = path.join(process.cwd(), 'server', 'wasm', 'bulletUpdate.wasm');
  const buffer = fs.readFileSync(wasmPath);

  const { instance } = await WebAssembly.instantiate(buffer, {
    env: {
      // if your WASM code uses memory, table, etc. from outside, define them here
    }
  });

  // instance.exports should have _updateBullets
  bulletWasmExports = instance.exports;
  return bulletWasmExports;
}
