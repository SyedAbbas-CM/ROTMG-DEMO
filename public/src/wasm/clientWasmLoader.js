/**
 * clientWasmLoader.js
 * Handles loading and interfacing with WebAssembly modules in the browser.
 * This replaces the server-side WASM loading with browser-compatible fetch.
 */

// Singleton instance of loaded WASM module
let collisionWasm = null;
let wasmMemory = null;

/**
 * Loads the collision WASM module for the browser
 * @returns {Promise<Object>} WebAssembly module exports
 */
export async function loadCollisionWasm() {
  // Return cached module if already loaded
  if (collisionWasm) return collisionWasm;
  
  try {
    // Fetch the WASM binary
    const response = await fetch('/wasm/collision.wasm');
    if (!response.ok) {
      throw new Error(`Failed to fetch WASM: ${response.statusText}`);
    }
    
    const wasmBuffer = await response.arrayBuffer();
    
    // Create shared memory
    wasmMemory = new WebAssembly.Memory({ 
      initial: 10,  // 10 pages = 640KB
      maximum: 100  // 100 pages = 6.4MB
    });
    
    // Instantiate the WebAssembly module
    const result = await WebAssembly.instantiate(wasmBuffer, {
      env: {
        memory: wasmMemory,
      }
    });
    
    // Store module exports
    collisionWasm = result.instance.exports;
    console.log('Collision WASM module loaded successfully');
    
    return collisionWasm;
  } catch (error) {
    console.error('Failed to load collision WASM module:', error);
    return null;
  }
}

/**
 * Gets the WebAssembly memory buffer
 * @returns {WebAssembly.Memory} WASM memory
 */
export function getWasmMemory() {
  return wasmMemory;
}