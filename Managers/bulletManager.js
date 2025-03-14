// File: /src/managers/BulletManager.js

import { loadBulletWasm } from '../../server/wasm/wasmLoader.js'; 
// Adjust path as needed

export default class BulletManager {
  constructor(maxBullets = 10000) {
    this.maxBullets = maxBullets;
    this.bulletCount = 0;

    // SoA data
    this.x    = new Float32Array(maxBullets);
    this.y    = new Float32Array(maxBullets);
    this.vx   = new Float32Array(maxBullets);
    this.vy   = new Float32Array(maxBullets);
    this.life = new Float32Array(maxBullets);

    this.width  = new Float32Array(maxBullets);
    this.height = new Float32Array(maxBullets);

    // Optionally extra arrays: ax, ay, etc.

    // For WASM usage:
    this.wasm = null;
    this.memoryBuffer = null;
    this.updateBulletsFn = null;
    this.bulletCountPtr = null;

    this.initWasm(); // Begin loading WASM
  }

  async initWasm() {
    try {
      this.wasm = await loadBulletWasm();
      this.updateBulletsFn = this.wasm._updateBullets; 
      // This is the function name if you used EXPORTED_FUNCTIONS='["_updateBullets"]'
      // Also define a memory approach below:
      // If you want to share memory, you'd do more advanced steps like create a memory buffer
      // or reference this.wasm.memory. For now, we'll do a naive copy approach in wasmUpdate().
    } catch (err) {
      console.error('Failed to load bullet WASM module:', err);
    }
  }

  addBullet(x, y, vx, vy, life, width = 5, height = 5) {
    if (this.bulletCount >= this.maxBullets) return;
    const i = this.bulletCount++;
    this.x[i] = x;
    this.y[i] = y;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.life[i] = life;
    this.width[i] = width;
    this.height[i] = height;
  }

  update(deltaTime) {
    // If WASM is loaded and we have the function, use it. Otherwise fallback.
    if (this.updateBulletsFn) {
      this.wasmUpdate(deltaTime);
    } else {
      this.jsUpdate(deltaTime);
    }
  }

  jsUpdate(deltaTime) {
    let count = this.bulletCount;
    for (let i = 0; i < count; i++) {
      this.x[i] += this.vx[i] * deltaTime;
      this.y[i] += this.vy[i] * deltaTime;
      this.life[i] -= deltaTime;

      if (this.life[i] <= 0) {
        this.swapRemove(i);
        count--;
        i--;
      }
    }
    this.bulletCount = count;
  }

  /**
   * A naive WASM approach: copy data to a small buffer, call updateBullets, copy back.
   * For big bullet arrays, this can be slow. A SharedArrayBuffer approach is faster.
   */
  wasmUpdate(deltaTime) {
    const count = this.bulletCount;
    if (count <= 0) return;

    // 1) Create a local buffer to hold x,y,vx,vy,life for all bullets
    //    Each array is 'count' floats, so total = count*5 floats, plus space for an int bulletCount
    const floatBytes = 4; 
    const totalFloats = count * 5;
    const totalBytes = totalFloats * floatBytes + 4; // 4 bytes for int bulletCount

    // Emscripten doesn't automatically provide malloc in side modules, so we do a manual approach
    // or rely on an environment-provided 'memory'. For simplicity, let's do a single typed array
    // that we read/write from the wasm memory buffer.

    // This demonstration is a placeholder. Typically, you'd do:
    // let memory = new Uint8Array(this.wasm.memory.buffer);
    // and then allocate some space, copy data in, call the function, copy data out.

    // We'll fallback for demonstration:
    return this.jsUpdate(deltaTime);
  }

  swapRemove(index) {
    const last = this.bulletCount - 1;
    if (index !== last) {
      this.x[index] = this.x[last];
      this.y[index] = this.y[last];
      this.vx[index] = this.vx[last];
      this.vy[index] = this.vy[last];
      this.life[index] = this.life[last];
      this.width[index] = this.width[last];
      this.height[index] = this.height[last];
    }
    this.bulletCount--;
  }

  getActiveBulletCount() {
    return this.bulletCount;
  }

  cleanup() {
    this.bulletCount = 0;
  }

  /**
   * For sending data to clients
   */
  getDataArray() {
    const arr = [];
    for (let i = 0; i < this.bulletCount; i++) {
      arr.push({
        x: this.x[i],
        y: this.y[i],
        vx: this.vx[i],
        vy: this.vy[i],
        life: this.life[i],
        width: this.width[i],
        height: this.height[i]
      });
    }
    return arr;
  }
}
