// public/src/render/webgl/WebGLStrategicRenderer.js
//--------------------------------------------------

import * as THREE from 'three';
import { TILE_IDS, TILE_SIZE } from '../../constants/constants.js';
import { gameState } from '../../game/gamestate.js';
import { TileLayer } from './TileLayer.js';

/**
 * A light‑weight WebGL renderer for the “strategic” zoomed‑out view.
 * Switch on by setting gameState.viewMode = 'strategic-webgl'
 * and calling its update() / render() hooks from the main loop.
 */
export class WebGLStrategicRenderer {
  /**
   * @param {number} viewRadius Grid‑radius (in tiles) to draw round the player
   * @param {number} scaling    Multiplier applied to TILE_SIZE to spread tiles apart visually
   */
  constructor(viewRadius = 64, scaling = 4) {
    this.viewRadius = viewRadius;
    this.scaling   = scaling;           // bigger gap between tiles keeps perspective clear
    this.scene     = new THREE.Scene();
    this.layers    = {};               // TileLayer instances, keyed by TILE_IDS

    this._setupCamera();
    this._setupLights();
    this._setupLayers();

    // track last integer‑grid position so we only rebuild when needed
    this._lastGX = null;
    this._lastGY = null;
  }

  // ───────────────────────────────────────────────────── private helpers

  _setupCamera() {
    // simple top‑down ortho camera
    const frustum = this.viewRadius * TILE_SIZE * this.scaling;

    this.camera = new THREE.OrthographicCamera(
      -frustum, frustum,           // left, right
       frustum, -frustum,          // top,  bottom  (note: y‑up in Three)
      0.1, 1000
    );
    this.camera.position.set(0, 100, 0);  // y‑axis is “up”
    this.camera.up.set(0, 0, -1);         // keep Z forward so +Z is south on the map
    this.camera.lookAt(0, 0, 0);
  }

  _setupLights() {
    const dir = new THREE.DirectionalLight(0xffffff, 1);
    dir.position.set(50, 100, 50);
    this.scene.add(dir);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  }

  _setupLayers() {
    [
      TILE_IDS.FLOOR,
      TILE_IDS.WATER,
      TILE_IDS.WALL,
      TILE_IDS.OBSTACLE,
      TILE_IDS.MOUNTAIN
    ].forEach(id => {
      this.layers[id] = new TileLayer(
        id,
        this.scene,
        this.viewRadius * 2 + 1 // max instance count in one axis, squared inside TileLayer
      );
    });
  }

  // ───────────────────────────────────────────────────── public API

  /** call if you hot‑reload the sprite sheet and need fresh materials */
  refreshMaterials() {
    Object.values(this.layers).forEach(l => l.refreshMaterial());
  }

  /**
   * Move camera with the player and rebuild instance buffers
   * when the player crosses a full‑tile boundary.
   */
  update() {
    const char = gameState.character;
    if (!char) return;

    // move camera
    this.camera.position.set(
      char.x * TILE_SIZE * this.scaling,
      100,
      char.y * TILE_SIZE * this.scaling
    );

    // determine integer‑grid location
    const gx = Math.floor(char.x);
    const gy = Math.floor(char.y);

    // only rebuild when we step into a new tile
    if (gx !== this._lastGX || gy !== this._lastGY) {
      this._lastGX = gx;
      this._lastGY = gy;
      this._rebuildTileBuffers(gx, gy);
    }
  }

  /**
   * Push all InstancedMeshes to GPU for the current neighbourhood.
   * Heavy lifting stays on the GPU once setMatrixAt() has done its work.
   */
  _rebuildTileBuffers(cx, cy) {
    const { map } = gameState;
    if (!map) return;

    const startX = cx - this.viewRadius;
    const endX   = cx + this.viewRadius;
    const startY = cy - this.viewRadius;
    const endY   = cy + this.viewRadius;

    // Pre‑bucket positions by tile‑type
    const batches = {};
    Object.keys(this.layers).forEach(id => batches[id] = []);

    for (let y = startY; y <= endY; y++) {
      for (let x = startX; x <= endX; x++) {
        const tile = map.getTile(x, y);
        if (!tile) continue;

        const layer = this.layers[tile.type];
        if (!layer) continue;         // unknown tile‑ID → ignore

        batches[tile.type].push({
          x: x * TILE_SIZE * this.scaling,
          z: y * TILE_SIZE * this.scaling,
          height: tile.height || 0
        });
      }
    }

    // push to each TileLayer
    Object.entries(batches).forEach(([id, arr]) => {
      this.layers[id].updateInstances(arr);
    });
  }

  /**
   * Draw the scene with a renderer you hand in (usually the global Three.WebGLRenderer).
   * Kept external so switching between render paths doesn’t instantiate multiple renderers.
   * @param {THREE.WebGLRenderer} rtRenderer
   */
  render(rtRenderer) {
    rtRenderer.render(this.scene, this.camera);
  }

  /** free GPU buffers – invoke when toggling off WebGL mode */
  dispose() {
    Object.values(this.layers).forEach(l => l.dispose());
    this.scene.clear();
  }
}

// convenience factory so other modules can do:
//    strategicRenderer = enableWebGLStrategic();
export function enableWebGLStrategic() {
  const renderer = new WebGLStrategicRenderer();
  gameState.strategicRenderer = renderer;
  return renderer;
}
