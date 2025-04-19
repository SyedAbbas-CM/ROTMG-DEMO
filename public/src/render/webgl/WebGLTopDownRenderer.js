// public/src/render/webgl/WebGLTopDownRenderer.js
//--------------------------------------------------

import * as THREE from 'three';
import { TILE_IDS, TILE_SIZE } from '../../constants/constants.js';
import { gameState } from '../../game/gamestate.js';
import { TileLayer } from './TileLayer.js';

/**
 * WebGL renderer for the “top‑down” (close overhead) map.
 * Enable by setting gameState.viewMode = 'topdown-webgl'
 * and wiring update()/render() into your main loop.
 */
export class WebGLTopDownRenderer {
  /**
   * @param {number} viewRadius  How many tiles (grid radius) to draw round the player
   * @param {number} heightY     Camera height in world units
   */
  constructor(viewRadius = 24, heightY = 40) {
    this.viewRadius = viewRadius;
    this.heightY    = heightY;

    this.scene   = new THREE.Scene();
    this.layers  = {};           // TileLayer cache

    this._buildCamera();
    this._lightScene();
    this._initLayers();

    this._lastGX = null;         // last integer grid‑coords we baked for
    this._lastGY = null;
  }

  // ───────────────────────────────────────────── setup

  _buildCamera() {
    const frustum = this.viewRadius * TILE_SIZE;

    // top‑down ortho camera centred on (0,0)
    this.camera = new THREE.OrthographicCamera(
      -frustum, frustum,   // left, right
       frustum, -frustum,  // top,  bottom  (Three’s Y‑up)
      0.1, 1000
    );
    this.camera.position.set(0, this.heightY, 0);
    this.camera.up.set(0, 0, -1);      // make +Z face “south” on the map
    this.camera.lookAt(0, 0, 0);
  }

  _lightScene() {
    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8);
    hemi.position.set(0, 1, 0);
    this.scene.add(hemi);
  }

  _initLayers() {
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
        this.viewRadius * 2 + 1  // enough instances for full square window
      );
    });
  }

  // ───────────────────────────────────────────── public API

  /** call if you hot‑reload the sprite sheet */
  refreshMaterials() {
    Object.values(this.layers).forEach(l => l.refreshMaterial());
  }

  update() {
    const char = gameState.character;
    if (!char) return;

    // centre camera over the player
    this.camera.position.set(
      char.x * TILE_SIZE,
      this.heightY,
      char.y * TILE_SIZE
    );

    const gx = Math.floor(char.x);
    const gy = Math.floor(char.y);

    if (gx !== this._lastGX || gy !== this._lastGY) {
      this._lastGX = gx;
      this._lastGY = gy;
      this._rebake(gx, gy);
    }
  }

  render(rtRenderer) {
    rtRenderer.render(this.scene, this.camera);
  }

  dispose() {
    Object.values(this.layers).forEach(l => l.dispose());
    this.scene.clear();
  }

  // ───────────────────────────────────────────── private

  _rebake(cx, cy) {
    const { map } = gameState;
    if (!map) return;

    const startX = cx - this.viewRadius;
    const endX   = cx + this.viewRadius;
    const startY = cy - this.viewRadius;
    const endY   = cy + this.viewRadius;

    const buckets = {};
    Object.keys(this.layers).forEach(k => (buckets[k] = []));

    for (let y = startY; y <= endY; y++) {
      for (let x = startX; x <= endX; x++) {
        const tile = map.getTile(x, y);
        if (!tile) continue;

        const layer = this.layers[tile.type];
        if (!layer) continue;

        buckets[tile.type].push({
          x: x * TILE_SIZE,
          z: y * TILE_SIZE,
          height: tile.height || 0
        });
      }
    }

    Object.entries(buckets).forEach(([id, arr]) =>
      this.layers[id].updateInstances(arr)
    );
  }
}

// convenience switcher
export function enableWebGLTopDown() {
  const renderer = new WebGLTopDownRenderer();
  gameState.topDownRenderer = renderer;
  return renderer;
}
