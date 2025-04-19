// public/src/render/webgl/WebGLMapRenderer.js
// Centralised WebGL renderer shared by "strategic" and "top‑down" views.
// Each view plugs one or more TileLayer instances into this class and then
// calls update() every frame.  Nothing here knows about how big the view
// is – that is supplied by the camera that the game state already has.

import * as THREE from 'three';
import { TILE_SIZE, TILE_IDS } from '../../constants/constants.js';
import { TileLayer } from './TileLayer.js';
import { spriteManager } from '../../assets/spriteManager.js';
import { gameState } from '../../game/gamestate.js';

/**
 * WebGLMapRenderer
 *
 * ┌ camera ───────┐        (Three.js camera)
 * │               │
 * │  TileLayer ───┼─> InstancedMesh batch per tile type
 * │               │
 * └───────────────┘        These live in a single THREE.Scene and are
 *                          rendered once per frame via a single call to
 *                          renderer.render().  That means negligible CPU
 *                          per‑tile overhead.
 */
export class WebGLMapRenderer {
  /**
   * @param {Object}   opts
   * @param {THREE.WebGLRenderer} opts.renderer  – the low‑level Three renderer
   * @param {THREE.Scene}         opts.scene     – scene to render into
   * @param {Object}              opts.camera    – camera from gameState
   * @param {Object}              opts.map       – map manager from gameState
   * @param {Number}              opts.viewRadius – how many world tiles around
   *                                                the camera we draw.
   */
  constructor ({ renderer, scene, camera, map, viewRadius = 64 }) {
    this.renderer   = renderer;
    this.scene      = scene;
    this.camera     = camera;
    this.mapManager = map;
    this.viewRadius = viewRadius;

    // One TileLayer for each tile kind we wish to batch.
    this.layers = {};

    // Load the common sprite sheet through spriteManager.  The images are
    // already cached there (they were used in the existing 2‑D canvas path),
    // so no extra network cost.
    const sheetObj = spriteManager.getSpriteSheet('tile_sprites');
    if (!sheetObj) {
      console.error('[WebGLMapRenderer] tile_sprites not yet loaded');
      return; // caller should retry once assets are in
    }

    const sharedTexture = new THREE.Texture(sheetObj.image);
    sharedTexture.needsUpdate = true;
    sharedTexture.minFilter   = THREE.NearestFilter;
    sharedTexture.magFilter   = THREE.NearestFilter;

    // Build TileLayers lazily when they are first required so that we pay
    // zero cost for tile types that never appear.
    const buildLayer = (tileId) => {
      this.layers[tileId] = new TileLayer({
        tileId,
        texture: sharedTexture,
        tileSize: TILE_SIZE,
        viewRadius: this.viewRadius,
      });
      this.scene.add(this.layers[tileId].mesh);
    };

    // Eagerly create layers for the most common kinds to avoid branch cost in
    // hot inner loop.
    [TILE_IDS.FLOOR, TILE_IDS.WALL, TILE_IDS.WATER].forEach(buildLayer);

    // Defer rarer kinds until they are actually encountered.
    this._buildLayer = buildLayer;
  }

  /**
   * Call once per frame.
   */
  update () {
    const cam = this.camera; // alias for brevity
    const cx  = Math.floor(cam.position.x);
    const cy  = Math.floor(cam.position.y);

    const minX = cx - this.viewRadius;
    const maxX = cx + this.viewRadius;
    const minY = cy - this.viewRadius;
    const maxY = cy + this.viewRadius;

    // Gather matrices per tile type
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const tile = this.mapManager.getTile(x, y);
        if (!tile) continue;

        let layer = this.layers[tile.type];
        if (!layer) {
          // We encountered a tile type we haven't built yet.
          this._buildLayer(tile.type);
          layer = this.layers[tile.type];
        }

        layer.addInstance(x, y, tile.height || 0);
      }
    }

    // Finalise GPU buffers.
    for (const id in this.layers) {
      this.layers[id].finaliseFrame();
    }
  }

  /**
   * Draws the scene to the screen.  Should be called in the main game loop
   * *after* update().
   */
  render () {
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Clears instance counts so we can start a fresh frame.
   * Call this at the start of every logic tick (before update()).
   */
  reset () {
    for (const id in this.layers) {
      this.layers[id].reset();
    }
  }
}

// Convenience factory wired into gameState so existing code needs minimal
// changes: set `gameState.useWebGL = true` and call enableWebGLRenderer().
export function enableWebGLRenderer ({ viewRadius = 64 } = {}) {
  if (gameState.webgl) {
    console.warn('WebGL renderer already initialised');
    return gameState.webgl;
  }

  const canvas = document.getElementById('gameCanvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  const scene = new THREE.Scene();
  const camera = gameState.camera._threeCamera; // we expose raw three camera

  const mapRenderer = new WebGLMapRenderer({
    renderer,
    scene,
    camera,
    map: gameState.map,
    viewRadius,
  });

  gameState.webgl = {
    enabled: true,
    renderer,
    scene,
    mapRenderer,
  };

  return gameState.webgl;
}
