// public/src/render/webgl/TileLayer.js
// ------------------------------------------------
// Manages a sparse "infinite" tilemap and renders it with a handful of
// InstancedMesh objects – *one per tile type*.  A light‑weight utility that
// WebGLMapRenderer and any other Three.js view can share.

import * as THREE from 'three';
import { TILE_IDS, TILE_SPRITES, TILE_SIZE } from '../../constants/constants.js';

/**
 * Tiny helper – builds a MeshStandardMaterial that shows *one* sprite from a
 * packed spritesheet.  (Wrap & repeat math identical to the canvas version.)
 */
function makeTileMaterial(sheetTexture, tileId) {
  const sprite = TILE_SPRITES[tileId];
  const tex   = sheetTexture.clone();
  tex.needsUpdate = true;

  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.repeat.set(
    TILE_SIZE / sheetTexture.image.width,
    TILE_SIZE / sheetTexture.image.height
  );
  tex.offset.set(
    sprite.x / sheetTexture.image.width,
    1 - (sprite.y + TILE_SIZE) / sheetTexture.image.height
  );

  return new THREE.MeshStandardMaterial({
    map: tex,
    transparent: true,
    side: THREE.DoubleSide
  });
}

/**
 * TileLayer
 * ========
 * Holds *just enough* state to render the tiles that surround the player –
 * nothing more.  We feed it discrete tiles via `setTile()` (typically from the
 * same Map manager you already have).  At runtime `update()` is called with the
 * player position; the layer figures out which of those tiles are inside the
 * view radius and crams them into InstancedMeshes.
 */
export class TileLayer {
  /**
   * @param {Object} opts
   *   - scene:       THREE.Scene to add meshes to
   *   - spriteSheet: THREE.Texture (already loaded)
   *   - viewRadius:  how many world units around player to draw (default 64)
   */
  constructor({ scene, spriteSheet, viewRadius = 64 }) {
    this.viewRadius = viewRadius;

    // --- data store --------------------------------------------------------
    // Sparse dictionary keyed "x,y" → tileId.  In practise you would stream
    // chunks into this from your existing map manager whenever chunks load.
    this.tiles = new Map();

    // --- InstancedMeshes ---------------------------------------------------
    // One InstancedMesh per tileId → avoids material switches.
    this.meshByTileId = {};

    const plane = new THREE.PlaneGeometry(TILE_SIZE, TILE_SIZE);
    plane.rotateX(-Math.PI / 2);

    const maxInstances = (viewRadius * 2 + 1) ** 2; // square around player

    for (const tileId of Object.values(TILE_IDS)) {
      const mat = makeTileMaterial(spriteSheet, tileId);
      const mesh = new THREE.InstancedMesh(plane, mat, maxInstances);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.visible = false; // no instances yet
      scene.add(mesh);
      this.meshByTileId[tileId] = mesh;
    }
  }

  /**
   * Register/replace a tile in our local cache.
   * @param {number} x grid/world x
   * @param {number} y grid/world y
   * @param {number} tileId see TILE_IDS
   */
  setTile(x, y, tileId) {
    this.tiles.set(`${x},${y}`, tileId);
  }

  /** Remove a tile from local cache. */
  clearTile(x, y) { this.tiles.delete(`${x},${y}`); }

  /**
   * Re‑populate visible InstancedMeshes around the focus point.
   * Called every time the camera / player moves a *full* tile (same heuristic
   * you used before).
   */
  update(focusX, focusY) {
    // Arrays to collect per‑type transforms first – faster than touching the
    // InstancedMesh every iteration.
    const transforms = {};
    for (const id of Object.values(TILE_IDS)) transforms[id] = [];

    const minX = Math.floor(focusX) - this.viewRadius;
    const maxX = Math.floor(focusX) + this.viewRadius;
    const minY = Math.floor(focusY) - this.viewRadius;
    const maxY = Math.floor(focusY) + this.viewRadius;

    for (let x = minX; x <= maxX; x++) {
      for (let y = minY; y <= maxY; y++) {
        const key = `${x},${y}`;
        const tileId = this.tiles.get(key);
        if (tileId === undefined) continue;

        const m = new THREE.Matrix4();
        m.makeTranslation(x * TILE_SIZE, 0, y * TILE_SIZE);
        transforms[tileId].push(m);
      }
    }

    // Blit into InstancedMeshes ------------------------------------------------
    for (const [tileId, mesh] of Object.entries(this.meshByTileId)) {
      const list = transforms[tileId];
      const count = Math.min(list.length, mesh.count);
      mesh.count = count;
      mesh.visible = count > 0;
      for (let i = 0; i < count; i++) mesh.setMatrixAt(i, list[i]);
      if (count) mesh.instanceMatrix.needsUpdate = true;
    }
  }
}
    