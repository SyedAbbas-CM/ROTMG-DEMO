/*
 * src/assets/TileDatabase.js
 *
 * Lightweight tile-definition database for the Node / server environment.
 * Mirrors the client-side implementation but uses the filesystem instead
 * of fetch(). In addition to string-key lookup (by tile id such as "floor")
 * it also supports lookup by the numeric TILE_IDS value so that server
 * systems that currently use those enums (e.g. MapManager) can easily fetch
 * the extended definition (walkable, height, slope …).
 */

import fs from 'fs';
import path from 'path';
import { TILE_IDS } from '../world/constants.js';

export class TileDatabase {
  constructor() {
    /** @type {Map<string,object>} */
    this.tiles = new Map();         // id(string)  -> def
    /** @type {Map<number,object>} */
    this.numToDef = new Map();      // TILE_IDS(num)-> def
    this.loaded = false;
  }

  /* ------------------------------------------------------------------ */
  /**
   * Load a JSON file that contains an array of tile definitions like:
   *   { "id": "floor", "sprite": "tile_sprites:floor", "walkable": true, "height": 0 }
   *
   * The method is synchronous because the server boots once at start-up and
   * reading a small JSON file synchronously keeps things simple.
   * @param {string} jsonFilePath – absolute or project-relative path
   */
  loadSync(jsonFilePath) {
    const resolved = path.isAbsolute(jsonFilePath)
      ? jsonFilePath
      : path.join(process.cwd(), jsonFilePath);

    if (!fs.existsSync(resolved)) {
      console.warn(`[TileDB] file not found: ${resolved}`);
      return;
    }

    try {
      const raw = fs.readFileSync(resolved, 'utf8');
      const data = JSON.parse(raw);
      this._populate(data);
      this.loaded = true;
      console.log(`[TileDB] loaded ${this.tiles.size} tile definitions from ${resolved}`);
    } catch (err) {
      console.error('[TileDB] failed to load', resolved, err);
    }
  }

  /** Populate internal maps from an array of defs */
  _populate(defArray) {
    if (!Array.isArray(defArray)) return;

    const numMap = {
      floor: TILE_IDS?.FLOOR,
      wall: TILE_IDS?.WALL,
      obstacle: TILE_IDS?.OBSTACLE,
      water: TILE_IDS?.WATER,
      mountain: TILE_IDS?.MOUNTAIN,
      // Additional mappings (ramps etc.) can be added when the enum exists
    };

    for (const def of defArray) {
      if (!def || !def.id) continue;
      this.tiles.set(def.id, def);

      if (numMap[def.id] !== undefined) {
        this.numToDef.set(numMap[def.id], def);
      }
    }
  }

  /**
   * Get by string id (e.g. "floor").
   * @param {string} id
   */
  get(id) {
    return this.tiles.get(id) || null;
  }

  /**
   * Get by numeric TILE_IDS value.
   * @param {number} num
   */
  getByNumeric(num) {
    return this.numToDef.get(num) || null;
  }

  merge(defArray=[]) {
    defArray.forEach(def=>{
      if (def && def.id) this.tiles.set(def.id, def);
    });
    console.log(`[TileDB] merged ${defArray.length} ext tile defs (total ${this.tiles.size})`);
  }
}

// -------------------------------------------------------------------------
// Default shared instance – auto-load the canonical tiles.json if present.
// -------------------------------------------------------------------------
export const tileDatabase = new TileDatabase();

try {
  // Same path as on the client but resolved from project root
  tileDatabase.loadSync('public/assets/database/tiles.json');
} catch (e) {
  // Ignore – server can still operate with default behaviour if file missing
} 