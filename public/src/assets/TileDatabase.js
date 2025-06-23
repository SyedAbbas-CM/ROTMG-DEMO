// public/src/assets/TileDatabase.js

/**
 * TileDatabase - loads and provides access to tile definitions (id, sprite, height, walkable, slope…)
 */
export class TileDatabase {
  constructor() {
    this.tiles = new Map(); // id -> def
  }

  /**
   * Load a JSON file containing an array of tile definitions.
   * Each entry: { id:string, sprite:string, walkable:boolean, height:number, slope?:string }
   * @param {string} url
   */
  async load(url) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      data.forEach(def => {
        this.tiles.set(def.id, def);
      });
      console.log(`[TileDB] loaded ${this.tiles.size} tiles from ${url}`);
    } catch (err) {
      console.warn('[TileDB] failed to load', url, err);
    }
  }

  /** @param {string} id */
  get(id) { return this.tiles.get(id) || null; }

  /** Merge in an array of tile defs coming from EntityDatabase */
  merge(defArray=[]) {
    defArray.forEach(def => {
      if (def && def.id) this.tiles.set(def.id, def);
    });
    console.log(`[TileDB] merged ${defArray.length} external tile defs (total ${this.tiles.size})`);
  }
}

export const tileDatabase = new TileDatabase(); 