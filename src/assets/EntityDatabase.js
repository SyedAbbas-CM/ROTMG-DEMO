/*
 * src/assets/EntityDatabase.js
 *
 * Loads and provides access to all in-game entity definitions (tiles, objects,
 * enemies, projectiles…).  Mirrors the client-side loader but uses synchronous
 * `fs` I/O because the Node server boots once at start-up.
 */

import fs from 'fs';
import path from 'path';
import { tileDatabase } from './TileDatabase.js';

export class EntityDatabase {
  constructor() {
    this.groups = {
      tiles: new Map(),
      objects: new Map(),
      enemies: new Map(),
      items: new Map()
    };
  }

  /**
   * Load the canonical JSON files from `public/assets/entities`.
   * This should be called exactly once during server start-up.
   */
  loadSync(baseDir = path.join(process.cwd(), 'public', 'assets', 'entities')) {
    ['tiles', 'objects', 'enemies', 'items'].forEach(group => {
      const file = path.join(baseDir, `${group}.json`);
      if (!fs.existsSync(file)) {
        console.warn(`[EntityDB] ${group}.json not found – skipping`);
        return;
      }
      if(group==='items') { return; } // temporarily skip items until definitions cleaned
      try {
        const arr = JSON.parse(fs.readFileSync(file, 'utf8'));
        arr.forEach(def => {
          if (!def || !def.id) return;
          this.groups[group].set(def.id, def);
        });
        if (group==='tiles' && tileDatabase?.merge) tileDatabase.merge(arr);
        console.log(`[EntityDB] Loaded ${this.groups[group].size} ${group}`);
      } catch (err) {
        console.error(`[EntityDB] Failed to load ${group}.json`, err);
      }
    });
  }

  /** Generic getter (tile / object / enemy) */
  get(id) {
    for (const map of Object.values(this.groups)) {
      if (map.has(id)) return map.get(id);
    }
    return null;
  }

  /** Convenience helpers */
  getTile(id) { return this.groups.tiles.get(id) || null; }
  getObject(id) { return this.groups.objects.get(id) || null; }
  getEnemy(id) { return this.groups.enemies.get(id) || null; }
  getItem(id)  { return this.groups.items.get(id)   || null; }
  getAll(group) { return Array.from(this.groups[group]?.values() || []); }
}

export const entityDatabase = new EntityDatabase();
try {
  entityDatabase.loadSync();
} catch (e) {
  console.error('[EntityDB] initial load failed', e);
} 