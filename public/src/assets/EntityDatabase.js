/*
 * public/src/assets/EntityDatabase.js
 *
 * Browser counterpart of the server-side database.  Fetches the consolidated
 * /api/entities payload that Server.js already exposes.
 */

export class EntityDatabase {
  constructor() {
    this.groups = {
      tiles: new Map(),
      objects: new Map(),
      enemies: new Map(),
      items: new Map()
    };
    this.loaded = false;
  }

  async load(url = '/api/entities') {
    try {
      const res = await fetch(url);
      const json = await res.json();
      ['tiles', 'objects', 'enemies', 'items'].forEach(group => {
        (json[group] || []).forEach(ent => {
          this.groups[group].set(ent.id, ent);
        });
      });
      this.loaded = true;
      console.log(`[EntityDB] Loaded – tiles:${this.groups.tiles.size} objects:${this.groups.objects.size} enemies:${this.groups.enemies.size}`);
    } catch (err) {
      console.error('[EntityDB] load failed', err);
    }
  }

  get(id) {
    for (const map of Object.values(this.groups)) {
      if (map.has(id)) return map.get(id);
    }
    return null;
  }
  getTile(id) { return this.groups.tiles.get(id) || null; }
  getObject(id) { return this.groups.objects.get(id) || null; }
  getEnemy(id) { return this.groups.enemies.get(id) || null; }
  getItem(id)  { return this.groups.items.get(id)   || null; }
  getAll(group) { return Array.from(this.groups[group]?.values() || []); }
}

export const entityDatabase = new EntityDatabase(); 