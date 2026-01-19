/*
 * public/src/assets/EntityDatabase.js
 *
 * Browser counterpart of the server-side database.  Fetches the consolidated
 * /api/entities payload that Server.js already exposes, with fallback to
 * static JSON files for local development.
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
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      ['tiles', 'objects', 'enemies', 'items'].forEach(group => {
        (json[group] || []).forEach(ent => {
          this.groups[group].set(ent.id, ent);
        });
      });
      this.loaded = true;
      console.log(`[EntityDB] Loaded from API – tiles:${this.groups.tiles.size} objects:${this.groups.objects.size} enemies:${this.groups.enemies.size}`);
    } catch (err) {
      console.warn('[EntityDB] API load failed, trying static JSON fallback...', err.message);
      await this._loadFromStaticFiles();
    }
  }

  async _loadFromStaticFiles() {
    // Fallback: load directly from static JSON files (for local dev without server)
    const files = [
      { group: 'enemies', path: '/assets/entities/enemies.json' },
      { group: 'objects', path: '/assets/entities/objects.json' },
      { group: 'tiles', path: '/assets/entities/tiles.json' }
    ];

    for (const { group, path } of files) {
      try {
        const res = await fetch(path);
        if (!res.ok) continue;
        const json = await res.json();
        const arr = Array.isArray(json) ? json : (json[group] || []);
        arr.forEach(ent => {
          if (ent.id) this.groups[group].set(ent.id, ent);
        });
      } catch (e) {
        console.warn(`[EntityDB] Could not load ${path}:`, e.message);
      }
    }
    this.loaded = true;
    console.log(`[EntityDB] Loaded from static files – tiles:${this.groups.tiles.size} objects:${this.groups.objects.size} enemies:${this.groups.enemies.size}`);
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