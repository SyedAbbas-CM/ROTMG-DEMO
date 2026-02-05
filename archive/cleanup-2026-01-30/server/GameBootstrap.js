import fs from 'fs';
import path from 'path';

/**
 * GameBootstrap loads a declarative configuration describing maps, portals,
 * and initial spawns, and then applies it to the running MapManager and world contexts.
 */
export class GameBootstrap {
  constructor({ mapManager, getWorldCtx, spawnMapEnemies }) {
    this.mapManager = mapManager;
    this.getWorldCtx = getWorldCtx;
    this.spawnMapEnemies = spawnMapEnemies;
  }

  loadConfig(filePath) {
    const abs = path.isAbsolute(filePath) ? filePath : path.join(process.cwd(), filePath);
    if (!fs.existsSync(abs)) {
      throw new Error(`Bootstrap config not found: ${abs}`);
    }
    const cfg = JSON.parse(fs.readFileSync(abs, 'utf8'));
    return cfg;
  }

  async apply(config) {
    const createdMapIds = new Map(); // name -> mapId

    // 1) Ensure maps exist
    for (const m of config.maps || []) {
      let mapId;
      if (m.file) {
        const mapPath = path.isAbsolute(m.file) ? m.file : path.join(process.cwd(), m.file);
        mapId = await this.mapManager.loadFixedMap(mapPath);
      } else {
        mapId = this.mapManager.createProceduralMap({ width: m.width || 64, height: m.height || 64, seed: m.seed || Date.now(), name: m.name || 'Map' });
      }
      createdMapIds.set(m.name || mapId, mapId);
    }

    // 2) Portals
    for (const p of config.portals || []) {
      const fromId = createdMapIds.get(p.from) || p.from;
      const toId = createdMapIds.get(p.to) || p.to;
      const meta = this.mapManager.getMapMetadata(fromId);
      if (!meta) continue;
      if (!meta.objects) meta.objects = [];
      meta.objects.push({ id: `portal_${Date.now()}_${Math.random().toString(36).slice(2,6)}`, type: 'portal', sprite: p.sprite || 'portal', x: p.x|0, y: p.y|0, destMap: toId });
    }

    // 3) Spawns (enemies)
    for (const s of config.spawns || []) {
      const mapId = createdMapIds.get(s.map) || s.map;
      try {
        this.spawnMapEnemies(mapId);
      } catch (_) {}
    }

    return { maps: Array.from(createdMapIds.values()) };
  }
}


