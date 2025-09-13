// src/world/worldRegistry.js

/**
 * Lightweight world registry utilities that wrap MapManager metadata and
 * provide consistent spawn-point selection for both initial spawns and portal
 * transfers. Keeping this thin avoids widespread imports churn.
 */

/**
 * Fetch world metadata via MapManager.
 * @param {import('./MapManager.js').MapManager} mapManager
 * @param {string} mapId
 * @returns {object} metadata or a minimal default
 */
export function getWorldMeta(mapManager, mapId){
  const meta = mapManager.getMapMetadata(mapId);
  return meta || { mapId, width: 64, height: 64, tileSize: 12, chunkSize: 16 };
}

/**
 * Choose a safe spawn point for a world.
 * Prefers explicit entryPoints, otherwise picks a safe margin position.
 * @param {import('./MapManager.js').MapManager} mapManager
 * @param {string} mapId
 */
export function getSpawnPoint(mapManager, mapId){
  const meta = getWorldMeta(mapManager, mapId);
  // Prefer provided entry points
  if (Array.isArray(meta.entryPoints) && meta.entryPoints.length > 0) {
    const p = meta.entryPoints[0];
    // Avoid spawning directly on a portal tile
    const objects = mapManager.getObjects ? (mapManager.getObjects(mapId) || []) : [];
    const isPortalHere = objects.some(o => o.type === 'portal' && o.x === (p.x ?? 5) && o.y === (p.y ?? 5));
    if (!isPortalHere) return { x: p.x ?? 5, y: p.y ?? 5 };
    const candidates = [
      { dx: 1, dy: 0 }, { dx: -1, dy: 0 }, { dx: 0, dy: 1 }, { dx: 0, dy: -1 },
      { dx: 2, dy: 0 }, { dx: 0, dy: 2 }, { dx: -2, dy: 0 }, { dx: 0, dy: -2 }
    ];
    for (const c of candidates) {
      const sx = (p.x ?? 5) + c.dx;
      const sy = (p.y ?? 5) + c.dy;
      const onPortal = objects.some(o => o.type === 'portal' && o.x === sx && o.y === sy);
      if (!onPortal && !mapManager.isWallOrOutOfBounds?.(sx, sy)) {
        return { x: sx, y: sy };
      }
    }
    return { x: (p.x ?? 5) + 2, y: (p.y ?? 5) };
  }

  // Fallback: choose a safe spawn near the map center, avoiding walls/portals
  const width  = Math.max(meta.width  || 64, 2);
  const height = Math.max(meta.height || 64, 2);
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);

  const objects = mapManager.getObjects ? (mapManager.getObjects(mapId) || []) : [];
  const isBlocked = (sx, sy) => {
    if (objects.some(o => o.type === 'portal' && o.x === sx && o.y === sy)) return true;
    return !!mapManager.isWallOrOutOfBounds?.(sx, sy);
  };

  // Spiral search outwards from center to find first safe tile
  const maxRadius = Math.max(Math.ceil(Math.max(width, height) / 2), 16);
  if (!isBlocked(cx, cy)) return { x: cx, y: cy };
  for (let r = 1; r <= maxRadius; r++) {
    for (let dx = -r; dx <= r; dx++) {
      const candidates = [
        { x: cx + dx, y: cy - r },
        { x: cx + dx, y: cy + r }
      ];
      for (const c of candidates) {
        if (c.x < 0 || c.y < 0 || c.x >= width || c.y >= height) continue;
        if (!isBlocked(c.x, c.y)) return { x: c.x, y: c.y };
      }
    }
    for (let dy = -r + 1; dy <= r - 1; dy++) {
      const candidates = [
        { x: cx - r, y: cy + dy },
        { x: cx + r, y: cy + dy }
      ];
      for (const c of candidates) {
        if (c.x < 0 || c.y < 0 || c.x >= width || c.y >= height) continue;
        if (!isBlocked(c.x, c.y)) return { x: c.x, y: c.y };
      }
    }
  }

  // Last resort: pick a random point within bounds and hope it's good
  const rx = Math.min(Math.max(1, cx), width - 2);
  const ry = Math.min(Math.max(1, cy), height - 2);
  return { x: rx, y: ry };
}


