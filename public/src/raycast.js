// raycast.js

import { TILE_SIZE, MAP_ROWS, MAP_COLS } from './constants/constants.js';
import { getTile } from './map/map.js';

export function castRays(player, canvasWidth, numRays) {
  const fov = Math.PI / 3; // 60 degrees field of view
  const halfFov = fov / 2;
  const angleStep = fov / numRays;
  const startAngle = player.rotation.yaw - halfFov;

  const rays = [];

  for (let i = 0; i < numRays; i++) {
    const rayAngle = startAngle + i * angleStep;
    const ray = castSingleRay(player, rayAngle);
    rays.push(ray);
  }

  return rays;
}

function castSingleRay(player, angle) {
  const sin = Math.sin(angle);
  const cos = Math.cos(angle);

  let distance = 0;
  let hit = false;
  let tileType = null;

  while (!hit && distance < 1000) {
    distance += 0.1;
    const x = player.x + cos * distance;
    const y = player.y + sin * distance;
    const tile = getTile(Math.floor(x / TILE_SIZE), Math.floor(y / TILE_SIZE));

    if (tile && tile.type !== TILE_IDS.FLOOR) {
      hit = true;
      tileType = tile.type;
    }
  }

  return {
    angle,
    distance,
    tileType,
  };
}
