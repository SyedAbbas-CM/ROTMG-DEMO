// render.js

import { assets } from './assets.js';
import {
  tSize,
  tHalf,
  xCenter,
  yCenter,
  min,
  max,
  WALL_SIZE,
  scan1,
  scan2,
  offCenter,
  offCenterDiff,
} from './constants.js';
import { player, enemies, bullets } from './gamestate.js';
import { map, mapSize, texMap, adjacentNWES } from './map.js';
import { rankArray } from './utils.js';
import { getOtherPlayersEntities } from './entities.js';

// Create and export the game context
export const canvas = document.getElementById('gameCanvas');
export const ctx = canvas.getContext('2d', { alpha: false });
ctx.imageSmoothingEnabled = false;

// Create offscreen canvas
export const gameCanvas = document.createElement('canvas');
gameCanvas.width = canvas.width;
gameCanvas.height = canvas.height;
export const gctx = gameCanvas.getContext('2d', { alpha: false });
gctx.imageSmoothingEnabled = false;

// Rest of the rendering functions using gctx
// ...

gctx.webkitImageSmoothingEnabled = false;

// === ENTITY DRAWING ===
export function renderEnemies() {
  enemies.forEach(enemy => {
    gctx.drawImage(
      assets.char,
      enemy.tx,
      enemy.ty,
      8,
      8,
      enemy.x - enemy.size / 2,
      enemy.y - enemy.size / 2,
      enemy.size,
      enemy.size
    );
  });
}

export function renderBullets() {
  bullets.forEach(bullet => {
    let screenX = tSize * (bullet.x - player.x) - xCenter;
    let screenY = tSize * (bullet.y - player.y) - yCenter;
    const cos = Math.cos(player.r);
    const sin = Math.sin(player.r);
    let rotatedScreenX = screenX * cos + screenY * -sin + xCenter;
    let rotatedScreenY = screenX * sin + screenY * cos + yCenter;

    gctx.drawImage(
      assets.obj4,
      bullet.tx,
      bullet.ty,
      8,
      8,
      rotatedScreenX - bullet.size / 2,
      rotatedScreenY - bullet.size / 2,
      bullet.size,
      bullet.size
    );
  });
}

export function renderMap() {
  const sin = Math.sin(player.r);
  const cos = Math.cos(player.r);

  let xDiff = Math.floor(player.x) - player.x;
  let yDiff = Math.floor(player.y) - player.y;

  let entity = [];

  // Add player entity
  entity.push({
    tex: assets.char,
    tx: player.tx,
    ty: player.ty,
    x: xCenter,
    y: yCenter,
    size: 40,
    wall: false,
    name: player.name,
  });

  // Add other players' entities
  const otherPlayerEntities = getOtherPlayersEntities();
  entity.push(...otherPlayerEntities);

  let roofs = [];

  gctx.save();
  gctx.translate(xCenter, yCenter);
  gctx.rotate(player.r);

  for (let x = scan1; x < scan2; x++) {
    for (let y = scan1; y < scan2; y++) {
      let outOfBounds =
        player.x + x < 0 ||
        player.x + x > mapSize ||
        player.y + y < 0 ||
        player.y + y > mapSize;

      let tileX = tSize * (xDiff + x) - tHalf;
      let tileY = tSize * (yDiff + y) - tHalf;

      // always top left corner of tile
      let rotatedTileX = tileX * cos + tileY * -sin + xCenter;
      let rotatedTileY = tileX * sin + tileY * cos + yCenter;

      if (
        rotatedTileX > min &&
        rotatedTileX < max &&
        rotatedTileY > min &&
        rotatedTileY < max
      ) {
        let arrayLocation =
          Math.floor(player.x) + x + mapSize * (Math.floor(player.y) + y);
        let texture = map[arrayLocation];
        let tileData = texMap.get(texture);

        if (tileData === undefined || outOfBounds) {
          tileData = {
            x: 48,
            y: 48,
            tex: assets.envi,
            wall: false,
            solid: false,
            obstacle: false,
            deco: false,
          };
        }

        gctx.drawImage(
          tileData.tex,
          tileData.x,
          tileData.y,
          8,
          8,
          tileX,
          tileY,
          tSize,
          tSize
        );

        if (tileData.deco)
          gctx.drawImage(
            tileData.tex,
            tileData.x2,
            tileData.y2,
            8,
            8,
            tileX,
            tileY,
            tSize,
            tSize
          );

        if (tileData.obstacle) {
          let obstX = tSize * (xDiff + x);
          let obstY = tSize * (yDiff + y);

          let obstacleX = obstX * cos + obstY * -sin + xCenter;
          let obstacleY = obstX * sin + obstY * cos + yCenter;

          entity.push({
            tex: tileData.tex,
            tx: tileData.x2,
            ty: tileData.y2,
            x: obstacleX,
            y: obstacleY,
            size: tileData.size,
          });
        }

        if (tileData.wall) {
          roofs.push({
            tex: tileData.tex,
            tx: tileData.x,
            ty: tileData.y,
            x: tileX,
            y: tileY,
          });

          // Implement wall rendering logic
          // Omitted for brevity
        }
      }
    }
  }

  gctx.restore();

  // Sort entities by y-coordinate for proper rendering order
  entity.sort((a, b) => a.y - b.y);

  // Draw entities
  entity.forEach(e => {
    if (e.wall) {
      gctx.drawImage(e.tex, e.tx, e.ty, 1, 8, e.x, e.y, WALL_SIZE, -tSize);
    } else {
      gctx.drawImage(
        e.tex,
        e.tx,
        e.ty,
        8,
        8,
        Math.round(e.x) - e.size * 0.5,
        Math.round(e.y) - e.size,
        e.size,
        e.size
      );

      if (e.name && e.name !== player.name) {
        gctx.fillStyle = 'rgb(255,255,255)';
        gctx.fillText(e.name, e.x - 20, e.y + 20);
      }
    }
  });

  // Draw roofs if any
  if (roofs.length) {
    gctx.save();
    gctx.translate(xCenter, yCenter - tSize);
    gctx.rotate(player.r);
    roofs.forEach(e => {
      gctx.drawImage(
        e.tex,
        e.tx,
        e.ty,
        8,
        8,
        e.x - 1,
        e.y - 1,
        tSize + 2,
        tSize + 2
      );
    });
    gctx.restore();
  }
}
