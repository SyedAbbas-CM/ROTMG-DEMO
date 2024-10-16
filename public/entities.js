// entities.js

import { enemies, bullets, player } from './gamestate.js';
import { xCenter, yCenter, offCenter, offCenterDiff, tSize } from './constants.js';
import { assets } from './assets.js';

// Define player speed
export const speed = 0.075;
const diagonalSpeed = 0.707 * speed;

// === PLAYER MANAGEMENT ===
export let playerList = {};

export function initializePlayers(playersData) {
  playerList = playersData;
}

export function updatePlayers(playersData) {
  playerList = playersData;
}

export function getOtherPlayersEntities() {
  let entities = [];
  const nearPX1 = player.x - 15;
  const nearPX2 = player.x + 15;
  const nearPY1 = player.y - 15;
  const nearPY2 = player.y + 15;

  for (const key in playerList) {
    if (Number(key) !== player.id && Object.prototype.hasOwnProperty.call(playerList, key)) {
      const e = playerList[key];
      if (e.x > nearPX1 && e.x < nearPX2 && e.y > nearPY1 && e.y < nearPY2) {
        let offCenterPlus = offCenter ? offCenterDiff : 0;
        let screenX = tSize * (e.x - player.x) - xCenter;
        let screenY = tSize * (e.y - player.y) - yCenter + offCenterPlus;
        const cos = Math.cos(player.r);
        const sin = Math.sin(player.r);
        let rotatedScreenX = screenX * cos + screenY * -sin + xCenter;
        let rotatedScreenY = screenX * sin + screenY * cos + yCenter;

        entities.push({
          x: rotatedScreenX,
          y: rotatedScreenY,
          tex: assets.char,
          tx: 0,
          ty: 0,
          size: 40,
          name: e.name,
        });
      }
    }
  }
  return entities;
}

// === ENEMY MANAGEMENT ===
export function createEnemy(x, y) {
  return {
    x: x,
    y: y,
    direction: Math.random() * 2 * Math.PI,
    speed: 0.5,
    shootCooldown: 100,
    shootTimer: 0,
    tx: 0,
    ty: 0,
    size: 40,
  };
}

export function updateEnemies() {
  enemies.forEach(enemy => {
    enemy.x += enemy.speed * Math.cos(enemy.direction);
    enemy.y += enemy.speed * Math.sin(enemy.direction);

    if (Math.random() < 0.01) {
      enemy.direction = Math.random() * 2 * Math.PI;
    }

    if (enemy.shootTimer <= 0) {
      shootEnemyBullet(enemy);
      enemy.shootTimer = enemy.shootCooldown;
    } else {
      enemy.shootTimer--;
    }
  });
}

export function shootBullet(mouseX, mouseY) {
  const dx = mouseX - xCenter;
  const dy = mouseY - yCenter;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const normalizedDx = dx / distance;
  const normalizedDy = dy / distance;
  const bulletSpeed = 2;

  const bullet = {
    x: player.x,
    y: player.y,
    velocityX: normalizedDx * bulletSpeed,
    velocityY: normalizedDy * bulletSpeed,
    tx: 16,
    ty: 16,
    size: 8,
    lifetime: 100,
  };

  bullets.push(bullet);
}

export function shootEnemyBullet(enemy) {
  const bulletSpeed = 1.5; // Adjust as needed
  const angle = Math.atan2(player.y - enemy.y, player.x - enemy.x);

  const bullet = {
    x: enemy.x,
    y: enemy.y,
    velocityX: Math.cos(angle) * bulletSpeed,
    velocityY: Math.sin(angle) * bulletSpeed,
    tx: 16,
    ty: 16,
    size: 8,
    lifetime: 100,
    owner: 'enemy', // To differentiate from player bullets
  };

  bullets.push(bullet);
}

export function updateBullets() {
  for (let i = bullets.length - 1; i >= 0; i--) {
    const bullet = bullets[i];
    bullet.x += bullet.velocityX;
    bullet.y += bullet.velocityY;
    bullet.lifetime--;

    if (bullet.lifetime <= 0) {
      bullets.splice(i, 1);
    }
  }
}
