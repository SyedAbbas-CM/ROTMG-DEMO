// server.js

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const fs = require('fs');
const mapRoutes = require('./routes/mapRoutes');
const spriteRoutes = require('./routes/spriteRoutes');

// Import the single GameManager
const { default: GameManager } = require('./src/GameManager.js');

// Create Express + WebSocket
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// We'll put routes in separate files
const mapRoutes = require('./src/routes/mapRoutes');
const spriteRoutes = require('./src/routes/spriteRoutes');

app.use(express.json());
app.use(express.static('public'));

// Use the external routes
app.use('/map', mapRoutes);
app.use('/assets/spritesheets', spriteRoutes); // etc.

// Create our single game manager
const gameManager = new GameManager();

// Keep track of players
let players = {};

// WebSocket Connection
wss.on('connection', (ws) => {
  console.log('[WS] new player connected');
  const playerId = Date.now();
  players[playerId] = { name: 'none', x: 2, y: 2 };

  // Send initial data
  try {
    ws.send(JSON.stringify({ type: 'YOUR_ID', playerId }));
    ws.send(JSON.stringify({ type: 'INIT_ENEMIES', enemies: gameManager.getEnemyData() }));
    ws.send(JSON.stringify({ type: 'INIT_BULLETS', bullets: gameManager.getBulletData() }));
  } catch (err) {
    console.error('[WS] Error sending data:', err);
  }

  ws.on('message', (msg) => {
    try {
      const data = JSON.parse(msg);
      switch (data.type) {
        case 'UPDATE_PLAYER':
          players[playerId] = data.playerData;
          broadcast({ type: 'UPDATE_PLAYER', players });
          break;

        case 'SHOOT':
          // We'll call gameManager.bulletManager.addBullet
          gameManager.bulletManager.addBullet(data.x, data.y, data.vx, data.vy, data.life);
          // Then broadcast new bullet
          broadcast({ type: 'NEW_BULLET', bullet: { x: data.x, y: data.y, vx: data.vx, vy: data.vy } });
          break;

        default:
          console.warn('[WS] Unhandled msg type:', data.type);
      }
    } catch (err) {
      console.error('[WS] message error:', err);
    }
  });

  ws.on('close', () => {
    console.log('[WS] player disconnected:', playerId);
    delete players[playerId];
    broadcast({ type: 'UPDATE_PLAYER', players });
  });

  ws.on('error', (err) => {
    console.error('[WS] error for player:', playerId, err);
  });
});

function broadcast(data) {
  const message = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
}

server.listen(3000, () => {
  console.log('[Server] listening on 3000');
});

// Our main update loop
setInterval(() => {
  // We do a very rough dt
  const dt = 1/30;
  gameManager.update(dt);

  // Then broadcast updated bullets/enemies
  broadcast({ type: 'UPDATE_ENEMIES', enemies: gameManager.getEnemyData() });
  broadcast({ type: 'UPDATE_BULLETS', bullets: gameManager.getBulletData() });
}, 1000/30);
