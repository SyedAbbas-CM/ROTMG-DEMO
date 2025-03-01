// server.js
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

// Create Express + WebSocket
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Optionally parse JSON request bodies for sprite editor
app.use(express.json());
 
// Serve static files from "public"
app.use(express.static('public'));

/* -----------------------
   GAME STATE MANAGEMENT
------------------------- */
const mapSize = 100 ** 2;
let map = new Array(mapSize).fill(0);
map[0] = 1;

// Player, enemy, bullet data
let players = {};
let enemies = [];
let bullets = [];

// Example texture map
const texMap = new Map();
texMap.set(0, { x: 48, y: 8, tex: "canvas3", wall: false, solid: false, x2: 48, y2: 8, deco: false });
texMap.set(1, { x: 32, y: 8, tex: "canvas3", wall: true, solid: true, x2: 8, y2: 8, deco: false });
let tex_map_size = texMap.size;

// Example: Create a couple enemies
function createEnemy(x, y) {
  const enemy = {
    id: Date.now() + Math.random(),
    x,
    y,
    direction: Math.random() * 2 * Math.PI,
    speed: 0.5,
    shootCooldown: 100,
    shootTimer: 0,
  };
  enemies.push(enemy);
  console.log('[createEnemy] Enemy at:', { x, y });
  return enemy;
}
function createBullet(x, y, direction, speed) {
  const bullet = {
    id: Date.now() + Math.random(),
    x,
    y,
    direction,
    speed,
  };
  bullets.push(bullet);
  console.log('[createBullet] Bullet created:', bullet);
  return bullet;
}
createEnemy(50, 50);
createEnemy(150, 150);

// Update game state
function updateGameState() {
  console.time('updateGameState');
  try {
    enemies.forEach(enemy => {
      enemy.x += enemy.speed * Math.cos(enemy.direction);
      enemy.y += enemy.speed * Math.sin(enemy.direction);

      // Random direction change
      if (Math.random() < 0.01) {
        enemy.direction = Math.random() * 2 * Math.PI;
      }

      // Shooting
      if (enemy.shootTimer <= 0) {
        createBullet(enemy.x, enemy.y, Math.random() * 2 * Math.PI, 1);
        enemy.shootTimer = enemy.shootCooldown;
        console.log('[updateGameState] Enemy shot a bullet', enemy);
      } else {
        enemy.shootTimer--;
      }
    });

    // Update bullets
    bullets.forEach((bullet, idx) => {
      bullet.x += bullet.speed * Math.cos(bullet.direction);
      bullet.y += bullet.speed * Math.sin(bullet.direction);

      if (bullet.x < 0 || bullet.x > 100 || bullet.y < 0 || bullet.y > 100) {
        bullets.splice(idx, 1);
        console.log('[updateGameState] Bullet removed at idx:', idx);
      }
    });
  } catch (error) {
    console.error('[updateGameState] Error:', error);
  }
  console.timeEnd('updateGameState');
}

// WebSocket events
wss.on('connection', ws => {
  console.log('[WebSocket] New player connected');
  const playerId = Date.now();
  players[playerId] = { name: "none", x: 2, y: 2 };

  // Send initial data
  try {
    ws.send(JSON.stringify({ type: 'MAP', map }));
    ws.send(JSON.stringify({ type: 'YOUR_ID', playerId }));
    ws.send(JSON.stringify({ type: 'INIT', players }));
    ws.send(JSON.stringify({ type: 'INIT_ENEMIES', enemies }));
    ws.send(JSON.stringify({ type: 'INIT_BULLETS', bullets }));
    ws.send(JSON.stringify({
      type: 'TEXTURE_MAP',
      texMap: Array.from(texMap.entries())
    }));
  } catch (error) {
    console.error('[WebSocket] Error sending init data:', error);
  }

  // Handle incoming messages
  ws.on('message', message => {
    console.time('messageHandling');
    try {
      const data = JSON.parse(message);
      console.log('[WebSocket] msg:', data);

      switch (data.type) {
        case 'UPDATE_PLAYER':
          players[playerId] = data.playerData;
          broadcast({ type: 'UPDATE_PLAYER', players });
          break;
        case 'MAP_CHANGE':
          if (data.location !== 5050) {
            map[data.location] = data.block;
            broadcast({ type: 'MAP_UPDATE', location: data.location, block: data.block });
          }
          break;
        case 'NEW_TEXTURE_MAP':
          texMap.set(tex_map_size, data.options);
          tex_map_size++;
          broadcast({ type: 'TEXTURE_MAP', texMap: Array.from(texMap.entries()) });
          break;
        case 'SHOOT':
          createBullet(data.x, data.y, data.direction, data.speed);
          broadcast({ type: 'NEW_BULLET', bullet: data });
          break;
        default:
          console.warn('[WebSocket] Unhandled msg type:', data.type);
      }
    } catch (err) {
      console.error('[messageHandling] Error:', err);
    }
    console.timeEnd('messageHandling');
  });

  ws.on('close', () => {
    console.log('[WebSocket] Player disconnected:', playerId);
    delete players[playerId];
    broadcast({ type: 'UPDATE_PLAYER', players });
  });
  ws.on('error', err => {
    console.error('[WebSocket] Error for player:', playerId, err);
  });
});

// Broadcast data to all WS clients
function broadcast(data) {
  console.time('broadcast');
  try {
    const message = JSON.stringify(data);
    wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    });
    console.log('[broadcast] Data broadcasted:', data.type);
  } catch (error) {
    console.error('[broadcast] Error broadcasting:', error);
  }
  console.timeEnd('broadcast');
}

// Periodic game update broadcast
setInterval(() => {
  console.time('broadcastUpdate');
  try {
    updateGameState();
    broadcast({ type: 'UPDATE_PLAYER', players });
    broadcast({ type: 'UPDATE_ENEMIES', enemies });
    broadcast({ type: 'UPDATE_BULLETS', bullets });
  } catch (error) {
    console.error('[broadcastUpdate] Error:', error);
  }
  console.timeEnd('broadcastUpdate');
}, 1000 / 30);

/* ------------------------
   SPRITE EDITOR ENDPOINTS
-------------------------- */

// (A) GET /assets/spritesheets
// Return a list of the .png/.jpg files in public/assets/spritesheets (no extension).
app.get('/assets/spritesheets', (req, res) => {
  try {
    const dirPath = path.join(__dirname, 'public', 'assets', 'spritesheets');
    fs.readdir(dirPath, (err, files) => {
      if (err) {
        return res.status(500).json({ error: 'Failed to read directory' });
      }
      // filter image extensions & strip .png/.jpg
      const sheetNames = files
        .filter(file => file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg'))
        .map(file => path.basename(file, path.extname(file)));

      res.json(sheetNames);
    });
  } catch (err) {
    console.error('[GET /assets/spritesheets] Error:', err);
    res.status(500).json({ error: err.message });
  }
});

// (B) POST /assets/spritesheets/:sheetName
// Receive bounding box or group data from sprite editor as JSON, then store it on the server.
app.post('/assets/spritesheets/:sheetName', (req, res) => {
  const { sheetName } = req.params;
  const jsonFilePath = path.join(__dirname, 'public', 'assets', 'spritesheets', `${sheetName}.json`);
  const data = req.body; // The sprite data posted from the editor

  // Could do a MERGE with existing data or just overwrite
  // For now, let's overwrite (or create new if none)
  fs.writeFile(jsonFilePath, JSON.stringify(data, null, 2), (err) => {
    if (err) {
      console.error('[POST /assets/spritesheets/:sheetName] Write error:', err);
      return res.status(500).json({ success: false, error: 'Could not save JSON' });
    }
    console.log(`[POST /assets/spritesheets/:sheetName] Wrote file: ${jsonFilePath}`);
    return res.json({ success: true });
  });
});

/* 
   If you want a route for "globalGroups" (like /assets/spritesheets/globalGroups), 
   you can add a similar POST endpoint below.
*/

// Start server
const PORT = 3000;
server.listen(PORT, () => {
  console.log(`[Server] Running on port ${PORT}`);
});

server.on('error', (error) => {
  console.error('[Server] Error:', error);
});
