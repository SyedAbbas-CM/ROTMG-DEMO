const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const mapSize = 100 ** 2;
let map = new Array(mapSize).fill(0);
map[0] = 1;

// Store player, enemy, and bullet data
let players = {};
let enemies = [];
let bullets = [];

const texMap = new Map();
texMap.set(0, { x: 48, y: 8, tex: "canvas3", wall: false, solid: false, x2: 48, y2: 8, deco: false });
texMap.set(1, { x: 32, y: 8, tex: "canvas3", wall: true, solid: true, x2: 8, y2: 8, deco: false });

let tex_map_size = texMap.size;

// Serve static files (e.g., your game files)
app.use(express.static('public'));

// Function to create a new enemy
function createEnemy(x, y) {
    const enemy = {
        id: Date.now() + Math.random(),
        x: x,
        y: y,
        direction: Math.random() * 2 * Math.PI,
        speed: 0.5,
        shootCooldown: 100,
        shootTimer: 0,
    };
    enemies.push(enemy);
    console.log('[createEnemy] Created enemy at position:', { x, y });
    return enemy;
}

// Function to create a new bullet
function createBullet(x, y, direction, speed) {
    const bullet = {
        id: Date.now() + Math.random(),
        x: x,
        y: y,
        direction: direction,
        speed: speed,
    };
    bullets.push(bullet);
    console.log('[createBullet] Bullet created at position:', { x, y }, 'with direction:', direction, 'and speed:', speed);
    return bullet;
}

// Initialize some enemies
createEnemy(50, 50);
createEnemy(150, 150);

// Update game state (enemies and bullets)
function updateGameState() {
    console.time('updateGameState');
    try {
        enemies.forEach(enemy => {
            enemy.x += enemy.speed * Math.cos(enemy.direction);
            enemy.y += enemy.speed * Math.sin(enemy.direction);

            if (Math.random() < 0.01) {
                enemy.direction = Math.random() * 2 * Math.PI;
            }

            if (enemy.shootTimer <= 0) {
                createBullet(enemy.x, enemy.y, Math.random() * 2 * Math.PI, 1);
                enemy.shootTimer = enemy.shootCooldown;
                console.log('[updateGameState] Enemy shot a bullet:', enemy);
            } else {
                enemy.shootTimer--;
            }
        });

        bullets.forEach((bullet, index) => {
            bullet.x += bullet.speed * Math.cos(bullet.direction);
            bullet.y += bullet.speed * Math.sin(bullet.direction);

            if (bullet.x < 0 || bullet.x > 100 || bullet.y < 0 || bullet.y > 100) {
                bullets.splice(index, 1);
                console.log('[updateGameState] Bullet removed at index:', index);
            }
        });
    } catch (error) {
        console.error('[updateGameState] Error updating game state:', error);
    }
    console.timeEnd('updateGameState');
}

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('[WebSocket] New player connected');

    const playerId = Date.now();
    players[playerId] = { name: "none", x: 2, y: 2 };

    try {
        ws.send(JSON.stringify({ type: 'MAP', map }));
        ws.send(JSON.stringify({ type: 'YOUR_ID', playerId }));
        ws.send(JSON.stringify({ type: 'INIT', players }));
        ws.send(JSON.stringify({ type: 'INIT_ENEMIES', enemies }));
        ws.send(JSON.stringify({ type: 'INIT_BULLETS', bullets }));
        ws.send(JSON.stringify({ type: 'TEXTURE_MAP', texMap: Array.from(texMap.entries()) }));
    } catch (error) {
        console.error('[WebSocket] Error sending initialization data:', error);
    }

    ws.on('message', (message) => {
        console.time('messageHandling');
        try {
            const data = JSON.parse(message);
            console.log('[WebSocket] Received message:', data);

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
                    console.warn('[WebSocket] Unhandled message type:', data.type);
            }
        } catch (error) {
            console.error('[messageHandling] Error handling message:', error);
        }
        console.timeEnd('messageHandling');
    });

    ws.on('close', () => {
        console.log('[WebSocket] Player disconnected:', playerId);
        delete players[playerId];
        broadcast({ type: 'UPDATE_PLAYER', players });
    });

    ws.on('error', (error) => {
        console.error('[WebSocket] Error occurred with player connection:', playerId, error);
    });
});

// Broadcast data to all connected clients
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
        console.error('[broadcast] Error broadcasting data:', error);
    }
    console.timeEnd('broadcast');
}

// Broadcast game state data at regular intervals
setInterval(() => {
    console.time('broadcastUpdate');
    try {
        updateGameState();
        broadcast({ type: 'UPDATE_PLAYER', players });
        broadcast({ type: 'UPDATE_ENEMIES', enemies });
        broadcast({ type: 'UPDATE_BULLETS', bullets });
    } catch (error) {
        console.error('[broadcastUpdate] Error broadcasting updates:', error);
    }
    console.timeEnd('broadcastUpdate');
}, 1000 / 30);

// Start the server
const PORT = 3000;
server.listen(PORT, () => {
    console.log(`[Server] Server is running on port ${PORT}`);
});

server.on('error', (error) => {
    console.error('[Server] Error occurred:', error);
});
