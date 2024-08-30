const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const mapSize = 100 ** 2; // Example map size
let map = new Array(mapSize).fill(0);
map[0] = 1;

// Store player data and bullets
let players = {};
let bullets = [];

const texMap = new Map();
texMap.set(0, { x: 48, y: 8, tex: "canvas3", wall: false, solid: false, x2: 48, y2: 8, deco: false });
texMap.set(1, { x: 32, y: 8, tex: "canvas3", wall: true, solid: true, x2: 8, y2: 8, deco: false });

let tex_map_size = texMap.size;

// Serve static files
app.use(express.static('public'));

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('New player connected');

    // Generate a unique ID for the player
    const playerId = Date.now();
    players[playerId] = { name: "none", x: 2, y: 2 }; // Example initial data

    // Notify the new player of existing players and bullets
    ws.send(JSON.stringify({ type: 'MAP', map }));
    ws.send(JSON.stringify({ type: 'YOUR_ID', playerId }));
    ws.send(JSON.stringify({ type: 'INIT', players }));
    ws.send(JSON.stringify({ type: 'TEXTURE_MAP', texMap: Array.from(texMap.entries()) }));
    ws.send(JSON.stringify({ type: 'INIT_BULLETS', bullets })); // Send existing bullets

    // Handle incoming messages from clients
    ws.on('message', (message) => {
        const data = JSON.parse(message);

        switch (data.type) {
            case 'UPDATE_PLAYER':
                // Update player data
                players[playerId] = data.playerData;

                // Broadcast updated player data to all clients
                const updateMessage = JSON.stringify({ type: 'UPDATE_PLAYER', players });
                wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(updateMessage);
                    }
                });
                break;

            case 'MAP_CHANGE':
                if (data.location !== 5050) {
                    map[data.location] = data.block;

                    // Broadcast map change to all clients
                    const mapUpdateMessage = JSON.stringify({ type: 'MAP_UPDATE', location: data.location, block: data.block });
                    wss.clients.forEach((client) => {
                        if (client.readyState === WebSocket.OPEN) {
                            client.send(mapUpdateMessage);
                        }
                    });
                }
                break;

            case 'NEW_TEXTURE_MAP':
                texMap.set(tex_map_size, data.options);
                tex_map_size++;

                const textureMapUpdate = JSON.stringify({ type: 'TEXTURE_MAP', texMap: Array.from(texMap.entries()) });
                wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(textureMapUpdate);
                    }
                });
                break;

            case 'SHOOT':
                // Create new bullet data
                const bullet = {
                    id: Date.now(),
                    playerId,
                    x: data.x,
                    y: data.y,
                    direction: data.direction,
                    speed: data.speed
                };
                bullets.push(bullet);

                // Broadcast new bullet to all clients
                const bulletMessage = JSON.stringify({ type: 'NEW_BULLET', bullet });
                wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(bulletMessage);
                    }
                });
                break;
        }
    });

    // Handle disconnection
    ws.on('close', () => {
        console.log('Player disconnected');
        delete players[playerId];

        // Notify all clients of the player removal
        const updateMessage = JSON.stringify({ type: 'UPDATE_PLAYER', players });
        wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(`player left ${playerId}`);
            }
        });
    });
});

// Broadcast player data and bullets at regular intervals
setInterval(() => {
    const updateMessage = JSON.stringify({ type: 'UPDATE_PLAYER', players });
    const bulletUpdateMessage = JSON.stringify({ type: 'UPDATE_BULLETS', bullets });

    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(updateMessage);
            client.send(bulletUpdateMessage);
        }
    });

    // Update bullets position
    bullets.forEach((bullet) => {
        bullet.x += bullet.speed * Math.cos(bullet.direction);
        bullet.y += bullet.speed * Math.sin(bullet.direction);
        // Remove bullet if it goes out of bounds (this is an example)
        if (bullet.x < 0 || bullet.x > mapSize || bullet.y < 0 || bullet.y > mapSize) {
            bullets = bullets.filter(b => b.id !== bullet.id);
        }
    });
}, 1000 / 30); // Update and broadcast 30 times per second

// Start the server
const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
