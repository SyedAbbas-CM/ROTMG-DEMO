const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const mapSize = 100**2 //  prompt("map size?",10)
let map = new Array(mapSize).fill(0)
map[0] = 1

// Store player data
let players = {};

const texMap = new Map();


texMap.set(0, { x: 48, y: 8, tex: "canvas3", wall: false, solid: false, x2: 48, y2: 8 , deco: false});
texMap.set(1, { x: 32, y: 8, tex: "canvas3", wall: true, solid: true, x2: 8, y2: 8 , deco: false});

let tex_map_size = texMap.size

// Serve static files (e.g., your game files)
app.use(express.static('public'));

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('New player connected');
    
    // Generate a unique ID for the player
    const playerId = Date.now(); // Simple unique ID based on timestamp
    players[playerId] = {name:"none", x: 2, y: 2 }; // Example initial data

    // Notify the new player of existing players
    ws.send(JSON.stringify({ type: 'MAP',  map }));
    ws.send(JSON.stringify({ type: 'YOUR_ID', playerId }));
    ws.send(JSON.stringify({ type: 'INIT', players }));
    ws.send(JSON.stringify({ type: 'TEXTURE_MAP', texMap: Array.from(texMap.entries()) }));

    // Handle incoming messages from clients
    ws.on('message', (message) => {
        const data = JSON.parse(message);

        if (data.type === 'UPDATE_PLAYER') {
            // Update player data
            players[playerId] = data.playerData;

            // Broadcast updated player data to all clients
            const updateMessage = JSON.stringify({ type: 'UPDATE_PLAYER', players });
            wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(updateMessage);
                }
            });
        }
        if (data.type === 'MAP_CHANGE' && data.location !== 5050) {
            // Update player data
            console.log(data)
            map[data.location] = data.block

             // Broadcast updated player data to all clients
            const updateMessage = JSON.stringify({type: 'MAP_UPDATE', location: data.location, block: data.block });
            wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(updateMessage);
                }
            }); 
        }
        if (data.type === 'NEW_TEXTURE_MAP') {
            console.log(data)

            texMap.set(tex_map_size, data.options);
            tex_map_size++

            const updateMessage = JSON.stringify({ type: 'TEXTURE_MAP', texMap: Array.from(texMap.entries()) })
            
            wss.clients.forEach((client) => {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(updateMessage);
                }
            });
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

// Broadcast player data at regular intervals
setInterval(() => {
    const updateMessage = JSON.stringify({ type: 'UPDATE_PLAYER', players });
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(updateMessage);
        }
    });
}, 1000); // Broadcast every 1 second

// Start the server
const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});