const express = require('express');
const http = require('http');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Store player data
let players = {};
let projectiles = {};


// Serve static files (e.g., your game files)
app.use(express.static('public'));

// Handle WebSocket connections
wss.on('connection', (ws) => {
    console.log('New player connected');

    const playerId = Date.now(); // Simple unique ID based on timestamp
    players[playerId] = {name:"none", x: 2, y: 2 }; // Example initial data
    projectiles[playerId] = []

    ws.send(JSON.stringify({ type: 'INIT', players, playerId }));

    // Handle incoming messages from clients
    ws.on('message', (message) => {
        let data = message
        try {
            data = JSON.parse(message);  // Attempt to parse the data
        } catch (error) {
            console.error("Malformed JSON received:", error);
            return;  // Stop further processing
        }
    
        if (data.type === 'UPDATE_PLAYER') {
            players[playerId] = data.playerData;
        }
        if (data.type === 'NEW_PROJECTILE') {
            let e = players[playerId]

           let index = projectiles[playerId].push({x: e.x, y: e.y, tan: data.tan, lifetime:250, spd: 0.2})
            
            const updateMessage = JSON.stringify({ type: 'NEW_PROJECTILE', projectile: projectiles[playerId][index -1] });
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


}, 100);

// Start the server
const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});


// projectiles = { id: [ {x,y}, {x,y}, {x,y} ], id2: [ {x,y}, {x,y}, {x,y} ] }
setInterval(() => {

    for(const key in projectiles){
        let array = projectiles[key]

        array = array.filter(e => e.lifetime > 0);

        array.forEach(e =>{
            e.lifetime -= 100
        })
    }

}, 100);