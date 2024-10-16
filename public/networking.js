// networking.js

import { map, setMap } from './map.js';
import { texMap, setTexMap } from './map.js';
import { initializePlayers, updatePlayers, bullets } from './entities.js';
import { player } from './gamestate.js';

// === NETWORKING SETUP ===
const socket = new WebSocket('ws://127.0.0.1:3000');

socket.addEventListener('open', () => {
  console.log('[WebSocket] Connected to the server');
});

socket.addEventListener('error', error => {
  console.error('[WebSocket] Error occurred:', error);
});

socket.addEventListener('close', (event) => {
  console.log('[WebSocket] Disconnected from the server', event);
  if (event.wasClean) {
    console.log(`[WebSocket] Connection closed cleanly, code=${event.code}, reason=${event.reason}`);
  } else {
    console.warn('[WebSocket] Connection died unexpectedly');
  }
});

socket.addEventListener('message', event => {
  try {
    const data = JSON.parse(event.data);
    console.log(`[WebSocket] Received message: ${data.type}`, data);

    switch (data.type) {
      case 'MAP_UPDATE':
        handleMapUpdate(data);
        break;
      case 'TEXTURE_MAP':
        handleTextureMap(data);
        break;
      case 'NEW_TEXTURE_MAP':
        handleNewTextureMap(data);
        break;
      case 'MAP':
        handleFullMap(data);
        break;
      case 'INIT':
        handleInit(data);
        break;
      case 'UPDATE_PLAYER':
        handleUpdatePlayer(data);
        break;
      case 'YOUR_ID':
        handleYourId(data);
        break;
      case 'NEW_BULLET':
        handleNewBullet(data);
        break;
      case 'UPDATE_BULLETS':
        handleUpdateBullets(data);
        break;
      default:
        console.warn('[WebSocket] Unhandled message type:', data.type);
    }
  } catch (err) {
    console.error('[WebSocket] Error parsing message:', err, event.data);
  }
});

// === MESSAGE HANDLING FUNCTIONS ===

// Handle 'MAP_UPDATE' messages
function handleMapUpdate(data) {
  const { location, block } = data;
  if (typeof location === 'number' && typeof block !== 'undefined') {
    map[location] = block;
    console.log('[MAP_UPDATE] Map updated at location:', location, 'with block:', block);
  } else {
    console.warn('[MAP_UPDATE] Invalid data:', data);
  }
}

// Handle 'TEXTURE_MAP' messages
function handleTextureMap(data) {
  const { texMap: receivedTexMap } = data;
  if (receivedTexMap && typeof receivedTexMap === 'object') {
    const updatedTexMap = new Map(receivedTexMap);

    // Update texMap with the received data
    setTexMap(updatedTexMap);
    console.log('[TEXTURE_MAP] Texture map updated');
  } else {
    console.warn('[TEXTURE_MAP] Invalid data:', data);
  }
}

// Handle 'NEW_TEXTURE_MAP' messages
function handleNewTextureMap(data) {
  console.log('[NEW_TEXTURE_MAP] Received new texture map:', data);
  // Update texMap or perform other actions here
}

// Handle 'MAP' messages
function handleFullMap(data) {
  const { map: receivedMap } = data;
  if (Array.isArray(receivedMap)) {
    setMap(receivedMap);
    console.log('[MAP] Full map received and set');
  } else {
    console.warn('[MAP] Invalid data:', data);
  }
}

// Handle 'INIT' messages
function handleInit(data) {
  const { players, playerId } = data;
  if (players && typeof players === 'object') {
    initializePlayers(players);
    handleYourId({ playerId });
    console.log('[INIT] Initialization complete with players data');
  } else {
    console.warn('[INIT] Invalid data:', data);
  }
}

// Handle 'UPDATE_PLAYER' messages
function handleUpdatePlayer(data) {
  const { players } = data;
  if (players && typeof players === 'object') {
    updatePlayers(players);
    console.log('[UPDATE_PLAYER] Players updated:', players);
  } else {
    console.warn('[UPDATE_PLAYER] Invalid data:', data);
  }
}

// Handle 'YOUR_ID' messages
let myId; // Export this if other modules need to access the player's ID
function handleYourId(data) {
  const { playerId } = data;
  if (typeof playerId === 'number') {
    myId = playerId;
    player.id = myId;
    player.tx = ((myId % 10) * 8) % 56;
    console.log('[YOUR_ID] Assigned player ID:', myId);
  } else {
    console.warn('[YOUR_ID] Invalid data:', data);
  }
}

// Handle 'NEW_BULLET' messages
function handleNewBullet(data) {
  const { bullet } = data;
  if (bullet && typeof bullet === 'object') {
    bullets.push(bullet);
    console.log('[NEW_BULLET] New bullet added:', bullet);
  } else {
    console.warn('[NEW_BULLET] Invalid data:', data);
  }
}

// Handle 'UPDATE_BULLETS' messages
function handleUpdateBullets(data) {
  const { bullets: receivedBullets } = data;
  if (Array.isArray(receivedBullets)) {
    bullets.length = 0; // Clear existing bullets
    receivedBullets.forEach(bullet => bullets.push(bullet));
    console.log('[UPDATE_BULLETS] Bullets array updated');
  } else {
    console.warn('[UPDATE_BULLETS] Invalid data:', data);
  }
}

// === PLAYER MANAGEMENT ===

// Function to send player data to the server
export function sendPlayerData(playerData) {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'UPDATE_PLAYER', playerData }));
    console.log('[WebSocket] Sent player data:', playerData);
  } else {
    console.warn('[WebSocket] Cannot send data. ReadyState:', socket.readyState);
  }
}

// === PLAYER DATA SYNC ===

setTimeout(() => {
  setInterval(() => {
    const playerData = {
      name: player.name,
      x: player.x,
      y: player.y,
      // Add other relevant player data here
    };
    sendPlayerData(playerData);
  }, 1000 / 60); // Sends data at ~60 times per second
}, 1000); // Initial delay of 1 second

// === EXPORTS ===
export { myId };
export { socket };
