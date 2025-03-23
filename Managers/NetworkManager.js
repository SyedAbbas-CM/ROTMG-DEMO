/**
 * NetworkManager.js (Server)
 * Handles WebSocket communication for the game server
 */
const WebSocket = require('ws');

class NetworkManager {
  /**
   * Creates the networking manager for the server
   * @param {Object} gameManager - Reference to the game manager
   */
  constructor(gameManager) {
    this.gameManager = gameManager;
    this.wss = null;
    this.clients = new Map(); // clientId -> {socket, player, lastUpdate}
    this.updateRate = 1000 / 20; // Send updates at 20 times per second by default
    this.lastBroadcastTime = 0;
    
    // Message handlers organized by type
    this.messageHandlers = {
      'PLAYER_UPDATE': this.handlePlayerUpdate.bind(this),
      'PLAYER_SHOOT': this.handlePlayerShoot.bind(this),
      'COLLISION': this.handleCollision.bind(this),
      'REQUEST_CHUNK': this.handleRequestChunk.bind(this),
      'PLAYER_JOIN': this.handlePlayerJoin.bind(this)
    };
  }
  
  /**
   * Initialize the WebSocket server
   * @param {Object} server - HTTP server instance
   */
  init(server) {
    this.wss = new WebSocket.Server({ server });
    
    // Set up event handlers
    this.wss.on('connection', (socket, req) => {
      this.handleConnection(socket, req);
    });
    
    console.log('WebSocket server initialized');
    
    // Start update loop
    this.startUpdateLoop();
  }
  
  /**
   * Handle a new WebSocket connection
   * @param {WebSocket} socket - New client socket
   * @param {Object} req - HTTP request object
   */
  handleConnection(socket, req) {
    // Generate a unique client ID
    const clientId = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
    
    // Store client information
    this.clients.set(clientId, {
      socket,
      player: {
        id: clientId,
        x: 50, // Default spawn position
        y: 50,
        name: `Player_${clientId.substr(0, 4)}`,
        health: 100
      },
      lastUpdate: Date.now()
    });
    
    console.log(`Client connected: ${clientId}`);
    
    // Set clientId property on socket for easy reference
    socket.clientId = clientId;
    
    // Send initial game state
    this.sendInitialState(socket, clientId);
    
    // Setup message handler
    socket.on('message', (message) => {
      this.handleMessage(socket, message);
    });
    
    // Setup disconnect handler
    socket.on('close', () => {
      this.handleDisconnect(clientId);
    });
    
    // Broadcast new player to everyone else
    this.broadcastPlayerJoin(clientId);
  }
  
  /**
   * Send initial game state to a newly connected client
   * @param {WebSocket} socket - Client socket
   * @param {string} clientId - Client ID
   */
  sendInitialState(socket, clientId) {
    try {
      // 1. Send client ID
      this.sendToClient(socket, {
        type: 'INIT_CLIENT',
        clientId: clientId,
        timestamp: Date.now()
      });
      
      // 2. Send map info (only metadata, actual chunks will be requested as needed)
      const mapInfo = this.gameManager.mapManager.getMapInfo();
      this.sendToClient(socket, {
        type: 'MAP_INFO',
        ...mapInfo
      });
      
      // 3. Send list of all players
      const players = {};
      this.clients.forEach((client, id) => {
        if (id !== clientId) { // Don't include the new player
          players[id] = client.player;
        }
      });
      
      this.sendToClient(socket, {
        type: 'PLAYERS_LIST',
        players: players
      });
      
      // 4. Send current enemies
      const enemies = this.gameManager.enemyManager.getEnemiesData();
      this.sendToClient(socket, {
        type: 'ENEMIES_LIST',
        enemies: enemies
      });
      
      // 5. Send current bullets
      const bullets = this.gameManager.bulletManager.getBulletsData();
      this.sendToClient(socket, {
        type: 'BULLETS_LIST',
        bullets: bullets
      });
      
      console.log(`Initial state sent to client ${clientId}`);
    } catch (error) {
      console.error('Error sending initial state:', error);
    }
  }
  
  /**
   * Handle a WebSocket message
   * @param {WebSocket} socket - Client socket
   * @param {string|Buffer} message - Raw message data
   */
  handleMessage(socket, message) {
    try {
      const data = JSON.parse(message);
      const handler = this.messageHandlers[data.type];
      
      if (handler) {
        handler(socket, data);
      } else {
        console.warn(`No handler for message type: ${data.type}`);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }
  
  /**
   * Handle client disconnection
   * @param {string} clientId - ID of disconnected client
   */
  handleDisconnect(clientId) {
    // Remove client from tracking
    this.clients.delete(clientId);
    
    // Broadcast player disconnect to remaining clients
    this.broadcast({
      type: 'PLAYER_LEAVE',
      clientId: clientId
    });
    
    console.log(`Client disconnected: ${clientId}`);
  }
  
  /**
   * Send a message to a specific client
   * @param {WebSocket} socket - Client socket
   * @param {Object} data - Message data
   */
  sendToClient(socket, data) {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(data));
    }
  }
  
  /**
   * Broadcast a message to all connected clients
   * @param {Object} data - Message data
   */
  broadcast(data) {
    this.wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(data));
      }
    });
  }
  
  /**
   * Broadcast to all clients except one
   * @param {Object} data - Message data
   * @param {string} excludeClientId - Client ID to exclude
   */
  broadcastExcept(data, excludeClientId) {
    this.wss.clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN && client.clientId !== excludeClientId) {
        client.send(JSON.stringify(data));
      }
    });
  }
  
  /**
   * Start the server update loop
   */
  startUpdateLoop() {
    setInterval(() => {
      this.sendWorldUpdates();
    }, this.updateRate);
  }
  
  /**
   * Send regular world state updates to clients
   */
  sendWorldUpdates() {
    const now = Date.now();
    
    // Don't send updates too frequently
    if (now - this.lastBroadcastTime < this.updateRate) {
      return;
    }
    
    this.lastBroadcastTime = now;
    
    try {
      // 1. Get updated positions of all enemies
      const enemyUpdates = this.gameManager.enemyManager.getEnemiesData();
      
      // 2. Get updated positions of all bullets
      const bulletUpdates = this.gameManager.bulletManager.getBulletsData();
      
      // 3. Get updated positions of all players
      const playerUpdates = {};
      this.clients.forEach((client, id) => {
        playerUpdates[id] = client.player;
      });
      
      // 4. Broadcast world updates to all clients
      this.broadcast({
        type: 'WORLD_UPDATE',
        enemies: enemyUpdates,
        bullets: bulletUpdates,
        players: playerUpdates,
        timestamp: now
      });
    } catch (error) {
      console.error('Error sending world updates:', error);
    }
  }
  
  /**
   * Broadcast player join event
   * @param {string} clientId - ID of new player
   */
  broadcastPlayerJoin(clientId) {
    const clientData = this.clients.get(clientId);
    if (!clientData) return;
    
    this.broadcastExcept({
      type: 'PLAYER_JOIN',
      player: clientData.player
    }, clientId);
  }
  
  /**
   * Handle player update message
   * @param {WebSocket} socket - Client socket
   * @param {Object} data - Message data
   */
  handlePlayerUpdate(socket, data) {
    const clientId = socket.clientId;
    const clientData = this.clients.get(clientId);
    
    if (!clientData) return;
    
    // Update player information
    const player = clientData.player;
    
    // Update position and other data
    if (data.x !== undefined) player.x = data.x;
    if (data.y !== undefined) player.y = data.y;
    if (data.rotation !== undefined) player.rotation = data.rotation;
    if (data.name !== undefined) player.name = data.name;
    
    // Update timestamp
    clientData.lastUpdate = Date.now();
  }
  
  /**
   * Handle player shoot message
   * @param {WebSocket} socket - Client socket
   * @param {Object} data - Message data
   */
  handlePlayerShoot(socket, data) {
    const clientId = socket.clientId;
    const clientData = this.clients.get(clientId);
    
    if (!clientData) return;
    
    // Extract bullet data
    const { x, y, angle, speed, lifetime } = data;
    
    // Create a new bullet
    const bulletId = this.gameManager.bulletManager.addBullet({
      x,
      y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      ownerId: clientId,
      lifetime: lifetime || 3000
    });
    
    // Broadcast new bullet to all clients
    this.broadcast({
      type: 'BULLET_CREATED',
      bulletId,
      x,
      y,
      angle,
      speed,
      ownerId: clientId,
      timestamp: Date.now()
    });
  }
  
  /**
   * Handle collision message
   * @param {WebSocket} socket - Client socket
   * @param {Object} data - Message data
   */
  handleCollision(socket, data) {
    const clientId = socket.clientId;
    
    // Add client ID to the collision data
    data.clientId = clientId;
    
    // Validate and process collision
    const result = this.gameManager.collisionManager.validateCollision(data);
    
    if (result.valid) {
      // Broadcast validated collision to all clients
      this.broadcast({
        type: 'COLLISION_RESULT',
        valid: true,
        bulletId: result.bulletId,
        enemyId: result.enemyId,
        damage: result.damage,
        enemyHealth: result.enemyHealth,
        enemyKilled: result.enemyKilled,
        timestamp: Date.now()
      });
    } else {
      // Send rejection only to the reporting client
      this.sendToClient(socket, {
        type: 'COLLISION_RESULT',
        valid: false,
        reason: result.reason,
        bulletId: result.bulletId,
        enemyId: result.enemyId
      });
    }
  }
  
  /**
   * Handle map chunk request
   * @param {WebSocket} socket - Client socket
   * @param {Object} data - Message data with chunk coordinates
   */
  handleRequestChunk(socket, data) {
    const { chunkX, chunkY } = data;
    
    // Get chunk data from the map manager
    const chunkData = this.gameManager.mapManager.getChunkData(chunkX, chunkY);
    
    if (chunkData) {
      this.sendToClient(socket, {
        type: 'CHUNK_DATA',
        chunkX,
        chunkY,
        data: chunkData
      });
    } else {
      this.sendToClient(socket, {
        type: 'CHUNK_NOT_FOUND',
        chunkX,
        chunkY
      });
    }
  }
  
  /**
   * Handle player join message (with custom info)
   * @param {WebSocket} socket - Client socket
   * @param {Object} data - Message data
   */
  handlePlayerJoin(socket, data) {
    const clientId = socket.clientId;
    const clientData = this.clients.get(clientId);
    
    if (!clientData) return;
    
    // Update player name or other customization
    if (data.name) {
      clientData.player.name = data.name;
    }
    
    // Broadcast updated player info
    this.broadcastPlayerJoin(clientId);
  }
  
  /**
   * Get player by client ID
   * @param {string} clientId - Client ID
   * @returns {Object|null} Player object
   */
  getPlayer(clientId) {
    const clientData = this.clients.get(clientId);
    return clientData ? clientData.player : null;
  }
  
  /**
   * Get all players
   * @returns {Object} Map of client IDs to player objects
   */
  getAllPlayers() {
    const players = {};
    this.clients.forEach((client, id) => {
      players[id] = client.player;
    });
    return players;
  }
}

module.exports = NetworkManager;