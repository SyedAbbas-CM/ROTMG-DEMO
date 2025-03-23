/**
 * NetworkManager.js (Client)
 * Handles WebSocket communication with the game server
 */
export class NetworkManager {
    /**
     * Creates the client networking manager
     * @param {string} serverUrl - WebSocket server URL
     * @param {Object} game - Reference to the main game object
     */
    constructor(serverUrl, game) {
      this.serverUrl = serverUrl;
      this.game = game;
      this.socket = null;
      this.connected = false;
      this.clientId = null;
      this.lastServerTime = 0;
      this.serverTimeOffset = 0;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.reconnectDelay = 2000; // ms
      
      // Message queue for messages that should be sent once connected
      this.messageQueue = [];
      
      // Register message handlers
      this.messageHandlers = {
        'INIT_CLIENT': this.handleInitClient.bind(this),
        'MAP_INFO': this.handleMapInfo.bind(this),
        'PLAYERS_LIST': this.handlePlayersList.bind(this),
        'ENEMIES_LIST': this.handleEnemiesList.bind(this),
        'BULLETS_LIST': this.handleBulletsList.bind(this),
        'WORLD_UPDATE': this.handleWorldUpdate.bind(this),
        'PLAYER_JOIN': this.handlePlayerJoin.bind(this),
        'PLAYER_LEAVE': this.handlePlayerLeave.bind(this),
        'BULLET_CREATED': this.handleBulletCreated.bind(this),
        'COLLISION_RESULT': this.handleCollisionResult.bind(this),
        'CHUNK_DATA': this.handleChunkData.bind(this),
        'CHUNK_NOT_FOUND': this.handleChunkNotFound.bind(this)
      };
    }
    
    /**
     * Connect to the WebSocket server
     * @returns {Promise} Resolves when connected
     */
    connect() {
      return new Promise((resolve, reject) => {
        try {
          console.log(`Connecting to server: ${this.serverUrl}`);
          this.socket = new WebSocket(this.serverUrl);
          
          this.socket.onopen = () => {
            console.log('Connected to server');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.drainMessageQueue();
            resolve();
          };
          
          this.socket.onmessage = (event) => {
            this.handleMessage(event.data);
          };
          
          this.socket.onclose = () => {
            console.log('Disconnected from server');
            this.connected = false;
            this.attemptReconnect();
            reject(new Error('Disconnected from server'));
          };
          
          this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            reject(error);
          };
        } catch (error) {
          console.error('Failed to connect to server:', error);
          reject(error);
        }
      });
    }
    
    /**
     * Attempt to reconnect to the server
     */
    attemptReconnect() {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.log('Maximum reconnect attempts reached');
        this.game.handleDisconnect();
        return;
      }
      
      this.reconnectAttempts++;
      
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect().catch(() => {
          // Error already logged in connect method
        });
      }, this.reconnectDelay);
    }
    
    /**
     * Send queued messages after connection is established
     */
    drainMessageQueue() {
      while (this.messageQueue.length > 0) {
        const message = this.messageQueue.shift();
        this.send(message);
      }
    }
    
    /**
     * Handle an incoming WebSocket message
     * @param {string} rawMessage - Raw message data
     */
    handleMessage(rawMessage) {
      try {
        const message = JSON.parse(rawMessage);
        
        // Update server time offset
        if (message.timestamp) {
          this.lastServerTime = message.timestamp;
          this.serverTimeOffset = Date.now() - message.timestamp;
        }
        
        // Find and call the appropriate handler
        const handler = this.messageHandlers[message.type];
        
        if (handler) {
          handler(message);
        } else {
          console.warn(`No handler for message type: ${message.type}`);
        }
      } catch (error) {
        console.error('Error handling message:', error);
      }
    }
    
    /**
     * Send a message to the server
     * @param {Object} data - Message data
     */
    send(data) {
      if (!this.connected) {
        this.messageQueue.push(data);
        return;
      }
      
      try {
        this.socket.send(JSON.stringify(data));
      } catch (error) {
        console.error('Error sending message:', error);
        this.messageQueue.push(data);
      }
    }
    
    /**
     * Send player update to server
     * @param {Object} playerData - Player state data
     */
    sendPlayerUpdate(playerData) {
      this.send({
        type: 'PLAYER_UPDATE',
        ...playerData,
        timestamp: Date.now()
      });
    }
    
    /**
     * Send shoot event to server
     * @param {Object} bulletData - Bullet data
     */
    sendShoot(bulletData) {
      this.send({
        type: 'PLAYER_SHOOT',
        ...bulletData,
        timestamp: Date.now()
      });
    }
    
    /**
     * Send collision to server
     * @param {Object} collisionData - Collision data
     */
    sendCollision(collisionData) {
      this.send({
        type: 'COLLISION',
        ...collisionData,
        timestamp: Date.now()
      });
    }
    
    /**
     * Request a map chunk from the server
     * @param {number} chunkX - Chunk X coordinate
     * @param {number} chunkY - Chunk Y coordinate
     */
    requestChunk(chunkX, chunkY) {
      this.send({
        type: 'REQUEST_CHUNK',
        chunkX,
        chunkY,
        timestamp: Date.now()
      });
    }
    
    /**
     * Send player join with custom info
     * @param {string} name - Player name
     */
    sendPlayerJoin(name) {
      this.send({
        type: 'PLAYER_JOIN',
        name,
        timestamp: Date.now()
      });
    }
    
    /**
     * Handle INIT_CLIENT message
     * @param {Object} data - Message data
     */
    handleInitClient(data) {
      this.clientId = data.clientId;
      this.game.setClientId(this.clientId);
      console.log(`Received client ID: ${this.clientId}`);
    }
    
    /**
     * Handle MAP_INFO message
     * @param {Object} data - Message data
     */
    handleMapInfo(data) {
      this.game.initMap(data);
      console.log('Received map info');
    }
    
    /**
     * Handle PLAYERS_LIST message
     * @param {Object} data - Message data
     */
    handlePlayersList(data) {
      this.game.setPlayers(data.players);
      console.log(`Received players list: ${Object.keys(data.players).length} players`);
    }
    
    /**
     * Handle ENEMIES_LIST message
     * @param {Object} data - Message data
     */
    handleEnemiesList(data) {
      this.game.setEnemies(data.enemies);
      console.log(`Received enemies list: ${data.enemies.length} enemies`);
    }
    
    /**
     * Handle BULLETS_LIST message
     * @param {Object} data - Message data
     */
    handleBulletsList(data) {
      this.game.setBullets(data.bullets);
      console.log(`Received bullets list: ${data.bullets.length} bullets`);
    }
    
    /**
     * Handle WORLD_UPDATE message
     * @param {Object} data - Message data
     */
    handleWorldUpdate(data) {
      this.game.updateWorld(data.enemies, data.bullets, data.players);
    }
    
    /**
     * Handle PLAYER_JOIN message
     * @param {Object} data - Message data
     */
    handlePlayerJoin(data) {
      this.game.addPlayer(data.player);
      console.log(`Player joined: ${data.player.id}`);
    }
    
    /**
     * Handle PLAYER_LEAVE message
     * @param {Object} data - Message data
     */
    handlePlayerLeave(data) {
      this.game.removePlayer(data.clientId);
      console.log(`Player left: ${data.clientId}`);
    }
    
    /**
     * Handle BULLET_CREATED message
     * @param {Object} data - Message data
     */
    handleBulletCreated(data) {
      this.game.addBullet({
        id: data.bulletId,
        x: data.x,
        y: data.y,
        vx: Math.cos(data.angle) * data.speed,
        vy: Math.sin(data.angle) * data.speed,
        ownerId: data.ownerId
      });
    }
    
    /**
     * Handle COLLISION_RESULT message
     * @param {Object} data - Message data
     */
    handleCollisionResult(data) {
      if (data.valid) {
        this.game.applyCollision(data);
        
        if (data.enemyKilled) {
          this.game.handleEnemyKilled(data.enemyId);
        }
      }
    }
    
    /**
     * Handle CHUNK_DATA message
     * @param {Object} data - Message data
     */
    handleChunkData(data) {
      this.game.setChunkData(data.chunkX, data.chunkY, data.data);
      console.log(`Received chunk data for (${data.chunkX}, ${data.chunkY})`);
    }
    
    /**
     * Handle CHUNK_NOT_FOUND message
     * @param {Object} data - Message data
     */
    handleChunkNotFound(data) {
      console.warn(`Chunk not found: (${data.chunkX}, ${data.chunkY})`);
      // Optionally generate a fallback client-side chunk
      this.game.generateFallbackChunk(data.chunkX, data.chunkY);
    }
    
    /**
     * Get the estimated server time
     * @returns {number} Server time in milliseconds
     */
    getServerTime() {
      return Date.now() - this.serverTimeOffset;
    }
    
    /**
     * Check if connected to server
     * @returns {boolean} True if connected
     */
    isConnected() {
      return this.connected;
    }
    
    /**
     * Get the client ID
     * @returns {string} Client ID
     */
    getClientId() {
      return this.clientId;
    }
    
    /**
     * Disconnect from the server
     */
    disconnect() {
      if (this.socket) {
        this.socket.close();
      }
    }
  } 