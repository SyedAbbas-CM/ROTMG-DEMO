// public/src/network/ClientNetworkManager.js

/**
 * ClientNetworkManager
 * Handles WebSocket communication with the game server using binary packet format
 */
export class ClientNetworkManager {
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
      this.connecting = false;
      this.clientId = null;
      this.lastServerTime = 0;
      this.serverTimeOffset = 0;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.reconnectDelay = 2000; // ms
      
      // Message queue for messages that should be sent once connected
      this.messageQueue = [];
      
      // Register message handlers for different message types
      this.handlers = {};
      Object.values(MessageType).forEach(type => {
          this.handlers[type] = () => {};
      });
      
      // Set up specific handlers
      this.setupMessageHandlers();
  }
  
  /**
   * Set up message type handlers
   */
  setupMessageHandlers() {
      this.handlers[MessageType.HANDSHAKE_ACK] = (data) => {
          this.clientId = data.clientId;
          console.log(`Received client ID: ${this.clientId}`);
          if (this.game.setClientId) {
              this.game.setClientId(this.clientId);
          }
      };
      
      this.handlers[MessageType.MAP_INFO] = (data) => {
          console.log('Received map info:', data);
          if (this.game.initMap) {
              this.game.initMap(data);
          }
      };
      
      this.handlers[MessageType.PLAYER_LIST] = (data) => {
          console.log(`Received players list: ${Object.keys(data.players).length} players`);
          if (this.game.setPlayers) {
              this.game.setPlayers(data.players);
          }
      };
      
      this.handlers[MessageType.ENEMY_LIST] = (data) => {
          console.log(`Received enemies list: ${data.enemies.length} enemies`);
          if (this.game.setEnemies) {
              this.game.setEnemies(data.enemies);
          }
      };
      
      this.handlers[MessageType.BULLET_LIST] = (data) => {
          console.log(`Received bullets list: ${data.bullets.length} bullets`);
          if (this.game.setBullets) {
              this.game.setBullets(data.bullets);
          }
      };
      
      this.handlers[MessageType.WORLD_UPDATE] = (data) => {
          if (this.game.updateWorld) {
              this.game.updateWorld(data.enemies, data.bullets, data.players);
          }
      };
      
      this.handlers[MessageType.PLAYER_JOIN] = (data) => {
          console.log(`Player joined: ${data.player.id}`);
          if (this.game.addPlayer) {
              this.game.addPlayer(data.player);
          }
      };
      
      this.handlers[MessageType.PLAYER_LEAVE] = (data) => {
          console.log(`Player left: ${data.clientId}`);
          if (this.game.removePlayer) {
              this.game.removePlayer(data.clientId);
          }
      };
      
      this.handlers[MessageType.BULLET_CREATE] = (data) => {
          if (this.game.addBullet) {
              this.game.addBullet({
                  id: data.id,
                  x: data.x,
                  y: data.y,
                  vx: Math.cos(data.angle) * data.speed,
                  vy: Math.sin(data.angle) * data.speed,
                  ownerId: data.ownerId,
                  damage: data.damage || 10,
                  lifetime: data.lifetime || 3.0
              });
          }
      };
      
      this.handlers[MessageType.COLLISION_RESULT] = (data) => {
          if (data.valid && this.game.applyCollision) {
              this.game.applyCollision(data);
              
              if (data.enemyKilled && this.game.handleEnemyKilled) {
                  this.game.handleEnemyKilled(data.enemyId);
              }
          }
      };
      
      this.handlers[MessageType.CHUNK_DATA] = (data) => {
          console.log(`Received chunk data for (${data.chunkX}, ${data.chunkY})`);
          if (this.game.setChunkData) {
              this.game.setChunkData(data.chunkX, data.chunkY, data.chunk);
          }
      };
      
      this.handlers[MessageType.CHUNK_NOT_FOUND] = (data) => {
          console.warn(`Chunk not found: (${data.chunkX}, ${data.chunkY})`);
          if (this.game.generateFallbackChunk) {
              this.game.generateFallbackChunk(data.chunkX, data.chunkY);
          }
      };
  }
  
  /**
   * Connect to the WebSocket server
   * @returns {Promise} Resolves when connected
   */
  connect() {
      return new Promise((resolve, reject) => {
          if (this.connected) {
              resolve();
              return;
          }
          
          if (this.connecting) {
              this.messageQueue.push({ resolve, reject });
              return;
          }
          
          this.connecting = true;
          
          try {
              console.log(`Connecting to server: ${this.serverUrl}`);
              this.socket = new WebSocket(this.serverUrl);
              
              // Set binary type for ArrayBuffer data
              this.socket.binaryType = 'arraybuffer';
              
              this.socket.onopen = () => {
                  console.log('Connected to server');
                  this.connected = true;
                  this.connecting = false;
                  this.reconnectAttempts = 0;
                  
                  // Send handshake
                  this.sendHandshake();
                  
                  // Start ping
                  this.startPing();
                  
                  // Drain message queue
                  this.drainMessageQueue();
                  
                  resolve();
              };
              
              this.socket.onmessage = (event) => {
                  this.handleMessage(event.data);
              };
              
              this.socket.onclose = () => {
                  console.log('Disconnected from server');
                  this.connected = false;
                  this.connecting = false;
                  this.stopPing();
                  this.attemptReconnect();
                  reject(new Error('Disconnected from server'));
              };
              
              this.socket.onerror = (error) => {
                  console.error('WebSocket error:', error);
                  this.connecting = false;
                  reject(error);
              };
          } catch (error) {
              console.error('Failed to connect to server:', error);
              this.connecting = false;
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
          if (this.game.handleDisconnect) {
              this.game.handleDisconnect();
          }
          return;
      }
      
      this.reconnectAttempts++;
      
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
          this.connect().catch(() => {
              // Error already logged in connect method
          });
      }, this.reconnectDelay * this.reconnectAttempts);
  }
  
  /**
   * Start the ping interval
   * @private
   */
  startPing() {
      this.stopPing();
      this.pingInterval = setInterval(() => {
          this.sendPing();
      }, 30000);
  }
  
  /**
   * Stop the ping interval
   * @private
   */
  stopPing() {
      if (this.pingInterval) {
          clearInterval(this.pingInterval);
          this.pingInterval = null;
      }
  }
  
  /**
   * Send a ping message
   * @private
   */
  sendPing() {
      this.lastPingTime = Date.now();
      this.send(MessageType.PING, { time: this.lastPingTime });
  }
  
  /**
   * Send handshake message
   * @private
   */
  sendHandshake() {
      this.send(MessageType.HANDSHAKE, {
          clientTime: Date.now(),
          screenWidth: window.innerWidth,
          screenHeight: window.innerHeight
      });
  }
  
  /**
   * Send queued messages after connection is established
   */
  drainMessageQueue() {
      while (this.messageQueue.length > 0) {
          const message = this.messageQueue.shift();
          if (message.resolve) {
              message.resolve();
          } else {
              this.send(message.type, message.data);
          }
      }
  }
  
  /**
   * Handle an incoming WebSocket message
   * @param {ArrayBuffer} data - Binary message data
   */
  handleMessage(data) {
      try {
          // Decode binary packet
          const packet = BinaryPacket.decode(data);
          const { type, data: messageData } = packet;
          
          // Update server time offset
          if (messageData.timestamp) {
              this.lastServerTime = messageData.timestamp;
              this.serverTimeOffset = Date.now() - messageData.timestamp;
          }
          
          // Call the handler for this message type
          if (this.handlers[type]) {
              this.handlers[type](messageData);
          } else {
              console.warn(`No handler for message type: ${type}`);
          }
      } catch (error) {
          console.error('Error handling message:', error);
      }
  }
  
  /**
   * Send a message to the server
   * @param {number} type - Message type
   * @param {Object} data - Message data
   */
  send(type, data) {
      if (!this.connected) {
          this.messageQueue.push({ type, data });
          return false;
      }
      
      try {
          // Encode binary packet
          const packet = BinaryPacket.encode(type, data);
          
          // Send packet
          this.socket.send(packet);
          return true;
      } catch (error) {
          console.error('Error sending message:', error);
          return false;
      }
  }
  
  /**
   * Send player update to server
   * @param {Object} playerData - Player state data
   */
  sendPlayerUpdate(playerData) {
      return this.send(MessageType.PLAYER_UPDATE, playerData);
  }
  
  /**
   * Send shoot event to server
   * @param {Object} bulletData - Bullet data
   */
  sendShoot(bulletData) {
      return this.send(MessageType.BULLET_CREATE, bulletData);
  }
  
  /**
   * Send collision to server
   * @param {Object} collisionData - Collision data
   */
  sendCollision(collisionData) {
      return this.send(MessageType.COLLISION, collisionData);
  }
  
  /**
   * Request a map chunk from the server
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   */
  requestChunk(chunkX, chunkY) {
      return this.send(MessageType.CHUNK_REQUEST, { chunkX, chunkY });
  }
  
  /**
   * Register a message handler
   * @param {number} type - Message type
   * @param {Function} handler - Message handler
   */
  on(type, handler) {
      if (typeof handler === 'function') {
          this.handlers[type] = handler;
      }
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
      this.stopPing();
      
      if (this.socket) {
          this.socket.close();
          this.socket = null;
      }
      
      this.connected = false;
      this.connecting = false;
  }
}

/**
* BinaryPacket - Utility for efficient binary packet encoding/decoding
*/
export class BinaryPacket {
  /**
   * Create a binary packet with a specific message type
   * @param {number} type - Message type ID
   * @param {Object} data - Message data
   * @returns {ArrayBuffer} Binary packet
   */
  static encode(type, data) {
      // Convert data to JSON string for flexibility
      const jsonStr = JSON.stringify(data);
      const jsonBytes = new TextEncoder().encode(jsonStr);
      
      // Create packet: [1 byte type][4 byte length][jsonBytes]
      const packet = new ArrayBuffer(5 + jsonBytes.byteLength);
      const view = new DataView(packet);
      
      // Write type and length
      view.setUint8(0, type);
      view.setUint32(1, jsonBytes.byteLength, true); // Little-endian
      
      // Write JSON data
      new Uint8Array(packet, 5).set(jsonBytes);
      
      return packet;
  }
  
  /**
   * Decode a binary packet
   * @param {ArrayBuffer} packet - Binary packet
   * @returns {Object} Decoded packet {type, data}
   */
  static decode(packet) {
      const view = new DataView(packet);
      
      // Read type and length
      const type = view.getUint8(0);
      const length = view.getUint32(1, true); // Little-endian
      
      // Read JSON data
      const jsonBytes = new Uint8Array(packet, 5, length);
      const jsonStr = new TextDecoder().decode(jsonBytes);
      
      // Parse JSON data
      try {
          const data = JSON.parse(jsonStr);
          return { type, data };
      } catch (error) {
          console.error('Error parsing packet JSON:', error);
          return { type, data: {} };
      }
  }
}

/**
* Message type constants
*/
export const MessageType = {
  // Connection messages
  HANDSHAKE: 1,
  HANDSHAKE_ACK: 2,
  PING: 3,
  PONG: 4,
  
  // Game state messages
  PLAYER_JOIN: 10,
  PLAYER_LEAVE: 11,
  PLAYER_UPDATE: 12,
  PLAYER_LIST: 13,
  
  // Entity messages
  ENEMY_LIST: 20,
  ENEMY_UPDATE: 21,
  ENEMY_DEATH: 22,
  
  // Bullet messages
  BULLET_CREATE: 30,
  BULLET_LIST: 31,
  BULLET_REMOVE: 32,
  
  // Collision messages
  COLLISION: 40,
  COLLISION_RESULT: 41,
  
  // Map messages
  MAP_INFO: 50,
  CHUNK_REQUEST: 51,
  CHUNK_DATA: 52,
  CHUNK_NOT_FOUND: 53,
  
  // World update
  WORLD_UPDATE: 60
};