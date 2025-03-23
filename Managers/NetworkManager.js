/**
 * OptimizedNetworkManager.js
 * Optimized network manager with binary packet encoding for efficiency
 */

/**
 * BinaryPacket - Utility for efficient binary packet encoding/decoding
 */
class BinaryPacket {
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
  const MessageType = {
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
    CHUNK_DATA: 52
  };
  
  /**
   * OptimizedNetworkManager - Client implementation
   */
  class OptimizedNetworkManager {
    /**
     * Create a network manager
     * @param {string} serverUrl - WebSocket server URL
     * @param {Object} handlers - Message handlers
     */
    constructor(serverUrl, handlers = {}) {
      this.serverUrl = serverUrl;
      this.socket = null;
      this.connected = false;
      this.connecting = false;
      this.clientId = null;
      this.messageQueue = [];
      this.lastPingTime = 0;
      this.pingInterval = null;
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.reconnectDelay = 2000; // ms
      
      // Message handlers - default empty object for each type
      this.handlers = {};
      Object.values(MessageType).forEach(type => {
        this.handlers[type] = handlers[type] || (() => {});
      });
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
          
          // Binary data
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
            
            // Resolve pending promises
            resolve();
            while (this.messageQueue.length > 0) {
              const { resolve } = this.messageQueue.shift();
              resolve();
            }
          };
          
          this.socket.onmessage = (event) => {
            this.handleMessage(event.data);
          };
          
          this.socket.onclose = () => {
            console.log('Disconnected from server');
            this.connected = false;
            this.connecting = false;
            this.stopPing();
            
            // Attempt reconnect
            this.attemptReconnect();
            
            // Reject pending promises
            reject(new Error('Disconnected from server'));
            while (this.messageQueue.length > 0) {
              const { reject } = this.messageQueue.shift();
              reject(new Error('Disconnected from server'));
            }
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
     * @private
     */
    attemptReconnect() {
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.log('Maximum reconnect attempts reached');
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
     * Handle an incoming message
     * @param {ArrayBuffer} data - Message data
     * @private
     */
    handleMessage(data) {
      try {
        // Decode binary packet
        const packet = BinaryPacket.decode(data);
        const { type, data: messageData } = packet;
        
        // Special handling for certain messages
        switch (type) {
          case MessageType.HANDSHAKE_ACK:
            this.clientId = messageData.clientId;
            console.log(`Received client ID: ${this.clientId}`);
            break;
            
          case MessageType.PONG:
            const pingTime = Date.now() - this.lastPingTime;
            console.log(`Ping: ${pingTime}ms`);
            break;
        }
        
        // Call handler for this message type
        if (this.handlers[type]) {
          this.handlers[type](messageData);
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
        console.warn('Cannot send message, not connected');
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
     * Send player update
     * @param {Object} player - Player data
     */
    sendPlayerUpdate(player) {
      return this.send(MessageType.PLAYER_UPDATE, player);
    }
    
    /**
     * Send bullet creation
     * @param {Object} bullet - Bullet data
     */
    sendBulletCreate(bullet) {
      return this.send(MessageType.BULLET_CREATE, bullet);
    }
    
    /**
     * Send collision
     * @param {Object} collision - Collision data
     */
    sendCollision(collision) {
      return this.send(MessageType.COLLISION, collision);
    }
    
    /**
     * Request map chunk
     * @param {number} chunkX - Chunk X coordinate
     * @param {number} chunkY - Chunk Y coordinate
     */
    requestChunk(chunkX, chunkY) {
      return this.send(MessageType.CHUNK_REQUEST, { chunkX, chunkY });
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
     * Check if connected to server
     * @returns {boolean} True if connected
     */
    isConnected() {
      return this.connected;
    }
    
    /**
     * Get client ID
     * @returns {string} Client ID
     */
    getClientId() {
      return this.clientId;
    }
  }
  
  /**
   * Export for browser
   */
  if (typeof window !== 'undefined') {
    window.OptimizedNetworkManager = OptimizedNetworkManager;
    window.MessageType = MessageType;
    window.BinaryPacket = BinaryPacket;
  }
  
  /**
   * Export for Node.js
   */
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      OptimizedNetworkManager,
      MessageType,
      BinaryPacket
    };
  }
  
  export { OptimizedNetworkManager, MessageType, BinaryPacket };