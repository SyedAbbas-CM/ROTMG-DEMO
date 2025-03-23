/**
 * Game.js
 * Main client-side game class that coordinates all game systems
 */
import { NetworkManager } from '../network/NetworkManager.js';
import { ClientBulletManager } from './ClientBulletManager.js';
import { ClientEnemyManager } from './ClientEnemyManager.js';
import { ClientCollisionManager } from '../collision/clientCollisionManager.js';
import { ClientMapManager } from '../map/ClientMapManager.js';
import { Player } from '../entities/Player.js';

export class Game {
  /**
   * Create a new game instance
   * @param {string} serverUrl - WebSocket server URL
   */
  constructor(serverUrl = 'ws://localhost:3000') {
    // Client ID assigned by server
    this.clientId = null;
    
    // Create managers
    this.networkManager = new NetworkManager(serverUrl, this);
    this.bulletManager = new ClientBulletManager(10000); // Up to 10,000 bullets
    this.enemyManager = new ClientEnemyManager(1000);    // Up to 1,000 enemies
    this.mapManager = new ClientMapManager();            // Client-side map manager
    this.collisionManager = new ClientCollisionManager(
      this.bulletManager,
      this.enemyManager,
      this.networkManager
    );
    
    // Create local player
    this.player = new Player();
    
    // Track other players
    this.players = {};
    
    // Game state
    this.isRunning = false;
    this.lastUpdateTime = 0;
    this.gameTime = 0;
    this.deltaTime = 0;
    
    // Input state
    this.keys = {};
    this.mouse = { x: 0, y: 0, isDown: false };
    
    // Initialize listeners
    this.initInputListeners();
  }
  
  /**
   * Initialize the game
   * @returns {Promise} Resolves when initialized
   */
  async init() {
    try {
      console.log('Initializing game...');
      
      // Connect to server
      await this.networkManager.connect();
      
      // Start game loop
      this.start();
      
      console.log('Game initialized successfully');
      return true;
    } catch (error) {
      console.error('Failed to initialize game:', error);
      return false;
    }
  }
  
  /**
   * Set client ID received from server
   * @param {string} clientId - Client ID
   */
  setClientId(clientId) {
    this.clientId = clientId;
    this.player.id = clientId;
    
    // Send player name or customization to server
    this.networkManager.sendPlayerJoin(this.player.name);
  }
  
  /**
   * Initialize input event listeners
   */
  initInputListeners() {
    // Keyboard events
    window.addEventListener('keydown', (e) => {
      this.keys[e.code] = true;
    });
    
    window.addEventListener('keyup', (e) => {
      this.keys[e.code] = false;
    });
    
    // Mouse events
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
    
    window.addEventListener('mousedown', (e) => {
      this.mouse.isDown = true;
      this.handleShoot();
    });
    
    window.addEventListener('mouseup', () => {
      this.mouse.isDown = false;
    });
    
    // Touch events for mobile
    window.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.mouse.x = touch.clientX;
      this.mouse.y = touch.clientY;
      this.mouse.isDown = true;
      this.handleShoot();
    });
    
    window.addEventListener('touchmove', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      this.mouse.x = touch.clientX;
      this.mouse.y = touch.clientY;
    });
    
    window.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.mouse.isDown = false;
    });
  }
  
  /**
   * Start the game loop
   */
  start() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.lastUpdateTime = performance.now();
    
    // Start the game loop
    requestAnimationFrame(this.gameLoop.bind(this));
    
    console.log('Game started');
  }
  
  /**
   * Main game loop
   * @param {number} timestamp - Current timestamp
   */
  gameLoop(timestamp) {
    if (!this.isRunning) return;
    
    // Calculate delta time
    this.deltaTime = (timestamp - this.lastUpdateTime) / 1000; // seconds
    this.lastUpdateTime = timestamp;
    this.gameTime += this.deltaTime;
    
    // Cap delta time to avoid huge jumps
    if (this.deltaTime > 0.1) this.deltaTime = 0.1;
    
    // Update game
    this.update(this.deltaTime);
    
    // Render
    this.render();
    
    // Schedule next frame
    requestAnimationFrame(this.gameLoop.bind(this));
  }
  
  /**
   * Update game state
   * @param {number} deltaTime - Time elapsed since last update in seconds
   */
  update(deltaTime) {
    // Update player based on input
    this.handleInput(deltaTime);
    
    // Update player position in collision manager
    this.collisionManager.updatePlayerPosition(this.player.x, this.player.y);
    
    // Update bullets
    this.bulletManager.update(deltaTime);
    
    // Update enemies (client-side prediction)
    this.enemyManager.update(deltaTime);
    
    // Run collision detection
    this.collisionManager.update();
    
    // Send player update to server periodically
    this.sendPlayerUpdate();
    
    // Check if we need to request map chunks
    this.checkAndRequestChunks();
  }
  
  /**
   * Render the game
   */
  render() {
    // This method should be implemented according to your rendering system
    // For now, we'll just assume it's handled elsewhere
  }
  
  /**
   * Handle player input
   * @param {number} deltaTime - Time elapsed since last update in seconds
   */
  handleInput(deltaTime) {
    // Movement
    let dx = 0;
    let dy = 0;
    
    // WASD or arrow keys
    if (this.keys['KeyW'] || this.keys['ArrowUp']) dy -= 1;
    if (this.keys['KeyS'] || this.keys['ArrowDown']) dy += 1;
    if (this.keys['KeyA'] || this.keys['ArrowLeft']) dx -= 1;
    if (this.keys['KeyD'] || this.keys['ArrowRight']) dx += 1;
    
    // Normalize diagonal movement
    if (dx !== 0 && dy !== 0) {
      const length = Math.sqrt(dx * dx + dy * dy);
      dx /= length;
      dy /= length;
    }
    
    // Apply movement
    const speed = this.player.speed * deltaTime;
    this.player.x += dx * speed;
    this.player.y += dy * speed;
    
    // Calculate rotation based on mouse position
    if (this.mouse) {
      // Get center of screen or player position converted to screen coordinates
      const centerX = window.innerWidth / 2;
      const centerY = window.innerHeight / 2;
      
      // Calculate angle
      const angle = Math.atan2(this.mouse.y - centerY, this.mouse.x - centerX);
      this.player.rotation = angle;
    }
  }
  
  /**
   * Handle player shooting
   */
  handleShoot() {
    // Don't shoot if not connected
    if (!this.networkManager.isConnected() || !this.clientId) return;
    
    // Create bullet data
    const bulletData = {
      x: this.player.x,
      y: this.player.y,
      angle: this.player.rotation,
      speed: 300, // Pixels per second
      lifetime: 2000 // Milliseconds
    };
    
    // Send to server
    this.networkManager.sendShoot(bulletData);
    
    // Create local bullet for immediate feedback
    this.bulletManager.addBullet({
      id: 'local_' + Date.now(),
      x: this.player.x,
      y: this.player.y,
      vx: Math.cos(this.player.rotation) * bulletData.speed,
      vy: Math.sin(this.player.rotation) * bulletData.speed,
      ownerId: this.clientId,
      lifetime: bulletData.lifetime
    });
  }
  
  /**
   * Send player update to server (throttled)
   */
  sendPlayerUpdate() {
    // Throttle updates to avoid network spam
    // For example, send every 50ms
    if (this.gameTime - (this._lastUpdateSent || 0) > 0.05) {
      this._lastUpdateSent = this.gameTime;
      
      this.networkManager.sendPlayerUpdate({
        x: this.player.x,
        y: this.player.y,
        rotation: this.player.rotation,
        name: this.player.name
      });
    }
  }
  
  /**
   * Check if we need to request map chunks based on player position
   */
  checkAndRequestChunks() {
    const visibleChunks = this.mapManager.getVisibleChunks(this.player.x, this.player.y);
    
    for (const chunk of visibleChunks) {
      if (!this.mapManager.hasChunk(chunk.x, chunk.y)) {
        this.networkManager.requestChunk(chunk.x, chunk.y);
      }
    }
  }
  
  /**
   * Initialize map based on server info
   * @param {Object} mapInfo - Map metadata from server
   */
  initMap(mapInfo) {
    this.mapManager.initMap(mapInfo);
  }
  
  /**
   * Set chunk data received from server
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @param {Object} data - Chunk data
   */
  setChunkData(chunkX, chunkY, data) {
    this.mapManager.setChunkData(chunkX, chunkY, data);
  }
  
  /**
   * Generate a fallback chunk if server doesn't have it
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   */
  generateFallbackChunk(chunkX, chunkY) {
    this.mapManager.generateFallbackChunk(chunkX, chunkY);
  }
  
  /**
   * Set initial players list
   * @param {Object} players - Map of client IDs to player objects
   */
  setPlayers(players) {
    this.players = { ...players };
  }
  
  /**
   * Add a new player
   * @param {Object} player - Player data
   */
  addPlayer(player) {
    this.players[player.id] = player;
  }
  
  /**
   * Remove a player
   * @param {string} clientId - Client ID of player to remove
   */
  removePlayer(clientId) {
    delete this.players[clientId];
  }
  
  /**
   * Set initial enemies list
   * @param {Array} enemies - Array of enemy data
   */
  setEnemies(enemies) {
    this.enemyManager.setEnemies(enemies);
  }
  
  /**
   * Set initial bullets list
   * @param {Array} bullets - Array of bullet data
   */
  setBullets(bullets) {
    this.bulletManager.setBullets(bullets);
  }
  
  /**
   * Add a new bullet
   * @param {Object} bullet - Bullet data
   */
  addBullet(bullet) {
    this.bulletManager.addBullet(bullet);
  }
  
  /**
   * Update world state based on server data
   * @param {Array} enemies - Updated enemy data
   * @param {Array} bullets - Updated bullet data
   * @param {Object} players - Updated player data
   */
  updateWorld(enemies, bullets, players) {
    // Update enemies
    this.enemyManager.updateEnemies(enemies);
    
    // Update bullets
    this.bulletManager.updateBullets(bullets);
    
    // Update other players (excluding self)
    for (const id in players) {
      if (id !== this.clientId) {
        this.players[id] = players[id];
      }
    }
  }
  
  /**
   * Apply a validated collision
   * @param {Object} collisionData - Collision data from server
   */
  applyCollision(collisionData) {
    const { bulletId, enemyId, damage, enemyHealth } = collisionData;
    
    // Remove the bullet
    this.bulletManager.removeBulletById(bulletId);
    
    // Update enemy health
    this.enemyManager.setEnemyHealth(enemyId, enemyHealth);
  }
  
  /**
   * Handle enemy killed event
   * @param {string} enemyId - ID of killed enemy
   */
  handleEnemyKilled(enemyId) {
    this.enemyManager.removeEnemyById(enemyId);
    
    // Could add particle effects, score updates, etc. here
  }
  
  /**
   * Handle disconnect from server
   */
  handleDisconnect() {
    console.log('Disconnected from server');
    
    // Show disconnect message or attempt to reconnect
    // For example:
    // document.getElementById('disconnectMessage').style.display = 'block';
  }
  
  /**
   * Stop the game
   */
  stop() {
    this.isRunning = false;
    console.log('Game stopped');
  }
  
  /**
   * Clean up resources
   */
  cleanup() {
    this.stop();
    this.networkManager.disconnect();
    
    // Remove event listeners?
    
    console.log('Game cleaned up');
  }
}