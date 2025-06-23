// public/src/game/GameManager.js

import { 
  ClientBulletManager,
  ClientEnemyManager,
  ClientNetworkManager as NetworkManager,
  ClientCollisionManager
} from '../managers.js';
import { SimpleCollisionSystem } from '../collision/SimpleCollisionSystem.js';
import { map } from '../map/map.js';
import { gameState } from './gamestate.js';

/**
 * GameManager - Coordinates game state, networking, and system integration
 */
export class GameManager {
  /**
   * Create a new game manager
   * @param {Object} options - Configuration options
   */
  constructor(options = {}) {
    // Initialize component managers
    this.bulletManager = new ClientBulletManager(options.maxBullets || 1000);
    this.enemyManager = new ClientEnemyManager(options.maxEnemies || 100);
    
    // Store references in gameState for global access
    gameState.bulletManager = this.bulletManager;
    gameState.enemyManager = this.enemyManager;
    
    // Initialize collision system
    this.collisionSystem = new SimpleCollisionSystem({
      bulletManager: this.bulletManager,
      enemyManager: this.enemyManager,
      mapManager: map
    });
    
    // Set up network manager if server URL provided
    this.serverUrl = options.serverUrl || 'ws://localhost:3000';
    this.networkManager = new NetworkManager(this.serverUrl, this);
    
    // Game state tracking
    this.clientId = null;
    this.players = {};
    this.isConnected = false;
    this.lastUpdateTime = 0;
    
    console.log('GameManager initialized');
  }
  
  /**
   * Connect to the game server
   * @returns {Promise} Resolves when connected
   */
  async connect() {
    try {
      await this.networkManager.connect();
      this.isConnected = true;
      console.log('Connected to game server');
      return true;
    } catch (error) {
      console.error('Failed to connect to game server:', error);
      return false;
    }
  }
  
  /**
   * Start the game
   */
  start() {
    // Connect to server if not already connected
    if (!this.isConnected) {
      this.connect().catch(error => {
        console.error('Error connecting to server:', error);
        // Continue in offline mode
        console.log('Running in offline mode');
      });
    }
    
    // Start the game loop
    this.lastUpdateTime = performance.now();
    requestAnimationFrame(this.update.bind(this));
    
    console.log('Game started');
  }
  
  /**
   * Main update loop
   * @param {number} timestamp - Current timestamp
   */
  update(timestamp) {
    // Calculate delta time in seconds
    const deltaTime = (timestamp - this.lastUpdateTime) / 1000;
    this.lastUpdateTime = timestamp;
    
    // Cap delta time to prevent large jumps
    const cappedDeltaTime = Math.min(deltaTime, 0.1);
    
    // Update local game state
    this.updateLocalState(cappedDeltaTime);
    
    // Send player update to server if connected
    if (this.isConnected) {
      this.sendPlayerUpdate();
    }
    
    // Continue the game loop
    requestAnimationFrame(this.update.bind(this));
  }
  
  /**
   * Update local game state
   * @param {number} deltaTime - Time elapsed since last update in seconds
   */
  updateLocalState(deltaTime) {
    // Update bullets
    this.bulletManager.update(deltaTime);
    
    // Update enemies
    this.enemyManager.update(deltaTime);
    
    // Update collision detection
    this.collisionSystem.update(deltaTime);
  }
  
  /**
   * Send player update to server
   */
  sendPlayerUpdate() {
    if (!this.networkManager.isConnected()) return;
    
    // Get player data from gameState
    const playerData = {
      x: gameState.character.x,
      y: gameState.character.y,
      rotation: gameState.character.rotation.yaw,
      health: gameState.character.health
    };
    
    // Send update
    this.networkManager.sendPlayerUpdate(playerData);
  }
  
  /**
   * Fire a bullet from the player
   * @param {number} angle - Firing angle in radians
   */
  firePlayerBullet(angle) {
    const character = gameState.character;
    
    // Create bullet data
    const bulletData = {
      x: character.x,
      y: character.y,
      vx: Math.cos(angle) * 300, // Speed of 300 units per second
      vy: Math.sin(angle) * 300,
      damage: 10,
      lifetime: 2.0,
      ownerId: this.clientId || 'local_player'
    };
    
    // Add bullet locally
    const bulletId = this.bulletManager.addBullet(bulletData);
    
    // Send to server if connected
    if (this.networkManager.isConnected()) {
      this.networkManager.sendShoot({
        x: bulletData.x,
        y: bulletData.y,
        angle: angle,
        speed: 300,
        damage: 10
      });
    }
    
    return bulletId;
  }
  
  /**
   * Set the client ID (called by NetworkManager)
   * @param {string} clientId - Client ID from server
   */
  setClientId(clientId) {
    this.clientId = clientId;
    console.log(`Set client ID: ${clientId}`);
  }
  
  /**
   * Initialize the game map
   * @param {Object} mapData - Map data from server
   */
  initMap(mapData) {
    // Store map metadata
    gameState.mapMetadata = mapData;
    
    console.log('Map initialized with data from server');
  }
  
  /**
   * Set initial players list
   * @param {Object} players - Players data from server
   */
  setPlayers(players) {
    this.players = players;
    console.log(`Received ${Object.keys(players).length} players from server`);
  }
  
  /**
   * Set initial enemies list
   * @param {Array} enemies - Enemies data from server
   */
  setEnemies(enemies) {
    this.enemyManager.setEnemies(enemies);
    console.log(`Set ${enemies.length} enemies from server`);
  }
  
  /**
   * Set initial bullets list
   * @param {Array} bullets - Bullets data from server
   */
  setBullets(bullets) {
    this.bulletManager.setBullets(bullets);
    console.log(`Set ${bullets.length} bullets from server`);
  }
  
  /**
   * Set world objects list
   * @param {Array} objects - Objects data from server
   */
  setObjects(objects){
    this.objects = objects || [];
    gameState.worldObjects = this.objects;
    if (objects && objects.length && console.debug) {
      console.debug(`[GAME] Received ${objects.length} objects`);
    }
  }
  
  /**
   * Update world state from server data
   * @param {Array} enemies - Updated enemies data
   * @param {Array} bullets - Updated bullets data
   * @param {Object} players - Updated players data
   * @param {Array} objects - Updated objects data
   */
  updateWorld(enemies, bullets, players, objects) {
    // Update enemies
    if (enemies && enemies.length > 0) {
      this.enemyManager.updateEnemies(enemies);
    }
    
    // Update bullets
    if (bullets && bullets.length > 0) {
      this.bulletManager.updateBullets(bullets);
    }
    
    // Update other players
    if (players) {
      this.players = players;
    }
    
    // Update objects
    if (objects) {
      this.setObjects(objects);
    }
  }
  
  /**
   * Add a new player
   * @param {Object} player - Player data
   */
  addPlayer(player) {
    this.players[player.id] = player;
    console.log(`Player added: ${player.id}`);
  }
  
  /**
   * Remove a player
   * @param {string} playerId - Player ID to remove
   */
  removePlayer(playerId) {
    delete this.players[playerId];
    console.log(`Player removed: ${playerId}`);
  }
  
  /**
   * Add a bullet
   * @param {Object} bulletData - Bullet data
   */
  addBullet(bulletData) {
    this.bulletManager.addBullet(bulletData);
  }
  
  /**
   * Apply collision result from server
   * @param {Object} data - Collision data
   */
  applyCollision(data) {
    const { bulletId, enemyId, damage, enemyHealth } = data;
    // Remove bullet
    this.bulletManager.removeBulletById(bulletId);
    // Update enemy health
    this.enemyManager.setEnemyHealth(enemyId, enemyHealth);
  }
  
  /**
   * Handle enemy killed event
   * @param {string} enemyId - ID of killed enemy
   */
  handleEnemyKilled(enemyId) {
    console.log(`Enemy killed: ${enemyId}`);
    // Potential place for score updates, effects, etc.
  }
  
  /**
   * Set chunk data for the map
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   * @param {Object} data - Chunk data
   */
  setChunkData(chunkX, chunkY, data) {
    // Always use the authoritative map instance stored in gameState
    if (gameState.map && typeof gameState.map.setChunkData === 'function') {
      gameState.map.setChunkData(chunkX, chunkY, data);
    } else {
      console.error('setChunkData: gameState.map is not ready');
    }
  }
  
  /**
   * Generate a fallback chunk
   * @param {number} chunkX - Chunk X coordinate
   * @param {number} chunkY - Chunk Y coordinate
   */
  generateFallbackChunk(chunkX, chunkY) {
    if (gameState.map && typeof gameState.map.generateFallbackChunk === 'function') {
      console.warn(`Fallback: creating dummy chunk (${chunkX},${chunkY})`);
      gameState.map.generateFallbackChunk(chunkX, chunkY);
    }
  }
  
  /**
   * Handle disconnection from server
   */
  handleDisconnect() {
    this.isConnected = false;
    console.log('Disconnected from server, running in offline mode');
  }
}

export default GameManager;