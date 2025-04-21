// Import the BehaviorSystem
import BehaviorSystem from './BehaviorSystem.js';

// Inside the Server class initialization:
constructor() {
  // ... existing code ...
  
  // Initialize the enemy manager and behavior system
  this.enemyManager = new EnemyManager(1000);
  this.behaviorSystem = new BehaviorSystem();
  
  // ... existing code ...
}

// Inside the update method:
update(deltaTime) {
  // ... existing code ...
  
  // Update all enemies with the behavior system
  if (this.gameState && this.gameState.player) {
    const target = {
      x: this.gameState.player.x,
      y: this.gameState.player.y
    };
    
    this.enemyManager.update(deltaTime, this.bulletManager, target);
  }
  
  // ... existing code ...
}

// Inside the broadcastGameState method:
broadcastGameState() {
  // ... existing code ...
  
  // Get enemy data for network transmission
  const enemiesData = this.enemyManager.getEnemiesData();
  
  // Send enemy data to all clients
  const gameState = {
    enemies: enemiesData,
    // ... other game state ...
  };
  
  this.networkManager.broadcastGameState(gameState);
  
  // ... existing code ...
} 