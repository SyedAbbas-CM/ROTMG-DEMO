/**
 * SpatialGrid.js
 * Spatial partitioning grid for efficient collision detection.
 * Compatible with both client and server environments.
 */

class SpatialGrid {
    /**
     * Creates a new spatial grid for collision optimization
     * @param {number} cellSize - Size of each grid cell
     * @param {number} width - Total width of the grid in world units
     * @param {number} height - Total height of the grid in world units
     */
    constructor(cellSize, width, height) {
      this.cellSize = cellSize;
      this.width = width;
      this.height = height;
      
      // Calculate grid dimensions in cells
      this.gridWidth = Math.ceil(width / cellSize);
      this.gridHeight = Math.ceil(height / cellSize);
      
      // Initialize empty grid cells
      this.grid = new Array(this.gridWidth);
      for (let x = 0; x < this.gridWidth; x++) {
        this.grid[x] = new Array(this.gridHeight);
        for (let y = 0; y < this.gridHeight; y++) {
          this.grid[x][y] = {
            bullets: [],
            enemies: []
          };
        }
      }
    }
    
    /**
     * Clears all entities from the grid
     */
    clear() {
      for (let x = 0; x < this.gridWidth; x++) {
        for (let y = 0; y < this.gridHeight; y++) {
          this.grid[x][y].bullets = [];
          this.grid[x][y].enemies = [];
        }
      }
    }
    
    /**
     * Determines which cells an object overlaps
     * @param {number} x - World X position of entity
     * @param {number} y - World Y position of entity
     * @param {number} width - Width of entity
     * @param {number} height - Height of entity
     * @returns {Object} Min/max cell coordinates that entity overlaps
     */
    getCellsForEntity(x, y, width, height) {
      // Clamp to grid boundaries
      const minCellX = Math.max(0, Math.floor(x / this.cellSize));
      const minCellY = Math.max(0, Math.floor(y / this.cellSize));
      const maxCellX = Math.min(this.gridWidth - 1, Math.floor((x + width) / this.cellSize));
      const maxCellY = Math.min(this.gridHeight - 1, Math.floor((y + height) / this.cellSize));
      
      return { minCellX, minCellY, maxCellX, maxCellY };
    }
    
    /**
     * Adds a bullet to all cells it overlaps
     * @param {number} index - Bullet index in the manager
     * @param {number} x - Bullet X position
     * @param {number} y - Bullet Y position
     * @param {number} width - Bullet width
     * @param {number} height - Bullet height
     */
    insertBullet(index, x, y, width, height) {
      const { minCellX, minCellY, maxCellX, maxCellY } = this.getCellsForEntity(x, y, width, height);
      
      for (let cellX = minCellX; cellX <= maxCellX; cellX++) {
        for (let cellY = minCellY; cellY <= maxCellY; cellY++) {
          this.grid[cellX][cellY].bullets.push(index);
        }
      }
    }
    
    /**
     * Adds an enemy to all cells it overlaps
     * @param {number} index - Enemy index in the manager
     * @param {number} x - Enemy X position
     * @param {number} y - Enemy Y position
     * @param {number} width - Enemy width
     * @param {number} height - Enemy height
     */
    insertEnemy(index, x, y, width, height) {
      const { minCellX, minCellY, maxCellX, maxCellY } = this.getCellsForEntity(x, y, width, height);
      
      for (let cellX = minCellX; cellX <= maxCellX; cellX++) {
        for (let cellY = minCellY; cellY <= maxCellY; cellY++) {
          this.grid[cellX][cellY].enemies.push(index);
        }
      }
    }
    
    /**
     * Gets all potential bullet-enemy collision pairs
     * @returns {Array} Array of [bulletIndex, enemyIndex] pairs
     */
    getPotentialCollisionPairs() {
      const potentialPairs = [];
      const processed = new Set(); // Avoid duplicate pairs
      
      // Check each grid cell
      for (let cellX = 0; cellX < this.gridWidth; cellX++) {
        for (let cellY = 0; cellY < this.gridHeight; cellY++) {
          const cell = this.grid[cellX][cellY];
          
          // For each bullet-enemy pair in this cell
          for (const bulletIndex of cell.bullets) {
            for (const enemyIndex of cell.enemies) {
              // Unique identifier for this pair
              const pairKey = `${bulletIndex},${enemyIndex}`;
              
              // Only add if not already processed
              if (!processed.has(pairKey)) {
                potentialPairs.push([bulletIndex, enemyIndex]);
                processed.add(pairKey);
              }
            }
          }
        }
      }
      
      return potentialPairs;
    }
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = SpatialGrid;
  } else {
    if (typeof window !== 'undefined') {
      window.SpatialGrid = SpatialGrid;
    }
  }
  
  // Allow ES modules import
  export default SpatialGrid;