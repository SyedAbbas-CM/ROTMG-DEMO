/**
 * Minimap component for displaying a small map of the game area
 */
import { UIComponent } from '../UIManager.js';

export class Minimap extends UIComponent {
  /**
   * Create a minimap component
   * @param {Object} gameState - Game state reference
   * @param {Object} manager - UI manager reference
   */
  constructor(gameState, manager) {
    super(gameState, manager);
    
    this.mapData = null;
    this.unitData = [];
  }
  
  /**
   * Initialize the component
   * @returns {HTMLElement} The component's DOM element
   */
  async init() {
    // Create panel container
    this.element = document.createElement('div');
    this.element.className = 'ui-minimap';
    this.element.style.position = 'absolute';
    this.element.style.top = '16px';
    this.element.style.right = '16px';
    this.element.style.width = '144px';
    this.element.style.height = '144px';
    this.element.style.backgroundColor = 'black';
    this.element.style.border = '2px solid #444';
    this.element.style.color = 'white';
    this.element.style.zIndex = '10';
    this.element.style.pointerEvents = 'auto';
    
    // Create panel header
    const header = document.createElement('div');
    header.className = 'panel-header';
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.padding = '2px 8px';
    header.style.borderBottom = '1px solid #333';
    header.style.backgroundColor = '#111';
    
    const title = document.createElement('div');
    title.className = 'panel-title';
    title.textContent = 'Map';
    title.style.fontSize = '12px';
    title.style.color = '#aaa';
    
    const controls = document.createElement('div');
    controls.className = 'panel-controls';
    controls.style.display = 'flex';
    
    const minimizeBtn = document.createElement('button');
    minimizeBtn.className = 'btn-minimize';
    minimizeBtn.innerHTML = '&minus;';
    minimizeBtn.title = 'Minimize';
    minimizeBtn.style.background = 'none';
    minimizeBtn.style.border = 'none';
    minimizeBtn.style.color = '#666';
    minimizeBtn.style.cursor = 'pointer';
    minimizeBtn.style.fontSize = '12px';
    minimizeBtn.style.width = '16px';
    minimizeBtn.style.height = '16px';
    minimizeBtn.style.display = 'flex';
    minimizeBtn.style.alignItems = 'center';
    minimizeBtn.style.justifyContent = 'center';
    minimizeBtn.style.marginLeft = '4px';
    
    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.title = 'Close';
    closeBtn.style.background = 'none';
    closeBtn.style.border = 'none';
    closeBtn.style.color = '#666';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.fontSize = '12px';
    closeBtn.style.width = '16px';
    closeBtn.style.height = '16px';
    closeBtn.style.display = 'flex';
    closeBtn.style.alignItems = 'center';
    closeBtn.style.justifyContent = 'center';
    closeBtn.style.marginLeft = '4px';
    
    controls.appendChild(minimizeBtn);
    controls.appendChild(closeBtn);
    
    header.appendChild(title);
    header.appendChild(controls);
    
    // Create map canvas
    const mapCanvas = document.createElement('canvas');
    mapCanvas.className = 'minimap-canvas';
    mapCanvas.width = 144;
    mapCanvas.height = 120;
    mapCanvas.style.display = 'block';
    
    // Setup event handlers
    minimizeBtn.addEventListener('click', () => this.minimize());
    closeBtn.addEventListener('click', () => this.hide());
    
    // Append elements to panel
    this.element.appendChild(header);
    this.element.appendChild(mapCanvas);
    
    this.canvas = mapCanvas;
    this.ctx = mapCanvas.getContext('2d');
    
    // Initial render
    this.render();
    
    return this.element;
  }
  
  /**
   * Minimize the panel
   */
  minimize() {
    this.hide();
    this.manager.trigger('minimizeComponent', { id: 'minimap' });
  }
  
  /**
   * Render the minimap
   */
  render() {
    if (!this.ctx) return;
    
    const ctx = this.ctx;
    
    // Clear the canvas
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw map base if available
    if (this.mapData) {
      // Draw map tiles
      // Implementation would depend on map data structure
    }
    
    // Draw units
    this.unitData.forEach(unit => {
      // Set color based on team/type
      if (unit.team === 'player') {
        ctx.fillStyle = '#4ade80'; // Green for player units
      } else if (unit.team === 'ally') {
        ctx.fillStyle = '#60a5fa'; // Blue for allies
      } else if (unit.team === 'enemy') {
        ctx.fillStyle = '#f87171'; // Red for enemies
      } else {
        ctx.fillStyle = '#a3a3a3'; // Gray for neutral
      }
      
      // Scale coordinates to fit the minimap
      const x = (unit.x / this.mapData?.width || 1000) * this.canvas.width;
      const y = (unit.y / this.mapData?.height || 1000) * this.canvas.height;
      
      // Draw unit dot
      ctx.beginPath();
      ctx.arc(x, y, unit.isPlayer ? 3 : 2, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Draw player
    const player = this.unitData.find(u => u.isPlayer);
    if (player) {
      // Draw player direction indicator
      const x = (player.x / this.mapData?.width || 1000) * this.canvas.width;
      const y = (player.y / this.mapData?.height || 1000) * this.canvas.height;
      
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  
  /**
   * Update the component with game state
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    if (!this.isVisible) return;
    
    if (gameState.map) {
      this.mapData = gameState.map;
    }
    
    if (gameState.units) {
      this.unitData = gameState.units;
    }
    
    // Render minimap with updated data
    this.render();
  }
} 