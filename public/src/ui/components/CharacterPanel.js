/**
 * Character panel component for displaying player stats and inventory
 */
import { UIComponent } from '../UIManager.js';

export class CharacterPanel extends UIComponent {
  /**
   * Create a character panel component
   * @param {Object} gameState - Game state reference
   * @param {Object} manager - UI manager reference
   */
  constructor(gameState, manager) {
    super(gameState, manager);
    
    this.inventory = [];
    this.stats = [];
    this.health = 0;
    this.maxHealth = 100;
    this.mana = 0;
    this.maxMana = 100;
    this.activeTab = 'equipment';
  }
  
  /**
   * Initialize the component
   * @returns {HTMLElement} The component's DOM element
   */
  async init() {
    // Create panel container
    this.element = document.createElement('div');
    this.element.className = 'ui-character-panel';
    this.element.style.position = 'absolute';
    this.element.style.top = '176px';
    this.element.style.right = '16px';
    this.element.style.width = '144px';
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
    title.textContent = 'Character';
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
    
    // Create panel content
    const content = document.createElement('div');
    content.className = 'stats-content';
    content.style.padding = '8px';
    
    // Health bar
    const healthContainer = document.createElement('div');
    healthContainer.className = 'resource-bar';
    healthContainer.style.display = 'flex';
    healthContainer.style.alignItems = 'center';
    healthContainer.style.gap = '4px';
    healthContainer.style.marginBottom = '4px';
    
    const healthIcon = document.createElement('div');
    healthIcon.className = 'resource-icon health-icon';
    healthIcon.style.width = '16px';
    healthIcon.style.height = '16px';
    healthIcon.style.backgroundColor = '#ef4444';
    healthIcon.style.clipPath = 'path("M8 2.748l-.717-.737C5.6.281 2.514.878 1.4 3.053c-.523 1.023-.641 2.5.314 4.385.92 1.815 2.834 3.989 6.286 6.357 3.452-2.368 5.365-4.542 6.286-6.357.955-1.886.838-3.362.314-4.385C13.486.878 10.4.28 8.717 2.01L8 2.748zM8 15C-7.333 4.868 3.279-3.04 7.824 1.143c.06.055.119.112.176.171a3.12 3.12 0 0 1 .176-.17C12.72-3.042 23.333 4.867 8 15z")';
    
    const healthBarContainer = document.createElement('div');
    healthBarContainer.className = 'bar-container';
    healthBarContainer.style.flexGrow = '1';
    healthBarContainer.style.height = '16px';
    healthBarContainer.style.backgroundColor = '#222';
    healthBarContainer.style.border = '1px solid #333';
    
    const healthBar = document.createElement('div');
    healthBar.className = 'bar health-bar';
    healthBar.style.height = '100%';
    healthBar.style.backgroundColor = '#dc2626';
    healthBar.style.width = '90%';
    healthBarContainer.appendChild(healthBar);
    
    const healthText = document.createElement('div');
    healthText.className = 'resource-text';
    healthText.textContent = '650/720';
    healthText.style.fontSize = '12px';
    healthText.style.textAlign = 'center';
    healthText.style.marginBottom = '8px';
    
    healthContainer.appendChild(healthIcon);
    healthContainer.appendChild(healthBarContainer);
    
    // Mana bar
    const manaContainer = document.createElement('div');
    manaContainer.className = 'resource-bar';
    manaContainer.style.display = 'flex';
    manaContainer.style.alignItems = 'center';
    manaContainer.style.gap = '4px';
    manaContainer.style.marginBottom = '4px';
    
    const manaIcon = document.createElement('div');
    manaIcon.className = 'resource-icon mana-icon';
    manaIcon.style.width = '16px';
    manaIcon.style.height = '16px';
    manaIcon.style.backgroundColor = '#3b82f6';
    manaIcon.style.clipPath = 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)';
    
    const manaBarContainer = document.createElement('div');
    manaBarContainer.className = 'bar-container';
    manaBarContainer.style.flexGrow = '1';
    manaBarContainer.style.height = '16px';
    manaBarContainer.style.backgroundColor = '#222';
    manaBarContainer.style.border = '1px solid #333';
    
    const manaBar = document.createElement('div');
    manaBar.className = 'bar mana-bar';
    manaBar.style.height = '100%';
    manaBar.style.backgroundColor = '#2563eb';
    manaBar.style.width = '65%';
    manaBarContainer.appendChild(manaBar);
    
    const manaText = document.createElement('div');
    manaText.className = 'resource-text';
    manaText.textContent = '252/385';
    manaText.style.fontSize = '12px';
    manaText.style.textAlign = 'center';
    manaText.style.marginBottom = '8px';
    
    manaContainer.appendChild(manaIcon);
    manaContainer.appendChild(manaBarContainer);
    
    // Tabs
    const tabs = document.createElement('div');
    tabs.className = 'tabs';
    tabs.style.display = 'flex';
    tabs.style.borderBottom = '2px solid #333';
    
    const equipmentTab = document.createElement('button');
    equipmentTab.className = 'tab active';
    equipmentTab.textContent = 'Equipment';
    equipmentTab.style.flex = '1';
    equipmentTab.style.backgroundColor = '#333';
    equipmentTab.style.color = '#f59e0b';
    equipmentTab.style.border = 'none';
    equipmentTab.style.padding = '4px 0';
    equipmentTab.style.fontSize = '12px';
    equipmentTab.style.cursor = 'pointer';
    
    const statsTab = document.createElement('button');
    statsTab.className = 'tab';
    statsTab.textContent = 'Stats';
    statsTab.style.flex = '1';
    statsTab.style.backgroundColor = 'black';
    statsTab.style.color = '#666';
    statsTab.style.border = 'none';
    statsTab.style.padding = '4px 0';
    statsTab.style.fontSize = '12px';
    statsTab.style.cursor = 'pointer';
    
    tabs.appendChild(equipmentTab);
    tabs.appendChild(statsTab);
    
    // Equipment content
    const equipmentContent = document.createElement('div');
    equipmentContent.className = 'tab-content';
    equipmentContent.style.padding = '8px 0';
    
    const inventoryGrid = document.createElement('div');
    inventoryGrid.className = 'inventory-grid';
    inventoryGrid.style.display = 'grid';
    inventoryGrid.style.gridTemplateColumns = 'repeat(4, 1fr)';
    inventoryGrid.style.gap = '4px';
    
    // Create inventory slots
    for (let i = 0; i < 12; i++) {
      const slot = document.createElement('div');
      slot.className = 'inventory-slot';
      slot.style.width = '28px';
      slot.style.height = '28px';
      slot.style.backgroundColor = '#222';
      slot.style.border = '1px solid #333';
      slot.style.display = 'flex';
      slot.style.alignItems = 'center';
      slot.style.justifyContent = 'center';
      
      if (i < 6) {
        // Add mock item
        const item = document.createElement('div');
        item.className = 'inventory-item';
        item.style.width = '20px';
        item.style.height = '20px';
        item.style.backgroundColor = '#333';
        slot.appendChild(item);
      }
      
      inventoryGrid.appendChild(slot);
    }
    
    equipmentContent.appendChild(inventoryGrid);
    
    // Stats content
    const statsContent = document.createElement('div');
    statsContent.className = 'tab-content';
    statsContent.style.padding = '8px 0';
    statsContent.style.display = 'none';
    
    const statsGrid = document.createElement('div');
    statsGrid.className = 'stats-grid';
    statsGrid.style.display = 'grid';
    statsGrid.style.gridTemplateColumns = 'repeat(2, 1fr)';
    statsGrid.style.gap = '4px';
    
    // Sample stats
    const statData = [
      { name: 'ATT', value: 65, max: 75 },
      { name: 'DEF', value: 40, max: 40 },
      { name: 'SPD', value: 55, max: 75 },
      { name: 'DEX', value: 50, max: 75 },
      { name: 'VIT', value: 40, max: 75 },
      { name: 'WIS', value: 60, max: 75 }
    ];
    
    // Create stat items
    statData.forEach(stat => {
      const statItem = document.createElement('div');
      statItem.className = 'stat-item';
      statItem.style.display = 'flex';
      statItem.style.flexDirection = 'column';
      
      const header = document.createElement('div');
      header.className = 'stat-header';
      header.style.display = 'flex';
      header.style.justifyContent = 'space-between';
      
      const name = document.createElement('span');
      name.className = 'stat-name';
      name.textContent = stat.name;
      name.style.fontSize = '12px';
      name.style.color = '#aaa';
      
      const value = document.createElement('span');
      value.className = 'stat-value';
      value.textContent = stat.value;
      value.style.fontSize = '12px';
      value.style.color = '#f59e0b';
      
      header.appendChild(name);
      header.appendChild(value);
      
      const barContainer = document.createElement('div');
      barContainer.className = 'stat-bar-container';
      barContainer.style.width = '100%';
      barContainer.style.height = '4px';
      barContainer.style.backgroundColor = '#222';
      barContainer.style.marginTop = '4px';
      
      const bar = document.createElement('div');
      bar.className = 'stat-bar';
      bar.style.height = '100%';
      bar.style.backgroundColor = '#22c55e';
      bar.style.width = `${(stat.value / stat.max) * 100}%`;
      
      barContainer.appendChild(bar);
      
      statItem.appendChild(header);
      statItem.appendChild(barContainer);
      statsGrid.appendChild(statItem);
    });
    
    statsContent.appendChild(statsGrid);
    
    // Add to content
    content.appendChild(healthContainer);
    content.appendChild(healthText);
    content.appendChild(manaContainer);
    content.appendChild(manaText);
    content.appendChild(tabs);
    content.appendChild(equipmentContent);
    content.appendChild(statsContent);
    
    this.element.appendChild(header);
    this.element.appendChild(content);
    
    // Store references to update later
    this.healthBar = healthBar;
    this.healthText = healthText;
    this.manaBar = manaBar;
    this.manaText = manaText;
    this.equipmentTab = equipmentTab;
    this.statsTab = statsTab;
    this.equipmentContent = equipmentContent;
    this.statsContent = statsContent;
    
    // Add event listeners
    equipmentTab.addEventListener('click', () => this.setActiveTab('equipment'));
    statsTab.addEventListener('click', () => this.setActiveTab('stats'));
    minimizeBtn.addEventListener('click', () => this.minimize());
    closeBtn.addEventListener('click', () => this.hide());
    
    return this.element;
  }
  
  /**
   * Set the active tab
   * @param {string} tab - Tab name
   */
  setActiveTab(tab) {
    this.activeTab = tab;
    
    if (tab === 'equipment') {
      this.equipmentTab.className = 'tab active';
      this.equipmentTab.style.backgroundColor = '#333';
      this.equipmentTab.style.color = '#f59e0b';
      
      this.statsTab.className = 'tab';
      this.statsTab.style.backgroundColor = 'black';
      this.statsTab.style.color = '#666';
      
      this.equipmentContent.style.display = 'block';
      this.statsContent.style.display = 'none';
    } else {
      this.statsTab.className = 'tab active';
      this.statsTab.style.backgroundColor = '#333';
      this.statsTab.style.color = '#f59e0b';
      
      this.equipmentTab.className = 'tab';
      this.equipmentTab.style.backgroundColor = 'black';
      this.equipmentTab.style.color = '#666';
      
      this.equipmentContent.style.display = 'none';
      this.statsContent.style.display = 'block';
    }
  }
  
  /**
   * Minimize the panel (to be implemented)
   */
  minimize() {
    // Call the manager to handle minimization
    if (this.manager) {
      // For now just hide it
      this.hide();
    }
  }
  
  /**
   * Update health display
   * @param {number} current - Current health
   * @param {number} max - Maximum health
   */
  updateHealth(current, max) {
    if (!this.healthBar || !this.healthText) return;
    
    this.health = current || 0;
    this.maxHealth = max || 100;
    
    // Update health bar
    const percent = Math.min(100, Math.max(0, (this.health / this.maxHealth) * 100));
    this.healthBar.style.width = `${percent}%`;
    
    // Update health text
    this.healthText.textContent = `${this.health}/${this.maxHealth}`;
  }
  
  /**
   * Update mana display
   * @param {number} current - Current mana
   * @param {number} max - Maximum mana
   */
  updateMana(current, max) {
    if (!this.manaBar || !this.manaText) return;
    
    this.mana = current || 0;
    this.maxMana = max || 100;
    
    // Update mana bar
    const percent = Math.min(100, Math.max(0, (this.mana / this.maxMana) * 100));
    this.manaBar.style.width = `${percent}%`;
    
    // Update mana text
    this.manaText.textContent = `${this.mana}/${this.maxMana}`;
  }
  
  /**
   * Update the component with game state data
   * @param {Object} gameState - Game state reference
   */
  update(gameState) {
    if (!gameState || !gameState.player) return;
    
    // Update health and mana
    this.updateHealth(gameState.player.health, gameState.player.maxHealth);
    this.updateMana(gameState.player.mana, gameState.player.maxMana);
    
    // TODO: Update inventory and stats when they're available in gameState
  }
} 