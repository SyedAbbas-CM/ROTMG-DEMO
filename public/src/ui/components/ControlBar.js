/**
 * Control bar component for game commands
 */
import { UIComponent } from '../UIManager.js';

export class ControlBar extends UIComponent {
  /**
   * Create a control bar component
   * @param {Object} gameState - Game state reference
   * @param {Object} manager - UI manager reference
   */
  constructor(gameState, manager) {
    super(gameState, manager);
    
    this.commands = [];
    this.abilities = [];
  }
  
  /**
   * Initialize the component
   * @returns {HTMLElement} The component's DOM element
   */
  async init() {
    // Create control bar container
    this.element = document.createElement('div');
    this.element.className = 'ui-control-bar';
    this.element.style.position = 'absolute';
    this.element.style.bottom = '8px';
    this.element.style.left = '50%';
    this.element.style.transform = 'translateX(-50%)';
    this.element.style.display = 'flex';
    this.element.style.flexDirection = 'column';
    this.element.style.alignItems = 'center';
    this.element.style.width = 'auto';
    this.element.style.zIndex = '10';
    this.element.style.pointerEvents = 'auto';
    
    // Create quick access bar for minimized panels
    const quickAccessBar = document.createElement('div');
    quickAccessBar.className = 'quick-access-bar';
    quickAccessBar.style.display = 'flex';
    quickAccessBar.style.gap = '4px';
    quickAccessBar.style.marginBottom = '4px';
    
    // Create main control bar
    const mainBar = document.createElement('div');
    mainBar.className = 'main-control-bar';
    mainBar.style.display = 'flex';
    mainBar.style.gap = '4px';
    mainBar.style.padding = '4px 8px';
    mainBar.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    mainBar.style.borderRadius = '4px';
    mainBar.style.border = '1px solid #444';
    
    // Create resource indicators
    const resourcesContainer = document.createElement('div');
    resourcesContainer.className = 'resources';
    resourcesContainer.style.display = 'flex';
    resourcesContainer.style.alignItems = 'center';
    resourcesContainer.style.marginRight = '8px';
    resourcesContainer.style.borderRight = '1px solid #444';
    resourcesContainer.style.paddingRight = '8px';
    
    // Health bar
    const healthContainer = document.createElement('div');
    healthContainer.className = 'resource-container';
    healthContainer.style.display = 'flex';
    healthContainer.style.alignItems = 'center';
    healthContainer.style.marginRight = '8px';
    
    const healthIcon = document.createElement('div');
    healthIcon.className = 'resource-icon health-icon';
    healthIcon.innerHTML = '♥';
    healthIcon.style.color = '#ef4444';
    healthIcon.style.fontSize = '16px';
    healthIcon.style.marginRight = '4px';
    
    const healthBarContainer = document.createElement('div');
    healthBarContainer.className = 'health-bar-container';
    healthBarContainer.style.width = '80px';
    healthBarContainer.style.height = '8px';
    healthBarContainer.style.backgroundColor = '#333';
    healthBarContainer.style.border = '1px solid #444';
    healthBarContainer.style.borderRadius = '2px';
    healthBarContainer.style.overflow = 'hidden';
    
    const healthBar = document.createElement('div');
    healthBar.className = 'health-bar';
    healthBar.style.width = '75%';
    healthBar.style.height = '100%';
    healthBar.style.backgroundColor = '#ef4444';
    
    const healthText = document.createElement('div');
    healthText.className = 'health-text';
    healthText.textContent = '750/1000';
    healthText.style.fontSize = '10px';
    healthText.style.color = '#eee';
    healthText.style.marginLeft = '4px';
    
    healthBarContainer.appendChild(healthBar);
    healthContainer.appendChild(healthIcon);
    healthContainer.appendChild(healthBarContainer);
    healthContainer.appendChild(healthText);
    
    // Mana bar
    const manaContainer = document.createElement('div');
    manaContainer.className = 'resource-container';
    manaContainer.style.display = 'flex';
    manaContainer.style.alignItems = 'center';
    
    const manaIcon = document.createElement('div');
    manaIcon.className = 'resource-icon mana-icon';
    manaIcon.innerHTML = '⚡';
    manaIcon.style.color = '#3b82f6';
    manaIcon.style.fontSize = '16px';
    manaIcon.style.marginRight = '4px';
    
    const manaBarContainer = document.createElement('div');
    manaBarContainer.className = 'mana-bar-container';
    manaBarContainer.style.width = '80px';
    manaBarContainer.style.height = '8px';
    manaBarContainer.style.backgroundColor = '#333';
    manaBarContainer.style.border = '1px solid #444';
    manaBarContainer.style.borderRadius = '2px';
    manaBarContainer.style.overflow = 'hidden';
    
    const manaBar = document.createElement('div');
    manaBar.className = 'mana-bar';
    manaBar.style.width = '50%';
    manaBar.style.height = '100%';
    manaBar.style.backgroundColor = '#3b82f6';
    
    const manaText = document.createElement('div');
    manaText.className = 'mana-text';
    manaText.textContent = '250/500';
    manaText.style.fontSize = '10px';
    manaText.style.color = '#eee';
    manaText.style.marginLeft = '4px';
    
    manaBarContainer.appendChild(manaBar);
    manaContainer.appendChild(manaIcon);
    manaContainer.appendChild(manaBarContainer);
    manaContainer.appendChild(manaText);
    
    resourcesContainer.appendChild(healthContainer);
    resourcesContainer.appendChild(manaContainer);
    
    // Command buttons container
    const commandsContainer = document.createElement('div');
    commandsContainer.className = 'commands-container';
    commandsContainer.style.display = 'flex';
    commandsContainer.style.gap = '4px';
    commandsContainer.style.alignItems = 'center';
    commandsContainer.style.marginRight = '8px';
    commandsContainer.style.borderRight = '1px solid #444';
    commandsContainer.style.paddingRight = '8px';
    
    // Add command buttons
    const moveCommand = this.createCommandButton('Move', 'M', '#22c55e');
    const attackCommand = this.createCommandButton('Attack', 'A', '#ef4444');
    const defendCommand = this.createCommandButton('Defend', 'D', '#3b82f6');
    const stopCommand = this.createCommandButton('Stop', 'S', '#a3a3a3');
    
    commandsContainer.appendChild(moveCommand);
    commandsContainer.appendChild(attackCommand);
    commandsContainer.appendChild(defendCommand);
    commandsContainer.appendChild(stopCommand);
    
    // Abilities container
    const abilitiesContainer = document.createElement('div');
    abilitiesContainer.className = 'abilities-container';
    abilitiesContainer.style.display = 'flex';
    abilitiesContainer.style.gap = '4px';
    abilitiesContainer.style.alignItems = 'center';
    
    // Append sections to main bar
    mainBar.appendChild(resourcesContainer);
    mainBar.appendChild(commandsContainer);
    mainBar.appendChild(abilitiesContainer);
    
    // Append bars to control bar
    this.element.appendChild(quickAccessBar);
    this.element.appendChild(mainBar);
    
    // Store references
    this.quickAccessBar = quickAccessBar;
    this.abilitiesContainer = abilitiesContainer;
    this.healthBar = healthBar;
    this.healthText = healthText;
    this.manaBar = manaBar;
    this.manaText = manaText;
    
    // Initialize with default abilities
    this.initializeDefaultAbilities();
    
    return this.element;
  }
  
  /**
   * Create a command button with icon and keybind
   * @param {string} name - Command name
   * @param {string} key - Keybind
   * @param {string} color - Button color
   * @returns {HTMLElement} Button element
   */
  createCommandButton(name, key, color) {
    const button = document.createElement('button');
    button.className = `command-button cmd-${name.toLowerCase()}`;
    button.title = `${name} (${key})`;
    button.style.width = '28px';
    button.style.height = '28px';
    button.style.backgroundColor = '#222';
    button.style.border = '1px solid #444';
    button.style.color = color;
    button.style.fontSize = '14px';
    button.style.fontWeight = 'bold';
    button.style.display = 'flex';
    button.style.alignItems = 'center';
    button.style.justifyContent = 'center';
    button.style.position = 'relative';
    button.style.cursor = 'pointer';
    
    button.textContent = key;
    
    // Add keybind indicator
    const keybind = document.createElement('div');
    keybind.className = 'keybind';
    keybind.textContent = key;
    keybind.style.position = 'absolute';
    keybind.style.bottom = '1px';
    keybind.style.right = '1px';
    keybind.style.fontSize = '8px';
    keybind.style.color = '#aaa';
    
    // Store command info
    button.dataset.command = name.toLowerCase();
    button.dataset.key = key;
    
    // Add click handler
    button.addEventListener('click', () => {
      this.executeCommand(name.toLowerCase());
    });
    
    return button;
  }
  
  /**
   * Create an ability button
   * @param {Object} ability - Ability data
   * @returns {HTMLElement} Button element
   */
  createAbilityButton(ability) {
    const button = document.createElement('button');
    button.className = 'ability-button';
    button.title = ability.name;
    button.style.width = '32px';
    button.style.height = '32px';
    button.style.backgroundColor = '#222';
    button.style.border = '1px solid #444';
    button.style.color = ability.color || 'white';
    button.style.fontSize = '16px';
    button.style.display = 'flex';
    button.style.alignItems = 'center';
    button.style.justifyContent = 'center';
    button.style.position = 'relative';
    button.style.cursor = 'pointer';
    
    // Set icon based on ability type
    if (ability.icon === 'shield') {
      button.innerHTML = '⛨';
      button.style.color = '#3b82f6';
    } else if (ability.icon === 'heart') {
      button.innerHTML = '♥';
      button.style.color = '#22c55e';
    } else if (ability.icon === 'sword') {
      button.innerHTML = '⚔';
      button.style.color = '#ef4444';
    } else if (ability.icon === 'zap') {
      button.innerHTML = '⚡';
      button.style.color = '#f59e0b';
    } else if (ability.icon === 'wall') {
      // Create a small square for wall ability
      const wallIcon = document.createElement('div');
      wallIcon.style.width = '12px';
      wallIcon.style.height = '12px';
      wallIcon.style.backgroundColor = '#555';
      wallIcon.style.border = '1px solid #444';
      button.appendChild(wallIcon);
    } else {
      button.textContent = ability.icon || '?';
    }
    
    // Add cooldown overlay (hidden by default)
    const cooldown = document.createElement('div');
    cooldown.className = 'ability-cooldown';
    cooldown.style.position = 'absolute';
    cooldown.style.bottom = '0';
    cooldown.style.left = '0';
    cooldown.style.width = '100%';
    cooldown.style.height = '0%';
    cooldown.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    cooldown.style.display = 'none';
    
    button.appendChild(cooldown);
    
    // Add keybind indicator if present
    if (ability.key) {
      const keybind = document.createElement('div');
      keybind.className = 'keybind';
      keybind.textContent = ability.key;
      keybind.style.position = 'absolute';
      keybind.style.bottom = '1px';
      keybind.style.right = '1px';
      keybind.style.fontSize = '8px';
      keybind.style.color = '#aaa';
      
      button.appendChild(keybind);
    }
    
    // Store ability ID
    button.dataset.abilityId = ability.id;
    
    // Add click handler
    button.addEventListener('click', () => {
      this.useAbility(ability.id);
    });
    
    return button;
  }
  
  /**
   * Initialize default abilities
   */
  initializeDefaultAbilities() {
    // Default abilities
    const defaultAbilities = [
      { id: 'heal', name: 'Heal', icon: 'heart', color: '#22c55e', key: '1' },
      { id: 'shield', name: 'Shield', icon: 'shield', color: '#3b82f6', key: '2' },
      { id: 'fireball', name: 'Fireball', icon: 'zap', color: '#f59e0b', key: '3' },
      { id: 'charge', name: 'Charge', icon: 'sword', color: '#ef4444', key: '4' }
    ];
    
    // Add ability buttons
    defaultAbilities.forEach(ability => {
      const button = this.createAbilityButton(ability);
      this.abilitiesContainer.appendChild(button);
    });
    
    // Store abilities
    this.abilities = defaultAbilities;
  }
  
  /**
   * Execute a command
   * @param {string} command - Command name
   */
  executeCommand(command) {
    console.log(`Executing command: ${command}`);
    
    // Trigger event for game to handle command
    this.manager.trigger('executeCommand', { command });
  }
  
  /**
   * Use an ability
   * @param {string} abilityId - Ability ID
   */
  useAbility(abilityId) {
    console.log(`Using ability: ${abilityId}`);
    
    // Find the ability
    const ability = this.abilities.find(a => a.id === abilityId);
    if (!ability) return;
    
    // Check if on cooldown
    if (ability.onCooldown) {
      console.log(`Ability ${abilityId} is on cooldown`);
      return;
    }
    
    // Trigger event for game to handle ability use
    this.manager.trigger('useAbility', { abilityId });
  }
  
  /**
   * Add a panel to quick access bar
   * @param {string} panelId - Panel ID
   * @param {string} title - Panel title
   */
  addToQuickAccess(panelId, title) {
    const button = document.createElement('button');
    button.className = 'quick-access-button';
    button.dataset.panel = panelId;
    button.title = `Restore ${title}`;
    button.textContent = title.charAt(0);
    button.style.width = '24px';
    button.style.height = '24px';
    button.style.backgroundColor = '#222';
    button.style.border = '1px solid #444';
    button.style.color = '#f59e0b';
    button.style.fontSize = '12px';
    button.style.display = 'flex';
    button.style.alignItems = 'center';
    button.style.justifyContent = 'center';
    button.style.cursor = 'pointer';
    
    button.addEventListener('click', () => {
      this.manager.trigger('restorePanel', { panelId });
      this.removeFromQuickAccess(panelId);
    });
    
    this.quickAccessBar.appendChild(button);
  }
  
  /**
   * Remove a panel from quick access bar
   * @param {string} panelId - Panel ID
   */
  removeFromQuickAccess(panelId) {
    const button = this.quickAccessBar.querySelector(`[data-panel="${panelId}"]`);
    if (button) {
      this.quickAccessBar.removeChild(button);
    }
  }
  
  /**
   * Start ability cooldown
   * @param {string} abilityId - Ability ID
   * @param {number} duration - Cooldown duration in seconds
   */
  startCooldown(abilityId, duration) {
    // Find the ability
    const ability = this.abilities.find(a => a.id === abilityId);
    if (!ability) return;
    
    // Mark as on cooldown
    ability.onCooldown = true;
    
    // Find the button
    const button = this.abilitiesContainer.querySelector(`[data-ability-id="${abilityId}"]`);
    if (!button) return;
    
    // Find cooldown overlay
    const cooldown = button.querySelector('.ability-cooldown');
    if (!cooldown) return;
    
    // Show cooldown overlay
    cooldown.style.display = 'block';
    cooldown.style.height = '100%';
    
    // Animate cooldown
    const startTime = Date.now();
    const animateCooldown = () => {
      const elapsed = Date.now() - startTime;
      const remaining = duration * 1000 - elapsed;
      
      if (remaining <= 0) {
        // Cooldown complete
        cooldown.style.display = 'none';
        cooldown.style.height = '0%';
        ability.onCooldown = false;
        return;
      }
      
      // Update cooldown height
      const percent = (remaining / (duration * 1000)) * 100;
      cooldown.style.height = `${percent}%`;
      
      // Continue animation
      requestAnimationFrame(animateCooldown);
    };
    
    // Start animation
    requestAnimationFrame(animateCooldown);
  }
  
  /**
   * Update the component with game state
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    // Update health
    if (gameState.health !== undefined && gameState.maxHealth !== undefined) {
      const healthPercent = (gameState.health / gameState.maxHealth) * 100;
      this.healthBar.style.width = `${healthPercent}%`;
      this.healthText.textContent = `${gameState.health}/${gameState.maxHealth}`;
    }
    
    // Update mana
    if (gameState.mana !== undefined && gameState.maxMana !== undefined) {
      const manaPercent = (gameState.mana / gameState.maxMana) * 100;
      this.manaBar.style.width = `${manaPercent}%`;
      this.manaText.textContent = `${gameState.mana}/${gameState.maxMana}`;
    }
    
    // Update abilities
    if (gameState.abilities) {
      // Update existing abilities or add new ones
      gameState.abilities.forEach(ability => {
        // Find existing ability
        const existingAbility = this.abilities.find(a => a.id === ability.id);
        
        if (existingAbility) {
          // Update existing ability
          Object.assign(existingAbility, ability);
        } else {
          // Add new ability
          this.abilities.push(ability);
          
          // Create and add button
          const button = this.createAbilityButton(ability);
          this.abilitiesContainer.appendChild(button);
        }
        
        // Check for cooldown
        if (ability.cooldownRemaining) {
          this.startCooldown(ability.id, ability.cooldownRemaining);
        }
      });
    }
  }
} 