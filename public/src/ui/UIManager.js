/**
 * UIManager - A component-based UI system for the game
 * This approach avoids iframes and creates UI elements directly in the DOM
 */

// Import the debug overlay
import { debugOverlay } from './DebugOverlay.js';

/**
 * Base class for UI components
 */
export class UIComponent {
  /**
   * Create a UI component
   * @param {Object} gameState - Game state reference
   * @param {UIManager} manager - UI manager reference
   */
  constructor(gameState, manager) {
    this.gameState = gameState;
    this.manager = manager;
    this.element = null;
    this.isVisible = true;
    this.wasVisible = true;
  }
  
  /**
   * Initialize the component
   * @returns {HTMLElement} The component's DOM element
   */
  async init() {
    // Create base element
    this.element = document.createElement('div');
    this.element.className = 'ui-component';
    this.element.style.position = 'absolute';
    this.element.style.pointerEvents = 'auto';
    
    return this.element;
  }
  
  /**
   * Show the component
   */
  show() {
    if (this.element) {
      this.element.style.display = 'block';
      this.isVisible = true;
    }
  }
  
  /**
   * Hide the component
   */
  hide() {
    if (this.element) {
      this.element.style.display = 'none';
      this.isVisible = false;
    }
  }
  
  /**
   * Toggle component visibility
   */
  toggle() {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }
  
  /**
   * Update the component
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    // Override in derived components
  }
}

export class UIManager {
  /**
   * Create a new UI manager
   * @param {Object} gameState - Reference to the game state
   */
  constructor(gameState) {
    this.gameState = gameState;
    this.container = null;
    this.isInitialized = false;
    this.components = {};
    this.callbacks = {};
    
    // Debug overlay reference
    this.debugOverlay = debugOverlay;
  }

  /**
   * Initialize the UI system
   * @returns {Promise} Resolves when UI is initialized
   */
  async init() {
    // Create main container
    this.container = document.getElementById('gameUIContainer') || this.createContainer();
    
    // Clear container if it already has content
    this.container.innerHTML = '';
    
    // Initialize component system
    await this.initComponents();
    
    // Set up key bindings
    this.setupKeyBindings();
    
    this.isInitialized = true;
    
    return Promise.resolve();
  }
  
  /**
   * Create the UI container if it doesn't exist
   * @returns {HTMLElement} The UI container
   */
  createContainer() {
    const container = document.createElement('div');
    container.id = 'gameUIContainer';
    container.style.position = 'absolute';
    container.style.top = '0';
    container.style.left = '0';
    container.style.width = '100%';
    container.style.height = '100%';
    container.style.pointerEvents = 'none';
    container.style.zIndex = '50';
    document.body.appendChild(container);
    return container;
  }
  
  /**
   * Initialize all UI components
   * @returns {Promise} Resolves when all components are loaded
   */
  async initComponents() {
    try {
      // Create UI toolbar
      this.createToolbar();
      
      // Import components dynamically to avoid circular dependencies
      try {
        // Dynamically import components
        const { Minimap } = await import('./components/Minimap.js');
        const { CharacterPanel } = await import('./components/CharacterPanel.js');
        const { ChatPanel } = await import('./components/ChatPanel.js');
        const { UnitBar } = await import('./components/UnitBar.js');
        const { ControlBar } = await import('./components/ControlBar.js');
        
        // Initialize components with loading promises
        const componentPromises = [
          this.initComponent('minimap', Minimap),
          this.initComponent('characterPanel', CharacterPanel),
          this.initComponent('chatPanel', ChatPanel),
          this.initComponent('unitBar', UnitBar),
          this.initComponent('controlBar', ControlBar)
        ];
        
        // Wait for all components to initialize
        await Promise.all(componentPromises.filter(p => p));
        
        console.log('All UI components initialized successfully');
      } catch (error) {
        console.error('Error importing UI components:', error);
      }
      
      return Promise.resolve();
    } catch (err) {
      console.error('Error initializing UI components:', err);
      return Promise.reject(err);
    }
  }
  
  /**
   * Initialize a single UI component
   * @param {string} id - Component ID
   * @param {Class} ComponentClass - Component class
   * @returns {Promise|null} Promise that resolves when component is loaded, or null if class not available
   */
  async initComponent(id, ComponentClass) {
    if (!ComponentClass) {
      console.warn(`Component class for ${id} not found`);
      return null;
    }
    
    try {
      // Create component instance
      const component = new ComponentClass(this.gameState, this);
      
      // Initialize the component
      const element = await component.init();
      
      // Add to container
      if (element) {
        this.container.appendChild(element);
        
        // Store reference to component
        this.components[id] = component;
        console.log(`Component ${id} initialized`);
      }
      
      return Promise.resolve(component);
    } catch (err) {
      console.error(`Error initializing component ${id}:`, err);
      return Promise.reject(err);
    }
  }
  
  /**
   * Create the toolbar for toggling UI components
   */
  createToolbar() {
    const toolbar = document.createElement('div');
    toolbar.className = 'ui-toolbar';
    toolbar.style.position = 'absolute';
    toolbar.style.top = '8px';
    toolbar.style.left = '50%';
    toolbar.style.transform = 'translateX(-50%)';
    toolbar.style.display = 'flex';
    toolbar.style.gap = '4px';
    toolbar.style.padding = '4px';
    toolbar.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    toolbar.style.borderRadius = '4px';
    toolbar.style.border = '1px solid #444';
    toolbar.style.zIndex = '100';
    toolbar.style.pointerEvents = 'auto';
    
    // Add buttons for each panel
    const buttons = [
      { id: 'minimap', label: 'Map' },
      { id: 'characterPanel', label: 'Character' },
      { id: 'chatPanel', label: 'Chat' },
      { id: 'unitBar', label: 'Units' }
    ];
    
    buttons.forEach(btn => {
      const button = document.createElement('button');
      button.textContent = btn.label;
      button.dataset.panel = btn.id;
      button.style.backgroundColor = '#222';
      button.style.color = '#aaa';
      button.style.border = '1px solid #444';
      button.style.padding = '2px 8px';
      button.style.cursor = 'pointer';
      button.style.fontSize = '12px';
      
      // Toggle panel when clicked
      button.addEventListener('click', () => {
        this.toggleComponent(btn.id);
        
        // Update button state
        if (this.components[btn.id] && this.components[btn.id].isVisible) {
          button.style.backgroundColor = '#333';
          button.style.color = '#f59e0b';
          button.style.borderColor = '#f59e0b';
        } else {
          button.style.backgroundColor = '#222';
          button.style.color = '#aaa';
          button.style.borderColor = '#444';
        }
      });
      
      toolbar.appendChild(button);
    });
    
    // Add debug button
    const debugButton = document.createElement('button');
    debugButton.textContent = 'Debug';
    debugButton.dataset.special = 'debug';
    debugButton.style.backgroundColor = '#222';
    debugButton.style.color = '#aaa';
    debugButton.style.border = '1px solid #444';
    debugButton.style.padding = '2px 8px';
    debugButton.style.cursor = 'pointer';
    debugButton.style.fontSize = '12px';
    debugButton.style.marginLeft = '8px'; // Separate it from the panel buttons
    
    // Toggle debug overlay when clicked
    debugButton.addEventListener('click', () => {
      this.toggleDebugOverlay();
      
      // Update button state
      if (this.debugOverlay.container.style.display !== 'none') {
        debugButton.style.backgroundColor = '#333';
        debugButton.style.color = '#10b981'; // Green for debug
        debugButton.style.borderColor = '#10b981';
      } else {
        debugButton.style.backgroundColor = '#222';
        debugButton.style.color = '#aaa';
        debugButton.style.borderColor = '#444';
      }
    });
    
    toolbar.appendChild(debugButton);
    this.debugButton = debugButton;
    
    this.container.appendChild(toolbar);
    this.toolbar = toolbar;
  }
  
  /**
   * Toggle the debug overlay
   */
  toggleDebugOverlay() {
    if (this.debugOverlay) {
      this.debugOverlay.toggle();
      
      // Update button state
      if (this.debugButton) {
        if (this.debugOverlay.container.style.display !== 'none') {
          this.debugButton.style.backgroundColor = '#333';
          this.debugButton.style.color = '#10b981';
          this.debugButton.style.borderColor = '#10b981';
        } else {
          this.debugButton.style.backgroundColor = '#222';
          this.debugButton.style.color = '#aaa';
          this.debugButton.style.borderColor = '#444';
        }
      }
    }
  }
  
  /**
   * Toggle a specific UI component
   * @param {string} id - Component ID
   */
  toggleComponent(id) {
    const component = this.components[id];
    if (component && typeof component.toggle === 'function') {
      component.toggle();
      
      // Update toolbar button state
      const button = this.toolbar.querySelector(`button[data-panel="${id}"]`);
      if (button) {
        if (component.isVisible) {
          button.style.backgroundColor = '#333';
          button.style.color = '#f59e0b';
          button.style.borderColor = '#f59e0b';
        } else {
          button.style.backgroundColor = '#222';
          button.style.color = '#aaa';
          button.style.borderColor = '#444';
        }
      }
    }
  }
  
  /**
   * Toggle the entire UI
   */
  toggle() {
    if (this.container.style.display === 'none') {
      this.show();
    } else {
      this.hide();
    }
  }
  
  /**
   * Show the UI
   */
  show() {
    this.container.style.display = 'block';
    
    // Show each component
    Object.values(this.components).forEach(component => {
      if (component.wasVisible) {
        component.show();
      }
    });
  }
  
  /**
   * Hide the UI
   */
  hide() {
    // Store component visibility state
    Object.values(this.components).forEach(component => {
      component.wasVisible = component.isVisible;
    });
    
    // Hide all components
    Object.values(this.components).forEach(component => {
      component.hide();
    });
    
    this.container.style.display = 'none';
  }
  
  /**
   * Update the UI with the current game state
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    if (!this.isInitialized) return;
    
    // Update each component
    Object.values(this.components).forEach(component => {
      if (typeof component.update === 'function') {
        component.update(gameState);
      }
    });
    
    // Update debug overlay with current time
    if (this.debugOverlay && this.debugOverlay.container.style.display !== 'none') {
      this.debugOverlay.update(gameState, performance.now());
    }
  }

  /**
   * Add a chat message
   * @param {string} message - Message text
   * @param {string} type - Message type (system, player, etc.)
   * @param {string} sender - Message sender
   */
  addChatMessage(message, type = 'system', sender = 'System') {
    const chatPanel = this.components['chatPanel'];
    if (chatPanel && typeof chatPanel.addMessage === 'function') {
      chatPanel.addMessage({ text: message, type, sender });
    }
  }
  
  /**
   * Update health display
   * @param {number} current - Current health
   * @param {number} max - Maximum health
   */
  updateHealth(current, max) {
    // Update in character panel
    const characterPanel = this.components['characterPanel'];
    if (characterPanel && typeof characterPanel.updateHealth === 'function') {
      characterPanel.updateHealth(current, max);
    }
    
    // Update in control bar
    const controlBar = this.components['controlBar'];
    if (controlBar && typeof controlBar.update === 'function') {
      controlBar.update({ health: current, maxHealth: max });
    }
  }
  
  /**
   * Update mana display
   * @param {number} current - Current mana
   * @param {number} max - Maximum mana
   */
  updateMana(current, max) {
    // Update in character panel
    const characterPanel = this.components['characterPanel'];
    if (characterPanel && typeof characterPanel.updateMana === 'function') {
      characterPanel.updateMana(current, max);
    }
    
    // Update in control bar
    const controlBar = this.components['controlBar'];
    if (controlBar && typeof controlBar.update === 'function') {
      controlBar.update({ mana: current, maxMana: max });
    }
  }
  
  /**
   * Register an event callback
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (!this.callbacks[event]) {
      this.callbacks[event] = [];
    }
    this.callbacks[event].push(callback);
  }
  
  /**
   * Trigger an event
   * @param {string} event - Event name
   * @param {Object} data - Event data
   */
  trigger(event, data) {
    if (this.callbacks[event]) {
      this.callbacks[event].forEach(callback => {
        callback(data);
      });
    }
  }

  /**
   * Set up key bindings for the UI
   */
  setupKeyBindings() {
    window.addEventListener('keydown', (e) => {
      // Alt+U to toggle entire UI
      if (e.altKey && e.key === 'u') {
        this.toggle();
        e.preventDefault();
      }
      
      // Alt+M to toggle minimap
      if (e.altKey && e.key === 'm') {
        this.toggleComponent('minimap');
        e.preventDefault();
      }
      
      // Alt+C to toggle character panel
      if (e.altKey && e.key === 'c') {
        this.toggleComponent('characterPanel');
        e.preventDefault();
      }
      
      // Alt+T to toggle chat
      if (e.altKey && e.key === 't') {
        this.toggleComponent('chatPanel');
        e.preventDefault();
      }
      
      // Alt+D to toggle debug overlay
      if (e.altKey && e.key === 'd') {
        this.toggleDebugOverlay();
        e.preventDefault();
      }
    });
  }
}

// Export a singleton instance
let uiManager = null;

/**
 * Initialize the UI manager
 * @param {Object} gameState - Game state reference
 * @returns {UIManager} The UI manager instance
 */
export function initUIManager(gameState) {
  if (!uiManager) {
    uiManager = new UIManager(gameState);
  }
  return uiManager;
}

/**
 * Get the UI manager instance
 * @returns {UIManager} The UI manager instance
 */
export function getUIManager() {
  return uiManager;
} 