// GameUI.js - UI Manager for integrating the HTML-based UI

/**
 * Manages the game UI overlay
 */
export class GameUI {
  /**
   * Initialize the game UI
   * @param {Object} gameState - Reference to the game state
   */
  constructor(gameState) {
    this.gameState = gameState;
    this.container = document.getElementById('gameUIContainer');
    this.uiWindow = null;
    this.uiDocument = null;
    this.isLoaded = false;
    this.callbacks = {};
    
    // Create an iframe to isolate the UI styles and scripts
    this.uiFrame = document.createElement('iframe');
    this.uiFrame.style.border = 'none';
    this.uiFrame.style.width = '100%';
    this.uiFrame.style.height = '100%';
    this.uiFrame.style.pointerEvents = 'none'; // Set to none by default
    this.uiFrame.style.backgroundColor = 'transparent';
    this.container.appendChild(this.uiFrame);
  }

  /**
   * Load the UI from the HTML file
   * @param {Object} options - Optional settings
   * @returns {Promise} Resolves when the UI is loaded
   */
  async load(options = {}) {
    return new Promise((resolve, reject) => {
      try {
        // Fetch the UI HTML content
        fetch('/UI/GameUI.html')
          .then(response => {
            if (!response.ok) {
              throw new Error(`Failed to load UI: ${response.status} ${response.statusText}`);
            }
            return response.text();
          })
          .then(html => {
            // Write the HTML content into the iframe
            this.uiFrame.onload = () => {
              try {
                this.uiWindow = this.uiFrame.contentWindow;
                this.uiDocument = this.uiFrame.contentDocument || this.uiFrame.contentWindow.document;
                
                // Apply transparency to the body
                if (this.uiDocument.body) {
                  this.uiDocument.body.style.backgroundColor = 'transparent';
                }
                
                // Set up communication between the game and UI
                this.setupCommunication();
                
                // Configure UI according to options
                this.configureUI(options);
                
                this.isLoaded = true;
                console.log('Game UI loaded successfully');
                resolve();
              } catch (err) {
                console.error('Error setting up UI:', err);
                reject(err);
              }
            };
            
            // Write the HTML to the iframe document
            const doc = this.uiFrame.contentDocument || this.uiFrame.contentWindow.document;
            doc.open();
            doc.write(html);
            doc.close();
          })
          .catch(error => {
            console.error('Error loading game UI:', error);
            reject(error);
          });
      } catch (err) {
        console.error('Failed to initialize game UI:', err);
        reject(err);
      }
    });
  }

  /**
   * Set up communication between the game and UI
   */
  setupCommunication() {
    if (!this.uiWindow || !this.uiDocument) return;
    
    // Expose game data to the UI
    this.uiWindow.gameAPI = {
      updateHealth: (current, max) => this.updateHealth(current, max),
      updateMana: (current, max) => this.updateMana(current, max),
      addChatMessage: (message, type, sender) => this.addChatMessage(message, type, sender),
      setPlayerStats: (stats) => this.setPlayerStats(stats),
      setInventory: (items) => this.setInventory(items),
      onCommand: (callback) => this.registerCallback('command', callback),
      onChatMessage: (callback) => this.registerCallback('chatMessage', callback)
    };
    
    // Handle chat input
    const chatInput = this.uiDocument.getElementById('chat-input');
    if (chatInput) {
      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
          const message = e.target.value.trim();
          if (this.callbacks['chatMessage']) {
            this.callbacks['chatMessage'](message);
          }
          e.target.value = '';
        }
      });
    }
    
    // Setup key bindings for UI controls
    this.setupKeyBindings();
    
    // Enable pointer events only on UI elements
    this.setupUIInteractivity();
  }
  
  /**
   * Set up keyboard shortcuts for UI functions
   */
  setupKeyBindings() {
    // Add a global key binding to toggle UI visibility (Alt+U)
    window.addEventListener('keydown', (e) => {
      // Alt+U to toggle UI visibility
      if (e.altKey && e.key === 'u') {
        this.toggle();
        e.preventDefault();
        console.log(`Game UI ${this.container.style.display === 'none' ? 'hidden' : 'shown'}`);
      }
      
      // Alt+M to toggle minimap
      if (e.altKey && e.key === 'm') {
        this.togglePanel('minimap');
        e.preventDefault();
      }
      
      // Alt+C to toggle character panel
      if (e.altKey && e.key === 'c') {
        this.togglePanel('stats-panel');
        e.preventDefault();
      }
      
      // Alt+T to toggle chat
      if (e.altKey && e.key === 't') {
        this.togglePanel('chat-panel');
        e.preventDefault();
      }
    });
  }
  
  /**
   * Toggle visibility of a specific panel
   * @param {string} panelId - ID of the panel to toggle
   */
  togglePanel(panelId) {
    if (!this.isLoaded || !this.uiDocument) return;
    
    const panel = this.uiDocument.getElementById(panelId);
    if (panel) {
      panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
      console.log(`Panel ${panelId} ${panel.style.display === 'none' ? 'hidden' : 'shown'}`);
    }
  }
  
  /**
   * Set up UI interactivity by enabling pointer events only on UI elements
   */
  setupUIInteractivity() {
    if (!this.uiDocument) return;
    
    // Make the game container's direct background transparent to mouse events
    const gameContainer = this.uiDocument.getElementById('game-container');
    if (gameContainer) {
      gameContainer.style.backgroundColor = 'transparent';
    }
    
    // Make the game-world div transparent to mouse events
    const gameWorld = this.uiDocument.getElementById('game-world');
    if (gameWorld) {
      gameWorld.style.pointerEvents = 'none';
    }
    
    // Make all UI elements interactive
    const uiElements = [
      this.uiDocument.getElementById('panel-toolbar'),
      this.uiDocument.getElementById('minimap'),
      this.uiDocument.getElementById('stats-panel'),
      this.uiDocument.getElementById('chat-panel'),
      this.uiDocument.getElementById('unit-bar'),
      this.uiDocument.getElementById('quick-access-bar'),
      this.uiDocument.querySelector('.control-bar')
    ];
    
    uiElements.forEach(element => {
      if (element) {
        element.style.pointerEvents = 'auto';
        
        // Find all buttons and interactive elements within this element
        const interactiveElements = element.querySelectorAll('button, input, .panel-controls, .inventory-slot');
        interactiveElements.forEach(el => {
          el.style.pointerEvents = 'auto';
        });
      }
    });
    
    // Handle the special case for the chat input
    const chatInput = this.uiDocument.getElementById('chat-input');
    if (chatInput) {
      chatInput.style.pointerEvents = 'auto';
    }
    
    console.log('UI interactivity setup complete');
  }

  /**
   * Update the player's health in the UI
   * @param {number} current - Current health
   * @param {number} max - Maximum health
   */
  updateHealth(current, max) {
    if (!this.isLoaded) return;
    
    try {
      const healthBar = this.uiDocument.getElementById('health-bar');
      const healthText = this.uiDocument.getElementById('health-text');
      
      if (healthBar) {
        healthBar.style.width = `${(current / max) * 100}%`;
      }
      
      if (healthText) {
        healthText.textContent = `${current}/${max}`;
      }
      
      // Update the internal state if it exists
      if (this.uiWindow.gameState) {
        this.uiWindow.gameState.health = current;
        this.uiWindow.gameState.maxHealth = max;
      }
    } catch (err) {
      console.warn('Failed to update health in UI:', err);
    }
  }

  /**
   * Update the player's mana in the UI
   * @param {number} current - Current mana
   * @param {number} max - Maximum mana
   */
  updateMana(current, max) {
    if (!this.isLoaded) return;
    
    try {
      const manaBar = this.uiDocument.getElementById('mana-bar');
      const manaText = this.uiDocument.getElementById('mana-text');
      
      if (manaBar) {
        manaBar.style.width = `${(current / max) * 100}%`;
      }
      
      if (manaText) {
        manaText.textContent = `${current}/${max}`;
      }
      
      // Update the internal state if it exists
      if (this.uiWindow.gameState) {
        this.uiWindow.gameState.mana = current;
        this.uiWindow.gameState.maxMana = max;
      }
    } catch (err) {
      console.warn('Failed to update mana in UI:', err);
    }
  }

  /**
   * Add a message to the chat
   * @param {string} message - Message content
   * @param {string} type - Message type (system, player, death)
   * @param {string} sender - Message sender
   */
  addChatMessage(message, type, sender) {
    if (!this.isLoaded) return;
    
    try {
      // Call the UI's addChatMessage function if it exists
      if (this.uiWindow.addChatMessage) {
        this.uiWindow.addChatMessage(message, type, sender);
      }
    } catch (err) {
      console.warn('Failed to add chat message in UI:', err);
    }
  }

  /**
   * Update player stats in the UI
   * @param {Array} stats - Array of stat objects
   */
  setPlayerStats(stats) {
    if (!this.isLoaded) return;
    
    try {
      // If the UI has its own stats array, update it
      if (this.uiWindow.stats) {
        for (let i = 0; i < stats.length && i < this.uiWindow.stats.length; i++) {
          this.uiWindow.stats[i].value = stats[i].value;
          this.uiWindow.stats[i].max = stats[i].max;
        }
        
        // Force UI refresh if there's a function for it
        if (typeof this.uiWindow.initUI === 'function') {
          this.uiWindow.initUI();
        }
      }
    } catch (err) {
      console.warn('Failed to update player stats in UI:', err);
    }
  }

  /**
   * Update inventory items in the UI
   * @param {Array} items - Array of inventory items
   */
  setInventory(items) {
    if (!this.isLoaded) return;
    
    try {
      // If the UI has its own inventory array, update it
      if (this.uiWindow.inventory) {
        this.uiWindow.inventory = items;
        
        // Force UI refresh if there's a function for it
        if (typeof this.uiWindow.initUI === 'function') {
          this.uiWindow.initUI();
        }
      }
    } catch (err) {
      console.warn('Failed to update inventory in UI:', err);
    }
  }

  /**
   * Register a callback function
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  registerCallback(event, callback) {
    this.callbacks[event] = callback;
  }

  /**
   * Show the UI
   */
  show() {
    this.container.style.display = 'block';
  }

  /**
   * Hide the UI
   */
  hide() {
    this.container.style.display = 'none';
  }

  /**
   * Toggle UI visibility
   */
  toggle() {
    if (this.container.style.display === 'none') {
      this.show();
    } else {
      this.hide();
    }
  }
  
  /**
   * Update the UI with the current game state
   * @param {Object} gameState - Current game state
   */
  update(gameState) {
    if (!this.isLoaded || !gameState) return;
    
    // Update player health and mana
    if (gameState.player) {
      this.updateHealth(gameState.player.health || 0, gameState.player.maxHealth || 100);
      this.updateMana(gameState.player.mana || 0, gameState.player.maxMana || 100);
    }
  }

  /**
   * Configure UI layout based on options
   * @param {Object} options - Configuration options
   */
  configureUI(options = {}) {
    if (!this.uiDocument) return;
    
    // Adjust z-index if provided
    const zIndex = options.zIndex || 50;
    
    // Add custom CSS to optimize UI rendering
    const styleElement = this.uiDocument.createElement('style');
    styleElement.textContent = `
      body, html, #game-container {
        background-color: transparent !important;
      }
      
      .game-world {
        background-color: transparent !important;
        pointer-events: none !important;
      }
      
      .panel {
        z-index: ${zIndex} !important;
      }
      
      .panel-toolbar {
        z-index: ${zIndex + 10} !important;
      }
      
      .quick-access-bar {
        z-index: ${zIndex + 10} !important;
      }
      
      .chat-panel {
        z-index: ${zIndex + 5} !important;
      }
      
      .unit-bar, .control-bar {
        z-index: ${zIndex + 5} !important;
      }
    `;
    
    this.uiDocument.head.appendChild(styleElement);
    
    // Start with only essential panels visible if minimizeUI option is set
    if (options.minimizeUI) {
      const panelsToMinimize = ['minimap', 'stats-panel', 'unit-bar'];
      
      panelsToMinimize.forEach(panelId => {
        const panel = this.uiDocument.getElementById(panelId);
        if (panel) {
          panel.style.display = 'none';
        }
      });
    }
  }
}

// Export a singleton instance for easy access
let gameUI = null;

/**
 * Initialize the game UI
 * @param {Object} gameState - Game state reference
 * @returns {GameUI} The game UI instance
 */
export function initGameUI(gameState) {
  if (!gameUI) {
    gameUI = new GameUI(gameState);
  }
  return gameUI;
}

/**
 * Get the game UI instance
 * @returns {GameUI} The game UI instance
 */
export function getGameUI() {
  return gameUI;
} 