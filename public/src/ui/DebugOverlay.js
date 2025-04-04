// public/src/ui/DebugOverlay.js

/**
 * Debug overlay for monitoring game status and performance
 */
export class DebugOverlay {
    /**
     * Create a new debug overlay
     */
    constructor() {
        this.container = document.createElement('div');
        this.container.style.position = 'absolute';
        this.container.style.top = '40px';
        this.container.style.left = '10px';
        this.container.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        this.container.style.color = '#fff';
        this.container.style.padding = '10px';
        this.container.style.borderRadius = '5px';
        this.container.style.fontFamily = 'monospace';
        this.container.style.fontSize = '12px';
        this.container.style.zIndex = '1000';
        this.container.style.maxWidth = '300px';
        this.container.style.display = 'none'; // Hidden by default
        
        // Create sections
        this.fpsSection = this.createSection('FPS');
        this.networkSection = this.createSection('Network');
        this.entitiesSection = this.createSection('Entities');
        this.playerSection = this.createSection('Player');
        this.mapSection = this.createSection('Map');
        
        // Add to document
        document.body.appendChild(this.container);
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastFpsUpdate = 0;
        this.fps = 0;
        
        // Toggle with F3 key
        window.addEventListener('keydown', (e) => {
            if (e.code === 'F3') {
                this.toggle();
            }
        });

        // Basic constructor
        this.enabled = false;
    }
    
    /**
     * Create a section in the debug overlay
     * @param {string} title - Section title
     * @returns {Object} Section elements
     */
    createSection(title) {
        const section = document.createElement('div');
        section.style.marginBottom = '5px';
        
        const titleElem = document.createElement('div');
        titleElem.textContent = title;
        titleElem.style.borderBottom = '1px solid #aaa';
        titleElem.style.marginBottom = '2px';
        
        const content = document.createElement('div');
        content.style.marginLeft = '10px';
        
        section.appendChild(titleElem);
        section.appendChild(content);
        this.container.appendChild(section);
        
        return { section, title: titleElem, content };
    }
    
    /**
     * Show the debug overlay
     */
    show() {
        this.container.style.display = 'block';
    }
    
    /**
     * Hide the debug overlay
     */
    hide() {
        this.container.style.display = 'none';
    }
    
    /**
     * Toggle the debug overlay
     */
    toggle() {
        if (this.container.style.display === 'none') {
            this.show();
        } else {
            this.hide();
        }
    }
    
    /**
     * Update the debug overlay with current game state
     * @param {Object} gameState - Current game state
     * @param {number} time - Current timestamp
     */
    update(gameState, time) {
        // Stub method
        if (!this.enabled) return;
        
        // Basic update
        console.log("Debug overlay updated", time);
        
        // Update FPS counter
        this.frameCount++;
        
        if (time - this.lastFpsUpdate >= 1000) {
            this.fps = Math.round(this.frameCount * 1000 / (time - this.lastFpsUpdate));
            this.frameCount = 0;
            this.lastFpsUpdate = time;
        }
        
        // Update sections
        this.updateFpsSection();
        this.updateNetworkSection(gameState);
        this.updateEntitiesSection(gameState);
        this.updatePlayerSection(gameState);
        this.updateMapSection(gameState);
    }
    
    /**
     * Update FPS section
     */
    updateFpsSection() {
        this.fpsSection.content.textContent = `${this.fps} FPS`;
        
        // Color code based on performance
        if (this.fps >= 50) {
            this.fpsSection.content.style.color = '#8f8';
        } else if (this.fps >= 30) {
            this.fpsSection.content.style.color = '#ff8';
        } else {
            this.fpsSection.content.style.color = '#f88';
        }
    }
    
    /**
     * Update network section
     * @param {Object} gameState - Current game state
     */
    updateNetworkSection(gameState) {
        const network = gameState.networkManager;
        
        if (!network) {
            this.networkSection.content.textContent = 'Network manager not initialized';
            return;
        }
        
        const status = network.isConnected() ? 'Connected' : 'Disconnected';
        const clientId = network.getClientId() || 'Unknown';
        
        this.networkSection.content.innerHTML = 
            `Status: ${status}<br>` +
            `Client ID: ${clientId}`;
    }
    
    /**
     * Update entities section
     * @param {Object} gameState - Current game state
     */
    updateEntitiesSection(gameState) {
        const bulletCount = gameState.bulletManager ? gameState.bulletManager.bulletCount : 0;
        const enemyCount = gameState.enemyManager ? gameState.enemyManager.enemyCount : 0;
        
        this.entitiesSection.content.innerHTML = 
            `Bullets: ${bulletCount}<br>` +
            `Enemies: ${enemyCount}`;
    }
    
    /**
     * Update player section
     * @param {Object} gameState - Current game state
     */
    updatePlayerSection(gameState) {
        const character = gameState.character;
        
        if (!character) {
            this.playerSection.content.textContent = 'Player not initialized';
            return;
        }
        
        this.playerSection.content.innerHTML = 
            `Position: (${character.x.toFixed(1)}, ${character.y.toFixed(1)})<br>` +
            `Health: ${character.health}<br>` +
            `Rotation: ${(character.rotation.yaw || 0).toFixed(2)} rad`;
    }
    
    /**
     * Update map section
     * @param {Object} gameState - Current game state
     */
    updateMapSection(gameState) {
        const map = gameState.map;
        
        if (!map) {
            this.mapSection.content.textContent = 'Map not initialized';
            return;
        }
        
        const chunkX = Math.floor(gameState.character.x / (map.chunkSize * map.tileSize));
        const chunkY = Math.floor(gameState.character.y / (map.chunkSize * map.tileSize));
        const chunksLoaded = map.chunks ? map.chunks.size : 0;
        
        this.mapSection.content.innerHTML = 
            `Current Chunk: (${chunkX}, ${chunkY})<br>` +
            `Chunks Loaded: ${chunksLoaded}<br>` +
            `View: ${gameState.camera.viewType}`;
    }
}

// Create a singleton instance
export const debugOverlay = new DebugOverlay();