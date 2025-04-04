// public/src/game/gamestate.js

import { Camera } from '../camera.js';
import { character } from '../entities/character.js';

/**
 * Game state object - holds references to all game components
 */
export const gameState = {
    // Core components
    character: character,
    camera: new Camera('top-down', { x: character.x, y: character.y }, 1),
    
    // Game managers - initialized in game.js
    networkManager: null,
    bulletManager: null, 
    enemyManager: null,
    map: null,
    collisionManager: null,
    
    // Tracking info
    lastUpdateX: character.x,
    lastUpdateY: character.y,
    
    // Game status
    isConnected: false,
    isPaused: false,
    
    // Settings
    settings: {
        soundEnabled: true,
        musicEnabled: true,
        showFPS: false
    }
};