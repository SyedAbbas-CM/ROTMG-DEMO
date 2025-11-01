/**
 * Centralized manager exports
 * This file re-exports all manager classes to avoid case sensitivity issues
 */

// Re-export all managers with consistent naming
export { ClientMapManager } from './map/ClientMapManager.js';
export { ClientNetworkManager } from './network/ClientNetworkManager.js';
// MessageType should be imported from /common/protocol.js directly, not from here
// export { MessageType } from './shared/messages.js'; // REMOVED - use /common/protocol.js instead
export { ClientBulletManager } from './game/ClientBulletManager.js';
export { ClientEnemyManager } from './game/ClientEnemyManager.js';
export { ClientCollisionManager } from './collision/ClientCollisionManager.js';

// Other important exports
export { debugOverlay } from './ui/DebugOverlay.js'; 

// UI System Exports - Two versions available:
// 1. Component-based UI system (recommended)
export { UIManager, initUIManager, getUIManager, UIComponent } from './ui/UIManager.js';

// 2. Legacy iframe-based UI system (kept for compatibility)
export { GameUI, initGameUI, getGameUI } from './ui/GameUI.js'; 