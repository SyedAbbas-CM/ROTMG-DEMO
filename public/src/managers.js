/**
 * Centralized manager exports
 * This file re-exports all manager classes to avoid case sensitivity issues
 */

// Re-export all managers with consistent naming
export { ClientMapManager } from './map/ClientMapManager.js';
export { ClientNetworkManager, MessageType } from './network/ClientNetworkManager.js';
export { ClientBulletManager } from './game/ClientBulletManager.js';
export { ClientEnemyManager } from './game/ClientEnemyManager.js';
export { ClientCollisionManager } from './collision/ClientCollisionManager.js';

// Other important exports
export { debugOverlay } from './ui/DebugOverlay.js'; 