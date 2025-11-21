// common/constants.js
// Small set of cross-environment tuning constants used by both server and client

export const NETWORK_SETTINGS = {
  UPDATE_RADIUS_TILES: 40,
  MAX_ENTITIES_PER_PACKET: 500,
  DELTA_COMPRESSION: true
};

// Lag Compensation Settings (server-side only)
// Note: These access process.env, so will only work server-side
export const LAG_COMPENSATION = typeof process !== 'undefined' && process.env ? {
  ENABLED: process.env.LAG_COMPENSATION_ENABLED !== 'false',
  MAX_REWIND_MS: parseInt(process.env.LAG_COMPENSATION_MAX_REWIND_MS || '200', 10),
  MIN_RTT_MS: parseInt(process.env.LAG_COMPENSATION_MIN_RTT_MS || '50', 10),
  DEBUG: process.env.LAG_COMPENSATION_DEBUG === 'true',
  HISTORY_SIZE: 10 // 300ms at 30Hz tick rate
} : {
  ENABLED: false,
  MAX_REWIND_MS: 200,
  MIN_RTT_MS: 50,
  DEBUG: false,
  HISTORY_SIZE: 10
};

// Movement Validation Settings (server-side only)
export const MOVEMENT_VALIDATION = typeof process !== 'undefined' && process.env ? {
  ENABLED: process.env.MOVEMENT_VALIDATION_ENABLED !== 'false',
  MAX_SPEED_TILES_PER_SEC: parseFloat(process.env.MOVEMENT_VALIDATION_MAX_SPEED || '7.2'),
  TELEPORT_THRESHOLD_TILES: parseFloat(process.env.MOVEMENT_VALIDATION_TELEPORT_THRESHOLD || '3.0'),
  LOG_INTERVAL_MS: 5000
} : {
  ENABLED: false,
  MAX_SPEED_TILES_PER_SEC: 7.2,
  TELEPORT_THRESHOLD_TILES: 3.0,
  LOG_INTERVAL_MS: 5000
};

// Collision Validation Settings (server-side only)
// Validates that collisions happen at reasonable positions based on player latency
export const COLLISION_VALIDATION = typeof process !== 'undefined' && process.env ? {
  ENABLED: process.env.COLLISION_VALIDATION_ENABLED !== 'false',
  // Mode: 'soft' = log suspicious only, 'strict' = reject invalid collisions
  MODE: process.env.COLLISION_VALIDATION_MODE || 'soft',
  // Max distance (tiles) player can be from server position, adjusted by latency
  MAX_DISTANCE_BASE_TILES: parseFloat(process.env.COLLISION_VALIDATION_MAX_DISTANCE || '2.0'),
  // Extra distance allowed per 100ms of RTT
  DISTANCE_PER_100MS_RTT: parseFloat(process.env.COLLISION_VALIDATION_DISTANCE_PER_RTT || '0.5'),
  // Suspicion threshold - distance multiplier that triggers logging
  SUSPICIOUS_THRESHOLD: parseFloat(process.env.COLLISION_VALIDATION_SUSPICIOUS_THRESHOLD || '1.5')
} : {
  ENABLED: false,
  MODE: 'soft',
  MAX_DISTANCE_BASE_TILES: 2.0,
  DISTANCE_PER_100MS_RTT: 0.5,
  SUSPICIOUS_THRESHOLD: 1.5
};


