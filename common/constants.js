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


