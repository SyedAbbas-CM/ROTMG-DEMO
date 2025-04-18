/**
 * Logger utility for consistent and configurable logging
 */

// Log levels
export const LOG_LEVELS = {
  NONE: 0,    // No logging at all
  ERROR: 1,   // Only errors
  WARN: 2,    // Errors and warnings
  INFO: 3,    // Normal information (default)
  DEBUG: 4,   // Detailed debug information
  VERBOSE: 5  // Everything including frequent updates
};

// Store log level settings for different modules
const moduleSettings = {
  // Default levels for modules
  'default': LOG_LEVELS.INFO,
  'player': LOG_LEVELS.INFO,
  'camera': LOG_LEVELS.INFO,
  'input': LOG_LEVELS.INFO,
  'render': LOG_LEVELS.INFO,
  'movement': LOG_LEVELS.INFO,
  'network': LOG_LEVELS.INFO,
  'collision': LOG_LEVELS.INFO,
  'coordinate': LOG_LEVELS.INFO
};

// Global log level - can be used to override all modules at once
let globalLogLevel = null;

/**
 * Set the log level for a specific module
 * @param {string} module - Module name
 * @param {number} level - Log level from LOG_LEVELS
 */
export function setLogLevel(module, level) {
  moduleSettings[module] = level;
  console.log(`Log level for ${module} set to ${getLogLevelName(level)}`);
}

/**
 * Set a global log level that overrides all module settings
 * @param {number} level - Log level from LOG_LEVELS, or null to use module settings
 */
export function setGlobalLogLevel(level) {
  globalLogLevel = level;
  const levelName = level === null ? 'module defaults' : getLogLevelName(level);
  console.log(`Global log level set to ${levelName}`);
}

/**
 * Get the name of a log level
 * @param {number} level - Log level
 * @returns {string} - Name of the log level
 */
function getLogLevelName(level) {
  return Object.keys(LOG_LEVELS).find(key => LOG_LEVELS[key] === level) || 'UNKNOWN';
}

/**
 * Get the effective log level for a module
 * @param {string} module - Module name
 * @returns {number} - Effective log level
 */
function getEffectiveLogLevel(module) {
  // Global setting overrides module settings if set
  if (globalLogLevel !== null) {
    return globalLogLevel;
  }
  
  // Use module setting if available, otherwise default
  return moduleSettings[module] !== undefined ? 
         moduleSettings[module] : 
         moduleSettings['default'];
}

/**
 * Check if a log message should be shown based on its level and module
 * @param {string} module - Module name
 * @param {number} level - Log level of the message
 * @returns {boolean} - True if the message should be logged
 */
function shouldLog(module, level) {
  return level <= getEffectiveLogLevel(module);
}

/**
 * Create a logger instance for a specific module
 * @param {string} module - Module name
 * @returns {Object} - Logger object with log methods
 */
export function createLogger(module) {
  return {
    error: (message, ...args) => {
      if (shouldLog(module, LOG_LEVELS.ERROR)) {
        console.error(`[${module}] ${message}`, ...args);
      }
    },
    
    warn: (message, ...args) => {
      if (shouldLog(module, LOG_LEVELS.WARN)) {
        console.warn(`[${module}] ${message}`, ...args);
      }
    },
    
    info: (message, ...args) => {
      if (shouldLog(module, LOG_LEVELS.INFO)) {
        console.log(`[${module}] ${message}`, ...args);
      }
    },
    
    debug: (message, ...args) => {
      if (shouldLog(module, LOG_LEVELS.DEBUG)) {
        console.log(`[${module} DEBUG] ${message}`, ...args);
      }
    },
    
    verbose: (message, ...args) => {
      if (shouldLog(module, LOG_LEVELS.VERBOSE)) {
        console.log(`[${module} VERBOSE] ${message}`, ...args);
      }
    },
    
    // Utility to log only occasionally (for high-frequency events)
    // probability should be between 0 and 1
    occasional: (probability, level, message, ...args) => {
      if (Math.random() < probability && shouldLog(module, level)) {
        console.log(`[${module} ${getLogLevelName(level)}] ${message}`, ...args);
      }
    }
  };
}

/**
 * Display current log level settings for all modules
 */
export function displayLogLevels() {
  console.log('=== Current Log Level Settings ===');
  
  console.log(`Global override: ${globalLogLevel === null ? 'Not set' : getLogLevelName(globalLogLevel)}`);
  
  console.log('\nModule settings:');
  Object.entries(moduleSettings).forEach(([module, level]) => {
    console.log(`- ${module}: ${getLogLevelName(level)} (${level})`);
  });
  
  console.log('\nUse window.gameLogger.setLogLevel(module, level) to change a specific module');
  console.log('Use window.gameLogger.setGlobalLogLevel(level) to override all modules');
  console.log('Use window.gameLogger.displayLogLevels() to show this information again');
  console.log('===============================');
}

// Initialize the logger
export function initLogger() {
  console.log('Logger system initialized');
  
  // Add logger to window for console access
  window.gameLogger = {
    setLogLevel,
    setGlobalLogLevel,
    LOG_LEVELS,
    displayLogLevels,
    currentLevel: null // Track the last set global level for UI toggle
  };
  
  return {
    setLogLevel,
    setGlobalLogLevel,
    LOG_LEVELS,
    displayLogLevels
  };
} 