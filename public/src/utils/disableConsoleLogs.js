/**
 * Disable console logs in production/when debugging is not needed
 * This helps reduce logs.txt spam while keeping error logging
 */

// Store original console methods
const originalConsoleLog = console.log;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;

// Flag to control logging (set to false to disable frontend logs)
const FRONTEND_LOGGING_ENABLED = false;

// Override console.log to no-op when disabled
if (!FRONTEND_LOGGING_ENABLED) {
  console.log = function() {
    // Silently ignore all console.log calls
  };

  console.debug = function() {
    // Silently ignore all console.debug calls
  };

  console.info = function() {
    // Silently ignore all console.info calls
  };
}

// Keep console.warn and console.error enabled always
// They are important for debugging real issues

// Export the flag and restore function for potential re-enabling
export function enableConsoleLogs() {
  console.log = originalConsoleLog;
  console.debug = originalConsoleDebug;
  console.info = originalConsoleInfo;
  console.log('Console logs re-enabled');
}

export function disableConsoleLogs() {
  console.log = function() {};
  console.debug = function() {};
  console.info = function() {};
}

export { FRONTEND_LOGGING_ENABLED };
