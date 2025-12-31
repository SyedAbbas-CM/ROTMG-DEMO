/**
 * Console log filtering for cleaner debugging
 * Suppresses most logs but ALWAYS shows:
 * - [BINARY] - Binary protocol status
 * - [NETWORK] - Network connection status
 * - Errors and warnings
 */

// Store original console methods
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleDebug = console.debug;
const originalConsoleInfo = console.info;

// Flag to control logging
const FRONTEND_LOGGING_ENABLED = false;

// Keywords that should ALWAYS be logged even when logging is disabled
const ALWAYS_LOG_KEYWORDS = ['[BINARY]', '[NETWORK]', '[FALLBACK]', '[WebTransport]', 'getNetworkStatus'];

// Check if a message should always be logged
function shouldAlwaysLog(args) {
  if (!args || args.length === 0) return false;
  const firstArg = String(args[0]);
  return ALWAYS_LOG_KEYWORDS.some(keyword => firstArg.includes(keyword));
}

// Override console.log to filter
if (!FRONTEND_LOGGING_ENABLED) {
  console.log = function(...args) {
    if (shouldAlwaysLog(args)) {
      originalConsoleLog.apply(console, args);
    }
    // Otherwise silently ignore
  };

  console.debug = function(...args) {
    if (shouldAlwaysLog(args)) {
      originalConsoleDebug.apply(console, args);
    }
  };

  console.info = function(...args) {
    if (shouldAlwaysLog(args)) {
      originalConsoleInfo.apply(console, args);
    }
  };

  // Also filter console.warn to reduce noise but keep important ones
  console.warn = function(...args) {
    if (shouldAlwaysLog(args)) {
      originalConsoleWarn.apply(console, args);
    }
    // Keep real warnings about errors
    const firstArg = String(args[0] || '');
    if (firstArg.includes('Error') || firstArg.includes('error') || firstArg.includes('failed') || firstArg.includes('Failed')) {
      originalConsoleWarn.apply(console, args);
    }
  };
}

// Keep console.error ALWAYS enabled

// Export functions for runtime control
export function enableConsoleLogs() {
  console.log = originalConsoleLog;
  console.warn = originalConsoleWarn;
  console.debug = originalConsoleDebug;
  console.info = originalConsoleInfo;
  originalConsoleLog('%c[Logger] All console logs enabled', 'color: #0f0');
}

export function disableConsoleLogs() {
  console.log = function(...args) {
    if (shouldAlwaysLog(args)) originalConsoleLog.apply(console, args);
  };
  console.warn = function(...args) {
    if (shouldAlwaysLog(args)) originalConsoleWarn.apply(console, args);
  };
  console.debug = function() {};
  console.info = function() {};
  originalConsoleLog('%c[Logger] Console logs filtered (only BINARY/NETWORK shown)', 'color: #888');
}

// Expose to window for easy control
if (typeof window !== 'undefined') {
  window.enableLogs = enableConsoleLogs;
  window.disableLogs = disableConsoleLogs;
}

export { FRONTEND_LOGGING_ENABLED };
