// src/utils/logger.js
// Simple logger utility for Node environment.
// Usage:
//   import { logger } from '../utils/logger.js';
//   logger.info('Subsystem','Message')
// LOG_LEVEL env var controls verbosity (NONE|ERROR|WARN|INFO|DEBUG|VERBOSE)

const LEVELS = {
  NONE: 0,
  ERROR: 1,
  WARN: 2,
  INFO: 3,
  DEBUG: 4,
  VERBOSE: 5
};

function parseLevel(str){
  if(!str) return LEVELS.INFO;
  const key = str.toUpperCase();
  return LEVELS[key] ?? LEVELS.INFO;
}

const GLOBAL_LEVEL = parseLevel(process.env.LOG_LEVEL || (process.env.NODE_ENV==='production' ? 'WARN' : 'INFO'));

function should(level){
  return level <= GLOBAL_LEVEL;
}

export const logger = {
  error(module,msg,...args){ if(should(LEVELS.ERROR)) console.error(`[${module}] ${msg}`,...args); },
  warn(module,msg,...args){ if(should(LEVELS.WARN)) console.warn(`[${module}] ${msg}`,...args); },
  info(module,msg,...args){ if(should(LEVELS.INFO)) console.log(`[${module}] ${msg}`,...args); },
  debug(module,msg,...args){ if(should(LEVELS.DEBUG)) console.log(`[${module} DEBUG] ${msg}`,...args); },
  verbose(module,msg,...args){ if(should(LEVELS.VERBOSE)) console.log(`[${module} VERBOSE] ${msg}`,...args); }
};

export { LEVELS as LOG_LEVELS }; 