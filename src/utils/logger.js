// src/utils/logger.js
// Logger with file output for different categories
// Files are written to /logs directory:
//   - server.log   : Server startup, shutdown, general
//   - game.log     : Entities, bullets, enemies, bosses
//   - players.log  : Player joins, leaves, deaths
//   - network.log  : Connections, packets
//   - anticheat.log: Suspicious activity
//   - errors.log   : All errors combined

import fs from 'fs';
import path from 'path';

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

// File logging setup
const LOG_DIR = path.join(process.cwd(), 'logs');
const streams = {};
const LOG_FILES = {
  server: 'server.log',
  game: 'game.log',
  players: 'players.log',
  network: 'network.log',
  anticheat: 'anticheat.log',
  errors: 'errors.log'
};

// Map module names to log categories
const MODULE_TO_CATEGORY = {
  // Server category
  'Server': 'server',
  'WebSocket': 'server',
  'WebTransport': 'server',
  'Database': 'server',
  'Maps': 'server',
  'Config': 'server',
  // Game category
  'EnemyManager': 'game',
  'BulletManager': 'game',
  'Enemy': 'game',
  'Bullet': 'game',
  'Boss': 'game',
  'AIPatternBoss': 'game',
  'PatternPlayer': 'game',
  'Behavior': 'game',
  'Collision': 'game',
  'Game': 'game',
  // Player category
  'Player': 'players',
  'PlayerManager': 'players',
  'Inventory': 'players',
  'Character': 'players',
  // Network category
  'Network': 'network',
  'Binary': 'network',
  'Protocol': 'network',
  'Packet': 'network',
  // Anticheat category
  'Anticheat': 'anticheat',
  'Validation': 'anticheat',
  'Security': 'anticheat'
};

function getCategory(module) {
  // Check exact match first
  if (MODULE_TO_CATEGORY[module]) return MODULE_TO_CATEGORY[module];
  // Check if module contains any known keyword
  for (const [key, cat] of Object.entries(MODULE_TO_CATEGORY)) {
    if (module.includes(key)) return cat;
  }
  return 'server'; // Default
}

function initLogDir() {
  try {
    if (!fs.existsSync(LOG_DIR)) {
      fs.mkdirSync(LOG_DIR, { recursive: true });
    }
    for (const [cat, file] of Object.entries(LOG_FILES)) {
      const filePath = path.join(LOG_DIR, file);
      streams[cat] = fs.createWriteStream(filePath, { flags: 'a' });
    }
    return true;
  } catch (e) {
    console.error('Failed to init log directory:', e.message);
    return false;
  }
}

// Initialize on first import
const logsInitialized = initLogDir();

function timestamp() {
  return new Date().toISOString();
}

function writeToFile(category, level, module, msg, args) {
  if (!logsInitialized) return;

  const stream = streams[category];
  if (!stream) return;

  let line = `[${timestamp()}] [${level}] [${module}] ${msg}`;
  if (args && args.length > 0) {
    const argsStr = args.map(a => {
      if (a === undefined) return 'undefined';
      if (a === null) return 'null';
      if (typeof a === 'object') {
        try { return JSON.stringify(a); }
        catch { return '[object]'; }
      }
      return String(a);
    }).join(' ');
    line += ' ' + argsStr;
  }
  line += '\n';

  stream.write(line);

  // Also write errors to errors.log
  if (level === 'ERROR' && category !== 'errors') {
    streams.errors?.write(`[${category.toUpperCase()}] ${line}`);
  }
}

export const logger = {
  error(module, msg, ...args) {
    if (should(LEVELS.ERROR)) console.error(`[${module}] ${msg}`, ...args);
    writeToFile(getCategory(module), 'ERROR', module, msg, args);
    writeToFile('errors', 'ERROR', module, msg, args);
  },
  warn(module, msg, ...args) {
    if (should(LEVELS.WARN)) console.warn(`[${module}] ${msg}`, ...args);
    writeToFile(getCategory(module), 'WARN', module, msg, args);
  },
  info(module, msg, ...args) {
    if (should(LEVELS.INFO)) console.log(`[${module}] ${msg}`, ...args);
    writeToFile(getCategory(module), 'INFO', module, msg, args);
  },
  debug(module, msg, ...args) {
    if (should(LEVELS.DEBUG)) console.log(`[${module} DEBUG] ${msg}`, ...args);
    writeToFile(getCategory(module), 'DEBUG', module, msg, args);
  },
  verbose(module, msg, ...args) {
    if (should(LEVELS.VERBOSE)) console.log(`[${module} VERBOSE] ${msg}`, ...args);
    writeToFile(getCategory(module), 'VERBOSE', module, msg, args);
  },

  // Direct category logging (bypasses module mapping)
  server(msg, ...args) {
    console.log(`[Server] ${msg}`, ...args);
    writeToFile('server', 'INFO', 'Server', msg, args);
  },
  game(msg, ...args) {
    writeToFile('game', 'INFO', 'Game', msg, args);
  },
  player(msg, ...args) {
    writeToFile('players', 'INFO', 'Player', msg, args);
  },
  network(msg, ...args) {
    writeToFile('network', 'INFO', 'Network', msg, args);
  },
  anticheat(msg, ...args) {
    console.warn(`[Anticheat] ${msg}`, ...args);
    writeToFile('anticheat', 'WARN', 'Anticheat', msg, args);
  },

  // Player-specific events
  playerJoin(id, name, data) {
    const msg = `${name} (${id}) joined`;
    console.log(`[Player] ${msg}`);
    writeToFile('players', 'JOIN', 'Player', msg, data ? [data] : []);
  },
  playerLeave(id, name, reason = 'disconnected') {
    const msg = `${name} (${id}) left: ${reason}`;
    console.log(`[Player] ${msg}`);
    writeToFile('players', 'LEAVE', 'Player', msg, []);
  },
  playerDeath(id, name, cause = 'unknown') {
    const msg = `${name} (${id}) died: ${cause}`;
    console.log(`[Player] ${msg}`);
    writeToFile('players', 'DEATH', 'Player', msg, []);
  },

  // Anticheat events
  flag(playerId, violation, data) {
    const msg = `FLAGGED player ${playerId}: ${violation}`;
    console.error(`[Anticheat] ${msg}`, data || '');
    writeToFile('anticheat', 'FLAG', 'Anticheat', msg, data ? [data] : []);
  },

  // Rotate logs (call on server start or periodically)
  rotate() {
    console.log('[Logger] Rotating logs...');
    for (const stream of Object.values(streams)) {
      stream?.end();
    }
    for (const [cat, file] of Object.entries(LOG_FILES)) {
      const filePath = path.join(LOG_DIR, file);
      const oldPath = filePath.replace('.log', '.old.log');
      try {
        if (fs.existsSync(filePath)) {
          if (fs.existsSync(oldPath)) fs.unlinkSync(oldPath);
          fs.renameSync(filePath, oldPath);
        }
        streams[cat] = fs.createWriteStream(filePath, { flags: 'a' });
      } catch (e) {
        console.error(`Failed to rotate ${file}:`, e.message);
      }
    }
    console.log('[Logger] Log rotation complete');
  },

  // Flush all streams
  flush() {
    for (const stream of Object.values(streams)) {
      // Node streams auto-flush, but we can force it
    }
  },

  // Close streams on shutdown
  close() {
    for (const stream of Object.values(streams)) {
      stream?.end();
    }
  }
};

export { LEVELS as LOG_LEVELS }; 