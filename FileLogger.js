// FileLogger.js
// Persistent file-based logging system with log rotation

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class FileLogger {
  constructor(options = {}) {
    this.logsDir = options.logsDir || path.join(__dirname, 'logs');
    this.maxFileSize = options.maxFileSize || 10 * 1024 * 1024; // 10MB default
    this.maxFiles = options.maxFiles || 10; // Keep last 10 log files
    this.enabled = options.enabled !== false;

    // Create logs directory if it doesn't exist
    if (this.enabled && !fs.existsSync(this.logsDir)) {
      fs.mkdirSync(this.logsDir, { recursive: true });
    }

    // Current log file
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
    this.currentLogFile = path.join(this.logsDir, `server-${timestamp}.log`);
    this.currentErrorFile = path.join(this.logsDir, `error-${timestamp}.log`);
    this.currentNetworkFile = path.join(this.logsDir, `network-${timestamp}.log`);
    this.currentCollisionFile = path.join(this.logsDir, `collision-${timestamp}.log`);

    // Log streams
    this.logStream = null;
    this.errorStream = null;
    this.networkStream = null;
    this.collisionStream = null;

    if (this.enabled) {
      this.initStreams();
      this.rotateOldLogs();
      console.log('[FileLogger] Initialized');
      console.log(`[FileLogger] Log files:`);
      console.log(`  - General:   ${this.currentLogFile}`);
      console.log(`  - Errors:    ${this.currentErrorFile}`);
      console.log(`  - Network:   ${this.currentNetworkFile}`);
      console.log(`  - Collision: ${this.currentCollisionFile}`);
    }
  }

  initStreams() {
    this.logStream = fs.createWriteStream(this.currentLogFile, { flags: 'a' });
    this.errorStream = fs.createWriteStream(this.currentErrorFile, { flags: 'a' });
    this.networkStream = fs.createWriteStream(this.currentNetworkFile, { flags: 'a' });
    this.collisionStream = fs.createWriteStream(this.currentCollisionFile, { flags: 'a' });

    // Handle stream errors
    this.logStream.on('error', (err) => console.error('[FileLogger] Log stream error:', err));
    this.errorStream.on('error', (err) => console.error('[FileLogger] Error stream error:', err));
    this.networkStream.on('error', (err) => console.error('[FileLogger] Network stream error:', err));
    this.collisionStream.on('error', (err) => console.error('[FileLogger] Collision stream error:', err));
  }

  rotateOldLogs() {
    // Get all log files
    const files = fs.readdirSync(this.logsDir)
      .filter(f => f.endsWith('.log'))
      .map(f => ({
        name: f,
        path: path.join(this.logsDir, f),
        time: fs.statSync(path.join(this.logsDir, f)).mtime.getTime()
      }))
      .sort((a, b) => b.time - a.time);

    // Delete old logs beyond maxFiles
    if (files.length > this.maxFiles * 4) { // 4 types of logs (server, error, network, collision)
      files.slice(this.maxFiles * 4).forEach(file => {
        try {
          fs.unlinkSync(file.path);
          console.log(`[FileLogger] Deleted old log: ${file.name}`);
        } catch (err) {
          console.error(`[FileLogger] Failed to delete ${file.name}:`, err);
        }
      });
    }
  }

  checkRotation() {
    if (!this.enabled) return;

    try {
      const stats = fs.statSync(this.currentLogFile);
      if (stats.size > this.maxFileSize) {
        console.log('[FileLogger] Rotating log files (size limit reached)');
        this.closeStreams();

        // Rename current files with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        fs.renameSync(this.currentLogFile, this.currentLogFile.replace('.log', `-${timestamp}.log`));
        fs.renameSync(this.currentErrorFile, this.currentErrorFile.replace('.log', `-${timestamp}.log`));
        fs.renameSync(this.currentNetworkFile, this.currentNetworkFile.replace('.log', `-${timestamp}.log`));

        // Create new files
        this.initStreams();
        this.rotateOldLogs();
      }
    } catch (err) {
      // File doesn't exist yet, ignore
    }
  }

  formatTimestamp() {
    return new Date().toISOString();
  }

  log(level, category, message, data = null) {
    if (!this.enabled) return;

    const timestamp = this.formatTimestamp();
    let logLine = `[${timestamp}] [${level}] [${category}] ${message}`;

    if (data) {
      logLine += ` ${JSON.stringify(data)}`;
    }

    logLine += '\n';

    try {
      // Write to general log
      this.logStream.write(logLine);

      // Write to error log if it's an error
      if (level === 'ERROR' || level === 'FATAL') {
        this.errorStream.write(logLine);
      }

      // Check if rotation is needed
      this.checkRotation();
    } catch (err) {
      console.error('[FileLogger] Failed to write log:', err);
    }
  }

  logNetwork(event, data) {
    if (!this.enabled) return;

    const timestamp = this.formatTimestamp();
    const logLine = `[${timestamp}] [NETWORK] [${event}] ${JSON.stringify(data)}\n`;

    try {
      this.networkStream.write(logLine);
    } catch (err) {
      console.error('[FileLogger] Failed to write network log:', err);
    }
  }

  logCollision(event, data) {
    if (!this.enabled) return;

    const timestamp = this.formatTimestamp();
    const logLine = `[${timestamp}] [COLLISION] [${event}] ${JSON.stringify(data)}\n`;

    try {
      this.collisionStream.write(logLine);
    } catch (err) {
      console.error('[FileLogger] Failed to write collision log:', err);
    }
  }

  // Collision-specific convenience methods
  bulletHit(bulletId, targetType, targetId, damage, position) {
    this.logCollision('BULLET_HIT', {
      bulletId,
      targetType, // 'enemy' or 'player'
      targetId,
      damage,
      position,
      timestamp: Date.now()
    });
  }

  contactDamage(enemyId, playerId, damage, knockback, positions) {
    this.logCollision('CONTACT_DAMAGE', {
      enemyId,
      playerId,
      damage,
      knockback,
      positions,
      timestamp: Date.now()
    });
  }

  collisionCheck(checkType, entityCount, playerCount) {
    this.logCollision('CHECK', {
      checkType,
      entityCount,
      playerCount,
      timestamp: Date.now()
    });
  }

  collisionValidation(result, details) {
    this.logCollision('VALIDATION', {
      result, // 'valid', 'suspicious', 'rejected'
      ...details,
      timestamp: Date.now()
    });
  }

  // Convenience methods
  info(category, message, data) {
    this.log('INFO', category, message, data);
  }

  warn(category, message, data) {
    this.log('WARN', category, message, data);
  }

  error(category, message, data) {
    this.log('ERROR', category, message, data);
  }

  debug(category, message, data) {
    this.log('DEBUG', category, message, data);
  }

  fatal(category, message, data) {
    this.log('FATAL', category, message, data);
  }

  // Network-specific logging
  connection(clientId, remoteAddress, action) {
    this.logNetwork('CONNECTION', {
      action,
      clientId,
      remoteAddress,
      timestamp: Date.now()
    });
  }

  message(clientId, direction, messageType, size) {
    this.logNetwork('MESSAGE', {
      clientId,
      direction, // 'in' or 'out'
      messageType,
      size,
      timestamp: Date.now()
    });
  }

  latency(clientId, rtt) {
    this.logNetwork('LATENCY', {
      clientId,
      rtt,
      timestamp: Date.now()
    });
  }

  stats(statistics) {
    this.logNetwork('STATS', {
      ...statistics,
      timestamp: Date.now()
    });
  }

  closeStreams() {
    if (this.logStream) this.logStream.end();
    if (this.errorStream) this.errorStream.end();
    if (this.networkStream) this.networkStream.end();
    if (this.collisionStream) this.collisionStream.end();
  }

  close() {
    console.log('[FileLogger] Closing log streams');
    this.closeStreams();
  }
}

export default FileLogger;
