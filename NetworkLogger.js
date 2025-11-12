// NetworkLogger.js
// Comprehensive network logging and diagnostics utility

class NetworkLogger {
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.logInterval = options.logInterval || 5000; // Log stats every 5 seconds
    this.verboseMode = options.verbose || false;

    // Statistics
    this.stats = {
      connections: {
        total: 0,
        active: 0,
        peak: 0,
        failed: 0,
        disconnects: 0
      },
      messages: {
        sent: 0,
        received: 0,
        byType: new Map(),
        errors: 0
      },
      bandwidth: {
        sent: 0,
        received: 0,
        sentPerSecond: 0,
        receivedPerSecond: 0
      },
      latency: {
        samples: [],
        min: Infinity,
        max: 0,
        avg: 0,
        p50: 0,
        p95: 0,
        p99: 0
      },
      packets: {
        tcp: 0,
        udp: 0,
        dropped: 0,
        retransmits: 0
      }
    };

    // Active connections tracking
    this.connections = new Map();

    // Message queue for rate limiting
    this.messageQueue = [];

    // Start periodic logging
    if (this.enabled) {
      this.startPeriodicLogging();
    }

    console.log('[NetworkLogger] Initialized with options:', {
      enabled: this.enabled,
      logInterval: this.logInterval,
      verbose: this.verboseMode
    });
  }

  // Connection Events
  onConnectionOpen(socketId, remoteAddress, protocol = 'ws') {
    this.stats.connections.total++;
    this.stats.connections.active++;
    this.stats.connections.peak = Math.max(this.stats.connections.peak, this.stats.connections.active);

    const connInfo = {
      id: socketId,
      remoteAddress,
      protocol,
      connectedAt: Date.now(),
      messagesSent: 0,
      messagesReceived: 0,
      bytesSent: 0,
      bytesReceived: 0,
      lastActivity: Date.now(),
      latencySamples: []
    };

    this.connections.set(socketId, connInfo);

    if (this.verboseMode) {
      console.log(`[NET:CONNECT] 🟢 Client ${socketId} connected from ${remoteAddress} (${protocol.toUpperCase()})`);
      console.log(`[NET:STATS] Active: ${this.stats.connections.active}, Total: ${this.stats.connections.total}, Peak: ${this.stats.connections.peak}`);
    }
  }

  onConnectionClose(socketId, code, reason) {
    const conn = this.connections.get(socketId);
    if (!conn) return;

    this.stats.connections.active--;
    this.stats.connections.disconnects++;

    const duration = Date.now() - conn.connectedAt;
    const durationSec = (duration / 1000).toFixed(2);

    if (this.verboseMode) {
      console.log(`[NET:DISCONNECT] 🔴 Client ${socketId} disconnected`);
      console.log(`  Duration: ${durationSec}s`);
      console.log(`  Messages: ${conn.messagesReceived} in, ${conn.messagesSent} out`);
      console.log(`  Bandwidth: ${this.formatBytes(conn.bytesReceived)} in, ${this.formatBytes(conn.bytesSent)} out`);
      if (code) console.log(`  Code: ${code}, Reason: ${reason || 'none'}`);
    }

    this.connections.delete(socketId);
  }

  onConnectionError(socketId, error) {
    this.stats.connections.failed++;

    if (this.verboseMode) {
      console.error(`[NET:ERROR] ❌ Client ${socketId} error:`, error.message);
    }
  }

  // Message Events
  onMessageReceived(socketId, messageType, size) {
    this.stats.messages.received++;
    this.stats.bandwidth.received += size;

    // Track by message type
    const typeName = this.getMessageTypeName(messageType);
    const typeCount = this.stats.messages.byType.get(typeName) || { sent: 0, received: 0 };
    typeCount.received++;
    this.stats.messages.byType.set(typeName, typeCount);

    // Update connection stats
    const conn = this.connections.get(socketId);
    if (conn) {
      conn.messagesReceived++;
      conn.bytesReceived += size;
      conn.lastActivity = Date.now();
    }

    if (this.verboseMode) {
      console.log(`[NET:RX] 📥 Client ${socketId} -> ${typeName} (${size} bytes)`);
    }
  }

  onMessageSent(socketId, messageType, size) {
    this.stats.messages.sent++;
    this.stats.bandwidth.sent += size;

    // Track by message type
    const typeName = this.getMessageTypeName(messageType);
    const typeCount = this.stats.messages.byType.get(typeName) || { sent: 0, received: 0 };
    typeCount.sent++;
    this.stats.messages.byType.set(typeName, typeCount);

    // Update connection stats
    const conn = this.connections.get(socketId);
    if (conn) {
      conn.messagesSent++;
      conn.bytesSent += size;
      conn.lastActivity = Date.now();
    }

    if (this.verboseMode) {
      console.log(`[NET:TX] 📤 Client ${socketId} <- ${typeName} (${size} bytes)`);
    }
  }

  onMessageError(socketId, error) {
    this.stats.messages.errors++;

    if (this.verboseMode) {
      console.error(`[NET:MSG_ERROR] ⚠️ Client ${socketId} message error:`, error.message);
    }
  }

  // Latency Tracking
  recordLatency(socketId, latencyMs) {
    this.stats.latency.samples.push(latencyMs);
    this.stats.latency.min = Math.min(this.stats.latency.min, latencyMs);
    this.stats.latency.max = Math.max(this.stats.latency.max, latencyMs);

    // Keep last 1000 samples
    if (this.stats.latency.samples.length > 1000) {
      this.stats.latency.samples.shift();
    }

    // Calculate percentiles
    this.calculateLatencyStats();

    const conn = this.connections.get(socketId);
    if (conn) {
      conn.latencySamples.push(latencyMs);
      if (conn.latencySamples.length > 100) {
        conn.latencySamples.shift();
      }
    }

    if (this.verboseMode) {
      console.log(`[NET:LATENCY] ⏱️ Client ${socketId} RTT: ${latencyMs}ms`);
    }
  }

  calculateLatencyStats() {
    if (this.stats.latency.samples.length === 0) return;

    const sorted = [...this.stats.latency.samples].sort((a, b) => a - b);
    const len = sorted.length;

    this.stats.latency.avg = sorted.reduce((a, b) => a + b, 0) / len;
    this.stats.latency.p50 = sorted[Math.floor(len * 0.5)];
    this.stats.latency.p95 = sorted[Math.floor(len * 0.95)];
    this.stats.latency.p99 = sorted[Math.floor(len * 0.99)];
  }

  // Packet Tracking (simulated for WebSocket)
  onTCPPacket(direction, size) {
    this.stats.packets.tcp++;
    if (direction === 'in') {
      this.stats.bandwidth.received += size;
    } else {
      this.stats.bandwidth.sent += size;
    }
  }

  onUDPPacket(direction, size) {
    this.stats.packets.udp++;
    if (direction === 'in') {
      this.stats.bandwidth.received += size;
    } else {
      this.stats.bandwidth.sent += size;
    }
  }

  onPacketDropped() {
    this.stats.packets.dropped++;
  }

  onPacketRetransmit() {
    this.stats.packets.retransmits++;
  }

  // Periodic Logging
  startPeriodicLogging() {
    this.logTimer = setInterval(() => {
      this.logStats();
    }, this.logInterval);
  }

  stopPeriodicLogging() {
    if (this.logTimer) {
      clearInterval(this.logTimer);
    }
  }

  logStats() {
    console.log('\n========== NETWORK STATISTICS ==========');
    console.log(`Timestamp: ${new Date().toISOString()}`);

    // Connections
    console.log('\n📡 CONNECTIONS:');
    console.log(`  Active: ${this.stats.connections.active}`);
    console.log(`  Total: ${this.stats.connections.total}`);
    console.log(`  Peak: ${this.stats.connections.peak}`);
    console.log(`  Failed: ${this.stats.connections.failed}`);
    console.log(`  Disconnects: ${this.stats.connections.disconnects}`);

    // Messages
    console.log('\n💬 MESSAGES:');
    console.log(`  Sent: ${this.stats.messages.sent}`);
    console.log(`  Received: ${this.stats.messages.received}`);
    console.log(`  Errors: ${this.stats.messages.errors}`);

    // Message types
    if (this.stats.messages.byType.size > 0) {
      console.log('\n📊 MESSAGE TYPES:');
      const sorted = [...this.stats.messages.byType.entries()]
        .sort((a, b) => (b[1].sent + b[1].received) - (a[1].sent + a[1].received));

      sorted.slice(0, 10).forEach(([type, counts]) => {
        console.log(`  ${type.padEnd(20)}: ${counts.received} in, ${counts.sent} out`);
      });
    }

    // Bandwidth
    console.log('\n📈 BANDWIDTH:');
    console.log(`  Sent: ${this.formatBytes(this.stats.bandwidth.sent)}`);
    console.log(`  Received: ${this.formatBytes(this.stats.bandwidth.received)}`);

    // Latency
    if (this.stats.latency.samples.length > 0) {
      console.log('\n⏱️  LATENCY:');
      console.log(`  Min: ${this.stats.latency.min.toFixed(2)}ms`);
      console.log(`  Max: ${this.stats.latency.max.toFixed(2)}ms`);
      console.log(`  Avg: ${this.stats.latency.avg.toFixed(2)}ms`);
      console.log(`  P50: ${this.stats.latency.p50.toFixed(2)}ms`);
      console.log(`  P95: ${this.stats.latency.p95.toFixed(2)}ms`);
      console.log(`  P99: ${this.stats.latency.p99.toFixed(2)}ms`);
      console.log(`  Samples: ${this.stats.latency.samples.length}`);
    }

    // Packets
    console.log('\n📦 PACKETS:');
    console.log(`  TCP: ${this.stats.packets.tcp}`);
    console.log(`  UDP: ${this.stats.packets.udp}`);
    console.log(`  Dropped: ${this.stats.packets.dropped}`);
    console.log(`  Retransmits: ${this.stats.packets.retransmits}`);

    // Active connections detail
    if (this.connections.size > 0) {
      console.log('\n👥 ACTIVE CONNECTIONS:');
      let connIndex = 0;
      for (const [id, conn] of this.connections.entries()) {
        const uptime = ((Date.now() - conn.connectedAt) / 1000).toFixed(0);
        const idle = ((Date.now() - conn.lastActivity) / 1000).toFixed(0);
        const avgLatency = conn.latencySamples.length > 0
          ? (conn.latencySamples.reduce((a, b) => a + b, 0) / conn.latencySamples.length).toFixed(2)
          : 'N/A';

        console.log(`  Client ${id}:`);
        console.log(`    Address: ${conn.remoteAddress}`);
        console.log(`    Uptime: ${uptime}s, Idle: ${idle}s`);
        console.log(`    Messages: ${conn.messagesReceived} in, ${conn.messagesSent} out`);
        console.log(`    Bandwidth: ${this.formatBytes(conn.bytesReceived)} in, ${this.formatBytes(conn.bytesSent)} out`);
        console.log(`    Avg Latency: ${avgLatency}ms`);

        connIndex++;
        if (connIndex >= 5) {
          console.log(`  ... and ${this.connections.size - 5} more`);
          break;
        }
      }
    }

    console.log('\n========================================\n');
  }

  // Utility Methods
  getMessageTypeName(messageType) {
    // This should match the MessageType enum from messages.js
    const typeNames = {
      0: 'HEARTBEAT',
      1: 'HANDSHAKE',
      2: 'HANDSHAKE_ACK',
      3: 'PING',
      4: 'PONG',
      10: 'PLAYER_JOIN',
      11: 'PLAYER_LEAVE',
      12: 'PLAYER_UPDATE',
      13: 'PLAYER_LIST',
      20: 'ENEMY_LIST',
      21: 'ENEMY_UPDATE',
      22: 'ENEMY_DEATH',
      30: 'BULLET_CREATE',
      31: 'BULLET_LIST',
      32: 'BULLET_REMOVE',
      33: 'BAG_LIST',
      34: 'PICKUP_ITEM',
      35: 'INVENTORY_UPDATE',
      36: 'BAG_REMOVE',
      37: 'PICKUP_DENIED',
      38: 'MOVE_ITEM',
      39: 'MOVE_DENIED',
      40: 'COLLISION',
      41: 'COLLISION_RESULT',
      50: 'MAP_INFO',
      51: 'CHUNK_REQUEST',
      52: 'CHUNK_DATA',
      53: 'CHUNK_NOT_FOUND',
      54: 'PORTAL_ENTER',
      55: 'WORLD_SWITCH',
      60: 'WORLD_UPDATE',
      70: 'MAP_REQUEST',
      80: 'PLAYER_LIST_REQUEST',
      89: 'PLAYER_TEXT',
      90: 'CHAT_MESSAGE',
      91: 'SPEECH',
      100: 'UNIT_SPAWN',
      101: 'UNIT_COMMAND',
      102: 'UNIT_SELECT'
    };

    return typeNames[messageType] || `UNKNOWN(${messageType})`;
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
  }

  // Export current stats
  getStats() {
    return {
      ...this.stats,
      connections: {
        ...this.stats.connections,
        list: Array.from(this.connections.values())
      }
    };
  }

  // Reset stats
  reset() {
    this.stats.messages.sent = 0;
    this.stats.messages.received = 0;
    this.stats.messages.byType.clear();
    this.stats.messages.errors = 0;
    this.stats.bandwidth.sent = 0;
    this.stats.bandwidth.received = 0;
    this.stats.latency.samples = [];
    this.stats.packets.tcp = 0;
    this.stats.packets.udp = 0;
    this.stats.packets.dropped = 0;
    this.stats.packets.retransmits = 0;

    console.log('[NetworkLogger] Stats reset');
  }

  // Enable/disable logging
  setVerbose(enabled) {
    this.verboseMode = enabled;
    console.log(`[NetworkLogger] Verbose mode ${enabled ? 'enabled' : 'disabled'}`);
  }
}

export default NetworkLogger;
