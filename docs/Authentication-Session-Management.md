# Authentication and Session Management System Documentation

## Overview
The ROTMG-DEMO game currently implements a **simplified session management system** without traditional authentication mechanisms. This document covers the existing session handling, connection management, and identifies areas for future authentication enhancement.

## Current Architecture

### 1. Session Management (`/Server.js`)

#### **Simplified Session Model**
The current system uses a minimal session approach with automatic client identification:

```javascript
// Global session storage
const clients = new Map(); // clientId -> { socket, player, lastUpdate, mapId }
let nextClientId = 1;       // Simple incrementing ID generator

// WebSocket connection creates immediate session
wss.on('connection', (socket, req) => {
  // Automatic session creation - no authentication required
  const clientId = nextClientId++;
  
  // Store client session data
  clients.set(clientId, {
    socket,                    // WebSocket connection
    player: {                  // Player state
      id: clientId,
      x: spawnX,
      y: spawnY,
      inventory: new Array(20).fill(null),
      health: 100,
      worldId: useMapId,
      lastUpdate: Date.now()
    },
    mapId: useMapId,          // World assignment
    lastUpdate: Date.now()    // Session activity timestamp
  });
  
  // Send session acknowledgment
  sendToClient(socket, MessageType.HANDSHAKE_ACK, {
    clientId,                 // Session identifier
    timestamp: Date.now()
  });
});
```

#### **Session Lifecycle**
```javascript
// Session Creation Flow
Connection Request → Auto-ID Assignment → World Placement → State Initialization → Game Ready

// Session Termination (Currently Incomplete)
WebSocket Close → handleClientDisconnect() → [MISSING IMPLEMENTATION]
```

### 2. Connection Security Analysis

#### **Current Security Model**
The system operates with **no authentication barriers**:

```javascript
// No authentication checks
// No credential validation  
// No session tokens
// No rate limiting
// No IP-based restrictions

// Direct access model:
const clientId = nextClientId++; // Anyone can connect and get an ID
```

#### **Potential Security Vulnerabilities**

**Connection Flooding**:
```javascript
// Risk: Unlimited connections can overwhelm server
// Current: No connection limits or rate limiting
// Impact: DoS attacks possible

// Mitigation needed:
const MAX_CONNECTIONS_PER_IP = 5;
const CONNECTION_RATE_LIMIT = 10; // per minute
const ipConnections = new Map(); // IP -> connection count
```

**Client ID Prediction**:
```javascript
// Risk: Sequential IDs are predictable
let nextClientId = 1; // Easily guessable: 1, 2, 3, 4...

// Better approach would be:
import crypto from 'crypto';
const clientId = crypto.randomUUID(); // e.g., "123e4567-e89b-12d3-a456-426614174000"
```

### 3. Session State Management

#### **In-Memory Session Storage**

**Session Data Structure**:
```javascript
const sessionSchema = {
  // Connection Layer
  socket: WebSocket,           // Active connection
  clientId: String,            // Unique session identifier
  
  // Timing and Lifecycle
  connectionTime: Date,        // When session started
  lastUpdate: Date,            // Last activity timestamp
  lastHeartbeat: Date,         // Last ping/pong
  
  // Game State
  player: PlayerObject,        // Character data
  mapId: String,               // Current world
  
  // Session Metadata
  ipAddress: String,           // Client IP (for security)
  userAgent: String,           // Client browser/version
  connectionSource: String,    // How they connected
  
  // Permissions (future)
  role: 'player',              // player/admin/moderator
  permissions: [],             // Specific capabilities
  banned: false,               // Account status
  
  // Statistics
  totalPlayTime: Number,       // Cumulative session time
  messagesReceived: Number,    // Message count for rate limiting
  lastMessageTime: Date        // For rate limiting
};
```

#### **Session Cleanup Requirements**

**Missing Disconnect Handler**:
```javascript
// Current incomplete implementation
socket.on('close', () => {
  handleClientDisconnect(clientId); // Function undefined!
});

// Required implementation:
function handleClientDisconnect(clientId) {
  const session = clients.get(clientId);
  if (!session) return;
  
  // Session cleanup tasks:
  // 1. Remove from active sessions
  clients.delete(clientId);
  
  // 2. Clean up world-specific tracking
  updateWorldClientTracking(session.mapId, clientId, 'remove');
  
  // 3. Handle player resources (inventory drops, etc.)
  handlePlayerResourceCleanup(session.player);
  
  // 4. Notify other players
  broadcastPlayerDisconnection(clientId, session.mapId);
  
  // 5. Log session statistics
  logSessionEnd(clientId, session);
  
  // 6. Release any reserved resources
  releaseSessionResources(session);
}
```

### 4. Proposed Authentication Enhancement

#### **JWT-Based Authentication System**

**Enhanced Authentication Flow**:
```javascript
// Future authentication architecture
import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';

class AuthenticationManager {
  constructor() {
    this.JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
    this.TOKEN_EXPIRY = '24h';
    this.sessions = new Map(); // token -> session data
  }
  
  // Registration (future feature)
  async registerUser(username, password, email) {
    // Validate input
    if (!this.validateUsername(username) || !this.validatePassword(password)) {
      throw new Error('Invalid credentials format');
    }
    
    // Check if user exists
    if (await this.userExists(username)) {
      throw new Error('Username already taken');
    }
    
    // Hash password
    const passwordHash = await bcrypt.hash(password, 12);
    
    // Store user in database
    const userId = await this.createUser({
      username,
      passwordHash,
      email,
      createdAt: new Date(),
      lastLogin: null,
      role: 'player'
    });
    
    return { userId, username };
  }
  
  // Login authentication
  async authenticateUser(username, password) {
    const user = await this.findUserByUsername(username);
    if (!user) {
      throw new Error('Invalid credentials');
    }
    
    const passwordValid = await bcrypt.compare(password, user.passwordHash);
    if (!passwordValid) {
      throw new Error('Invalid credentials');
    }
    
    // Update last login
    await this.updateLastLogin(user.id);
    
    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id, 
        username: user.username,
        role: user.role 
      },
      this.JWT_SECRET,
      { expiresIn: this.TOKEN_EXPIRY }
    );
    
    return { token, user: this.sanitizeUser(user) };
  }
  
  // Token validation
  validateToken(token) {
    try {
      const decoded = jwt.verify(token, this.JWT_SECRET);
      return { valid: true, payload: decoded };
    } catch (error) {
      return { valid: false, error: error.message };
    }
  }
}
```

#### **Secure WebSocket Connection**

**Token-Based WebSocket Authentication**:
```javascript
// Enhanced connection handler with authentication
wss.on('connection', async (socket, req) => {
  try {
    // Extract authentication token from query params or headers
    const url = new URL(req.url, `http://${req.headers.host}`);
    const token = url.searchParams.get('token') || req.headers.authorization?.replace('Bearer ', '');
    
    if (!token) {
      socket.close(1008, 'Authentication required');
      return;
    }
    
    // Validate token
    const authResult = authManager.validateToken(token);
    if (!authResult.valid) {
      socket.close(1008, 'Invalid authentication token');
      return;
    }
    
    const { userId, username, role } = authResult.payload;
    
    // Check for duplicate sessions
    if (await this.hasActiveSession(userId)) {
      // Option 1: Reject new connection
      socket.close(1008, 'Already connected from another location');
      return;
      
      // Option 2: Terminate old session
      // await this.terminateExistingSession(userId);
    }
    
    // Create authenticated session
    const sessionId = crypto.randomUUID();
    const session = {
      sessionId,
      socket,
      userId,
      username,
      role,
      connectionTime: new Date(),
      lastActivity: new Date(),
      ipAddress: req.socket.remoteAddress,
      userAgent: req.headers['user-agent'],
      authenticated: true
    };
    
    // Store session
    authenticatedSessions.set(sessionId, session);
    userSessions.set(userId, sessionId); // Track by user ID
    
    // Continue with game initialization...
    initializeGameSession(session);
    
  } catch (error) {
    console.error('[AUTH] Connection authentication failed:', error);
    socket.close(1011, 'Authentication error');
  }
});
```

### 5. Rate Limiting and Security

#### **Connection Rate Limiting**

**IP-Based Rate Limiting**:
```javascript
class ConnectionRateLimiter {
  constructor() {
    this.connections = new Map(); // IP -> { count, windowStart, blocked }
    this.WINDOW_SIZE = 60000;     // 1 minute window
    this.MAX_CONNECTIONS = 10;    // Max connections per window
    this.BLOCK_DURATION = 300000; // 5 minute block
  }
  
  checkRateLimit(ipAddress) {
    const now = Date.now();
    const record = this.connections.get(ipAddress) || { 
      count: 0, 
      windowStart: now, 
      blocked: false,
      blockStart: null
    };
    
    // Check if currently blocked
    if (record.blocked && (now - record.blockStart) < this.BLOCK_DURATION) {
      return { allowed: false, reason: 'IP temporarily blocked' };
    }
    
    // Reset window if expired
    if ((now - record.windowStart) > this.WINDOW_SIZE) {
      record.count = 0;
      record.windowStart = now;
      record.blocked = false;
      record.blockStart = null;
    }
    
    record.count++;
    
    // Check if limit exceeded
    if (record.count > this.MAX_CONNECTIONS) {
      record.blocked = true;
      record.blockStart = now;
      this.connections.set(ipAddress, record);
      return { allowed: false, reason: 'Rate limit exceeded' };
    }
    
    this.connections.set(ipAddress, record);
    return { allowed: true };
  }
}

// Usage in connection handler
const rateLimiter = new ConnectionRateLimiter();

wss.on('connection', (socket, req) => {
  const clientIP = req.socket.remoteAddress;
  const rateCheck = rateLimiter.checkRateLimit(clientIP);
  
  if (!rateCheck.allowed) {
    socket.close(1008, rateCheck.reason);
    return;
  }
  
  // Continue with connection...
});
```

#### **Message Rate Limiting**

**Per-Session Message Limiting**:
```javascript
class MessageRateLimiter {
  constructor() {
    this.WINDOW_SIZE = 10000;      // 10 second window
    this.MAX_MESSAGES = 50;        // Max 50 messages per window
    this.MESSAGE_SIZE_LIMIT = 8192; // 8KB max message size
  }
  
  checkMessageRate(sessionId, messageSize) {
    const session = clients.get(sessionId);
    if (!session) return { allowed: false, reason: 'Invalid session' };
    
    const now = Date.now();
    
    // Initialize rate limiting data
    if (!session.rateLimit) {
      session.rateLimit = {
        messageCount: 0,
        windowStart: now,
        violations: 0
      };
    }
    
    const rl = session.rateLimit;
    
    // Check message size
    if (messageSize > this.MESSAGE_SIZE_LIMIT) {
      return { allowed: false, reason: 'Message too large' };
    }
    
    // Reset window if expired
    if ((now - rl.windowStart) > this.WINDOW_SIZE) {
      rl.messageCount = 0;
      rl.windowStart = now;
    }
    
    rl.messageCount++;
    
    // Check rate limit
    if (rl.messageCount > this.MAX_MESSAGES) {
      rl.violations++;
      
      // Escalating penalties
      if (rl.violations >= 3) {
        // Disconnect repeat offenders
        return { allowed: false, reason: 'Excessive rate limit violations', disconnect: true };
      }
      
      return { allowed: false, reason: 'Message rate limit exceeded' };
    }
    
    return { allowed: true };
  }
}
```

### 6. Session Persistence

#### **Database Session Storage**

**Persistent Session Management**:
```javascript
// Future enhancement: Database-backed sessions
class SessionManager {
  constructor(database) {
    this.db = database;
    this.activeSessions = new Map(); // In-memory cache
  }
  
  async createSession(userId, connectionData) {
    const session = {
      id: crypto.randomUUID(),
      userId,
      createdAt: new Date(),
      lastActivity: new Date(),
      ipAddress: connectionData.ip,
      userAgent: connectionData.userAgent,
      gameState: {
        mapId: 'default',
        playerData: this.initializePlayer(userId)
      }
    };
    
    // Store in database
    await this.db.sessions.insert(session);
    
    // Cache in memory
    this.activeSessions.set(session.id, session);
    
    return session;
  }
  
  async updateSessionActivity(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (session) {
      session.lastActivity = new Date();
      // Async database update (don't wait)
      this.db.sessions.update(sessionId, { lastActivity: session.lastActivity });
    }
  }
  
  async cleanupExpiredSessions() {
    const EXPIRY_TIME = 30 * 60 * 1000; // 30 minutes
    const cutoff = new Date(Date.now() - EXPIRY_TIME);
    
    // Find expired sessions
    const expired = await this.db.sessions.findWhere('lastActivity < ?', cutoff);
    
    for (const session of expired) {
      // Remove from memory
      this.activeSessions.delete(session.id);
      
      // Clean up game resources
      await this.cleanupGameResources(session);
    }
    
    // Remove from database
    await this.db.sessions.deleteWhere('lastActivity < ?', cutoff);
  }
}
```

### 7. Security Monitoring

#### **Connection Monitoring**

**Security Event Logging**:
```javascript
class SecurityMonitor {
  constructor() {
    this.suspiciousIPs = new Set();
    this.connectionAttempts = new Map(); // IP -> attempts[]
    this.maxAttemptsPerHour = 100;
  }
  
  logConnectionAttempt(ip, success, reason = null) {
    const now = Date.now();
    const hourAgo = now - (60 * 60 * 1000);
    
    // Clean old attempts
    if (!this.connectionAttempts.has(ip)) {
      this.connectionAttempts.set(ip, []);
    }
    
    const attempts = this.connectionAttempts.get(ip);
    this.connectionAttempts.set(ip, attempts.filter(time => time > hourAgo));
    
    // Log this attempt
    attempts.push(now);
    
    // Check for suspicious activity
    if (attempts.length > this.maxAttemptsPerHour) {
      this.flagSuspiciousIP(ip, 'Excessive connection attempts');
    }
    
    // Log to security system
    this.logSecurityEvent({
      type: 'connection_attempt',
      ip,
      success,
      reason,
      timestamp: now,
      attemptsInLastHour: attempts.length
    });
  }
  
  flagSuspiciousIP(ip, reason) {
    this.suspiciousIPs.add(ip);
    
    console.warn(`[SECURITY] Flagged IP ${ip}: ${reason}`);
    
    // Could integrate with firewall or IP blocking service
    // this.blockIP(ip);
  }
  
  logSecurityEvent(event) {
    // In production, send to security monitoring service
    console.log(`[SECURITY] ${JSON.stringify(event)}`);
  }
}
```

### 8. Current System Analysis

#### **Security Assessment**

**Current Vulnerabilities**:
```javascript
// CRITICAL: No authentication
// Anyone can connect and play

// HIGH: Predictable session IDs
let nextClientId = 1; // Easy to guess/enumerate

// HIGH: No rate limiting
// Vulnerable to DoS attacks

// MEDIUM: No session validation
// No checks for session integrity

// LOW: Basic connection handling
// Missing proper cleanup on disconnect
```

**Immediate Security Improvements Needed**:
```javascript
// 1. Implement secure session ID generation
const clientId = crypto.randomUUID();

// 2. Add basic rate limiting
const rateLimiter = new Map(); // IP -> { count, resetTime }

// 3. Complete disconnect handling
function handleClientDisconnect(clientId) {
  // Full implementation needed
}

// 4. Add connection logging
function logConnection(ip, userAgent, success) {
  console.log(`[CONN] ${ip} ${userAgent} ${success ? 'SUCCESS' : 'FAILED'}`);
}
```

### 9. Integration Points Summary

#### **System Dependencies**
- **NetworkManager**: Binary protocol for secure message encoding
- **Logger**: Centralized logging for security events and session activity
- **MapManager**: World assignment and spawn point generation
- **PlayerManager**: Character state initialization and management

#### **Performance Characteristics**
- **Session Storage**: In-memory Map structure for real-time access
- **Connection Overhead**: ~500 bytes per active session
- **Cleanup Frequency**: Manual cleanup on disconnect (when implemented)
- **Scalability**: Current design supports 1000+ concurrent sessions

#### **Data Flow**
```
Connection Request → Rate Limit Check → Session Creation → Game Initialization → Active Session
        ↓                  ↓                ↓                    ↓                  ↓
    WebSocket Open → IP Validation → ID Assignment → Player Spawn → Game Loop Integration
        ↓                  ↓                ↓                    ↓                  ↓
    Protocol Setup → Security Check → State Storage → World Placement → Message Handling
```

**Future Enhancement Roadmap**:
1. **Authentication Layer**: JWT-based user authentication
2. **Session Persistence**: Database-backed session storage
3. **Rate Limiting**: Comprehensive DoS protection
4. **Security Monitoring**: Real-time threat detection
5. **User Management**: Registration, roles, and permissions

The current system provides basic session functionality but lacks essential security features for production deployment. The simplified approach is suitable for development and testing but requires significant enhancement for public deployment.