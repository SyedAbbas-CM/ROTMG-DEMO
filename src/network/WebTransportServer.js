// src/network/WebTransportServer.js
// WebTransport/QUIC server for UDP-like low-latency game communication
//
// Requirements:
// - Node.js 21.6.1+ (for WebTransport support in @fails-components)
// - TLS certificate (WebTransport requires HTTPS)
// - npm install @fails-components/webtransport @fails-components/webtransport-transport-http3-quiche
//
// Environment variables:
// - WEBTRANSPORT_ENABLED=true (enable WebTransport server)
// - WEBTRANSPORT_PORT=4433 (UDP port for QUIC)
// - WEBTRANSPORT_CERT=/path/to/cert.pem
// - WEBTRANSPORT_KEY=/path/to/key.pem
// - WEBTRANSPORT_HOST=quic.eternalconquests.com (public hostname)

let Http3Server, quicheLoaded;
let webtransportAvailable = false;

// Try to load WebTransport packages
try {
    const wt = await import('@fails-components/webtransport');

    console.log('[WebTransport] Package exports:', Object.keys(wt));
    console.log('[WebTransport] Http3Server from import:', typeof wt.Http3Server);

    Http3Server = wt.Http3Server;
    quicheLoaded = wt.quicheLoaded;

    console.log('[WebTransport] Http3Server assigned:', typeof Http3Server);

    // Wait for quiche native library to load
    if (quicheLoaded) {
        await quicheLoaded;
        console.log('[WebTransport] quiche native library loaded');
    } else {
        console.log('[WebTransport] No quicheLoaded promise (using bundled quiche)');
    }

    webtransportAvailable = true;
    console.log('[WebTransport] @fails-components/webtransport loaded successfully');
} catch (error) {
    console.warn('[WebTransport] Package not available:', error.message);
    console.warn('[WebTransport] Install with: npm install @fails-components/webtransport @fails-components/webtransport-transport-http3-quiche');
}

// Import protocol for binary encoding/decoding
let BinaryPacket, MessageType;
try {
    const protocol = await import('../../common/protocol-native.js');
    BinaryPacket = protocol.BinaryPacket;
    MessageType = protocol.MessageType;
} catch {
    const protocol = await import('../../common/protocol.js');
    BinaryPacket = protocol.BinaryPacket;
    MessageType = protocol.MessageType;
}

import fs from 'fs';
import path from 'path';

/**
 * WebTransport session handler for a single client
 */
class WebTransportSession {
    constructor(session, clientId, onMessage, onClose) {
        this.session = session;
        this.clientId = clientId;
        this.onMessage = onMessage;
        this.onClose = onClose;
        this.streams = new Map(); // streamId -> stream
        this.isReady = false;
        this.datagramsSupported = false;

        this.setupSession();
    }

    async setupSession() {
        try {
            // Check if datagrams are supported (true UDP-like behavior)
            this.datagramsSupported = this.session.datagrams?.readable != null;

            if (this.datagramsSupported) {
                console.log(`[WebTransport] Client ${this.clientId}: Datagrams supported - using true UDP mode`);
                this.setupDatagramHandler();
            } else {
                console.log(`[WebTransport] Client ${this.clientId}: Datagrams not supported - using streams`);
            }

            // Handle incoming unidirectional streams (fallback for large messages)
            this.handleIncomingStreams();

            // Handle session close
            this.session.closed.then(() => {
                console.log(`[WebTransport] Client ${this.clientId}: Session closed`);
                this.isReady = false;
                this.onClose(this.clientId);
            }).catch(err => {
                console.error(`[WebTransport] Client ${this.clientId}: Session error:`, err);
                this.isReady = false;
                this.onClose(this.clientId);
            });

            this.isReady = true;
            console.log(`[WebTransport] Client ${this.clientId}: Session ready`);
        } catch (error) {
            console.error(`[WebTransport] Client ${this.clientId}: Setup error:`, error);
            this.isReady = false;
        }
    }

    /**
     * Set up datagram handler for UDP-like messages
     */
    async setupDatagramHandler() {
        const reader = this.session.datagrams.readable.getReader();

        const readLoop = async () => {
            try {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    // Process incoming datagram
                    this.handleIncomingData(value);
                }
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error(`[WebTransport] Client ${this.clientId}: Datagram read error:`, error);
                }
            }
        };

        readLoop();
    }

    /**
     * Handle incoming bidirectional/unidirectional streams
     */
    async handleIncomingStreams() {
        // Handle bidirectional streams
        if (this.session.incomingBidirectionalStreams) {
            const reader = this.session.incomingBidirectionalStreams.getReader();
            this.readStreams(reader, 'bidirectional');
        }

        // Handle unidirectional streams
        if (this.session.incomingUnidirectionalStreams) {
            const reader = this.session.incomingUnidirectionalStreams.getReader();
            this.readStreams(reader, 'unidirectional');
        }
    }

    async readStreams(reader, type) {
        try {
            while (true) {
                const { value: stream, done } = await reader.read();
                if (done) break;

                this.handleStream(stream, type);
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error(`[WebTransport] Client ${this.clientId}: Stream reader error:`, error);
            }
        }
    }

    async handleStream(stream, type) {
        const streamReader = stream.readable.getReader();
        const chunks = [];

        try {
            while (true) {
                const { value, done } = await streamReader.read();
                if (done) break;
                chunks.push(value);
            }

            // Combine chunks and process
            const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const combined = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }

            this.handleIncomingData(combined);
        } catch (error) {
            console.error(`[WebTransport] Client ${this.clientId}: Stream read error:`, error);
        }
    }

    /**
     * Process incoming binary data
     */
    handleIncomingData(data) {
        try {
            const buffer = data.buffer || data;
            const packet = BinaryPacket.decode(buffer);

            if (this.onMessage) {
                this.onMessage(this.clientId, packet.type, packet.data);
            }
        } catch (error) {
            console.error(`[WebTransport] Client ${this.clientId}: Decode error:`, error);
        }
    }

    /**
     * Send message to client
     * Uses datagrams for small messages (UDP-like), streams for large ones
     */
    send(type, data) {
        if (!this.isReady) return false;

        try {
            const packet = BinaryPacket.encode(type, data);

            // Use datagrams for small messages (< 1200 bytes for MTU safety)
            if (this.datagramsSupported && packet.byteLength < 1200) {
                const writer = this.session.datagrams.writable.getWriter();
                writer.write(new Uint8Array(packet));
                writer.releaseLock();
                return true;
            }

            // Use unidirectional stream for larger messages
            return this.sendViaStream(packet);
        } catch (error) {
            console.error(`[WebTransport] Client ${this.clientId}: Send error:`, error);
            return false;
        }
    }

    async sendViaStream(data) {
        try {
            const stream = await this.session.createUnidirectionalStream();
            const writer = stream.getWriter();
            await writer.write(new Uint8Array(data));
            await writer.close();
            return true;
        } catch (error) {
            console.error(`[WebTransport] Client ${this.clientId}: Stream send error:`, error);
            return false;
        }
    }

    /**
     * Close session
     */
    close() {
        try {
            this.session.close();
        } catch (error) {
            // Ignore close errors
        }
        this.isReady = false;
    }
}

/**
 * WebTransport Server Manager
 */
export class WebTransportServer {
    constructor(options = {}) {
        this.port = options.port || parseInt(process.env.WEBTRANSPORT_PORT || '4433', 10);
        this.certPath = options.cert || process.env.WEBTRANSPORT_CERT;
        this.keyPath = options.key || process.env.WEBTRANSPORT_KEY;
        this.host = options.host || process.env.WEBTRANSPORT_HOST || 'localhost';

        this.server = null;
        this.sessions = new Map(); // clientId -> WebTransportSession
        this.enabled = webtransportAvailable && process.env.WEBTRANSPORT_ENABLED === 'true';

        // Callbacks
        this.onMessage = null; // (clientId, type, data) => void
        this.onConnect = null; // (clientId) => void
        this.onDisconnect = null; // (clientId) => void

        if (!webtransportAvailable) {
            console.log('[WebTransport Server] Disabled - package not available');
        } else if (!this.enabled) {
            console.log('[WebTransport Server] Disabled - set WEBTRANSPORT_ENABLED=true to enable');
        }
    }

    /**
     * Start the WebTransport server
     */
    async start() {
        if (!this.enabled || !webtransportAvailable) {
            return false;
        }

        // Validate certificate files
        if (!this.certPath || !this.keyPath) {
            console.error('[WebTransport Server] Certificate paths not configured');
            console.error('[WebTransport Server] Set WEBTRANSPORT_CERT and WEBTRANSPORT_KEY environment variables');
            return false;
        }

        if (!fs.existsSync(this.certPath) || !fs.existsSync(this.keyPath)) {
            console.error('[WebTransport Server] Certificate files not found');
            console.error(`  WEBTRANSPORT_CERT: ${this.certPath}`);
            console.error(`  WEBTRANSPORT_KEY: ${this.keyPath}`);
            return false;
        }

        try {
            const cert = fs.readFileSync(this.certPath, 'utf8');
            const key = fs.readFileSync(this.keyPath, 'utf8');

            // Debug: Check if Http3Server is available
            console.log('[WebTransport Server] Http3Server type:', typeof Http3Server);
            console.log('[WebTransport Server] webtransportAvailable:', webtransportAvailable);

            if (!Http3Server) {
                console.error('[WebTransport Server] Http3Server is undefined!');
                return false;
            }

            this.server = new Http3Server({
                port: this.port,
                host: '0.0.0.0', // Listen on all interfaces
                secret: 'webtransport_secret_' + Date.now(),
                cert,
                privKey: key
            });

            // Start the server
            await this.server.startServer();
            console.log(`[WebTransport Server] Started on UDP port ${this.port}`);

            // Listen for WebTransport sessions
            this.handleSessions();

            console.log(`[WebTransport Server] Connect URL: https://${this.host}:${this.port}/game`);
            return true;
        } catch (error) {
            console.error('[WebTransport Server] Failed to start:', error);
            return false;
        }
    }

    async handleSessions() {
        const sessionStream = await this.server.sessionStream('/game');
        const reader = sessionStream.getReader();

        const acceptLoop = async () => {
            try {
                while (true) {
                    const { value: session, done } = await reader.read();
                    if (done) break;

                    await this.handleNewSession(session);
                }
            } catch (error) {
                console.error('[WebTransport Server] Session accept error:', error);
            }
        };

        acceptLoop();
    }

    async handleNewSession(session) {
        try {
            // Session is already accepted from the stream
            // Generate client ID (could be passed from WebSocket handshake)
            const clientId = `wt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

            console.log(`[WebTransport Server] New session: ${clientId}`);

            // Create session handler
            const wtSession = new WebTransportSession(
                session,
                clientId,
                (cid, type, data) => {
                    if (this.onMessage) {
                        this.onMessage(cid, type, data);
                    }
                },
                (cid) => {
                    this.sessions.delete(cid);
                    if (this.onDisconnect) {
                        this.onDisconnect(cid);
                    }
                }
            );

            this.sessions.set(clientId, wtSession);

            if (this.onConnect) {
                this.onConnect(clientId, wtSession);
            }
        } catch (error) {
            console.error('[WebTransport Server] Session accept error:', error);
        }
    }

    /**
     * Send message to specific client
     */
    send(clientId, type, data) {
        const session = this.sessions.get(clientId);
        if (session) {
            return session.send(type, data);
        }
        return false;
    }

    /**
     * Broadcast to all connected clients
     */
    broadcast(type, data, excludeClientId = null) {
        let sentCount = 0;
        for (const [clientId, session] of this.sessions) {
            if (clientId !== excludeClientId && session.isReady) {
                if (session.send(type, data)) {
                    sentCount++;
                }
            }
        }
        return sentCount;
    }

    /**
     * Check if client is connected via WebTransport
     */
    isClientConnected(clientId) {
        const session = this.sessions.get(clientId);
        return session?.isReady || false;
    }

    /**
     * Remove client session
     */
    removeClient(clientId) {
        const session = this.sessions.get(clientId);
        if (session) {
            session.close();
            this.sessions.delete(clientId);
            console.log(`[WebTransport Server] Client ${clientId} removed`);
        }
    }

    /**
     * Get server stats
     */
    getStats() {
        let readySessions = 0;
        let datagramSessions = 0;

        for (const session of this.sessions.values()) {
            if (session.isReady) readySessions++;
            if (session.datagramsSupported) datagramSessions++;
        }

        return {
            enabled: this.enabled,
            running: this.server != null,
            port: this.port,
            totalSessions: this.sessions.size,
            readySessions,
            datagramSessions
        };
    }

    /**
     * Stop the server
     */
    stop() {
        // Close all sessions
        for (const session of this.sessions.values()) {
            session.close();
        }
        this.sessions.clear();

        if (this.server) {
            try {
                this.server.stopServer();
            } catch (error) {
                // Ignore stop errors
            }
            this.server = null;
        }

        console.log('[WebTransport Server] Stopped');
    }
}

// Singleton instance
let webTransportServer = null;

export function getWebTransportServer() {
    if (!webTransportServer) {
        webTransportServer = new WebTransportServer();
    }
    return webTransportServer;
}

export default WebTransportServer;
