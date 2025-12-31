// public/src/network/WebTransportManager.js
// Client-side WebTransport manager for UDP-like communication
//
// WebTransport provides true UDP-like datagrams over QUIC protocol
// - No ICE/TURN needed (direct client->server connection)
// - Works through NAT via tunnel (PlayIt.gg)
// - Unordered, unreliable datagrams for position updates
// - Ordered streams for reliable messages

import { BinaryPacket, MessageType, UDP_MESSAGES } from '/common/protocol.js';

/**
 * WebTransportManager - Handles WebTransport connection for low-latency game updates
 * Runs alongside WebSocket for fallback on messages that need reliability
 */
export class WebTransportManager {
    constructor(networkManager) {
        this.networkManager = networkManager;
        this.transport = null;
        this.isReady = false;
        this.isSupported = typeof WebTransport !== 'undefined';

        // Connection settings
        this.url = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 3;
        this.reconnectDelay = 2000;

        // Stats
        this.messagesSent = 0;
        this.messagesReceived = 0;
        this.bytesSent = 0;
        this.bytesReceived = 0;
        this.datagramsUsed = 0;
        this.streamsUsed = 0;

        if (!this.isSupported) {
            console.warn('[WebTransport] Not supported in this browser');
        } else {
            console.log('[WebTransport] Manager initialized');
        }
    }

    /**
     * Initialize WebTransport connection
     * @param {string} url - WebTransport server URL (e.g., https://quic.example.com:4433/game)
     */
    async initialize(url) {
        if (!this.isSupported) {
            console.warn('[WebTransport] Not supported - using WebSocket only');
            return false;
        }

        this.url = url;
        console.log(`[WebTransport] Connecting to ${url}...`);

        try {
            // Create WebTransport connection
            this.transport = new WebTransport(url);

            // Wait for connection to be ready
            await this.transport.ready;
            console.log('[WebTransport] Connection established!');

            // Set up handlers
            this.setupHandlers();

            this.isReady = true;
            this.reconnectAttempts = 0;

            // Update UI
            this.updateConnectionStatus('Connected (QUIC/UDP)');

            // Debug: Log successful connection
            console.log(`[WebTransport] CONNECTION READY - Binary protocol enabled, datagrams: ${this.transport.datagrams?.writable ? 'YES' : 'NO'}`);

            // Link WebTransport session to WebSocket client by sending clientId
            const clientId = this.networkManager.getClientId();
            if (clientId) {
                console.log(`[WebTransport] Linking to WebSocket client: ${clientId}`);
                this.send(MessageType.WT_LINK, { clientId });
            } else {
                console.warn('[WebTransport] No clientId available yet, will link on next update');
                // Retry linking after a short delay
                setTimeout(() => {
                    const cid = this.networkManager.getClientId();
                    if (cid && this.isReady) {
                        console.log(`[WebTransport] Delayed link to client: ${cid}`);
                        this.send(MessageType.WT_LINK, { clientId: cid });
                    }
                }, 500);
            }

            return true;
        } catch (error) {
            console.error('%c[WebTransport] CONNECTION FAILED', 'color: #f00; font-weight: bold');
            console.error('[WebTransport] Error:', error.message || error);
            console.error('[WebTransport] URL was:', url);
            this.isReady = false;

            // Try to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                console.log(`[WebTransport] Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectDelay}ms`);
                setTimeout(() => this.initialize(url), this.reconnectDelay);
            } else {
                console.warn('%c[WebTransport] GIVING UP - using WebSocket only (binary protocol disabled)', 'color: #f90; font-weight: bold');
            }

            return false;
        }
    }

    /**
     * Set up WebTransport event handlers
     */
    setupHandlers() {
        // Handle connection close
        this.transport.closed.then(() => {
            console.log('[WebTransport] Connection closed');
            this.isReady = false;
            this.updateConnectionStatus('Disconnected');
        }).catch(error => {
            console.error('[WebTransport] Connection error:', error);
            this.isReady = false;
        });

        // Start reading datagrams
        this.readDatagrams();

        // Handle incoming streams (for larger messages)
        this.readIncomingStreams();
    }

    /**
     * Read incoming datagrams (UDP-like messages)
     */
    async readDatagrams() {
        if (!this.transport.datagrams?.readable) {
            console.warn('[WebTransport] Datagrams not available');
            return;
        }

        const reader = this.transport.datagrams.readable.getReader();

        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                this.messagesReceived++;
                this.bytesReceived += value.byteLength;

                // CRITICAL: value.buffer might be a larger shared buffer
                // We need to slice out just our data portion
                const buffer = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
                this.handleMessage(buffer);
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('[WebTransport] Datagram read error:', error);
            }
        }
    }

    /**
     * Read incoming unidirectional streams
     */
    async readIncomingStreams() {
        if (!this.transport.incomingUnidirectionalStreams) {
            return;
        }

        const reader = this.transport.incomingUnidirectionalStreams.getReader();

        try {
            while (true) {
                const { value: stream, done } = await reader.read();
                if (done) break;

                this.readStream(stream);
            }
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error('[WebTransport] Stream reader error:', error);
            }
        }
    }

    /**
     * Read data from a stream
     */
    async readStream(stream) {
        const reader = stream.getReader();
        const chunks = [];

        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                // Copy the chunk data to avoid shared buffer issues
                const copy = new Uint8Array(value.byteLength);
                copy.set(value);
                chunks.push(copy);
            }

            // Combine chunks
            const totalLength = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
            const combined = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
                combined.set(chunk, offset);
                offset += chunk.byteLength;
            }

            this.messagesReceived++;
            this.bytesReceived += totalLength;
            // combined is a fresh array, its .buffer is exactly the right size
            this.handleMessage(combined.buffer);
        } catch (error) {
            console.error('[WebTransport] Stream read error:', error);
        }
    }

    /**
     * Handle incoming binary message
     * Supports both raw binary (WORLD_DELTA) and JSON-wrapped packets
     */
    handleMessage(buffer) {
        try {
            // Check first byte to determine packet type
            const view = new DataView(buffer);
            const firstByte = view.getUint8(0);

            // Raw binary WORLD_DELTA starts with 0x10 (BinaryPacketType.WORLD_DELTA)
            if (firstByte === 0x10) {
                // Route directly to BINARY_WORLD_DELTA handler with raw buffer
                if (this.networkManager.handlers[MessageType.BINARY_WORLD_DELTA]) {
                    this.networkManager.handlers[MessageType.BINARY_WORLD_DELTA](buffer);
                }
                return;
            }

            // Client binary messages (0x01-0x04) shouldn't come from server, but handle just in case
            if (firstByte >= 0x01 && firstByte <= 0x04) {
                console.warn('[WebTransport] Received client-type binary message from server:', firstByte);
                return;
            }

            // Otherwise, try JSON-wrapped BinaryPacket decode
            const packet = BinaryPacket.decode(buffer);

            // Forward to network manager's handler
            if (this.networkManager.handlers[packet.type]) {
                this.networkManager.handlers[packet.type](packet.data);
            }
        } catch (error) {
            console.error('[WebTransport] Message decode error:', error);
        }
    }

    /**
     * Send message over WebTransport
     * @param {number} type - Message type
     * @param {Object} data - Message data
     * @returns {boolean} True if sent successfully
     */
    send(type, data) {
        if (!this.isReady || !this.transport) {
            return false; // Fall back to WebSocket
        }

        try {
            const packet = BinaryPacket.encode(type, data);

            // Use datagrams for small UDP-appropriate messages
            if (this.transport.datagrams?.writable && packet.byteLength < 1200) {
                const writer = this.transport.datagrams.writable.getWriter();
                writer.write(new Uint8Array(packet));
                writer.releaseLock();

                this.messagesSent++;
                this.bytesSent += packet.byteLength;
                this.datagramsUsed++;
                return true;
            }

            // Use stream for larger messages
            return this.sendViaStream(packet);
        } catch (error) {
            console.error('[WebTransport] Send error:', error);
            return false;
        }
    }

    /**
     * Send raw binary data (no BinaryPacket wrapper)
     * Used for client→server binary protocol (type embedded in payload)
     * @param {ArrayBuffer} buffer - Raw binary payload
     * @returns {boolean} True if sent successfully
     */
    sendBinary(buffer) {
        if (!this.isReady || !this.transport) {
            // Debug: Log why we're not ready
            if (!this._notReadyLogged || Date.now() - this._notReadyLogged > 10000) {
                this._notReadyLogged = Date.now();
                console.warn(`[WebTransport] sendBinary failed: isReady=${this.isReady}, transport=${!!this.transport}`);
            }
            return false;
        }

        try {
            // Use datagrams for small messages (< MTU)
            if (this.transport.datagrams?.writable && buffer.byteLength < 1200) {
                const writer = this.transport.datagrams.writable.getWriter();
                writer.write(new Uint8Array(buffer));
                writer.releaseLock();

                this.messagesSent++;
                this.bytesSent += buffer.byteLength;
                this.datagramsUsed++;

                // Debug: Log first binary send
                if (!this._firstBinarySent) {
                    this._firstBinarySent = true;
                    console.log(`[WebTransport] First binary datagram sent: ${buffer.byteLength} bytes`);
                }
                return true;
            }

            // Fall back to stream for larger messages
            return this.sendBinaryViaStream(buffer);
        } catch (error) {
            console.error('[WebTransport] SendBinary error:', error);
            return false;
        }
    }

    /**
     * Send raw binary via stream
     */
    async sendBinaryViaStream(buffer) {
        try {
            const stream = await this.transport.createUnidirectionalStream();
            const writer = stream.getWriter();
            await writer.write(new Uint8Array(buffer));
            await writer.close();

            this.messagesSent++;
            this.bytesSent += buffer.byteLength;
            this.streamsUsed++;
            return true;
        } catch (error) {
            console.error('[WebTransport] Binary stream send error:', error);
            return false;
        }
    }

    /**
     * Send large message via unidirectional stream
     */
    async sendViaStream(packet) {
        try {
            const stream = await this.transport.createUnidirectionalStream();
            const writer = stream.getWriter();
            await writer.write(new Uint8Array(packet));
            await writer.close();

            this.messagesSent++;
            this.bytesSent += packet.byteLength;
            this.streamsUsed++;
            return true;
        } catch (error) {
            console.error('[WebTransport] Stream send error:', error);
            return false;
        }
    }

    /**
     * Check if a message type should use UDP/datagrams
     */
    shouldUseUDP(type) {
        return UDP_MESSAGES.has(type);
    }

    /**
     * Update connection status in UI
     */
    updateConnectionStatus(status) {
        const statusEl = document.getElementById('connectionStatus');
        if (statusEl) {
            statusEl.textContent = status;
            if (status.includes('QUIC') || status.includes('UDP')) {
                statusEl.style.backgroundColor = 'rgba(0, 200, 0, 0.5)';
            } else if (status.includes('Disconnected')) {
                statusEl.style.backgroundColor = 'rgba(200, 0, 0, 0.5)';
            }
        }
    }

    /**
     * Get connection stats
     */
    getStats() {
        return {
            isSupported: this.isSupported,
            isReady: this.isReady,
            messagesSent: this.messagesSent,
            messagesReceived: this.messagesReceived,
            bytesSent: this.bytesSent,
            bytesReceived: this.bytesReceived,
            datagramsUsed: this.datagramsUsed,
            streamsUsed: this.streamsUsed,
            reconnectAttempts: this.reconnectAttempts
        };
    }

    /**
     * Close WebTransport connection
     */
    close() {
        if (this.transport) {
            try {
                this.transport.close();
            } catch (error) {
                // Ignore close errors
            }
            this.transport = null;
        }
        this.isReady = false;
        console.log('[WebTransport] Connection closed');
    }
}

export default WebTransportManager;
