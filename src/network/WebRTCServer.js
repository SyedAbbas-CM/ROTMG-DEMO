// src/network/WebRTCServer.js
// Server-side WebRTC handler for UDP-like DataChannels

// Use dynamic import path based on environment
let BinaryPacket, MessageType, UDP_MESSAGES;
try {
    const protocol = await import('../../common/protocol-native.js');
    BinaryPacket = protocol.BinaryPacket;
    MessageType = protocol.MessageType;
    // UDP_MESSAGES might not exist in protocol-native, define locally if needed
    UDP_MESSAGES = protocol.UDP_MESSAGES || new Set([12, 21, 30, 31, 60, 40, 41]);
} catch {
    const protocol = await import('../../common/protocol.js');
    BinaryPacket = protocol.BinaryPacket;
    MessageType = protocol.MessageType;
    UDP_MESSAGES = protocol.UDP_MESSAGES;
}

// Try to load wrtc package (required for Node.js WebRTC)
// Try multiple forks - some have better prebuilt binary support
let wrtc = null;
const wrtcPackages = [
    '@roamhq/wrtc',  // Most actively maintained, has prebuilt binaries
    '@koush/wrtc',    // Another maintained fork
    'wrtc'            // Original package (often needs compilation)
];

for (const packageName of wrtcPackages) {
    try {
        wrtc = await import(packageName);
        // Handle default export
        if (wrtc.default) wrtc = wrtc.default;
        console.log(`[WebRTC Server] Loaded ${packageName} successfully`);
        break;
    } catch {
        // Try next package
    }
}

if (!wrtc) {
    console.warn('[WebRTC Server] No wrtc package available - WebRTC disabled');
    console.warn('[WebRTC Server] Install with: npm install @roamhq/wrtc');
}

/**
 * WebRTC peer connection manager for a single client
 */
class RTCPeer {
    constructor(clientId, sendToClient) {
        this.clientId = clientId;
        this.sendToClient = sendToClient; // Function to send via WebSocket
        this.peerConnection = null;
        this.dataChannel = null;
        this.isReady = false;
        this.pendingCandidates = [];

        // ICE servers
        this.iceServers = [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' }
        ];
    }

    /**
     * Handle incoming SDP offer from client
     */
    async handleOffer(offer) {
        if (!wrtc) {
            console.warn(`[WebRTC] Cannot handle offer - wrtc not available`);
            return false;
        }

        try {
            console.log(`[WebRTC] Client ${this.clientId}: Handling offer`);

            // Create peer connection
            this.peerConnection = new wrtc.RTCPeerConnection({
                iceServers: this.iceServers
            });

            this.setupPeerConnectionHandlers();

            // Set remote description (client's offer)
            await this.peerConnection.setRemoteDescription(
                new wrtc.RTCSessionDescription(offer)
            );

            // Apply any pending candidates
            for (const candidate of this.pendingCandidates) {
                await this.peerConnection.addIceCandidate(candidate);
            }
            this.pendingCandidates = [];

            // Create answer
            const answer = await this.peerConnection.createAnswer();
            await this.peerConnection.setLocalDescription(answer);

            // Send answer to client via WebSocket
            this.sendToClient(MessageType.RTC_ANSWER, {
                sdp: answer.sdp,
                type: answer.type
            });

            console.log(`[WebRTC] Client ${this.clientId}: Answer sent`);
            return true;
        } catch (error) {
            console.error(`[WebRTC] Client ${this.clientId}: Error handling offer:`, error);
            return false;
        }
    }

    /**
     * Set up RTCPeerConnection event handlers
     */
    setupPeerConnectionHandlers() {
        // Handle incoming data channel from client
        this.peerConnection.ondatachannel = (event) => {
            console.log(`[WebRTC] Client ${this.clientId}: DataChannel received`);
            this.dataChannel = event.channel;
            this.setupDataChannelHandlers();
        };

        // ICE candidate generated - send to client
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                this.sendToClient(MessageType.RTC_ICE_CANDIDATE, {
                    candidate: event.candidate.candidate,
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex
                });
            }
        };

        // Connection state changes
        this.peerConnection.onconnectionstatechange = () => {
            console.log(`[WebRTC] Client ${this.clientId}: Connection state:`,
                this.peerConnection.connectionState);

            if (this.peerConnection.connectionState === 'failed' ||
                this.peerConnection.connectionState === 'disconnected') {
                this.isReady = false;
            }
        };
    }

    /**
     * Set up DataChannel event handlers
     */
    setupDataChannelHandlers() {
        this.dataChannel.onopen = () => {
            console.log(`[WebRTC] Client ${this.clientId}: DataChannel OPEN`);
            this.isReady = true;
        };

        this.dataChannel.onclose = () => {
            console.log(`[WebRTC] Client ${this.clientId}: DataChannel closed`);
            this.isReady = false;
        };

        this.dataChannel.onerror = (error) => {
            console.error(`[WebRTC] Client ${this.clientId}: DataChannel error:`, error);
            this.isReady = false;
        };

        this.dataChannel.onmessage = (event) => {
            // Messages from DataChannel are handled via onMessage callback
            if (this.onMessage) {
                this.onMessage(event.data);
            }
        };
    }

    /**
     * Handle ICE candidate from client
     */
    async handleIceCandidate(candidateData) {
        try {
            const candidate = new wrtc.RTCIceCandidate(candidateData);

            if (this.peerConnection && this.peerConnection.remoteDescription) {
                await this.peerConnection.addIceCandidate(candidate);
            } else {
                this.pendingCandidates.push(candidate);
            }
        } catch (error) {
            console.error(`[WebRTC] Client ${this.clientId}: Error adding ICE candidate:`, error);
        }
    }

    /**
     * Send message via DataChannel
     */
    send(type, data) {
        if (!this.isReady || !this.dataChannel || this.dataChannel.readyState !== 'open') {
            return false;
        }

        try {
            const packet = BinaryPacket.encode(type, data);
            this.dataChannel.send(packet);
            return true;
        } catch (error) {
            console.error(`[WebRTC] Client ${this.clientId}: Send error:`, error);
            return false;
        }
    }

    /**
     * Close connection
     */
    close() {
        if (this.dataChannel) {
            this.dataChannel.close();
            this.dataChannel = null;
        }
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        this.isReady = false;
    }
}

/**
 * WebRTC Server Manager - manages all peer connections
 */
export class WebRTCServer {
    constructor() {
        this.peers = new Map(); // clientId -> RTCPeer
        this.enabled = wrtc !== null;
        this.onMessage = null; // Callback for incoming messages

        if (this.enabled) {
            console.log('[WebRTC Server] Initialized and ready');
        } else {
            console.log('[WebRTC Server] Disabled - wrtc package not available');
        }
    }

    /**
     * Check if WebRTC is available
     */
    isEnabled() {
        return this.enabled;
    }

    /**
     * Handle RTC offer from client
     */
    async handleOffer(clientId, offer, sendToClient) {
        if (!this.enabled) return false;

        // Create or get existing peer
        let peer = this.peers.get(clientId);
        if (!peer) {
            peer = new RTCPeer(clientId, sendToClient);
            peer.onMessage = (data) => {
                if (this.onMessage) {
                    this.onMessage(clientId, data);
                }
            };
            this.peers.set(clientId, peer);
        }

        return await peer.handleOffer(offer);
    }

    /**
     * Handle ICE candidate from client
     */
    async handleIceCandidate(clientId, candidateData) {
        const peer = this.peers.get(clientId);
        if (peer) {
            await peer.handleIceCandidate(candidateData);
        }
    }

    /**
     * Send message to client via DataChannel
     */
    send(clientId, type, data) {
        const peer = this.peers.get(clientId);
        if (peer && peer.isReady) {
            return peer.send(type, data);
        }
        return false;
    }

    /**
     * Check if client has ready DataChannel
     */
    isClientReady(clientId) {
        const peer = this.peers.get(clientId);
        return peer?.isReady || false;
    }

    /**
     * Remove client peer connection
     */
    removeClient(clientId) {
        const peer = this.peers.get(clientId);
        if (peer) {
            peer.close();
            this.peers.delete(clientId);
            console.log(`[WebRTC Server] Client ${clientId} removed`);
        }
    }

    /**
     * Broadcast to all ready clients via DataChannel
     */
    broadcast(type, data, excludeClientId = null) {
        let sentCount = 0;
        for (const [clientId, peer] of this.peers) {
            if (clientId !== excludeClientId && peer.isReady) {
                if (peer.send(type, data)) {
                    sentCount++;
                }
            }
        }
        return sentCount;
    }

    /**
     * Get stats for all connections
     */
    getStats() {
        const stats = {
            enabled: this.enabled,
            totalPeers: this.peers.size,
            readyPeers: 0
        };

        for (const peer of this.peers.values()) {
            if (peer.isReady) stats.readyPeers++;
        }

        return stats;
    }
}

// Singleton instance
let webrtcServer = null;

export function getWebRTCServer() {
    if (!webrtcServer) {
        webrtcServer = new WebRTCServer();
    }
    return webrtcServer;
}

export default WebRTCServer;
