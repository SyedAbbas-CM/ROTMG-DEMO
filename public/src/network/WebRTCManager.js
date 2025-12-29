// public/src/network/WebRTCManager.js
// WebRTC DataChannel manager for UDP-like communication

import { BinaryPacket, MessageType, UDP_MESSAGES } from '/common/protocol.js';

/**
 * WebRTCManager - Handles WebRTC DataChannel for low-latency game updates
 * Works alongside WebSocket for reliable messages
 */
export class WebRTCManager {
    constructor(networkManager) {
        this.networkManager = networkManager;
        this.peerConnection = null;
        this.dataChannel = null;
        this.isReady = false;
        this.pendingCandidates = [];

        // Stats
        this.messagesSent = 0;
        this.messagesReceived = 0;
        this.bytesSent = 0;
        this.bytesReceived = 0;

        // ICE servers for NAT traversal
        this.iceServers = [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' }
        ];

        console.log('[WebRTC] Manager initialized');
    }

    /**
     * Initialize WebRTC connection
     * Called after WebSocket connection is established
     */
    async initialize() {
        try {
            console.log('[WebRTC] Starting initialization...');

            // Create peer connection
            this.peerConnection = new RTCPeerConnection({
                iceServers: this.iceServers
            });

            // Set up event handlers
            this.setupPeerConnectionHandlers();

            // Create data channel with unreliable settings (UDP-like)
            this.dataChannel = this.peerConnection.createDataChannel('gamedata', {
                ordered: false,        // Don't guarantee order (faster)
                maxRetransmits: 0      // No retransmits (true UDP behavior)
            });

            this.setupDataChannelHandlers();

            // Create and send offer
            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            console.log('[WebRTC] Sending offer to server...');
            this.networkManager.send(MessageType.RTC_OFFER, {
                sdp: offer.sdp,
                type: offer.type
            });

            return true;
        } catch (error) {
            console.error('[WebRTC] Initialization failed:', error);
            return false;
        }
    }

    /**
     * Set up RTCPeerConnection event handlers
     */
    setupPeerConnectionHandlers() {
        // ICE candidate generated - send to server
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                console.log('[WebRTC] Sending ICE candidate');
                this.networkManager.send(MessageType.RTC_ICE_CANDIDATE, {
                    candidate: event.candidate.candidate,
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex
                });
            }
        };

        // Connection state changes
        this.peerConnection.onconnectionstatechange = () => {
            console.log('[WebRTC] Connection state:', this.peerConnection.connectionState);

            if (this.peerConnection.connectionState === 'connected') {
                console.log('[WebRTC] Peer connection established!');
            } else if (this.peerConnection.connectionState === 'failed') {
                console.error('[WebRTC] Connection failed, falling back to WebSocket only');
                this.isReady = false;
            }
        };

        // ICE connection state
        this.peerConnection.oniceconnectionstatechange = () => {
            console.log('[WebRTC] ICE state:', this.peerConnection.iceConnectionState);
        };
    }

    /**
     * Set up DataChannel event handlers
     */
    setupDataChannelHandlers() {
        this.dataChannel.onopen = () => {
            console.log('[WebRTC] DataChannel OPEN - UDP transport ready!');
            this.isReady = true;

            // Notify server that DataChannel is ready
            this.networkManager.send(MessageType.RTC_READY, { ready: true });

            // Update UI
            const statusEl = document.getElementById('connectionStatus');
            if (statusEl) {
                statusEl.textContent = 'Connected (UDP)';
                statusEl.style.backgroundColor = 'rgba(0, 200, 0, 0.5)';
            }
        };

        this.dataChannel.onclose = () => {
            console.log('[WebRTC] DataChannel closed');
            this.isReady = false;
        };

        this.dataChannel.onerror = (error) => {
            console.error('[WebRTC] DataChannel error:', error);
            this.isReady = false;
        };

        this.dataChannel.onmessage = (event) => {
            this.messagesReceived++;
            this.bytesReceived += event.data.byteLength || event.data.length;

            // Handle incoming message (binary)
            this.handleMessage(event.data);
        };
    }

    /**
     * Handle incoming message from DataChannel
     */
    handleMessage(data) {
        try {
            // Convert to ArrayBuffer if needed
            let buffer;
            if (data instanceof ArrayBuffer) {
                buffer = data;
            } else if (data instanceof Blob) {
                // Handle Blob (shouldn't happen with binaryType = arraybuffer)
                data.arrayBuffer().then(ab => this.handleMessage(ab));
                return;
            } else {
                console.warn('[WebRTC] Unexpected data type:', typeof data);
                return;
            }

            // Decode binary packet
            const packet = BinaryPacket.decode(buffer);

            // Forward to network manager's handler
            if (this.networkManager.handlers[packet.type]) {
                this.networkManager.handlers[packet.type](packet.data);
            }
        } catch (error) {
            console.error('[WebRTC] Error handling message:', error);
        }
    }

    /**
     * Handle SDP answer from server
     */
    async handleAnswer(answer) {
        try {
            console.log('[WebRTC] Received answer from server');
            await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

            // Apply any pending ICE candidates
            for (const candidate of this.pendingCandidates) {
                await this.peerConnection.addIceCandidate(candidate);
            }
            this.pendingCandidates = [];

            console.log('[WebRTC] Remote description set successfully');
        } catch (error) {
            console.error('[WebRTC] Error handling answer:', error);
        }
    }

    /**
     * Handle ICE candidate from server
     */
    async handleIceCandidate(candidateData) {
        try {
            const candidate = new RTCIceCandidate(candidateData);

            if (this.peerConnection.remoteDescription) {
                await this.peerConnection.addIceCandidate(candidate);
            } else {
                // Queue candidate until remote description is set
                this.pendingCandidates.push(candidate);
            }
        } catch (error) {
            console.error('[WebRTC] Error adding ICE candidate:', error);
        }
    }

    /**
     * Send message over DataChannel
     * @returns {boolean} True if sent via DataChannel, false if should use WebSocket
     */
    send(type, data) {
        if (!this.isReady || !this.dataChannel || this.dataChannel.readyState !== 'open') {
            return false; // Fallback to WebSocket
        }

        try {
            const packet = BinaryPacket.encode(type, data);
            this.dataChannel.send(packet);
            this.messagesSent++;
            this.bytesSent += packet.byteLength;
            return true;
        } catch (error) {
            console.error('[WebRTC] Send error:', error);
            return false;
        }
    }

    /**
     * Check if a message type should use UDP
     */
    shouldUseUDP(type) {
        return UDP_MESSAGES.has(type);
    }

    /**
     * Get connection stats
     */
    getStats() {
        return {
            isReady: this.isReady,
            messagesSent: this.messagesSent,
            messagesReceived: this.messagesReceived,
            bytesSent: this.bytesSent,
            bytesReceived: this.bytesReceived,
            connectionState: this.peerConnection?.connectionState,
            iceState: this.peerConnection?.iceConnectionState,
            channelState: this.dataChannel?.readyState
        };
    }

    /**
     * Close WebRTC connection
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
        console.log('[WebRTC] Connection closed');
    }
}

export default WebRTCManager;
