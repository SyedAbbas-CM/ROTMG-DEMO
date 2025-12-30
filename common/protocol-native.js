// Native C++ protocol wrapper
// Drop-in replacement for common/protocol.js with better performance

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

let nativeProtocol = null;
let useNative = false;

// Try to load native module
try {
    nativeProtocol = require('../build/Release/protocol.node');
    useNative = true;
    console.log('[PROTOCOL] ✓ Using C++ native protocol (2x faster)');
} catch (err) {
    console.log('[PROTOCOL] ⚠ Falling back to JavaScript protocol');
    console.log('[PROTOCOL]   Build native module with: npx node-gyp rebuild');
}

// Message types (must match C++ implementation)
export const MessageType = {
    // System / heartbeat
    HEARTBEAT: 0,

    // Connection
    HANDSHAKE: 1,
    HANDSHAKE_ACK: 2,
    PING: 3,
    PONG: 4,

    // Players
    PLAYER_JOIN: 10,
    PLAYER_LEAVE: 11,
    PLAYER_UPDATE: 12,
    PLAYER_LIST: 13,
    PLAYER_DEATH: 14,
    PLAYER_RESPAWN: 15,

    // Enemies
    ENEMY_LIST: 20,
    ENEMY_UPDATE: 21,
    ENEMY_DEATH: 22,

    // Bullets
    BULLET_CREATE: 30,
    BULLET_LIST: 31,
    BULLET_REMOVE: 32,

    // Loot / inventory
    BAG_LIST: 33,
    PICKUP_ITEM: 34,
    INVENTORY_UPDATE: 35,
    BAG_REMOVE: 36,
    PICKUP_DENIED: 37,
    MOVE_ITEM: 38,
    MOVE_DENIED: 39,

    // Collisions
    COLLISION: 40,
    COLLISION_RESULT: 41,

    // Map & world
    MAP_INFO: 50,
    CHUNK_REQUEST: 51,
    CHUNK_DATA: 52,
    CHUNK_NOT_FOUND: 53,
    PORTAL_ENTER: 54,
    WORLD_SWITCH: 55,
    WORLD_UPDATE: 60,
    MAP_REQUEST: 70,

    // Diagnostics
    PLAYER_LIST_REQUEST: 80,

    // Chat / speech
    CHAT_MESSAGE: 90,
    SPEECH: 91,
    PLAYER_TEXT: 92,

    // Units
    UNIT_COMMAND: 100,
    UNIT_UPDATE: 101,
    UNIT_SPAWN: 102,

    // Abilities
    USE_ABILITY: 110,
    ABILITY_RESULT: 111,

    // WebRTC Signaling (for UDP DataChannel)
    RTC_OFFER: 120,
    RTC_ANSWER: 121,
    RTC_ICE_CANDIDATE: 122,
    RTC_READY: 123,

    // WebTransport Session Linking
    WT_LINK: 124,
    WT_LINK_ACK: 125
};

// Messages that should go over UDP (DataChannel) when available
export const UDP_MESSAGES = new Set([
    12, // PLAYER_UPDATE
    21, // ENEMY_UPDATE
    30, // BULLET_CREATE
    31, // BULLET_LIST
    60, // WORLD_UPDATE
    40, // COLLISION
    41  // COLLISION_RESULT
]);

// Fallback JavaScript implementation
class JavaScriptBinaryPacket {
    static encode(type, data) {
        const jsonStr = JSON.stringify(data ?? {});
        const jsonBytes = new TextEncoder().encode(jsonStr);
        const packet = new ArrayBuffer(5 + jsonBytes.byteLength);
        const view = new DataView(packet);
        view.setUint8(0, type);
        view.setUint32(1, jsonBytes.byteLength, true);
        new Uint8Array(packet, 5).set(jsonBytes);
        return packet;
    }

    static decode(packet) {
        const view = new DataView(packet);
        const type = view.getUint8(0);
        const length = view.getUint32(1, true);

        const maxLength = packet.byteLength - 5;
        if (length > maxLength || length < 0) {
            console.error(`[Protocol] Invalid packet length: ${length}, max: ${maxLength}`);
            return { type, data: {} };
        }

        const jsonBytes = new Uint8Array(packet, 5, length);
        const jsonStr = new TextDecoder().decode(jsonBytes);
        try {
            const data = JSON.parse(jsonStr);
            return { type, data };
        } catch {
            return { type, data: {} };
        }
    }
}

// Optimized C++ wrapper
class NativeBinaryPacket {
    static encode(type, data) {
        const jsonStr = JSON.stringify(data ?? {});
        const jsonBuffer = Buffer.from(jsonStr);
        const encoded = nativeProtocol.encode(type, jsonBuffer);
        // Convert Buffer to ArrayBuffer for compatibility
        return encoded.buffer.slice(encoded.byteOffset, encoded.byteOffset + encoded.byteLength);
    }

    static decode(packet) {
        // Convert ArrayBuffer to Buffer if needed
        const buffer = packet instanceof ArrayBuffer
            ? Buffer.from(packet)
            : packet;

        const result = nativeProtocol.decode(buffer);
        const jsonStr = result.payload.toString();

        try {
            const data = JSON.parse(jsonStr);
            return { type: result.type, data };
        } catch {
            return { type: result.type, data: {} };
        }
    }
}

// Export the appropriate implementation
export const BinaryPacket = useNative ? NativeBinaryPacket : JavaScriptBinaryPacket;

// Export stats
export const ProtocolStats = {
    usingNative: useNative,
    implementation: useNative ? 'C++ (Fast)' : 'JavaScript (Fallback)',
    getTypeName: useNative ? nativeProtocol.getTypeName : (type) => {
        for (const [key, value] of Object.entries(MessageType)) {
            if (value === type) return key;
        }
        return 'UNKNOWN';
    }
};
