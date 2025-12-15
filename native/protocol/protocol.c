#include "protocol.h"
#include <string.h>
#include <arpa/inet.h> // For endian conversion

int protocol_encode(uint8_t type, const uint8_t* payload, uint32_t payload_size,
                    uint8_t* out_buffer, uint32_t out_buffer_size) {
    // Check if output buffer is large enough
    uint32_t total_size = PROTOCOL_HEADER_SIZE + payload_size;
    if (total_size > out_buffer_size) {
        return -1; // Buffer too small
    }

    // Write header
    out_buffer[0] = type;

    // Write payload size (little-endian)
    out_buffer[1] = (payload_size) & 0xFF;
    out_buffer[2] = (payload_size >> 8) & 0xFF;
    out_buffer[3] = (payload_size >> 16) & 0xFF;
    out_buffer[4] = (payload_size >> 24) & 0xFF;

    // Copy payload
    if (payload_size > 0 && payload != NULL) {
        memcpy(out_buffer + PROTOCOL_HEADER_SIZE, payload, payload_size);
    }

    return total_size;
}

int protocol_decode(const uint8_t* buffer, uint32_t buffer_size,
                    uint8_t* out_type, const uint8_t** out_payload) {
    // Check minimum size
    if (buffer_size < PROTOCOL_HEADER_SIZE) {
        return -1; // Buffer too small
    }

    // Read type
    *out_type = buffer[0];

    // Read payload size (little-endian)
    uint32_t payload_size =
        ((uint32_t)buffer[1]) |
        ((uint32_t)buffer[2] << 8) |
        ((uint32_t)buffer[3] << 16) |
        ((uint32_t)buffer[4] << 24);

    // Validate payload size
    if (payload_size > buffer_size - PROTOCOL_HEADER_SIZE) {
        return -1; // Invalid payload size
    }

    // Set payload pointer
    if (payload_size > 0) {
        *out_payload = buffer + PROTOCOL_HEADER_SIZE;
    } else {
        *out_payload = NULL;
    }

    return payload_size;
}

const char* message_type_name(uint8_t type) {
    switch (type) {
        case MSG_HEARTBEAT: return "HEARTBEAT";
        case MSG_HANDSHAKE: return "HANDSHAKE";
        case MSG_HANDSHAKE_ACK: return "HANDSHAKE_ACK";
        case MSG_PING: return "PING";
        case MSG_PONG: return "PONG";
        case MSG_PLAYER_JOIN: return "PLAYER_JOIN";
        case MSG_PLAYER_LEAVE: return "PLAYER_LEAVE";
        case MSG_PLAYER_UPDATE: return "PLAYER_UPDATE";
        case MSG_PLAYER_LIST: return "PLAYER_LIST";
        case MSG_PLAYER_DEATH: return "PLAYER_DEATH";
        case MSG_PLAYER_RESPAWN: return "PLAYER_RESPAWN";
        case MSG_ENEMY_LIST: return "ENEMY_LIST";
        case MSG_ENEMY_UPDATE: return "ENEMY_UPDATE";
        case MSG_ENEMY_DEATH: return "ENEMY_DEATH";
        case MSG_BULLET_CREATE: return "BULLET_CREATE";
        case MSG_BULLET_LIST: return "BULLET_LIST";
        case MSG_BULLET_REMOVE: return "BULLET_REMOVE";
        case MSG_BAG_LIST: return "BAG_LIST";
        case MSG_PICKUP_ITEM: return "PICKUP_ITEM";
        case MSG_INVENTORY_UPDATE: return "INVENTORY_UPDATE";
        case MSG_BAG_REMOVE: return "BAG_REMOVE";
        case MSG_PICKUP_DENIED: return "PICKUP_DENIED";
        case MSG_MOVE_ITEM: return "MOVE_ITEM";
        case MSG_MOVE_DENIED: return "MOVE_DENIED";
        case MSG_COLLISION: return "COLLISION";
        case MSG_COLLISION_RESULT: return "COLLISION_RESULT";
        case MSG_MAP_INFO: return "MAP_INFO";
        case MSG_CHUNK_REQUEST: return "CHUNK_REQUEST";
        case MSG_CHUNK_DATA: return "CHUNK_DATA";
        case MSG_CHUNK_NOT_FOUND: return "CHUNK_NOT_FOUND";
        case MSG_PORTAL_ENTER: return "PORTAL_ENTER";
        case MSG_WORLD_SWITCH: return "WORLD_SWITCH";
        case MSG_WORLD_UPDATE: return "WORLD_UPDATE";
        case MSG_MAP_REQUEST: return "MAP_REQUEST";
        case MSG_PLAYER_LIST_REQUEST: return "PLAYER_LIST_REQUEST";
        case MSG_CHAT_MESSAGE: return "CHAT_MESSAGE";
        case MSG_SPEECH: return "SPEECH";
        case MSG_PLAYER_TEXT: return "PLAYER_TEXT";
        case MSG_UNIT_COMMAND: return "UNIT_COMMAND";
        case MSG_UNIT_UPDATE: return "UNIT_UPDATE";
        default: return "UNKNOWN";
    }
}
