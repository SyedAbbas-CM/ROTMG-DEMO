#ifndef ROTMG_PROTOCOL_H
#define ROTMG_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Message types (must match JavaScript MessageType enum)
typedef enum {
    // System / heartbeat
    MSG_HEARTBEAT = 0,

    // Connection
    MSG_HANDSHAKE = 1,
    MSG_HANDSHAKE_ACK = 2,
    MSG_PING = 3,
    MSG_PONG = 4,

    // Players
    MSG_PLAYER_JOIN = 10,
    MSG_PLAYER_LEAVE = 11,
    MSG_PLAYER_UPDATE = 12,
    MSG_PLAYER_LIST = 13,
    MSG_PLAYER_DEATH = 14,
    MSG_PLAYER_RESPAWN = 15,

    // Enemies
    MSG_ENEMY_LIST = 20,
    MSG_ENEMY_UPDATE = 21,
    MSG_ENEMY_DEATH = 22,

    // Bullets
    MSG_BULLET_CREATE = 30,
    MSG_BULLET_LIST = 31,
    MSG_BULLET_REMOVE = 32,

    // Loot / inventory
    MSG_BAG_LIST = 33,
    MSG_PICKUP_ITEM = 34,
    MSG_INVENTORY_UPDATE = 35,
    MSG_BAG_REMOVE = 36,
    MSG_PICKUP_DENIED = 37,
    MSG_MOVE_ITEM = 38,
    MSG_MOVE_DENIED = 39,

    // Collisions
    MSG_COLLISION = 40,
    MSG_COLLISION_RESULT = 41,

    // Map & world
    MSG_MAP_INFO = 50,
    MSG_CHUNK_REQUEST = 51,
    MSG_CHUNK_DATA = 52,
    MSG_CHUNK_NOT_FOUND = 53,
    MSG_PORTAL_ENTER = 54,
    MSG_WORLD_SWITCH = 55,
    MSG_WORLD_UPDATE = 60,
    MSG_MAP_REQUEST = 70,

    // Diagnostics
    MSG_PLAYER_LIST_REQUEST = 80,

    // Chat / speech
    MSG_CHAT_MESSAGE = 90,
    MSG_SPEECH = 91,
    MSG_PLAYER_TEXT = 92,

    // Units
    MSG_UNIT_COMMAND = 100,
    MSG_UNIT_UPDATE = 101
} MessageType;

// Binary protocol header (optimized, no JSON)
typedef struct __attribute__((packed)) {
    uint8_t type;           // Message type
    uint32_t payload_size;  // Payload size in bytes (little-endian)
} ProtocolHeader;

#define PROTOCOL_HEADER_SIZE 5

// Encode a message (type + payload) into a buffer
// Returns size of encoded message, or -1 on error
int protocol_encode(uint8_t type, const uint8_t* payload, uint32_t payload_size,
                    uint8_t* out_buffer, uint32_t out_buffer_size);

// Decode a message from a buffer
// Returns payload size on success, or -1 on error
int protocol_decode(const uint8_t* buffer, uint32_t buffer_size,
                    uint8_t* out_type, const uint8_t** out_payload);

// Get message type name (for debugging)
const char* message_type_name(uint8_t type);

#ifdef __cplusplus
}
#endif

#endif // ROTMG_PROTOCOL_H
