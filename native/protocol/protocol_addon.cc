#include <node_api.h>
#include "protocol.h"
#include <string.h>
#include <stdlib.h>

// Encode function wrapper for Node.js
// encode(type: number, payload: Buffer) -> Buffer
static napi_value Encode(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 2;
    napi_value args[2];

    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (status != napi_ok || argc < 2) {
        napi_throw_error(env, nullptr, "Expected 2 arguments: type and payload");
        return nullptr;
    }

    // Get message type
    uint32_t type;
    status = napi_get_value_uint32(env, args[0], &type);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "First argument must be a number");
        return nullptr;
    }

    // Get payload buffer
    bool is_buffer;
    status = napi_is_buffer(env, args[1], &is_buffer);
    if (status != napi_ok || !is_buffer) {
        napi_throw_error(env, nullptr, "Second argument must be a Buffer");
        return nullptr;
    }

    uint8_t* payload_data;
    size_t payload_size;
    status = napi_get_buffer_info(env, args[1], (void**)&payload_data, &payload_size);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to get buffer info");
        return nullptr;
    }

    // Allocate output buffer
    size_t output_size = PROTOCOL_HEADER_SIZE + payload_size;
    uint8_t* output_buffer = (uint8_t*)malloc(output_size);
    if (!output_buffer) {
        napi_throw_error(env, nullptr, "Failed to allocate output buffer");
        return nullptr;
    }

    // Encode
    int result = protocol_encode((uint8_t)type, payload_data, payload_size,
                                 output_buffer, output_size);
    if (result < 0) {
        free(output_buffer);
        napi_throw_error(env, nullptr, "Encoding failed");
        return nullptr;
    }

    // Create Node.js Buffer from output
    napi_value output;
    status = napi_create_buffer_copy(env, result, output_buffer, nullptr, &output);
    free(output_buffer);

    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create output buffer");
        return nullptr;
    }

    return output;
}

// Decode function wrapper for Node.js
// decode(buffer: Buffer) -> { type: number, payload: Buffer }
static napi_value Decode(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];

    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (status != napi_ok || argc < 1) {
        napi_throw_error(env, nullptr, "Expected 1 argument: buffer");
        return nullptr;
    }

    // Get input buffer
    bool is_buffer;
    status = napi_is_buffer(env, args[0], &is_buffer);
    if (status != napi_ok || !is_buffer) {
        napi_throw_error(env, nullptr, "Argument must be a Buffer");
        return nullptr;
    }

    uint8_t* input_data;
    size_t input_size;
    status = napi_get_buffer_info(env, args[0], (void**)&input_data, &input_size);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to get buffer info");
        return nullptr;
    }

    // Decode
    uint8_t type;
    const uint8_t* payload_ptr;
    int payload_size = protocol_decode(input_data, input_size, &type, &payload_ptr);

    if (payload_size < 0) {
        napi_throw_error(env, nullptr, "Decoding failed");
        return nullptr;
    }

    // Create result object
    napi_value result;
    status = napi_create_object(env, &result);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create result object");
        return nullptr;
    }

    // Set type
    napi_value type_value;
    status = napi_create_uint32(env, type, &type_value);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create type value");
        return nullptr;
    }
    napi_set_named_property(env, result, "type", type_value);

    // Set payload
    napi_value payload_value;
    if (payload_size > 0 && payload_ptr != nullptr) {
        status = napi_create_buffer_copy(env, payload_size, payload_ptr, nullptr, &payload_value);
    } else {
        status = napi_create_buffer(env, 0, nullptr, &payload_value);
    }

    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create payload buffer");
        return nullptr;
    }
    napi_set_named_property(env, result, "payload", payload_value);

    return result;
}

// Get message type name for debugging
// getTypeName(type: number) -> string
static napi_value GetTypeName(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 1;
    napi_value args[1];

    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (status != napi_ok || argc < 1) {
        napi_throw_error(env, nullptr, "Expected 1 argument: type");
        return nullptr;
    }

    uint32_t type;
    status = napi_get_value_uint32(env, args[0], &type);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Argument must be a number");
        return nullptr;
    }

    const char* name = message_type_name((uint8_t)type);

    napi_value result;
    status = napi_create_string_utf8(env, name, NAPI_AUTO_LENGTH, &result);
    if (status != napi_ok) {
        napi_throw_error(env, nullptr, "Failed to create string");
        return nullptr;
    }

    return result;
}

// Module initialization
static napi_value Init(napi_env env, napi_value exports) {
    napi_status status;
    napi_value fn;

    // Export encode function
    status = napi_create_function(env, nullptr, 0, Encode, nullptr, &fn);
    if (status != napi_ok) return nullptr;
    status = napi_set_named_property(env, exports, "encode", fn);
    if (status != napi_ok) return nullptr;

    // Export decode function
    status = napi_create_function(env, nullptr, 0, Decode, nullptr, &fn);
    if (status != napi_ok) return nullptr;
    status = napi_set_named_property(env, exports, "decode", fn);
    if (status != napi_ok) return nullptr;

    // Export getTypeName function
    status = napi_create_function(env, nullptr, 0, GetTypeName, nullptr, &fn);
    if (status != napi_ok) return nullptr;
    status = napi_set_named_property(env, exports, "getTypeName", fn);
    if (status != napi_ok) return nullptr;

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
