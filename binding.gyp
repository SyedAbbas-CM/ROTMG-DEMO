{
  "targets": [
    {
      "target_name": "protocol",
      "sources": [
        "native/protocol/protocol.c",
        "native/protocol/protocol_addon.cc"
      ],
      "include_dirs": [
        "native/protocol"
      ],
      "cflags": ["-O3", "-march=native", "-Wall"],
      "cflags_cc": ["-O3", "-march=native", "-Wall", "-std=c++14"],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_OPTIMIZATION_LEVEL": "3",
            "OTHER_CFLAGS": ["-march=native"],
            "CLANG_CXX_LANGUAGE_STANDARD": "c++14"
          }
        }]
      ]
    },
    {
      "target_name": "collision",
      "sources": [
        "native/collision/collision.c"
      ],
      "include_dirs": [
        "native/collision"
      ],
      "cflags": ["-O3", "-march=native", "-Wall", "-ffast-math"],
      "conditions": [
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_OPTIMIZATION_LEVEL": "3",
            "OTHER_CFLAGS": ["-march=native", "-ffast-math"]
          }
        }]
      ]
    }
  ]
}
