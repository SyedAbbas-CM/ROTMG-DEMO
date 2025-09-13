# 🌍 Overworld Demo - Isolated System

A standalone 5000x5000 tile procedural overworld system with chunk-based loading and sprite rendering.

## 🚀 Quick Start

1. **Start the server:**
   ```bash
   node server.js
   ```

2. **Open in browser:**
   ```
   http://localhost:3001
   ```

## 🎮 Controls

- **WASD** or **Arrow Keys**: Move around the world
- **Mouse Wheel**: Zoom in/out (10% - 800%)
- **Click**: Teleport to clicked position
- **+/-**: Zoom controls
- **C**: Center view (2500, 2500)

## ✨ Features

### World Generation
- **Size**: 5000x5000 tiles (25 million tiles)
- **Chunks**: 100x100 tile chunks (2,500 total chunks)  
- **Procedural**: Deterministic generation with seed
- **Memory Management**: Only 25 chunks loaded at once

### Terrain Types
- **Plains** 🌱 (center regions)
- **Forest** 🌲 (mixed regions)
- **Mountains** ⛰️ (outer regions)
- **Water** 🌊 (rivers/lakes)
- **Desert** 🏜️ (harsh outer regions)
- **Wasteland** ☠️ (far outer regions)

### Technical Details
- **Rendering**: Canvas 2D with sprite support
- **Sprites**: Uses lofiEnvironment.png (8x8 sprites)
- **Fallback**: Colored rectangles if sprites fail to load
- **Performance**: Optimized viewport rendering
- **Debug Mode**: Toggle chunk boundaries

## 📁 Files

- `index.html` - Main demo page with embedded game logic
- `EfficientWorldManager.js` - Core world management system
- `server.js` - Simple HTTP server for local testing
- `README.md` - This documentation

## 🔧 Architecture

```
EfficientWorldManager
├── Chunk Loading (on-demand)
├── Memory Management (LRU eviction)
├── Procedural Generation (seeded)
├── Terrain Distribution (distance-based)
└── Sprite Mapping (lofiEnvironment.png)
```

## 🎯 Integration Ready

This system is completely isolated and ready for integration into larger game projects. The `EfficientWorldManager` class can be imported and used independently.

## 🐛 Troubleshooting

- **CORS Errors**: Use the provided server.js instead of opening index.html directly
- **Black Screen**: Check browser console for errors
- **No Sprites**: System will fall back to colored rectangles automatically
- **Port Issues**: Change PORT in server.js if 3001 is occupied