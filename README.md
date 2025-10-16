# ROTMG-DEMO – LLM-Powered Game Server

A revolutionary multiplayer action RPG server that combines classic *Realm of the Mad God* mechanics with cutting-edge **LLM-driven AI bosses**. Features real-time AI decision making using Google Gemini or local Ollama models, creating dynamic and emergent gameplay experiences.

## 🚀 Key Features

- **LLM Boss System**: AI-powered bosses that adapt strategies in real-time
- **Multiplayer Architecture**: WebSocket-based real-time gameplay
- **High-Performance ECS**: Structure of Arrays (SoA) design for 1000+ entities
- **Modular Capabilities**: Plugin-based boss ability system
- **Comprehensive Documentation**: Detailed system architecture guides
- **OpenTelemetry Integration**: Production-ready monitoring and metrics

---

## 📚 Documentation

### Core Systems Documentation
- **[System Architecture Overview](./docs/System-Architecture-Overview.md)** - High-level project architecture
- **[Enemy System](./docs/Enemy-System.md)** - Enemy management and network synchronization
- **[LLM Boss System](./docs/LLM-Boss-System.md)** - AI-powered boss architecture
- **[Drop System](./docs/Drop-System.md)** - Loot generation and probability mechanics
- **[Item System](./docs/Item-System.md)** - Item management and binary serialization
- **[Bag System](./docs/Bag-System.md)** - Loot bag management and TTL system
- **[Frontend-Backend Data Flow](./docs/Frontend-Backend-Data-Flow.md)** - Network protocols and client-server communication

---

## 🛠️ Quick Start

### Development Setup

```bash
# 1. Install dependencies
npm install

# 2. Generate capability schemas and types
npm run generate:union
npm run generate:types

# 3. Configure environment
cp env.example.txt .env
# Edit .env with your API keys (see Configuration section)

# 4. Run tests
npm test

# 5. Start the server
npm start
```

The server starts on `http://localhost:3000` with WebSocket support for real-time gameplay.

Shared protocol and constants are served under `/common` so the browser can import them directly.
 - Server imports from `./common/*`
 - Client imports from `/common/*`

### Testing the LLM Boss
1. Open `http://localhost:3000` in your browser
2. A boss will spawn automatically after 5 seconds
3. The boss behavior is controlled by the configured LLM provider
4. Watch the console for LLM decision-making logs

---

## ⚙️ Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `gemini` | AI provider: `gemini` or `ollama` |
| `LLM_MODEL` | `gemini-pro` | Model name (e.g., `gemini-pro`, `llama3`) |
| `LLM_TEMP` | `0.7` | Generation temperature (0.0-1.0) |
| `LLM_MAXTOKENS` | `256` | Maximum response tokens |
| `GOOGLE_API_KEY` | — | **Required** for Gemini provider |
| `OLLAMA_HOST` | `127.0.0.1` | Ollama server host |

### LLM Configuration

Edit `src/config/llmConfig.js` for advanced settings:
- `planPeriodSec`: Minimum time between LLM calls (default: 3.0s)
- `backoffSec`: Cooldown after failed calls (default: 1.0s)
- `maxRetries`: Maximum retry attempts (default: 3)

---

## 🏗️ Project Architecture

### Server Structure (`/src/`)

#### Core Game Systems
- **EnemyManager.js** - High-performance enemy management (SoA layout)
- **BulletManager.js** - Projectile physics and collision
- **CollisionManager.js** - Spatial collision detection
- **ItemManager.js** - Item definitions and binary serialization
- **BagManager.js** - Loot bag management with TTL
- **DropSystem.js** - Probabilistic loot generation
- **MapManager.js** - World and tile management

#### LLM Boss System
- **BossManager.js** - Boss physics and state management
- **LLMBossController.js** - AI provider coordination
- **ScriptBehaviourRunner.js** - Action execution engine
- **BossSpeechController.js** - Dynamic boss dialogue

#### Capability Registry
- **registry/Registry.js** - Modular action system
- **registry/DirectoryLoader.js** - Dynamic capability discovery
- **capabilities/** - Plugin-based boss abilities
  - Core: Wait, basic actions
  - Movement: Dash, teleport
  - Emitter: RadialBurst, ProjectileSpread

#### Infrastructure
- **NetworkManager.js** - WebSocket message handling
- **BehaviorSystem.js** - Enemy AI state machines
- **telemetry/** - OpenTelemetry monitoring
- **llm/providers/** - AI provider implementations

### Client Structure (`/public/src/`)

#### Game Core
- **game/game.js** - Main client game loop
- **game/ClientEnemyManager.js** - Client-side enemy state
- **game/ClientBulletManager.js** - Projectile interpolation
- **entities/** - Player and entity management

#### Rendering
- **render/** - Multi-view rendering system
  - Top-down view (main gameplay)
  - Strategic view (map overview)
  - WebGL optimized renderers

#### Assets & Database
- **assets/EntityDatabase.js** - Game object definitions
- **assets/SpriteDatabase.js** - Sprite atlas management
- **assets/TileDatabase.js** - Tilemap system

---

## 🔧 Development Tools

### Code Generation
```bash
npm run generate:union   # Generate capability union schema
npm run generate:types   # Generate TypeScript types
```

### Testing
```bash
npm test                 # Run Jest test suite
```

Test files:
- `tests/hash.test.js` - Snapshot hashing verification
- `tests/providerFactory.test.js` - LLM provider instantiation
- `tests/dsl-interpreter.spec.js` - Boss behavior DSL

### File Extraction Tools
- `extract_files/extractfiles.cjs` - Extract project files for LLM analysis
- `extract_systems.js` - System architecture extraction

---

## 🗺️ Map Editing & Live Mode

### Map Editor
- Open `public/tools/map-editor.html` in the browser.
- Paint Tiles, Enemies, Portals, and Spawn points on a grid.
- Use the sidebar “Live Mode” to apply changes to the running server.

### Live Mode Workflow
1. Set an admin token:
   - In the editor, click “Editor Settings” and enter `ADMIN_TOKEN` (or run `localStorage.setItem('ADMIN_TOKEN','your-secret')`).
2. Set the current `mapId` in settings (defaults to `default`).
3. Toggle “Live Mode”.
4. Paint:
   - Portals → POST `/api/portals/add` (right‑click to remove).
   - Enemies → POST `/api/enemies/spawn` (right‑click to remove at tile).
   - Spawn → POST `/api/maps/entrypoints/set`.
5. Save persistent changes:
   - Click “Save Active” (or Shift+Ctrl+S) → POST `/api/maps/save-active` to `maps/<mapId>.json`.

### Admin & Security
- Set `ADMIN_TOKEN` in environment for the server to require `x-admin-token` on write routes.
- Basic rate limits are applied to write endpoints.

### Routes Reference
- Portals: `GET /api/portals/list`, `POST /api/portals/add`, `POST /api/portals/remove`, `POST /api/portals/link-both`
- Enemies: `GET /api/enemies/list?mapId=...`, `POST /api/enemies/spawn`, `POST /api/enemies/remove`, `POST /api/enemies/remove-at`
- Maps: `GET /api/maps/:id/meta`, `POST /api/maps/entrypoints/set`, `POST /api/maps/reload`, `POST /api/maps/save-active`
- Status: `GET /api/status`

---

## 📊 Monitoring & Telemetry

### OpenTelemetry Integration
Automatic instrumentation for:
- **`llm.generate.*`** - LLM API call latency and tokens
- **`boss.plan`** - High-level boss planning operations
- **`mutator.<name>`** - Individual capability execution
- **`registry.compile`** - Capability compilation performance

### Console Exporter
By default, telemetry data is output to console. Configure custom exporters in `src/telemetry/index.js`.

---

## 🎮 Gameplay Features

### Core Mechanics
- **Real-time Multiplayer** - WebSocket-based with client prediction
- **Bullet Hell Combat** - High-performance projectile system
- **Loot System** - Color-coded bags with TTL mechanics
- **Character Classes** - Multiple player archetypes
- **World System** - Multi-map support with context isolation

### LLM Boss Features
- **Adaptive AI** - Bosses analyze player behavior and adapt
- **Dynamic Abilities** - Modular capability system for varied attacks
- **Contextual Decision Making** - AI considers player health, position, and class
- **Emergent Gameplay** - Unpredictable boss behaviors create unique encounters

---

## 🔍 System Integration

### Network Architecture
```
Client ←─→ WebSocket ←─→ Server.js ←─→ Game Systems
   │                        │              │
   │                        │              ├── EnemyManager
   │                        │              ├── BulletManager  
   │                        │              ├── LLMBossController
   │                        │              └── CollisionManager
   │                        │
   └── ClientManagers ←─────┤
       ├── ClientEnemyManager (interpolation)
       ├── ClientBulletManager (prediction)
       └── Rendering Pipeline
```

### Data Flow
1. **Server Tick** (60 FPS)
   - Update all game systems
   - Process LLM boss decisions  
   - Generate world snapshots
   - Broadcast to clients

2. **Client Tick** (60 FPS)
   - Receive server updates
   - Interpolate entity movement
   - Handle player input
   - Render frame

### Performance Optimizations
- **Structure of Arrays** - Cache-friendly data layout
- **Interest Management** - Only send nearby entities
- **Binary Serialization** - Compact network packets
- **Client Prediction** - Responsive player movement

---

## 📁 Unused/Legacy Components

The following directories contain unused or legacy code that could be removed:

### Unused Directories
- **`src/units/`** - Legacy unit system (not integrated)
- **`src/wasm/`** - WebAssembly experiments (not loaded)
- **`public/src/wasm/`** - Client-side WASM loader (unused)
- **`public/src/game/backup/`** - Old manager implementations
- **`extract_files/chunks/`** - Generated extraction chunks

### Experimental Features
- **`public/src/world/`** - Alternative world system
- **`public/src/shared/spatialGrid.js`** - Spatial optimization (unused)
- **`src/shared/spatialGrid.js`** - Server spatial grid (unused)

### Development Tools
- **`ROTMG-CORE-FILES/`** - Reference implementation (gitignored)
- **`logs/`** - Runtime log files
- **`scripts/testLLM.js`** - LLM testing script

---

## 🚀 Extending the System

### Adding New Capabilities

1. **Create Directory Structure**:
```bash
mkdir -p capabilities/YourCategory/YourCapability/1.0.0
```

2. **Define Schema** (`schema.json`):
```json
{
  "$id": "YourCategory:YourCapability@1.0.0",
  "title": "YourCategory:YourCapability",
  "type": "object",
  "required": ["type"],
  "properties": {
    "type": { "const": "YourCategory:YourCapability@1.0.0" },
    "parameter": { "type": "number", "default": 1.0 }
  }
}
```

3. **Implement Logic** (`implementation.js`):
```javascript
export function compile(brick) {
  return {
    ability: 'your_ability',
    args: { parameter: brick.parameter ?? 1.0 },
    _capType: brick.type
  };
}

export function invoke(node, state, { dt, bossMgr, bulletMgr }) {
  // Your capability logic here
  return completed; // boolean
}
```

4. **Regenerate Schemas**:
```bash
npm run generate:union && npm run generate:types
```

### Adding New Game Systems

1. Follow the **Structure of Arrays** pattern for performance
2. Integrate with `Server.js` main game loop
3. Add client-side counterpart in `public/src/`
4. Document in `docs/` directory

---

## 🧪 Testing Strategy

### Unit Tests
- Individual system functionality
- LLM provider integration
- Hash consistency verification

### Integration Tests
- Client-server communication
- Boss behavior execution
- Multiplayer synchronization

### Performance Tests
- Entity capacity limits (1000+ enemies)
- Network bandwidth optimization
- Memory usage profiling

---

## 🤝 Contributing

### Development Guidelines
1. Follow existing **Structure of Arrays** patterns
2. Add comprehensive documentation for new systems
3. Include telemetry spans for performance monitoring
4. Write tests for new functionality
5. Update this README for architectural changes

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests and documentation
4. Ensure all tests pass
5. Submit pull request with detailed description

---

## 📜 License

ISC License - See package.json for details.

---

## 🔗 Links

- **Documentation**: Complete system guides in `/docs/`
- **Examples**: Sample capabilities in `/capabilities/`
- **Assets**: Game sprites and data in `/public/assets/`
- **Tests**: Test suite in `/tests/`

This project represents a significant advancement in game AI, demonstrating how Large Language Models can create dynamic, emergent gameplay experiences that go far beyond traditional scripted behaviors.