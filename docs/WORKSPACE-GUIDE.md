### Workspace guide

Focus areas you can open independently:

- Networking (shared)
  - `common/protocol.js` – MessageType and BinaryPacket
  - `common/constants.js` – network tuning
  - Client: `public/src/network/ClientNetworkManager.js`
  - Server: `Server.js` (WebSocket handling, broadcast)

- LLM Boss system
  - Server files: `src/BossManager.js`, `src/LLMBossController.js`, `src/BossSpeechController.js`
  - Routes/logging: `src/routes/llmRoutes.js`, `src/telemetry/`

- World/Simulation
  - `src/MapManager.js`, `src/EnemyManager.js`, `src/BulletManager.js`, `src/CollisionManager.js`, `src/BagManager.js`

- Client rendering/UI
  - `public/src/game/game.js` (entry)
  - `public/src/render/*`, `public/src/ui/*`, `public/src/assets/*`

Near-term plan:
1) Verify protocol unification (run the game; ensure connections and player lists work)
2) Remove duplicate protocol definitions from client
3) Archive legacy/unused files to reduce sidebar clutter
4) Isolate LLM Boss files under `server/world/llm/` (in a later pass)


