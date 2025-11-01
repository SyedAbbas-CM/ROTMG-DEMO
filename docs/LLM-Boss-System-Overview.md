## LLM Boss System – Architecture Overview

### Goals
- Adaptive, emergent boss behavior powered by an LLM
- Safe, bounded actions via a capability system and critic
- Predictable cadence with hashing, rate limits, and backoff

### High-Level Data Flow
```
Players & World ──> BossManager.buildSnapshot() ──> LLMBossController
                                │                          │
                                └─> xxhash (change?) ──────┤
                                                           ▼
                                                ProviderFactory → Provider (Gemini/Ollama)
                                                           │
                                      JSON plan/actions ◄──┘
                                                           │
                                   Capability Registry/Runner → Game Systems
```

### Runtime Components
- BossManager (`src/boss/BossManager.js`)
  - Maintains boss state (position, HP, cooldowns, phase)
  - `buildSnapshot(players)` collects boss, players, environment, recent history
- LLMBossController (`src/boss/LLMBossController.js`)
  - Rate limiting: `planPeriodSec`, `backoffSec`, `maxRetries`
  - Snapshot hashing (`xxhash-wasm`) to avoid redundant calls
  - Calls provider; parses and executes returned actions
  - Fallback behavior if provider fails
- ProviderFactory (`src/boss/llm/ProviderFactory.js`)
  - Chooses backend: Gemini vs. Ollama (via env)
  - Normalizes generate() interface
- Providers (`src/boss/llm/providers/*`)
  - Gemini: `@google/generative-ai`
  - Ollama: `ollama` client to local server
- Capability System (`src/capabilities/*` + registry)
  - Compiles validated bricks to invokable actions
  - Executes actions against `bossMgr`, `bulletMgr`, etc.
- Difficulty Critic (`src/boss/critic/DifficultyCritic.js`)
  - Checks DPS, action rates, resource limits before enactment
  - Rejects unsafe plans

### Control Flow (Sequence)
```
Server Tick (60 FPS)
  -> Every planPeriodSec (cooldown <= 0)
       -> BossManager.buildSnapshot(players)
       -> hashSnapshot(snapshot)
       -> if hash != lastHash
            -> ProviderFactory.create().generate(prompt, snapshot)
            -> parse/validate response
            -> DifficultyCritic.evaluate(metrics)
            -> if ok -> CapabilityRunner.execute(actions)
            -> else -> fallback/minimal plan
       -> set cooldown (or backoff on failure)
```

### Provider API Details
- Gemini (cloud)
  - Env: `LLM_BACKEND=gemini`, `GOOGLE_API_KEY`, `LLM_MODEL=gemini-pro`
  - SDK: `@google/generative-ai`
- Ollama (local)
  - Env: `LLM_BACKEND=ollama`, `OLLAMA_HOST=127.0.0.1`, `OLLAMA_PORT=11434`, `LLM_MODEL=llama3`
  - SDK: `ollama`

### Telemetry & Observability
- Spans: `llm.generate.*`, `boss.plan`, `mutator.<capability>`, `registry.compile`
- Metrics: latency, tokens, errors; server tick timing

### Key Safety Boundaries
- Rate limiting and backoff to contain cost and instability
- Snapshot hashing to limit calls to meaningful changes
- Capability limits: `maxProjectiles`, `maxSpeed`, `maxRadius`, `maxDuration`
- DifficultyCritic enforces DPS and action rate ceilings

### Interaction With Game Systems
```
LLM Actions → CapabilityRunner
              ├─ Movement (dash/teleport) → BossManager position/state
              ├─ Emitters (patterns)      → BulletManager spawns
              ├─ Speech                   → BossSpeechController (chat)
              └─ Wait/Analyze             → Internal cooldowns/phase
```

### Sequence Diagram (Simplified)
```
Client(s)            Server Loop                 LLM Controller            Provider
   |                     |                              |                      |
   |  Inputs/State      |                              |                      |
   |------------------->| buildSnapshot(players)       |                      |
   |                     |---- hash != lastHash? ----->|                      |
   |                     |                              |--- generate(...) --->|
   |                     |                              |<-- actions JSON -----|
   |                     |<----- actions/cooldowns -----|                      |
   |<---- WORLD_UPDATE --| apply actions (capabilities) |                      |
```

### Redesign Recommendations
1) Extract Capability Registry to its own module with explicit interfaces
2) Add streaming responses for partial plans (progressive action enactment)
3) Expand critic to include spatial safety checks (e.g., wall proximity)
4) Add plan cache keyed by (hash, model, parameters) for rapid replay/tests
5) Introduce sandbox runner with time budgets and per-cap cost accounting
6) Provide a “dry run” mode to measure plan complexity before execution

### Minimal Checklist
- Env configured (`LLM_BACKEND`, `LLM_MODEL`, provider keys/host)
- Telemetry enabled for `llm.generate.*`
- Safety limits set in `src/boss/config/llmConfig.js`
- Capability schemas/types generated if adding new abilities


