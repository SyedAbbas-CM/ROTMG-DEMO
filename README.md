# ROTMG-DEMO – Hyper-Boss Playground

This repository is a fork of the classic *Realm of the Mad God* demo server.  
It has been retro-fitted with an **LLM-driven Hyper-Boss** that plans attacks
in real-time using Google Gemini (or a local Ollama model).

---

## Quick start (dev)

```bash
# 1. install node modules
npm install

# 2. generate the union JSON-Schema & TS types
npm run generate:union   # node ci-tools/generateUnionSchema.js
npm run generate:types   # node ci-tools/generateTypes.js

# 3. copy environment variables & edit as needed
cp .env.example .env

# 4. run tests (hash & provider factory)
npm test

# 5. start the server (opens Express + WebSocket)
npm start
```

Visit `http://localhost:3000` in a browser to join the game.  A single boss will
spawn automatically; its behaviour is planned by the configured LLM provider.

---

## Environment variables (.env)

| Var                 | Default     | Purpose                                             |
| ------------------- | ----------- | --------------------------------------------------- |
| `LLM_BACKEND`       | `gemini`    | `gemini` or `ollama`                                |
| `LLM_MODEL`         | `gemini-pro`| Model name (e.g. `gemini-pro`, `llama3`)           |
| `LLM_TEMP`          | `0.7`       | Generation temperature                              |
| `LLM_MAXTOKENS`     | `256`       | Completion token cap                                |
| `GOOGLE_API_KEY`    | —           | Required when `LLM_BACKEND=gemini`                  |
| `OLLAMA_HOST`       | `127.0.0.1` | Optional Ollama host                                |

Create `.env` by copying `.env.example` and filling in your API key.

---

## Telemetry

OpenTelemetry spans are emitted for:

* `llm.generate.*` – each LLM round-trip (latency & tokens)
* `boss.plan`      – high-level plan acquisition
* `mutator.<name>` – execution of each mutator per frame

By default a `ConsoleSpanExporter` prints to stdout.  Swap
`src/telemetry/index.js` for your own exporter (Jaeger, OTLP, …).

---

## Tests

* **Hashing** – verifies `xxhash32` produces stable digests (`tests/hash.test.js`).
* **Provider factory** – checks environment-driven instantiation (`tests/providerFactory.test.js`).

Run with `npm test` (Jest).

---

## Contributing

See `ROADMAP.md` for upcoming milestones.  Pull-requests welcome!
