# Network Devices Inventory

Last updated: 2026-01-24

## Windows Machines

| Device | IP | Username | Password | OS | RAM | GPU | VRAM |
|--------|-----|----------|----------|-----|-----|-----|------|
| Dell Latitude (Game Server) | 192.168.0.100 | newadmin | 12345 | Windows | 8GB | Intel HD 5500 | 1GB |
| Acer E5 | 192.168.0.203 | scrapenode | 12345 | Windows 10 Pro | 16GB | NVIDIA 940MX + Intel HD 520 | 2GB + 1GB |

## Linux / ARM Devices

| Device | IP | Username | Password | OS | RAM | GPU | VRAM/Notes |
|--------|-----|----------|----------|-----|-----|-----|------------|
| Raspberry Pi 4 | 192.168.0.204 | abbas | 123456 | Raspbian (Bookworm) | 4GB | VideoCore VI | 76MB |
| Raspberry Pi 5 | 192.168.0.205 | fast | 123456 | Raspbian (Bookworm) | 8GB | VideoCore VII | 4MB (configurable) |
| Jetson Nano | 192.168.0.206 | a344 | 123456 | Ubuntu 18.04 (L4T) | 4GB shared | Maxwell 128 CUDA | 4GB shared |

## Other Devices (from notes)

| Device | IP | Username | Password | Notes |
|--------|-----|----------|----------|-------|
| Developer Mac | 192.168.0.200 | developer-cloudprimero | devcp123 | - |
| Thinkpad | 192.168.0.201 | scraper | 123 | - |
| Dell Latitude (alt) | 192.168.0.202 | newadmin | 12345 | - |

## GPU Summary

| Device | GPU | CUDA Cores | VRAM | Best For |
|--------|-----|------------|------|----------|
| Jetson Nano | Maxwell | 128 | 4GB shared | ML inference, CUDA |
| Acer E5 | 940MX | 384 | 2GB | Light ML, CUDA |
| Pi 5 | VideoCore VII | - | 4MB-512MB | Video decode |
| Pi 4 | VideoCore VI | - | 76MB | Video decode |

## SSH Quick Connect

```bash
# Game Server (Windows)
sshpass -p "12345" ssh newadmin@192.168.0.100

# Acer E5 (Windows)
sshpass -p "12345" ssh scrapenode@192.168.0.203

# Raspberry Pi 4
sshpass -p "123456" ssh abbas@192.168.0.204

# Raspberry Pi 5
sshpass -p "123456" ssh fast@192.168.0.205

# Jetson Nano
sshpass -p "123456" ssh a344@192.168.0.206
```

## Services Running

### 192.168.0.100 (Dell Latitude - Game Server)
- Node.js game server (port 4000)
- Cloudflare tunnel (rotmg-tunnel)
- Playit tunnel
- Auto-start configured via Task Scheduler

### 192.168.0.206 (Jetson Nano - LLM Server)
- Ollama v0.14.3 (port 11434)
- Model: lfm2.5-1.2b (1.2B params, Q8_0)
- CPU-only inference (CUDA 10.2 incompatible with Ollama)
- ~1-2 tokens/sec, ~60-120s per response
- Auto-start via systemd

### 192.168.0.203 (Acer E5 - LLM Server)
- Ollama v0.13.4 (port 11434)
- Models: LFM2.5-1.2B (~24 tok/s), Qwen2.5-0.5B (~37 tok/s)
- GPU: 940MX with CUDA v11 (384 cores, 2GB VRAM)
- Auto-start via Task Scheduler
- Note: LFM2.5 requires Ollama 0.13.4 (0.14.x has missing tensor bug)

## LLM API Endpoints

| Server | URL | Model | GPU Accel | Speed |
|--------|-----|-------|-----------|-------|
| Acer E5 | http://192.168.0.203:11434 | LFM2.5-1.2B | Yes (940MX) | ~24 tok/s |
| Acer E5 | http://192.168.0.203:11434 | Qwen2.5-0.5B | Yes (940MX) | ~37 tok/s |
| Jetson Nano | http://192.168.0.206:11434 | lfm2.5-1.2b | No (CPU) | ~2 tok/s |

OpenAI-compatible API at `/v1/chat/completions`
