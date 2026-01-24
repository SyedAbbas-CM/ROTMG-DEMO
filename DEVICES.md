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
