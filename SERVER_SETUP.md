# ROTMG Server Setup & Recovery Guide

## Current Status (January 2026)

### Running Services
| Service | PID | Memory | Status |
|---------|-----|--------|--------|
| `node.exe` (Game Server) | 8200 | ~228 MB | Running |
| `cloudflared.exe` (WS Tunnel) | 13776 | ~43 MB | Running |
| `playit.exe` (UDP Tunnel) | 4000 | ~48 MB | Running |

### Listening Ports
| Protocol | Port | Service |
|----------|------|---------|
| TCP | 4000 | WebSocket game server |
| UDP | 4433 | WebTransport/QUIC server |
| TCP | 22 | SSH (OpenSSH) |

---

## Server Location
- **Windows Laptop**: `newadmin@192.168.0.202`
- **Password**: `12345`
- **Server Directory**: `C:\ROTMG-DEMO`
- **Node.js**: `C:\node-v22.12.0-win-x64\node.exe` (v22.12.0)
- **OS**: Windows 10 (Build 19045.6466)

---

## CRITICAL WARNINGS

### DO NOT RUN `npm install` ON THE SERVER

The `@fails-components/webtransport` packages are **manually installed** and NOT in `package.json`. Running `npm install` will:

1. **Delete** the WebTransport packages from `node_modules`
2. **Break** the native binary (`webtransport.node`)
3. **Cause** WebTransport to fail silently (port 4433 won't listen)

If you accidentally ran `npm install`, see "WebTransport Recovery" below.

### DO NOT DEPLOY `.env` OR `certs/`

These contain server-specific paths:
```
WEBTRANSPORT_CERT=C:/ROTMG-DEMO/certs/fullchain.pem
WEBTRANSPORT_KEY=C:/ROTMG-DEMO/certs/privkey.pem
```

The `./deploy.sh` commands already exclude these files.

### DO NOT UPDATE NODE.JS WITHOUT UPDATING FIREWALL

Firewall rules are tied to the Node.js executable path. If you upgrade Node.js:
1. The old rules will stop working
2. UDP traffic will be blocked
3. WebTransport will fail with `QUIC_NETWORK_IDLE_TIMEOUT`

See "Firewall Rules" section below.

---

## Server Configuration

### Ports & Tunneling
| Service | Protocol | Local Port | External | Tunnel |
|---------|----------|------------|----------|--------|
| WebSocket | TCP | 4000 | eternalconquests.com | Cloudflare Tunnel |
| WebTransport | UDP | 4433 | quic.eternalconquests.com:10615 | PlayIt.gg |

### Scheduled Tasks (Auto-Start on Boot)
- `CloudflareTunnel` - Starts cloudflared.exe
- `GameServer` - Starts Node.js server

### Firewall Rules (CRITICAL for WebTransport)
```
Node.js v22 TCP - C:\node-v22.12.0-win-x64\node.exe
Node.js v22 UDP - C:\node-v22.12.0-win-x64\node.exe
ROTMG Server - Custom rule
```

---

## Directory Structure

### Server Files
```
C:\ROTMG-DEMO\
├── .env                    # Server config - DO NOT OVERWRITE
├── Server.js               # Main server entry point
├── certs\
│   ├── fullchain.pem       # Let's Encrypt cert for quic.eternalconquests.com
│   └── privkey.pem         # EC private key (241 bytes)
├── node_modules\
│   └── @fails-components\
│       ├── webtransport\                           # WebTransport API
│       └── webtransport-transport-http3-quiche\
│           └── build\Release\
│               └── webtransport.node               # Native binary (4.9 MB)
├── startup.bat             # Manual startup script
├── restart_server.bat      # Restart script
└── logs\                   # Server logs

C:\ROTMG-BACKUP\
├── .env.backup             # Backup of working .env
├── fullchain.pem           # Backup of certs
├── privkey.pem
└── RESTORE_INSTRUCTIONS.txt

C:\node-v22.12.0-win-x64\   # Node.js installation
└── node.exe                # 82 MB executable

C:\Users\newadmin\
├── playit.exe              # PlayIt tunnel client
└── AppData\Local\playit_gg\
    └── playit.toml         # PlayIt configuration (contains secret key)
```

### Package Management

**In `package.json`** (safe to npm install):
- express, ws, dotenv, uuid, sql.js, etc.

**NOT in `package.json`** (manually installed):
- `@fails-components/webtransport`
- `@fails-components/webtransport-transport-http3-quiche`

These require a native binary that must be downloaded from GitHub releases.

---

## Recovery Procedures

### WebTransport Recovery (After Accidental npm install)

If WebTransport stopped working after `npm install`:

```bash
# SSH into server
sshpass -p "12345" ssh newadmin@192.168.0.202

# Navigate to project
cd C:\ROTMG-DEMO

# Reinstall WebTransport packages (without scripts to avoid build failure)
npm install @fails-components/webtransport --ignore-scripts
npm install @fails-components/webtransport-transport-http3-quiche --ignore-scripts

# Download prebuilt native binary
cd node_modules\@fails-components\webtransport-transport-http3-quiche
curl -L -o prebuild.tar.gz "https://github.com/fails-components/webtransport/releases/download/v1.5.1/webtransport-transport-http3-quiche-v1.5.1-napi-v6-win32-x64.tar.gz"
tar -xzf prebuild.tar.gz

# Verify binary exists (should be ~5MB)
dir build\Release\webtransport.node

# Restart server
cd C:\ROTMG-DEMO
taskkill /IM node.exe /F
C:\node-v22.12.0-win-x64\node.exe Server.js
```

### .env Recovery

```bash
# From local machine:
./deploy.sh restore-env

# Or manually via SSH:
sshpass -p "12345" ssh newadmin@192.168.0.202 "copy C:\ROTMG-BACKUP\.env.backup C:\ROTMG-DEMO\.env"
```

### Certificate Recovery

```bash
sshpass -p "12345" ssh newadmin@192.168.0.202 "copy C:\ROTMG-BACKUP\fullchain.pem C:\ROTMG-DEMO\certs\"
sshpass -p "12345" ssh newadmin@192.168.0.202 "copy C:\ROTMG-BACKUP\privkey.pem C:\ROTMG-DEMO\certs\"
```

### Firewall Recovery (After Node.js Upgrade)

If you upgraded Node.js and WebTransport stopped working:

```bash
# SSH into server and add firewall rules for new Node path
netsh advfirewall firewall add rule name="Node.js vXX UDP" dir=in action=allow protocol=UDP program="C:\node-vXX.XX.X-win-x64\node.exe"
netsh advfirewall firewall add rule name="Node.js vXX TCP" dir=in action=allow protocol=TCP program="C:\node-vXX.XX.X-win-x64\node.exe"
```

---

## Deployment Commands

### From Local Machine (Mac)

```bash
./deploy.sh client    # Client files only (no restart needed)
./deploy.sh server    # Server files + restart
./deploy.sh all       # Full deploy (excludes .env, certs) + restart
./deploy.sh restart   # Just restart the server
./deploy.sh status    # Check running processes
./deploy.sh logs      # View last 50 lines of logs
./deploy.sh logs-follow # Tail logs continuously
./deploy.sh tunnel    # Get current Cloudflare tunnel URL
./deploy.sh restore-env # Restore .env from backup
./deploy.sh show-env  # Display current server .env
```

### Manual Server Control

```bash
# Kill server
sshpass -p "12345" ssh newadmin@192.168.0.202 "taskkill /IM node.exe /F"

# Start server (background)
sshpass -p "12345" ssh newadmin@192.168.0.202 "wmic process call create \"cmd /c cd /d C:\ROTMG-DEMO && C:\node-v22.12.0-win-x64\node.exe Server.js\""

# Check what's running
sshpass -p "12345" ssh newadmin@192.168.0.202 "tasklist | findstr \"node cloudflared playit\""

# Check ports
sshpass -p "12345" ssh newadmin@192.168.0.202 "netstat -an | findstr \"4000 4433\""
```

---

## Server .env Reference

The server's `.env` contains:
```ini
# Server Configuration
ARTIFICIAL_LATENCY_MS=0
NETWORK_LOGGER_ENABLED=true
NETWORK_LOGGER_VERBOSE=false
FILE_LOGGER_ENABLED=true

# Lag Compensation
LAG_COMPENSATION_ENABLED=true
LAG_COMPENSATION_MAX_REWIND_MS=200
LAG_COMPENSATION_MIN_RTT_MS=50
LAG_COMPENSATION_DEBUG=false

# Movement Validation
MOVEMENT_VALIDATION_ENABLED=true
MOVEMENT_VALIDATION_MAX_SPEED=7.2
MOVEMENT_VALIDATION_TELEPORT_THRESHOLD=3.0

# Collision Validation
COLLISION_VALIDATION_ENABLED=true
COLLISION_VALIDATION_MODE=soft
COLLISION_VALIDATION_MAX_DISTANCE=2.0
COLLISION_VALIDATION_DISTANCE_PER_RTT=0.5
COLLISION_VALIDATION_SUSPICIOUS_THRESHOLD=1.5

# WebTransport/QUIC (critical - must match server paths)
WEBTRANSPORT_ENABLED=true
WEBTRANSPORT_PORT=4433
WEBTRANSPORT_HOST=quic.eternalconquests.com
WEBTRANSPORT_CERT=C:/ROTMG-DEMO/certs/fullchain.pem
WEBTRANSPORT_KEY=C:/ROTMG-DEMO/certs/privkey.pem

# PlayIt.gg tunnel
PLAYIT_HOST=147.185.221.20
PLAYIT_UDP_PORT=10615
```

---

## Troubleshooting

### WebTransport Not Connecting
1. Check UDP port 4433: `netstat -an | findstr UDP | findstr 4433`
2. Check PlayIt is running: `tasklist | findstr playit`
3. Check firewall rules exist for Node v22
4. Check certs exist at `C:\ROTMG-DEMO\certs\`
5. Check `webtransport.node` exists (see WebTransport Recovery)

### Game Not Loading (502 Bad Gateway)
1. Server is down: `./deploy.sh restart`
2. Cloudflare tunnel is down: check `cloudflared.exe` is running

### QUIC_NETWORK_IDLE_TIMEOUT in Browser
1. Firewall blocking UDP (see Firewall Recovery)
2. PlayIt tunnel not running
3. DNS not pointing to PlayIt IP (should be 147.185.221.20)

### Server Crashes on Startup
1. Check logs: `./deploy.sh logs`
2. Missing WebTransport binary: see WebTransport Recovery
3. Certificate issues: see Certificate Recovery

---

## Past Issues & Solutions

### January 2026: npm install Broke WebTransport
**Cause**: Ran `npm install` which deleted manually-installed `@fails-components` packages
**Solution**: Downloaded prebuilt binary from GitHub releases

### December 2025: Firewall Blocked UDP
**Cause**: Firewall rules were for Node v20, but server used Node v22
**Solution**: Added new firewall rules for `C:\node-v22.12.0-win-x64\node.exe`

### December 2025: .env Overwritten
**Cause**: `deploy.sh all` copied local .env to server
**Solution**: Fixed deploy.sh to exclude .env, created backup at `C:\ROTMG-BACKUP\`

---

## Architecture

```
Client Browser
    │
    ├─► wss://eternalconquests.com (WebSocket)
    │       │
    │       └─► Cloudflare Tunnel → localhost:4000 (TCP)
    │
    └─► https://quic.eternalconquests.com:10615 (WebTransport)
            │
            └─► DNS: 147.185.221.20 (PlayIt)
                    │
                    └─► PlayIt Tunnel → localhost:4433 (UDP)
                            │
                            └─► Node.js WebTransport Server
```

**Key Points:**
- WebSocket uses Cloudflare Tunnel (TCP)
- WebTransport uses PlayIt.gg (UDP)
- `quic` subdomain must NOT be proxied through Cloudflare (grey cloud)
- Both require TLS certificates for `quic.eternalconquests.com`
