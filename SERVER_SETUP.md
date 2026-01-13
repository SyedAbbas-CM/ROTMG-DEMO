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
- `ROTMGServerStartup` - Runs `C:\ROTMG-DEMO\start_all.bat` on login

### Auto-Start Script: `C:\ROTMG-DEMO\start_all.bat`
```batch
@echo off
echo Starting ROTMG Server Services...

:: Start PlayIt tunnel (if not already running)
tasklist /FI "IMAGENAME eq playit.exe" | find /I "playit.exe" >nul
if errorlevel 1 (
    echo Starting PlayIt...
    start "" "C:\Users\newadmin\playit.exe"
    timeout /t 3 /nobreak >nul
) else (
    echo PlayIt already running
)

:: Start Cloudflared tunnel (if not already running)
tasklist /FI "IMAGENAME eq cloudflared.exe" | find /I "cloudflared.exe" >nul
if errorlevel 1 (
    echo Starting Cloudflared...
    start "" "C:\Users\newadmin\cloudflared.exe" tunnel run
    timeout /t 3 /nobreak >nul
) else (
    echo Cloudflared already running
)

:: Start Node server (if not already running)
tasklist /FI "IMAGENAME eq node.exe" | find /I "node.exe" >nul
if errorlevel 1 (
    echo Starting Node server...
    cd /d C:\ROTMG-DEMO
    start "" "C:\node-v22.12.0-win-x64\node.exe" Server.js
) else (
    echo Node server already running
)

echo All services started!
```

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

### Quick Recovery After Laptop Restart

If domain shows Error 1033 or 502, services may be down. Check and restart:

```bash
# Check what's running
sshpass -p "12345" ssh newadmin@192.168.0.202 "tasklist | findstr -i \"node playit cloudflared\""

# Start cloudflared (if not running)
sshpass -p "12345" ssh newadmin@192.168.0.202 "wmic process call create \"C:\\Users\\newadmin\\cloudflared.exe tunnel run\""

# Start node server (if not running)
sshpass -p "12345" ssh newadmin@192.168.0.202 "wmic process call create \"cmd /c cd /d C:\\ROTMG-DEMO && C:\\node-v22.12.0-win-x64\\node.exe Server.js\""

# Verify domain works
curl -s -o /dev/null -w "%{http_code}" https://eternalconquests.com/
```

**Note**: The `start_all.bat` script works when run locally on the laptop but the `start` command doesn't work reliably over SSH. Use `wmic process call create` for remote starts.

The scheduled task `ROTMGServerStartup` auto-starts everything on login (works because it runs locally, not over SSH).

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

### January 13, 2026: WebTransport QUIC_NETWORK_IDLE_TIMEOUT

**Symptom**: Browser showed `ERR_QUIC_PROTOCOL_ERROR.QUIC_NETWORK_IDLE_TIMEOUT` with `num_undecryptable_packets: 0`

**What We Investigated (but wasn't the problem):**
1. PlayIt tunnel config - Dashboard showed correct setup (127.0.0.1:4433, TCP+UDP)
2. DNS resolution - `quic.eternalconquests.com` correctly resolved to PlayIt IP (147.185.221.20)
3. PlayIt service status - Was running and connected ("agent has 1 tunnels")
4. Certificate validity - Certs existed at `C:\ROTMG-DEMO\certs\`
5. Port binding - `netstat` showed UDP 4433 was listening
6. Scheduled task config - Services were actually running

**The Key Clue**: `num_undecryptable_packets: 0` meant NO packets were reaching the server at all, not a crypto/cert issue.

**The Actual Problem**: Windows Firewall was blocking incoming UDP traffic on port 4433. No firewall rules existed specifically for port 4433 (only for the Node.exe executable, which may not have been enough).

**The Fix**:
```batch
netsh advfirewall firewall add rule name="ROTMG WebTransport UDP" dir=in action=allow protocol=UDP localport=4433
netsh advfirewall firewall add rule name="ROTMG WebTransport TCP" dir=in action=allow protocol=TCP localport=4433
```

**Why It Was Confusing**:
- PlayIt dashboard showed traffic (it was receiving packets externally)
- The tunnel config looked perfect
- UDP port was "listening" locally but firewall silently dropped incoming packets from localhost (PlayIt forwards as localhost)
- We spent time checking DNS, certs, PlayIt config - none were the issue

**Lesson Learned**: When debugging "no packets received" issues, check Windows Firewall rules FIRST. Add explicit port-based rules, not just executable-based rules.

---

### January 10, 2026: Laptop Restart - Cloudflared Not Running

**Symptom**: Domain `eternalconquests.com` returned Cloudflare Error 1033 (Argo Tunnel error)

**What Happened**:
1. Laptop restarted (sleep/power cycle)
2. PlayIt auto-started via Windows Services (was configured correctly)
3. Node server auto-started via scheduled task
4. **Cloudflared did NOT auto-start** - no scheduled task existed

**Initial Wrong Assumptions**:
- Thought port was wrong (changed 4000 to 3000, then back)
- Thought PlayIt tunnel was the issue
- Tried restarting PlayIt multiple times

**Root Cause**: `cloudflared.exe` was not configured to auto-start. It was only running from a previous manual session.

**Fix Applied**:
```bash
# SSH and manually start cloudflared
sshpass -p "12345" ssh newadmin@192.168.0.202 "C:\Users\newadmin\cloudflared.exe tunnel run"
```

**Permanent Fix**: Created `C:\ROTMG-DEMO\start_all.bat` that starts all services (see Auto-Start section below)

**Lesson Learned**: After laptop restart, check ALL three services:
1. `node.exe` - Game server
2. `playit.exe` - UDP tunnel for WebTransport
3. `cloudflared.exe` - TCP tunnel for WebSocket

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
