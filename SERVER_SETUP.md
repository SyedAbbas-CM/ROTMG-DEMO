# ROTMG Server Setup & Recovery Guide

## Server Location
- **Windows Laptop**: `newadmin@192.168.0.202`
- **Password**: `12345`
- **Server Directory**: `C:\ROTMG-DEMO`
- **Node.js**: `C:\node-v22.12.0-win-x64\node.exe`

---

## What Broke (December 31, 2025)

### Root Cause
The `deploy.sh` script had multiple issues:
1. **Wrong directory**: Used `C:/Users/newadmin/rotmg-server` instead of `C:\ROTMG-DEMO`
2. **.env overwriting**: `./deploy.sh all` copied local .env to server, wiping WebTransport settings
3. **Failed restart**: Used non-existent scheduled task instead of WMIC

### What Was Lost
Server-specific settings in `.env`:
```
WEBTRANSPORT_ENABLED=true
WEBTRANSPORT_PORT=4433
WEBTRANSPORT_CERT=C:/ROTMG-DEMO/certs/fullchain.pem
WEBTRANSPORT_KEY=C:/ROTMG-DEMO/certs/privkey.pem
```

### Result
- WebTransport/QUIC stopped working
- Client fell back to WebSocket-only mode
- Binary protocol was disabled

---

## Server Configuration

### Ports
| Service | Protocol | Port | Notes |
|---------|----------|------|-------|
| HTTP/WebSocket | TCP | 4000 | Game server, tunneled via Cloudflare |
| WebTransport | UDP | 4433 | QUIC, tunneled via PlayIt.gg (external 10615) |

### Tunneling
- **Cloudflare Tunnel**: `eternalconquests.com` → localhost:4000
- **PlayIt.gg**: `quic.eternalconquests.com:10615` → localhost:4433

### Critical Files on Server
```
C:\ROTMG-DEMO\
├── .env                    # Server config - DO NOT OVERWRITE
├── certs\
│   ├── fullchain.pem       # Let's Encrypt cert for quic.eternalconquests.com
│   └── privkey.pem         # Private key
├── Server.js               # Main server entry point
└── restart_server.bat      # Uses WMIC for background start

C:\ROTMG-BACKUP\
├── .env.backup             # Backup of working .env
├── fullchain.pem           # Backup of certs
├── privkey.pem
└── RESTORE_INSTRUCTIONS.txt
```

---

## Recovery Procedures

### If .env Gets Corrupted
```bash
# From local machine:
./deploy.sh restore-env

# Or manually via SSH:
sshpass -p "12345" ssh newadmin@192.168.0.202 "copy C:\ROTMG-BACKUP\.env.backup C:\ROTMG-DEMO\.env"
```

### If Server Won't Start
```bash
# Kill and restart:
./deploy.sh restart

# Or manually:
sshpass -p "12345" ssh newadmin@192.168.0.202 "taskkill /IM node.exe /F"
sshpass -p "12345" ssh newadmin@192.168.0.202 "wmic process call create \"cmd /c cd /d C:\ROTMG-DEMO && C:\node-v22.12.0-win-x64\node.exe Server.js\""
```

### Check Server Status
```bash
./deploy.sh status

# Or manually check ports:
sshpass -p "12345" ssh newadmin@192.168.0.202 "tasklist | findstr node && netstat -ano | findstr \"4433 4000\""
```

---

## Deployment Rules

### NEVER Deploy These Files
- `.env` - Contains server-specific paths and settings
- `certs/` - SSL certificates are server-specific

### Safe to Deploy
- `Server.js` - Main server code
- `src/` - Server-side modules
- `common/` - Shared protocol code
- `public/` - Client-side code

### Deployment Commands
```bash
./deploy.sh client    # Client files only (no restart)
./deploy.sh server    # Server files only (restarts server)
./deploy.sh all       # Full deploy (excludes .env and certs)
./deploy.sh restart   # Just restart the server
```

---

## Server .env Reference

The server's `.env` should contain:
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
1. Check server is running: `./deploy.sh status`
2. Check UDP port 4433 is listening
3. Check PlayIt.gg is running on server
4. Verify certs exist at `C:\ROTMG-DEMO\certs\`

### Game Not Loading (502 Bad Gateway)
1. Server is down - run `./deploy.sh restart`
2. Cloudflare tunnel is down - check `cloudflared.exe` is running

### Render Crash (NaN bullet positions)
This is a client-side bug where binary protocol sends invalid bullet data.
Check console for `[RENDER] Skipping bullet with invalid position` warnings.
