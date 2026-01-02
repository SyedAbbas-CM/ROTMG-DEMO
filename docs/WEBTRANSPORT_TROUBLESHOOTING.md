# WebTransport Troubleshooting Guide

This document covers common issues with the WebTransport/QUIC setup and how to resolve them.

## Issue 1: WebTransport Package Missing Native Binary

### Symptoms
```
Cannot find module '../build/Release/webtransport.node'
```
Server starts but port 4433 doesn't listen, or WebTransport fails silently.

### Cause
The `@fails-components/webtransport-transport-http3-quiche` package requires a native binary that either:
1. Wasn't downloaded (if installed with `--ignore-scripts`)
2. Failed to compile (requires Visual Studio on Windows)
3. Was deleted when running `npm install`

### Diagnosis
```bash
# Check if binary exists
dir C:\ROTMG-DEMO\node_modules\@fails-components\webtransport-transport-http3-quiche\build\Release\
# Should contain: webtransport.node (~5MB)
```

### Solution
Download prebuilt binary from GitHub releases:
```bash
cd C:\ROTMG-DEMO\node_modules\@fails-components\webtransport-transport-http3-quiche
curl -L -o prebuild.tar.gz "https://github.com/fails-components/webtransport/releases/download/v1.5.1/webtransport-transport-http3-quiche-v1.5.1-napi-v6-win32-x64.tar.gz"
tar -xzf prebuild.tar.gz
```

### Prevention
- Never run `npm install` without `--ignore-scripts` unless you have Visual Studio
- The webtransport packages are NOT in package.json - they're manually installed
- Keep a backup of node_modules or document the manual install steps

---

## Issue 2: Windows Firewall Blocking Node.js

### Symptoms
- Server logs show "WebTransport server started on port 4433"
- `netstat` shows UDP 4433 listening
- Client gets `QUIC_NETWORK_IDLE_TIMEOUT`
- No connection attempts visible in server logs

### Cause
Windows Firewall rules were configured for an OLD version of Node.js:
- Rules existed for `C:\users\newadmin\node-v20.18.0-win-x64\node.exe`
- Server runs with `C:\node-v22.12.0-win-x64\node.exe`
- UDP traffic was blocked because the executable path didn't match

### Diagnosis
```bash
# Check firewall rules for node
netsh advfirewall firewall show rule name=all dir=in | findstr -i node

# Check the program path in existing rules
netsh advfirewall firewall show rule name="node.exe" verbose
```

### Solution
Add firewall rules for the correct Node.js version:
```bash
netsh advfirewall firewall add rule name="Node.js v22 UDP" dir=in action=allow protocol=UDP program="C:\node-v22.12.0-win-x64\node.exe"
netsh advfirewall firewall add rule name="Node.js v22 TCP" dir=in action=allow protocol=TCP program="C:\node-v22.12.0-win-x64\node.exe"
```

### Prevention
- When upgrading Node.js, update firewall rules
- Use program path rules, not port-based rules (more secure)

---

## Issue 3: PlayIt Tunnel Not Running

### Symptoms
- WebTransport connection times out
- PlayIt.exe not in task list

### Diagnosis
```bash
tasklist | findstr playit
```

### Solution
Start PlayIt:
```bash
wmic process call create "C:\Users\newadmin\playit.exe"
```

Or run interactively to see status:
```bash
cd C:\Users\newadmin
playit.exe
```

PlayIt output should show:
```
tunnel running, 1 tunnels registered
TUNNELS
wildfire-tanzania.gl.at.ply.gg:10615 => 127.0.0.1:4433 (proto: Both, port count: 1)
```

---

## Issue 4: npm install Corrupts Manual Packages

### Symptoms
After running `npm install`, WebTransport stops working.

### Cause
The `@fails-components/webtransport` packages are manually installed (not in package.json). Running `npm install`:
1. May delete packages not in package.json
2. May overwrite with incompatible versions
3. Removes prebuilt native binaries

### Solution
1. Reinstall the packages:
```bash
npm install @fails-components/webtransport --ignore-scripts
npm install @fails-components/webtransport-transport-http3-quiche --ignore-scripts
```

2. Download prebuilt binary (see Issue 1)

### Prevention
- Add packages to package.json if they should persist
- Or use `--ignore-scripts` and manually manage binaries
- Document the manual installation process

---

## Issue 5: Port 4433 Not Listening

### Symptoms
```bash
netstat -an | findstr 4433
# Returns nothing
```

### Possible Causes

1. **Missing native binary** - See Issue 1
2. **Server crashed** - Check if node.exe is running
3. **Port conflict** - Another process using 4433
4. **WEBTRANSPORT_ENABLED=false** - Check .env

### Diagnosis
```bash
# Is node running?
tasklist | findstr node

# Is something else on 4433?
netstat -ano | findstr 4433

# Check .env
type C:\ROTMG-DEMO\.env | findstr WEBTRANSPORT
```

### Solution
Ensure .env has:
```
WEBTRANSPORT_ENABLED=true
WEBTRANSPORT_PORT=4433
```

---

## Issue 6: Certificate Mismatch

### Symptoms
- Client shows certificate errors
- QUIC handshake fails

### Diagnosis
```javascript
// Check certificate details
const crypto = require('crypto');
const fs = require('fs');
const x509 = new crypto.X509Certificate(fs.readFileSync('C:/ROTMG-DEMO/certs/fullchain.pem'));
console.log('Subject:', x509.subject);
console.log('SubjectAltName:', x509.subjectAltName);
console.log('Valid to:', x509.validTo);
```

Should show:
- Subject: CN=quic.eternalconquests.com
- SubjectAltName: DNS:quic.eternalconquests.com
- Valid date in the future

### Solution
If certificate is wrong or expired, generate new one:
```bash
certbot certonly --dns-cloudflare --dns-cloudflare-credentials ~/cloudflare.ini -d quic.eternalconquests.com
```

Then copy to Windows server.

---

## Issue 7: DNS Pointing to Wrong IP

### Symptoms
- Client connects but times out
- nslookup shows wrong IP

### Diagnosis
```bash
nslookup quic.eternalconquests.com
# Should return: 147.185.221.20 (PlayIt IP)

nslookup wildfire-tanzania.gl.at.ply.gg
# Should return same IP
```

### Solution
Update Cloudflare DNS:
- Type: A
- Name: quic
- Content: 147.185.221.20
- Proxy: OFF (grey cloud - important!)

---

## Issue 8: Cloudflare Proxy Enabled for QUIC Subdomain

### Symptoms
- DNS resolves to Cloudflare IPs (188.114.x.x) instead of PlayIt IP
- WebTransport fails because Cloudflare doesn't proxy UDP

### Diagnosis
```bash
dig +short quic.eternalconquests.com
# Should be 147.185.221.20, NOT 188.114.x.x
```

### Solution
In Cloudflare dashboard:
1. Go to DNS settings
2. Find the `quic` A record
3. Click the orange cloud to turn it grey (proxy OFF)

---

## Debugging Commands Reference

### Check All Services
```bash
tasklist | findstr -i "node playit cloudflared"
netstat -an | findstr "4000 4433"
```

### Check Server Logs
```bash
type C:\ROTMG-DEMO\logs\server-YYYY-MM-DD.log | findstr -i webtransport
```

### Check PlayIt Status
Run `playit.exe` interactively - it shows tunnel status every few seconds.

### Test UDP Connectivity
From external machine:
```bash
echo "test" | nc -u -w2 147.185.221.20 10615
```

### Restart Everything
```bash
# Stop all
taskkill /IM node.exe /F
taskkill /IM playit.exe /F

# Start PlayIt
wmic process call create "C:\Users\newadmin\playit.exe"

# Start Server
wmic process call create "cmd /c cd /d C:\ROTMG-DEMO && C:\node-v22.12.0-win-x64\node.exe Server.js"
```

---

## Architecture Reminder

```
Client Browser
    │
    ├─► wss://eternalconquests.com (WebSocket)
    │       │
    │       └─► Cloudflare Tunnel → localhost:4000
    │
    └─► https://quic.eternalconquests.com:10615 (WebTransport)
            │
            └─► DNS: 147.185.221.20 (PlayIt)
                    │
                    └─► PlayIt Tunnel → localhost:4433
                            │
                            └─► Node.js WebTransport Server
```

Key points:
- WebSocket uses Cloudflare Tunnel (TCP)
- WebTransport uses PlayIt.gg (UDP)
- quic subdomain must NOT be proxied through Cloudflare
- Both use TLS certificates for quic.eternalconquests.com
