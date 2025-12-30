# WebTransport/QUIC Setup Guide

This document describes the complete setup for WebTransport (UDP-like transport over QUIC) for the ROTMG-DEMO game server.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WINDOWS SERVER LAPTOP                        │
│                         (192.168.0.202)                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   Game Server   │  │   cloudflared   │  │      playit.exe     │  │
│  │   Port 4000     │  │                 │  │                     │  │
│  │   (TCP/WS)      │  │                 │  │                     │  │
│  ├─────────────────┤  └────────┬────────┘  └──────────┬──────────┘  │
│  │  WebTransport   │           │                      │             │
│  │   Port 4433     │           │                      │             │
│  │   (UDP/QUIC)    │           │                      │             │
│  └─────────────────┘           │                      │             │
└────────────────────────────────┼──────────────────────┼─────────────┘
                                 │                      │
                                 ▼                      ▼
┌────────────────────────────────────────┐  ┌─────────────────────────┐
│         CLOUDFLARE TUNNEL              │  │      PLAYIT.GG          │
│                                        │  │                         │
│  eternalconquests.com → localhost:4000 │  │  147.185.221.20:10615   │
│  (WebSocket over HTTPS)                │  │         ↓               │
│                                        │  │  localhost:4433 (UDP)   │
└────────────────────────────────────────┘  └─────────────────────────┘
```

## Components

### 1. Cloudflare Tunnel (TCP/WebSocket)
- **Purpose**: Routes WebSocket traffic for game state, chat, reliable messages
- **Domain**: `eternalconquests.com`
- **Local Port**: 4000
- **Protocol**: WSS (WebSocket Secure)

### 2. PlayIt.gg Tunnel (UDP/QUIC)
- **Purpose**: Routes WebTransport/QUIC traffic for low-latency updates (bullets, positions)
- **Domain**: `quic.eternalconquests.com`
- **External**: `147.185.221.20:10615`
- **Local Port**: 4433
- **Protocol**: QUIC/HTTP3

## DNS Configuration (Cloudflare)

| Type | Name | Content | Proxy |
|------|------|---------|-------|
| CNAME | @ | (Cloudflare Tunnel) | Yes |
| A | quic | 147.185.221.20 | **No** (grey cloud) |

**Important**: The `quic` subdomain must have proxy OFF (grey cloud) because WebTransport needs direct UDP access.

## TLS Certificate

WebTransport requires TLS. Certificate obtained via Let's Encrypt with DNS-01 challenge:

```bash
# On Mac (with Cloudflare API token)
pip install certbot certbot-dns-cloudflare

# Create credentials file
cat > ~/cloudflare.ini << EOF
dns_cloudflare_api_token = YOUR_CLOUDFLARE_API_TOKEN
EOF
chmod 600 ~/cloudflare.ini

# Issue certificate
certbot certonly \
  --dns-cloudflare \
  --dns-cloudflare-credentials ~/cloudflare.ini \
  -d quic.eternalconquests.com \
  --config-dir ~/certbot-certs \
  --work-dir ~/certbot-certs \
  --logs-dir ~/certbot-certs
```

Certificate files copied to Windows server:
- `C:\ROTMG-DEMO\certs\fullchain.pem`
- `C:\ROTMG-DEMO\certs\privkey.pem`

## PlayIt.gg Configuration

### Dashboard Setup (playit.gg/account/tunnels)

1. **Tunnel Type**: Custom TCP+UDP (or UDP only)
2. **Port Count**: 1
3. **Local Address**: 127.0.0.1
4. **Local Port**: 4433
5. **Region**: Global Anycast

### Agent Setup

```bash
# Download playit.exe to C:\Users\newadmin\
# First time setup:
playit.exe claim

# Start agent:
playit.exe start
```

Config stored at: `C:\Users\newadmin\AppData\Local\playit_gg\playit.toml`

## Server Configuration

### Environment Variables (.env)

```env
# WebTransport/QUIC Settings
WEBTRANSPORT_ENABLED=true
WEBTRANSPORT_PORT=4433
WEBTRANSPORT_HOST=quic.eternalconquests.com
WEBTRANSPORT_CERT=C:/ROTMG-DEMO/certs/fullchain.pem
WEBTRANSPORT_KEY=C:/ROTMG-DEMO/certs/privkey.pem

# PlayIt.gg tunnel config (for reference)
PLAYIT_HOST=147.185.221.20
PLAYIT_UDP_PORT=10615
```

### Required Node.js Packages

```bash
npm install @fails-components/webtransport @fails-components/webtransport-transport-http3-quiche
```

**Note**: Requires Node.js 21.6.1+ (we use 22.12.0)

## Code Files

### Server Side

- `src/network/WebTransportServer.js` - WebTransport server using @fails-components/webtransport
- `Server.js` - Main server, initializes WebTransport in `onListening` callback

### Client Side

- `public/src/network/WebTransportManager.js` - Client WebTransport manager
- `public/src/network/ClientNetworkManager.js` - Integrates WebTransport with game networking

### Client Connection URL

The client auto-detects the WebTransport URL:
```javascript
// In ClientNetworkManager.js
wtUrl = `https://quic.${wsUrl.hostname}:10615/game`;
// Results in: https://quic.eternalconquests.com:10615/game
```

## Message Flow

### TCP (WebSocket via Cloudflare)
- Handshake, authentication
- Chat messages
- Inventory updates
- Any reliable/ordered data

### UDP (WebTransport via PlayIt)
- Player position updates
- Bullet creation/updates
- Enemy positions
- Any latency-sensitive data

## Browser Requirements

WebTransport is supported in:
- Chrome 97+
- Edge 97+
- Opera 83+
- Brave (must enable in brave://flags)

**NOT supported**: Firefox, Safari

## Troubleshooting

### "WebTransport not supported in this browser"
- Check browser version
- For Brave: enable WebTransport in `brave://flags`
- Must be on HTTPS (not HTTP)

### No UDP packets reaching server
1. Check PlayIt tunnel is configured for correct local port (4433)
2. Verify DNS `quic.eternalconquests.com` points to PlayIt IP
3. Ensure Cloudflare proxy is OFF for `quic` subdomain

### Certificate errors
- Certificate must match `quic.eternalconquests.com`
- Certificate must not be expired
- Private key must be accessible by Node.js

### Session accept errors
- Ensure using correct @fails-components/webtransport API
- Session from stream is already accepted, don't call `.accept()`

## Maintenance

### Certificate Renewal
Let's Encrypt certificates expire every 90 days. Renew with:
```bash
certbot renew --config-dir ~/certbot-certs --work-dir ~/certbot-certs --logs-dir ~/certbot-certs
```

Then copy new certs to Windows server and restart.

### PlayIt IP Changes
If you delete/recreate the PlayIt tunnel, the IP may change. Update:
1. Cloudflare DNS A record for `quic`
2. No code changes needed (client uses hostname)
