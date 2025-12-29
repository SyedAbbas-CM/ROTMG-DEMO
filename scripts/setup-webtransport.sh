#!/bin/bash
# setup-webtransport.sh
# Script to set up WebTransport/QUIC for the ROTMG game server
#
# Prerequisites:
# - PlayIt.gg Premium account with custom domain support
# - Domain configured in Cloudflare (eternalconquests.com)
# - Node.js 21.6.1+ on the server

set -e

echo "==========================================="
echo "WebTransport/QUIC Setup for ROTMG Server"
echo "==========================================="
echo ""

# Configuration
DOMAIN="${WEBTRANSPORT_DOMAIN:-quic.eternalconquests.com}"
PLAYIT_IP="${PLAYIT_IP:-147.185.221.20}"
PLAYIT_PORT="${PLAYIT_PORT:-10615}"
CERT_DIR="${CERT_DIR:-./certs}"

echo "Configuration:"
echo "  Domain: $DOMAIN"
echo "  PlayIt IP: $PLAYIT_IP"
echo "  PlayIt Port: $PLAYIT_PORT"
echo "  Certificate Directory: $CERT_DIR"
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Please install Node.js 21.6.1+"
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//')
NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
if [ "$NODE_MAJOR" -lt 21 ]; then
    echo "WARNING: Node.js $NODE_VERSION detected. WebTransport requires 21.6.1+"
    echo "Please upgrade Node.js or the WebTransport server may not work correctly."
fi

echo "  Node.js: $(node -v)"

# Step 2: Create certificate directory
echo ""
echo "Step 2: Creating certificate directory..."
mkdir -p "$CERT_DIR"
echo "  Created: $CERT_DIR"

# Step 3: Install certbot if needed
echo ""
echo "Step 3: Checking certbot installation..."

if ! command -v certbot &> /dev/null; then
    echo "  certbot not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install certbot
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y certbot
    else
        echo "  ERROR: Please install certbot manually"
        exit 1
    fi
fi
echo "  certbot: $(certbot --version 2>&1 | head -1)"

# Step 4: DNS Configuration Instructions
echo ""
echo "==========================================="
echo "Step 4: DNS Configuration Required"
echo "==========================================="
echo ""
echo "Before getting the certificate, you need to configure DNS:"
echo ""
echo "Option A - Point subdomain to PlayIt (Recommended):"
echo "  1. Go to your Cloudflare dashboard"
echo "  2. Select eternalconquests.com"
echo "  3. Go to DNS settings"
echo "  4. Add an A record:"
echo "     Name: quic"
echo "     Content: $PLAYIT_IP"
echo "     Proxy: OFF (DNS only - grey cloud)"
echo "     TTL: Auto"
echo ""
echo "Option B - Use PlayIt's nameservers:"
echo "  1. Go to https://playit.gg/account/settings/domains/add-external"
echo "  2. Add quic.eternalconquests.com"
echo "  3. Follow PlayIt's instructions to add NS records"
echo ""
echo "IMPORTANT: The Cloudflare proxy must be OFF for WebTransport/QUIC!"
echo "WebTransport uses UDP which Cloudflare's proxy doesn't support."
echo ""
read -p "Press Enter once DNS is configured... "

# Step 5: Get Let's Encrypt Certificate
echo ""
echo "==========================================="
echo "Step 5: Getting Let's Encrypt Certificate"
echo "==========================================="
echo ""
echo "We'll use DNS-01 challenge (works without opening ports)"
echo ""

# Check if Cloudflare credentials are available
if [ -f ~/.cloudflare/credentials ]; then
    echo "Cloudflare credentials found. Using DNS challenge..."

    sudo certbot certonly \
        --dns-cloudflare \
        --dns-cloudflare-credentials ~/.cloudflare/credentials \
        -d "$DOMAIN" \
        --cert-path "$CERT_DIR/cert.pem" \
        --key-path "$CERT_DIR/key.pem" \
        --fullchain-path "$CERT_DIR/fullchain.pem"
else
    echo "No Cloudflare credentials found."
    echo ""
    echo "Option 1: Set up Cloudflare DNS credentials:"
    echo "  1. Get your Cloudflare API token from:"
    echo "     https://dash.cloudflare.com/profile/api-tokens"
    echo "  2. Create ~/.cloudflare/credentials with:"
    echo "     dns_cloudflare_api_token = YOUR_API_TOKEN"
    echo ""
    echo "Option 2: Use manual DNS challenge:"
    echo "  Running certbot with manual DNS verification..."
    echo ""

    sudo certbot certonly \
        --manual \
        --preferred-challenges dns \
        -d "$DOMAIN"
fi

# Step 6: Copy certificates to project
echo ""
echo "Step 6: Copying certificates..."

# Find the certificate location
LETSENCRYPT_DIR="/etc/letsencrypt/live/$DOMAIN"
if [ -d "$LETSENCRYPT_DIR" ]; then
    sudo cp "$LETSENCRYPT_DIR/fullchain.pem" "$CERT_DIR/cert.pem"
    sudo cp "$LETSENCRYPT_DIR/privkey.pem" "$CERT_DIR/key.pem"
    sudo chown $(whoami) "$CERT_DIR"/*.pem
    chmod 600 "$CERT_DIR/key.pem"
    chmod 644 "$CERT_DIR/cert.pem"
    echo "  Certificates copied to $CERT_DIR"
else
    echo "  WARNING: Certificates not found at $LETSENCRYPT_DIR"
    echo "  Please copy your certificates manually to $CERT_DIR"
fi

# Step 7: Install npm packages
echo ""
echo "Step 7: Installing WebTransport npm packages..."
npm install @fails-components/webtransport @fails-components/webtransport-transport-http3-quiche

# Step 8: Update .env file
echo ""
echo "Step 8: Updating .env configuration..."

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    # Backup existing .env
    cp "$ENV_FILE" "${ENV_FILE}.backup"
fi

# Add WebTransport settings
cat >> "$ENV_FILE" << EOF

# WebTransport/QUIC Settings
WEBTRANSPORT_ENABLED=true
WEBTRANSPORT_PORT=4433
WEBTRANSPORT_HOST=$DOMAIN
WEBTRANSPORT_CERT=$CERT_DIR/cert.pem
WEBTRANSPORT_KEY=$CERT_DIR/key.pem
EOF

echo "  Updated $ENV_FILE with WebTransport settings"

# Step 9: PlayIt Configuration
echo ""
echo "==========================================="
echo "Step 9: PlayIt.gg Configuration"
echo "==========================================="
echo ""
echo "Make sure PlayIt is configured to forward UDP traffic:"
echo ""
echo "1. Go to https://playit.gg/account"
echo "2. Select your tunnel"
echo "3. Add a new allocation:"
echo "   - Protocol: UDP"
echo "   - Local Port: 4433"
echo "   - Public Port: (Note the assigned port)"
echo ""
echo "4. Update .env with the public port if different from 10615"
echo ""

# Final summary
echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "Configuration:"
echo "  WebTransport URL: https://$DOMAIN:4433/game"
echo "  Certificate: $CERT_DIR/cert.pem"
echo "  Key: $CERT_DIR/key.pem"
echo ""
echo "Next steps:"
echo "1. Verify DNS propagation: nslookup $DOMAIN"
echo "2. Upload to server: sync certificates to Windows server"
echo "3. Start server with: node server.js"
echo "4. Test connection in browser"
echo ""
echo "Client connection URL:"
echo "  WebTransport: https://$DOMAIN:4433/game"
echo "  WebSocket:    wss://eternalconquests.com (via Cloudflare tunnel)"
echo ""
