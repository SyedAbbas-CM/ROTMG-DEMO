#!/bin/bash
# =============================================================================
# ROTMG Server Deploy Script
# =============================================================================
# Target: Windows Laptop at 192.168.0.202
# Server Directory: C:\ROTMG-DEMO
#
# WARNING: NEVER DEPLOY THESE FILES (they contain server-specific settings):
#   - .env (WebTransport ports, cert paths)
#   - certs/ (SSL certificates for QUIC)
#
# If you accidentally overwrote .env, run: ./deploy.sh restore-env
# Backup location on server: C:\ROTMG-BACKUP\
#
# See SERVER_SETUP.md for full documentation.
# =============================================================================

REMOTE="newadmin@192.168.0.202"
PASS="12345"
REMOTE_DIR="C:/ROTMG-DEMO"
BACKUP_DIR="C:/ROTMG-BACKUP"
NODE_PATH="C:\\node-v22.12.0-win-x64\\node.exe"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== ROTMG Quick Deploy ===${NC}"

case "${1:-help}" in
    client)
        echo -e "${GREEN}Deploying client files only...${NC}"
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no public/* "$REMOTE:$REMOTE_DIR/public/"
        echo -e "${GREEN}Client deployed! Just refresh browser.${NC}"
        ;;
    server)
        echo -e "${GREEN}Deploying server files (preserving .env)...${NC}"
        sshpass -p "$PASS" scp -o StrictHostKeyChecking=no Server.js "$REMOTE:$REMOTE_DIR/"
        # NOTE: .env is NOT deployed to preserve server-specific settings (WebTransport certs, ports)
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no src/* "$REMOTE:$REMOTE_DIR/src/" 2>/dev/null
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no common/* "$REMOTE:$REMOTE_DIR/common/"
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no server/* "$REMOTE:$REMOTE_DIR/server/" 2>/dev/null
        echo -e "${YELLOW}Restarting server...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "taskkill /IM node.exe /F 2>nul"
        sleep 2
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "wmic process call create \"cmd /c cd /d $REMOTE_DIR && $NODE_PATH Server.js\""
        sleep 3
        echo -e "${GREEN}Server deployed and restarted!${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting server...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "taskkill /IM node.exe /F 2>nul"
        sleep 2
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "wmic process call create \"cmd /c cd /d $REMOTE_DIR && $NODE_PATH Server.js\""
        sleep 5
        $0 status
        ;;
    logs)
        echo -e "${GREEN}Fetching latest logs...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "powershell -Command \"Get-ChildItem '$REMOTE_DIR\\logs\\server-*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { Write-Host 'File:' \$_.Name; Get-Content \$_ -Tail 50 }\""
        ;;
    logs-follow)
        echo -e "${GREEN}Following logs (Ctrl+C to stop)...${NC}"
        while true; do
            sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "powershell -Command \"Get-ChildItem '$REMOTE_DIR\\logs\\server-*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content -Tail 5\""
            sleep 2
        done
        ;;
    status)
        echo -e "${GREEN}Checking status...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "tasklist | findstr -i \"node cloudflared\" && echo. && powershell -Command \"(Invoke-WebRequest -Uri 'http://127.0.0.1:20241/quicktunnel' -UseBasicParsing -TimeoutSec 2).Content\" 2>nul || echo Tunnel URL not available"
        ;;
    tunnel)
        echo -e "${GREEN}Getting tunnel URL...${NC}"
        URL=$(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "powershell -Command \"(Invoke-WebRequest -Uri 'http://127.0.0.1:20241/quicktunnel' -UseBasicParsing).Content\"" 2>/dev/null | grep -o '"hostname":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}https://$URL${NC}"
        ;;
    all)
        echo -e "${GREEN}Full deploy (excluding node_modules, .env, certs)...${NC}"
        # IMPORTANT: Excludes .env and certs/ to preserve server-specific WebTransport settings
        tar --exclude='node_modules' --exclude='.git' --exclude='.env' --exclude='certs' \
            --exclude='ml/models/*.pth' --exclude='ml/__pycache__' --exclude='logs/*.log' -cf - . | \
            sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "cd $REMOTE_DIR && tar -xf -"
        echo -e "${YELLOW}Restarting server...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "taskkill /IM node.exe /F 2>nul"
        sleep 2
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "wmic process call create \"cmd /c cd /d $REMOTE_DIR && $NODE_PATH Server.js\""
        sleep 5
        $0 status
        ;;
    start-tunnel)
        echo -e "${GREEN}Starting Cloudflare tunnel...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "schtasks /Run /TN CloudflareTunnel"
        sleep 5
        $0 tunnel
        ;;
    restore-env)
        echo -e "${YELLOW}Restoring .env from backup...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "copy $BACKUP_DIR\\.env.backup $REMOTE_DIR\\.env /Y"
        echo -e "${GREEN}.env restored from C:\\ROTMG-BACKUP\\.env.backup${NC}"
        echo -e "${YELLOW}Restarting server to apply changes...${NC}"
        $0 restart
        ;;
    backup-env)
        echo -e "${YELLOW}Creating backup of current .env...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "mkdir $BACKUP_DIR 2>nul & copy $REMOTE_DIR\\.env $BACKUP_DIR\\.env.backup /Y"
        echo -e "${GREEN}Backup saved to C:\\ROTMG-BACKUP\\.env.backup${NC}"
        ;;
    show-env)
        echo -e "${GREEN}Current server .env:${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "type $REMOTE_DIR\\.env"
        ;;
    *)
        echo "Usage: ./deploy.sh <command>"
        echo ""
        echo -e "${GREEN}Deploy Commands:${NC}"
        echo "  client       - Deploy only public/ folder (no restart needed)"
        echo "  server       - Deploy server files and restart (preserves .env)"
        echo "  all          - Full deploy and restart (excludes .env, certs)"
        echo ""
        echo -e "${GREEN}Server Commands:${NC}"
        echo "  restart      - Just restart the server"
        echo "  status       - Check if node and cloudflared are running"
        echo "  logs         - Show latest 50 lines of server logs"
        echo "  logs-follow  - Tail logs continuously"
        echo ""
        echo -e "${GREEN}Tunnel Commands:${NC}"
        echo "  tunnel       - Get current tunnel URL"
        echo "  start-tunnel - Start the Cloudflare tunnel"
        echo ""
        echo -e "${YELLOW}Recovery Commands:${NC}"
        echo "  restore-env  - Restore .env from backup (if you broke it)"
        echo "  backup-env   - Create new backup of current .env"
        echo "  show-env     - Display current server .env contents"
        echo ""
        echo -e "${RED}WARNING: .env and certs/ are NEVER deployed to preserve WebTransport settings.${NC}"
        echo "See SERVER_SETUP.md for full documentation."
        ;;
esac
