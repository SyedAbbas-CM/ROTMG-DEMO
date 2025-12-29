#!/bin/bash
# Quick deploy script for ROTMG server to Windows laptop

REMOTE="newadmin@192.168.0.202"
PASS="12345"
REMOTE_DIR="C:/Users/newadmin/rotmg-server"

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
        echo -e "${GREEN}Deploying server files...${NC}"
        sshpass -p "$PASS" scp -o StrictHostKeyChecking=no Server.js "$REMOTE:$REMOTE_DIR/"
        sshpass -p "$PASS" scp -o StrictHostKeyChecking=no .env "$REMOTE:$REMOTE_DIR/"
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no src/* "$REMOTE:$REMOTE_DIR/src/" 2>/dev/null
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no common/* "$REMOTE:$REMOTE_DIR/common/"
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no server/* "$REMOTE:$REMOTE_DIR/server/" 2>/dev/null
        echo -e "${YELLOW}Restarting server...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "taskkill /IM node.exe /F 2>nul & schtasks /Run /TN GameServer"
        sleep 3
        echo -e "${GREEN}Server deployed and restarted!${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting server...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "taskkill /IM node.exe /F 2>nul & schtasks /Run /TN GameServer"
        sleep 2
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
        echo -e "${GREEN}Full deploy (excluding node_modules)...${NC}"
        tar --exclude='node_modules' --exclude='.git' --exclude='ml/models/*.pth' --exclude='ml/__pycache__' --exclude='logs/*.log' -cf - . | \
            sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "cd $REMOTE_DIR && tar -xf -"
        echo -e "${YELLOW}Restarting server...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "taskkill /IM node.exe /F 2>nul & schtasks /Run /TN GameServer"
        sleep 3
        $0 status
        ;;
    start-tunnel)
        echo -e "${GREEN}Starting Cloudflare tunnel...${NC}"
        sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no "$REMOTE" "schtasks /Run /TN CloudflareTunnel"
        sleep 5
        $0 tunnel
        ;;
    *)
        echo "Usage: ./deploy.sh <command>"
        echo ""
        echo "Commands:"
        echo "  client       - Deploy only public/ folder (no restart needed)"
        echo "  server       - Deploy server files and restart"
        echo "  restart      - Just restart the server"
        echo "  logs         - Show latest 50 lines of server logs"
        echo "  logs-follow  - Tail logs continuously"
        echo "  status       - Check if node and cloudflared are running"
        echo "  tunnel       - Get current tunnel URL"
        echo "  start-tunnel - Start the Cloudflare tunnel"
        echo "  all          - Full deploy and restart"
        echo ""
        echo "Quick workflow:"
        echo "  1. Make changes locally"
        echo "  2. ./deploy.sh client   # For frontend-only changes"
        echo "  3. ./deploy.sh server   # For backend changes"
        echo "  4. Refresh browser"
        ;;
esac
