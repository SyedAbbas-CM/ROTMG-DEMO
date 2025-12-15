#!/bin/bash

# ROTMG Game Server Deployment Script
# Deploys the game backend to remote servers

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVER_202_IP="192.168.0.202"
SERVER_202_USER="admin"
SERVER_202_NAME="Server-202"

SERVER_203_IP="192.168.0.203"
SERVER_203_USER="scrapenode"
SERVER_203_NAME="Server-203"

DEPLOY_DIR="/opt/ROTMG-DEMO"
BACKUP_DIR="/opt/ROTMG-DEMO-backup"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ROTMG Game Server Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to deploy to a server
deploy_to_server() {
    local SERVER_IP=$1
    local SERVER_USER=$2
    local SERVER_NAME=$3
    local PASSWORD=$4

    echo -e "\n${YELLOW}Deploying to ${SERVER_NAME} (${SERVER_USER}@${SERVER_IP})...${NC}"

    # Test SSH connection
    echo "Testing SSH connection..."
    if ! sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${SERVER_USER}@${SERVER_IP} "echo 'Connection successful'"; then
        echo -e "${RED}Failed to connect to ${SERVER_NAME}${NC}"
        return 1
    fi

    # Create backup of existing deployment
    echo "Creating backup..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
        if [ -d /opt/ROTMG-DEMO ]; then
            sudo rm -rf /opt/ROTMG-DEMO-backup 2>/dev/null || true
            sudo cp -r /opt/ROTMG-DEMO /opt/ROTMG-DEMO-backup
            echo "Backup created at /opt/ROTMG-DEMO-backup"
        fi
ENDSSH

    # Create deployment directory
    echo "Preparing deployment directory..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
        sudo mkdir -p /opt/ROTMG-DEMO
        sudo chown -R $(whoami):$(whoami) /opt/ROTMG-DEMO
ENDSSH

    # Sync files (exclude node_modules, build artifacts, logs)
    echo "Syncing files..."
    sshpass -p "$PASSWORD" rsync -avz --progress \
        --exclude 'node_modules' \
        --exclude 'build' \
        --exclude 'logs' \
        --exclude '.git' \
        --exclude '.env' \
        --exclude '*.log' \
        ./ ${SERVER_USER}@${SERVER_IP}:${DEPLOY_DIR}/

    # Install dependencies and build native modules
    echo "Installing dependencies and building native modules..."
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
        cd /opt/ROTMG-DEMO

        # Install Node.js if not present
        if ! command -v node &> /dev/null; then
            echo "Installing Node.js..."
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt-get install -y nodejs build-essential
        fi

        # Install dependencies
        npm install --production

        # Build native modules
        npm install --save-dev node-gyp
        npx node-gyp rebuild

        echo "Deployment complete!"
ENDSSH

    echo -e "${GREEN}Successfully deployed to ${SERVER_NAME}${NC}"
}

# Main deployment flow
echo -e "\n${YELLOW}This script will deploy to:${NC}"
echo "  1. ${SERVER_202_NAME} (${SERVER_202_USER}@${SERVER_202_IP})"
echo "  2. ${SERVER_203_NAME} (${SERVER_203_USER}@${SERVER_203_IP})"
echo ""

read -p "Enter password for servers: " -s PASSWORD
echo ""

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo -e "${YELLOW}Installing sshpass...${NC}"
    brew install sshpass 2>/dev/null || {
        echo -e "${RED}Please install sshpass manually: brew install sshpass${NC}"
        exit 1
    }
fi

# Deploy to both servers
deploy_to_server ${SERVER_202_IP} ${SERVER_202_USER} ${SERVER_202_NAME} ${PASSWORD}
deploy_to_server ${SERVER_203_IP} ${SERVER_203_USER} ${SERVER_203_NAME} ${PASSWORD}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start the server on each machine:"
echo "  ssh ${SERVER_202_USER}@${SERVER_202_IP}"
echo "  cd /opt/ROTMG-DEMO && npm start"
echo ""
