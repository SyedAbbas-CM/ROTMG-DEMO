#!/bin/bash
# Remote log monitoring for ROTMG-DEMO server
# Usage: ./remote-logs.sh [tail|errors|search <pattern>|watch]

SERVER="newadmin@192.168.0.202"
PASS="12345"
LOG_PATH="C:/ROTMG-DEMO/server.log"

ssh_cmd() {
    sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SERVER" "$1"
}

case "$1" in
    tail|"")
        # Default: show last 50 lines
        LINES="${2:-50}"
        ssh_cmd "powershell -Command \"Get-Content $LOG_PATH -Tail $LINES\""
        ;;
    errors)
        # Show only error lines
        LINES="${2:-100}"
        ssh_cmd "powershell -Command \"Get-Content $LOG_PATH -Tail $LINES | Select-String -Pattern 'error|ERROR|Error|exception|Exception|EXCEPTION|failed|Failed|FAILED|crash|Crash|CRASH'\""
        ;;
    search)
        # Search for pattern
        PATTERN="$2"
        LINES="${3:-200}"
        if [ -z "$PATTERN" ]; then
            echo "Usage: $0 search <pattern> [lines]"
            exit 1
        fi
        ssh_cmd "powershell -Command \"Get-Content $LOG_PATH -Tail $LINES | Select-String -Pattern '$PATTERN'\""
        ;;
    watch)
        # Continuous tail (refreshes every 2 seconds)
        echo "Watching server logs (Ctrl+C to stop)..."
        while true; do
            clear
            echo "=== Server Logs (last 30 lines) - $(date) ==="
            ssh_cmd "powershell -Command \"Get-Content $LOG_PATH -Tail 30\""
            sleep 2
        done
        ;;
    status)
        # Server status
        echo "=== Server Status ==="
        ssh_cmd "tasklist | findstr node"
        echo ""
        echo "=== Recent Errors ==="
        ssh_cmd "powershell -Command \"Get-Content $LOG_PATH -Tail 50 | Select-String -Pattern 'error|ERROR|exception|failed|crash' | Select-Object -Last 10\""
        ;;
    *)
        echo "Usage: $0 [tail [lines]|errors [lines]|search <pattern> [lines]|watch|status]"
        echo ""
        echo "Commands:"
        echo "  tail [N]          - Show last N lines (default 50)"
        echo "  errors [N]        - Show errors from last N lines"
        echo "  search <pat> [N]  - Search pattern in last N lines"
        echo "  watch             - Continuous log monitoring"
        echo "  status            - Server status + recent errors"
        ;;
esac
