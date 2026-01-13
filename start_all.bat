@echo off
echo ========================================
echo   ROTMG Server - Starting All Services
echo ========================================
echo.

:: Start PlayIt tunnel (if not already running)
tasklist /FI "IMAGENAME eq playit.exe" 2>nul | find /I "playit.exe" >nul
if errorlevel 1 (
    echo [1/3] Starting PlayIt tunnel...
    start "" "C:\Users\newadmin\playit.exe"
    timeout /t 3 /nobreak >nul
) else (
    echo [1/3] PlayIt already running - OK
)

:: Start Cloudflared tunnel (if not already running)
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | find /I "cloudflared.exe" >nul
if errorlevel 1 (
    echo [2/3] Starting Cloudflared tunnel...
    start "" "C:\Users\newadmin\cloudflared.exe" tunnel run
    timeout /t 3 /nobreak >nul
) else (
    echo [2/3] Cloudflared already running - OK
)

:: Start Node server (if not already running)
tasklist /FI "IMAGENAME eq node.exe" 2>nul | find /I "node.exe" >nul
if errorlevel 1 (
    echo [3/3] Starting Node.js game server...
    cd /d C:\ROTMG-DEMO
    start "" "C:\node-v22.12.0-win-x64\node.exe" Server.js
) else (
    echo [3/3] Node server already running - OK
)

echo.
echo ========================================
echo   All services started!
echo   - PlayIt: UDP tunnel for WebTransport
echo   - Cloudflared: TCP tunnel for WebSocket
echo   - Node: Game server on port 4000
echo ========================================
timeout /t 5
