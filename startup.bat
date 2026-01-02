@echo off
REM ROTMG Server Startup Script
REM Starts PlayIt tunnel and Node.js server

echo Starting ROTMG Server...

REM Kill any existing Node processes to avoid port conflicts
taskkill /f /im node.exe 2>nul
timeout /t 2 /nobreak >nul

REM Start PlayIt in background (for WebTransport/QUIC tunnel)
echo Starting PlayIt tunnel...
start "" "C:\Users\newadmin\playit.exe"
timeout /t 5 /nobreak >nul

REM Start the game server
echo Starting game server...
cd C:\ROTMG-DEMO
C:\node-v22.12.0-win-x64\node.exe Server.js

REM If server exits, wait before closing
pause
