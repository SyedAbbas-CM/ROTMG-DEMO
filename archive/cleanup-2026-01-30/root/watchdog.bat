@echo off
REM ROTMG Server Watchdog - Auto-restarts server if it crashes
REM Run this at Windows startup

cd /d C:\ROTMG-DEMO
set NODE=C:\node-v22.12.0-win-x64\node.exe

echo [WATCHDOG] Starting ROTMG server watchdog...
echo [WATCHDOG] Log: %date% %time%

:loop
echo [WATCHDOG] Starting server at %time%...
%NODE% Server.js

echo [WATCHDOG] Server exited at %time%. Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
