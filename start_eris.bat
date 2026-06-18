@echo off
cd /d "%~dp0"

echo Cleaning up old processes...
taskkill /F /IM node.exe >nul 2>&1

echo Starting Eris Vitals Backend...
start cmd /k "python -m eris.server.app"

echo Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

echo Starting Eris Web UI...
cd eris-ui
start cmd /k "npm run dev"

echo Eris Systems Online.
exit
