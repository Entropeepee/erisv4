@echo off
cd /d "%~dp0"

echo Starting Eris Vitals Backend...
start cmd /k "python -m eris.server.app"

echo Starting Eris Web UI...
cd eris-ui
start cmd /k "npm run dev"

echo Eris Systems Online.
exit
