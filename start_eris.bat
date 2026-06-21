@echo off
cd /d "%~dp0"
title Eris Echo
echo ============================================
echo   Eris Echo - starting language model + cognitive backend
echo   (metacognition loop + dreams + nightly study
echo    all start automatically with the server)
echo ============================================

echo Starting Eris's language model (Ollama)...
echo (If its window says "address already in use", that is fine -
echo  it just means Ollama was already running.)
start "Eris LLM (Ollama)" cmd /k "ollama serve"
timeout /t 4 /nobreak >nul

start "Eris Backend" cmd /k "python -m eris.server.app"

echo Waiting for the field engine to warm up...
timeout /t 6 /nobreak >nul

echo Opening the Eris cockpit...
start "" http://localhost:8001/

echo.
echo Eris is online. Cockpit: http://localhost:8001/
echo (The old React UI is still available via: cd eris-ui ^&^& npm run dev)
exit
