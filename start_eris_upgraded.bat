@echo off
cd /d "%~dp0"
title Eris Echo (UPGRADED - optional new functions ON)
echo ============================================
echo   Eris Echo - UPGRADED MODE
echo.
echo   The web-reading cleanup, the new voice, the dream-loop
echo   fix, and the chat "answer the person" fix are ALREADY ON
echo   by default - you do not need any flag for those.
echo.
echo   This launcher additionally turns ON the optional:
echo     * Web reader proxy - lets Eris read bot-blocked or
echo       VPN-blocked articles (fetches via r.jina.ai).
echo       NOTE: this sends the page URL to a third party.
echo ============================================

REM === optional upgrade flags (environment variables - no code editing) ===
set ERIS_WEB_PROXY=on

REM  Want "smarter but slower" replies? Eris will sample several answers
REM  and return the consensus. It is 3-8x slower per message on a local
REM  model, so it is OFF here. To try it: delete the word REM (and the
REM  space) at the start of the next line, then save and run again.
REM set ERIS_TTC=on

REM  Smart local->cloud router (only escalates to a cloud model on genuine
REM  outliers, and ONLY if you have a cloud API key set). Harmless without
REM  a key. To try it, un-REM the next line:
REM set ERIS_ORCHESTRATION=on

echo Starting Eris's language model (Ollama)...
echo (If its window says "address already in use", that is fine -
echo  it just means Ollama was already running.)
start "Eris LLM (Ollama)" cmd /k "ollama serve"
timeout /t 4 /nobreak >nul

start "Eris Backend (UPGRADED)" cmd /k "python -m eris.server.app"

echo Waiting for the field engine to warm up...
timeout /t 6 /nobreak >nul

echo Opening the Eris cockpit...
start "" http://localhost:8001/

echo.
echo Eris is online (UPGRADED). Cockpit: http://localhost:8001/
echo To go back to plain default mode, just run start_eris.bat instead.
exit
