@echo off
cd /d "%~dp0"
title Eris Echo (UPGRADED - all no-install upgrade flags ON)
echo ==================================================================
echo   Eris Echo - UPGRADED MODE
echo.
echo   The web-reading cleanup, the new voice, the dream-loop fix, and
echo   the chat "answer the person" fix are ALREADY ON by default - no
echo   flag needed. This launcher also turns on every upgrade flag that
echo   needs NO extra install:
echo.
echo     ERIS_TTC           smarter replies - drafts several answers and
echo                        returns the consensus.  *** 3-8x SLOWER per
echo                        reply on a local model; this is the one you
echo                        will actually notice. ***
echo     ERIS_WEB_PROXY     read bot-blocked / VPN-blocked articles
echo                        (fetches via r.jina.ai - sends the URL to a
echo                        third party; that is the tradeoff).
echo     ERIS_AGENT_TOOLS   arms the ReAct tools (factual_lookup /
echo                        remember_fact / recall_facts). No visible
echo                        change in normal chat yet - nothing in the
echo                        cockpit triggers the agent loop so far.
echo     ERIS_ORCHESTRATION the smart local-^>cloud router. Only escalates
echo                        if you have a cloud API key set; with just the
echo                        local model it stays local and looks the same.
echo.
echo   If replies feel too slow, close this and run start_eris.bat instead.
echo ==================================================================

REM === upgrade flags that need NO extra install (environment vars; no code edits) ===
set ERIS_TTC=on
set ERIS_WEB_PROXY=on
set ERIS_AGENT_TOOLS=on
set ERIS_ORCHESTRATION=on

REM === these need extra setup - leave OFF until you have it ===
REM  Point the language model at a vLLM / llama-server you started yourself:
REM     set ERIS_LLM_BASE_URL=http://localhost:8000/v1
REM  Durable memory: built-in "local" is the default and works now.
REM  Do NOT set this to mem0 - that adapter is deferred and will error:
REM     set ERIS_MEMORY_BACKEND=local
REM  Vision: only after you pull a vision model and serve it:
REM     set ERIS_VISION_BASE_URL=http://localhost:8000/v1

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
