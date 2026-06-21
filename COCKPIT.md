# Eris Cockpit (Tier 7) — setup & guide

A single-page control room served by the Eris FastAPI server. Replaces the
cramped React layout. Open **http://localhost:8001/** after starting the server.

## What's on screen
- **Left:** conversation history — condensed title + description, started/last-seen
  dates, turn count. Click to reload a thread; “+ New conversation” starts fresh.
- **Center:** the avatar (pulses + lip-syncs to TTS, shows emotion) over a
  **scrolling** chat. Voice controls (speak toggle, voice picker, volume) sit
  under the composer and are always visible.
- **Right (scrolls):** Living-Field visualizer with a **pop-out** button (opens
  the big `/visualizer` in its own window); **Cognitive Vitals**; **System
  Vitals** (CPU/RAM/GPU/VRAM/temps); **Dreams & Reflection**; **Studied Recently**.

## System vitals
- CPU% and RAM come from `psutil` (`pip install psutil`).
- GPU util / VRAM / temperature come from **`nvidia-smi`** (you have an RTX 5080) —
  it must be on PATH. No NVIDIA → that row shows “no nvidia-smi”, everything else
  still works.
- **CPU temperature** on Windows needs a helper exposing WMI sensors
  (LibreHardwareMonitor running in the background) + `pip install wmi`. Without it,
  CPU temp shows “—”. Everything else is unaffected.

## The avatar (Live2D)
The cockpit drives lip-sync (from your Emily TTS audio) and emotion (from her
regime + reply) on **two** render paths:
1. **Portrait (default, works now):** drop a PNG at
   `eris/server/static/eris_portrait.png` (a default is seeded). It pulses with
   speech and glows by emotion.
2. **Live2D (when you add a model):** put a Cubism-4 model at
   `eris/server/static/models/live2d/eris/eris.model3.json`. The cockpit auto-detects
   it, loads pixi-live2d-display + the Cubism core from CDN, and drives
   `ParamMouthOpenY` for lip-sync. Free riggable/sample models: the official
   Live2D sample collection, and community VTuber models (check each model's
   license). To match your “Eris” art, a model can be rigged in the free Live2D
   Cubism Editor or commissioned. Until a model is present, the portrait path is
   used — no errors.

Emotion mapping (portrait glow / Live2D expression): plastic→pink (restructuring),
transfixed→amber (stuck), warmup→grey, otherwise calm-cyan; a question in her
reply → “curious”.

## Voice (Emily)
The voice picker auto-selects a voice whose name contains “Emily” if your TTS
engine exposes one. Otherwise pick it manually; the choice is used for `/api/tts/generate`.

## Self-directed learning (nightly)
- Eris studies a **topic list you control**. Click **topics** in the “Studied
  Recently” panel to edit it (reliable nonfiction — defaults to Wikipedia + an
  allow-list of reputable domains). **study now** runs a session immediately.
- A nightly job runs at **03:00 local** by default. Configure:
  - `ERIS_STUDY_HOUR=3` — hour of day to study.
  - `ERIS_STUDY_ENABLED=0` — disable the nightly job.
- Each session writes a **study report** (topics, passages ingested, a short
  summary) you can click to read. Everything read is ingested dual-track into
  memory so it informs future answers.

## Dreams & on-demand pondering
- The dream/metacognition loop already runs every 15 min and on cognitive
  dissonance; now every reflection is written to a **journal** shown in the
  Dreams panel (summary line; ❓ flag when she needs your input). Click to read
  her fuller thoughts.
- **＋ ponder…** lets you direct her into a focused dream state on a specific
  question — she evolves the field on it, researches it, and records what she found.

## Run
```
pip install -r requirements.txt        # fastapi, uvicorn, psutil, …
# optional upgrades:
pip install sentence-transformers       # + set ERIS_EMBEDDINGS=on  (semantic memory)
pip install anthropic                   # + set ANTHROPIC_API_KEY   (research oracle)
python -m eris.server.app               # serves on http://0.0.0.0:8001
```
