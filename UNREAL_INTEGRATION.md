# Eris ↔ Unreal Engine — integration contract

The Eris server (`python -m eris.server.app`, default `http://localhost:8001`) is
the single integration surface. Unreal talks to it over HTTP + WebSocket; nothing
in UE needs to import Python. This doc is the stable contract the UE side codes
against.

## 1. Conversation (the main bridge)

Eris speaks **OpenAI-compatible chat completions**, which is exactly what NVIDIA
ACE / most UE LLM plugins expect.

`POST /v1/chat/completions`
```json
{
  "model": "eris",
  "messages": [
    {"role": "system", "content": "optional persona/scene context"},
    {"role": "user",   "content": "what the player said"}
  ]
}
```
Response (OpenAI shape; the spoken line is `choices[0].message.content`):
```json
{ "object": "chat.completion",
  "choices": [{ "message": { "role": "assistant", "content": "Eris's reply" } }] }
```
- `content` is clean spoken text (markdown is for the cockpit; for TTS use the
  cleaned text — see §3). No cognitive telemetry leaks into it.
- A native endpoint `POST /chat` also exists and returns richer fields
  (`regime`, `archetype`, `dCdX`, `coherence`, `specialist`, `latency_ms`) if UE
  wants the cognitive state inline. Same engine; pick whichever fits the plugin.

## 2. Emotion / cognitive state (drive the avatar)

Eris's felt state maps to her **field regime**. Two ways to read it:

- **Live stream** — `WS /ws` pushes vitals continuously:
  `{ regime, coherence, dCdX, dissonance, archetype, stm_size, mtm_size, ltm_size, llm_backends }`
- **Inline** — the `/chat` response carries `regime` + `archetype` per turn.

Suggested avatar mapping (matches the cockpit):
`plastic`→restructuring (pink), `transfixed`→stuck (amber), `warmup`→grey,
`elastic`/other→calm (cyan); a question mark in the reply → "curious".

The live field itself (φ/θ PDE grid) streams on `WS /ws/field`
(`{ size, phi[][], theta[][], coherence, dCdX, regime_str }`) if you want to
render her "mind" as a material in-world.

## 3. Voice (TTS)

- `GET /api/tts/voices` → `{ voices: [{id, name}], ... }` (auto-prefers an "Emily" voice).
- `POST /api/tts/generate` `{ "text": "...", "voice_id": "..." }` → audio bytes
  (`audio/wav`). The server already strips markdown before synthesis.
- For lip-sync, drive the mouth from the audio amplitude envelope (the cockpit
  uses a Web-Audio analyser; UE can use its own audio-amplitude tap).

## 4. Host telemetry (optional HUD)

`GET /api/system` → `{ cpu_pct, ram_*, gpus:[{gpu_util_pct, vram_used_mb, vram_total_mb, gpu_temp_c}], cpu_temp_c }`.

## 5. OSC bridge

`eris_unreal_osc.py` is the optional OSC glue for UE Blueprints that prefer OSC
over HTTP. Keep it alongside the server; it forwards the same state.

## 6. VRAM budget (important on a 16 GB card)

Ollama `gpt-oss:20b` uses ~13 GB. When UE + NVIDIA ACE are also on the GPU:
- launch Eris with `ERIS_GPU=0` to keep her field math on CPU and hand the card
  to UE/ACE, **or**
- switch her to a smaller local model: `ERIS_LOCAL_MODEL=mistral` (≈4.4 GB).

Both are runtime env vars — no code change.

## 7. Smoke test before wiring UE

```
python -m eris.server.app
curl -s localhost:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"eris","messages":[{"role":"user","content":"hello"}]}'
```
A JSON reply with non-empty `choices[0].message.content` means the bridge is live.
