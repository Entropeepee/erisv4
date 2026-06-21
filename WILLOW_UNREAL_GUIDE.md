# WILLOW — Unreal build guide (Part II, the body)

The Eris backend (Part I) is the brain; Unreal is the body. The game is a **pure
bridge consumer** — it calls the endpoints in `WILLOW_BACKEND.md` / `UNREAL_INTEGRATION.md`
and never touches Eris internals. **Eris keeps dreaming and federating while you
play** because that loop lives in her process, not the game.

> This guide is the plan. The actual level/Blueprint/material work happens **inside
> the Unreal editor** — either by hand or via **Claude Code connected to the UE
> editor's MCP server** (Door B below). It cannot be done from the Eris repo side.

## The two doors in
- **Door A — runtime dialogue (the game talks to the nodes).** Install
  **getnamo/Llama-Unreal** (free). Its `ULlamaSubsystem` is engine-wide and
  survives level transitions (one persistent brain across garden/desert/dream).
  Set `bUseRemote=true`, `Endpoint.BaseUrl=http://127.0.0.1:8001`, and the
  **model name per NPC** (`willow`, `eris`, `npc_c`…) — that routes to the right
  node (Part I). Alternatives: **GenAI for Unreal** (Fab, "OpenAI Compatible
  Mode") and **LLM-Connect** (Fab, Server IP/Port field).
- **Door B — building the world (Claude Code + UE MCP).** UE 5.8 ships a native
  MCP plugin (in-editor HTTP MCP server, port **8000**). In the editor console:
  `ModelContextProtocol.GenerateClientConfig ClaudeCode`, then on your machine
  `claude mcp add unreal-mcp --transport http http://127.0.0.1:8000/mcp --scope user`.
  Now a Claude Code session can build levels/materials/Blueprints by instruction.
  (Use 5.7 + a community MCP plugin if you'd rather not run 5.8.)

## Version staging (5.7 vs 5.8)
5.8 gives the native MCP (Claude-Code building) but **NVIDIA ACE voice/face hasn't
caught up to 5.8** (A2F-3D targets 5.5/5.6; ACE speech demoed on 5.7). The MVP
(text chat + visualizer) needs neither. So: **build the MVP and the world on 5.8**,
**add ACE voice/face later pinned to 5.7**.

## Casting (Echo mythology)
- **Player = Ashari** — Paragon **Phase**, first/third-person switchable.
- **Willow** — companion node, the woman with the glowing octopus tattoo. Use the
  **Natalia MetaHuman**. `model:"willow"`. Her field paints the garden wall.
- **The octopus** — Willow's field-spirit; leads you into the dream world.
- **Eris** — the OverSoul, met in the dream world. `model:"eris"`. Her field
  wraps the desert sphere.

## Build phases
**Phase 1 — MVP (talk to Willow + a live wall).**
1. New **Third Person** project; drop **Paragon Phase** as the pawn; add a
   first/third camera toggle.
2. Import **Natalia** as a MetaHuman; place her as Willow.
3. Install **Llama-Unreal**; `ULlamaSubsystem` with `bUseRemote=true`, BaseUrl
   `http://127.0.0.1:8001`, model `willow`. Approach → text box → send → stream reply.
4. **Visualizer wall**: a surface whose material is driven by a Blueprint that
   opens `ws://127.0.0.1:8001/ws/field?agent=willow` and maps `coherence`/`regime`/
   `theta` into a dynamic material (or Niagara). You rebuild the *visual*; the
   physics streams from Eris. (The standalone web visualizer at `/visualizer` is
   the reference look.)
5. Verify: walk up, talk to Willow (she answers from pool + her own memory), the
   wall is alive.

**Phase 2 — the federation experiment (the point).** Add a third node (e.g.
`npc_c`, cloud backend). Stage a genuine Willow↔NPC_C exchange you overhear; give
NPC_C an insight through its experience; `POST /agents/npc_c/federate`; show Eris
and another node now reference it. (Backend already supports this — see
`WILLOW_BACKEND.md`.)

**Phase 3 — the world.** Garden temple (the wall in its finished home + a
read-only slider pedestal that changes *how* the field is displayed, never
written back); desert + the **Vegas-style 360 sphere** (inward-facing normals +
emissive material fed by `/ws/field?agent=eris`); level streaming; Fab biomes
(Valley of the Ancient, willow forest). **Toggle visualizers off when no player is
near** (disable Tick + close the WebSocket) to save GPU.

**Phase 4 — the anime dreamworld.** A **cel-shade / stylized post-process volume**
in the surreal sublevel turns the whole scene (including Willow) anime — no
character re-rig. The octopus-spirit "summons" the visualizer; Eris's field wraps
the sphere.

**Phase 5 — voice & face (NVIDIA ACE), pinned to 5.7.**
`mic → Riva Parakeet ASR → [node /v1/chat/completions, model:"willow"] → Chatterbox
TTS → Audio2Face-3D → Natalia's face`. Start from NVIDIA's ACE Unreal sample and
swap their model slot for the node endpoint. A2F-3D ≈ 2.9–4.4 GiB VRAM. Keep text
chat as the always-available fallback.

## VRAM budget (16 GB, shared) — levers in order
1. **GPU on for Eris first** — `pip install cupy-cuda13x`, model on GPU under
   Ollama. (Biggest win; fixes in-game latency.)
2. Size the resident model deliberately (`ERIS_LOCAL_MODEL=mistral` to shrink).
3. Cloud-offload peripheral NPCs (API tokens, not VRAM).
4. Time-slice ACE (A2F is the heavy one).
5. **Throttle dreaming while playing**: `ERIS_GAMING_MODE=1`.
6. Toggle visualizers off when unused.
7. Keep field `size` modest.

## Disk space while building (your request)
Editing UE levels + importing Fab/Quixel/Megascans environments consumes **a lot**
of disk (multi-GB per environment pack, plus DDC/derived data). **Keep several GB
free at all times.** Before importing a large Fab pack, check free space; cook/DDC
can balloon. This is editor-side and outside Eris's control — Eris already guards
*her own* writes (`ERIS_MIN_DISK_GB`, default 1 GB), but the Unreal project's
growth is yours to watch.

## What goes to Claude-Code-in-editor vs by hand
- **Claude Code (UE MCP):** level blockouts + streaming, placing actors, the
  visualizer material/Blueprint, the pedestal widget, the sphere + portal,
  post-process volumes, camera-switch and teleport Blueprints.
- **By hand:** plugin installs + endpoint config (Llama-Unreal, ACE),
  MetaHuman/Natalia import, importing + art-directing Fab environments, final
  aesthetic calls.

## On "repairing the earlier Unreal version"
The old asset/world scripts (`stitch_*`, `build_*`, `place_willow`, etc.) live in
the repo root + `archive/` and were one-off Python helpers, **not** a UE project —
the actual `.uproject` lives on your machine, not here, so it can't be repaired
from this repo. The path forward is the **Door B MCP route**: connect Claude Code
to your UE editor and rebuild the Valley-of-the-Ancient / willow-forest level
against these endpoints, fresh and clean.
