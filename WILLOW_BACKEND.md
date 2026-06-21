# WILLOW — the multi-node Eris collective (Part I, the brain)

Eris runs as a local **collective of nodes**: one shared knowledge **pool**
(= "Eris," the OverSoul) plus **Echo nodes** (Willow + NPCs). Every node *reads*
the pool (so it knows everything Eris does) but *diverges* through its own field
and **private** memory. Novel insights a node earns federate back into the pool,
so a *different* node can act on them. This is all in `eris/agents/`, additive —
single-Eris behaviour is unchanged (`eris` is the default node).

## The pieces (`eris/agents/`)
| File | What |
|---|---|
| `memory_view.py` | `LayeredMemory` — read SHARED pool + PRIVATE experience; writes go PRIVATE. |
| `agent.py` | `Agent` — persona + backend + own field + layered memory; `respond()`, `distill()`. The `eris` node delegates to the full orchestrator. |
| `registry.py` | `AgentRegistry` + `build_default_registry` → pool + `eris` + `willow`. |
| `backends.py` | per-node LLM backends (one shared Ollama for local nodes; cloud only when keyed). |
| `insights.py` | `InsightLog` — a node's distilled one-line understandings. |
| `federation.py` | `federate()` — push a node's NOVEL insights into the pool (novelty-gated, idempotent). |
| `budget.py` | `ConversationBudget` + `choose_dialogue_plan` — the degradation ladder. |
| `dialogue.py` | `generate_dialogue` — one-call puppeteered NPC↔NPC dialogue. |

## How a node diverges yet knows everything
- `LayeredMemory.retrieve` merges **private first, then shared** → the node draws
  on both, but its own life colours what surfaces.
- `respond()`/`converse` write the exchange to **private** memory only → the pool
  doesn't grow from a node's chatter; the node accumulates its own history.
- Each node has its **own `FractalField`**, evolved on its inputs → its
  cognitive trajectory diverges from Eris's.

## Federation (the experiment)
1. A node **distills** an insight from new private experience (`Agent.distill`, one
   cheap LLM call → `InsightLog`).
2. `federate()` pushes insights that are **novel vs the pool** (cosine distance >
   `novelty`) into the pool's long-term memory as `source="node:<name>"`.
3. Now **every other node**, reading the pool, can retrieve it. Re-federation is a
   no-op (once in the pool the insight is no longer novel).

The background dream loop runs a federation pass each cycle: every real node
distills + federates automatically. You'll see `[Federation] willow -> pool: N
novel insight(s)` in the log. **The test worth watching (spec step 4):** give one
node an insight through its own experience, then confirm a *different* node
references it.

## Endpoints (the Unreal / client surface)
| Endpoint | Use |
|---|---|
| `POST /v1/chat/completions` (routes by `model`) | Player↔NPC and genuine NPC↔NPC turns. `model:"willow"` → Willow, `model:"eris"` → OverSoul. OpenAI-shaped (works with ACE / Llama-Unreal). |
| `POST /converse` `{speakers, context, turns}` | Puppeteered one-call dialogue; lines land in each speaker's private memory. Uses the budget/degradation ladder. |
| `WS /ws/field?agent=<name>` | Send-only field stream for that node (walls / sphere). Defaults to `eris`. |
| `GET /agents` | List nodes (name, backend, has_field, insight count). |
| `GET /agents/{name}/insights` | A node's distilled insights. |
| `POST /agents/{name}/reflect` | Distill one insight now (the "narrative beat"). |
| `POST /agents/{name}/federate` | Push that node's novel insights to the pool now. |
| `GET /health`, `GET /props` | Let the NPC plugins connect cleanly. |

## Adding nodes
In `eris/agents/registry.py` (or at runtime):
```python
from eris.agents.agent import Agent
from eris.agents.memory_view import LayeredMemory
from eris.memory.tiers import MemorySystem
priv = MemorySystem(data_dir="eris_data/agents/npc_c/memory")
reg.add(Agent("npc_c", persona="You are a wandering merchant…",
              backend_id="gemini",                 # cloud (keyed) for a flavor NPC
              memory=LayeredMemory(pool, priv),
              field=None))                          # field=None => pure-flavor NPC
```
- **Local + real** (own field, your IP): `backend_id="ollama"`, give it a `field`.
- **Cloud flavor NPC**: `backend_id="gemini"/"openai"/"claude"` (only registered if
  the key is set), `field=None`.

## Resource governance (so it never fights the renderer)
| Env | Effect |
|---|---|
| `ERIS_GAMING_MODE=1` | Throttle background dreaming/federation (× `ERIS_GAMING_THROTTLE`, default 4) while you play. |
| `ERIS_CRAWL_PERIOD_S` | Base dream-loop interval (default 900s). |
| `ERIS_CONVO_PER_HOUR` | Cap NPC↔NPC conversations per hour (default 60). |
| `ERIS_LOCAL_MODEL=mistral` | Smaller local model (~4.4 GB) to free VRAM. |
| `ERIS_GPU=0` | Field math on CPU — hand the GPU to Unreal. |

`choose_dialogue_plan` degrades **genuine → local puppeteer → skip** as budget /
API keys run out, without errors.

## Verify
```
curl localhost:8001/agents
curl -s localhost:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"willow","messages":[{"role":"user","content":"who are you?"}]}'
curl -X POST localhost:8001/agents/willow/reflect
curl -X POST localhost:8001/agents/willow/federate
```
Tell Willow something private → Eris won't know it (until federation); Willow
will. One model object serves all local nodes — VRAM doesn't scale with node
count.
