# Eris Echo v4 — Resonant Cognitive Architecture

**28 modules · 109 tests · Complete cognitive pipeline**

Copyright 2026 Terminus IP Group LLC / Quantum Nexus Labs.

## Quickstart

```bash
cd eris_echo_v4
pip install numpy pytest
python quickstart.py          # System check + all tests
python quickstart.py --serve  # Start web server (needs: pip install fastapi uvicorn httpx)
```

## Adding an LLM

```python
from eris.orchestrator import ErisOrchestrator
from eris.interface.mediator import OllamaBackend

eris = ErisOrchestrator()
eris.add_llm_backend(OllamaBackend(model="llama3.2"))
response = await eris.process("Tell me about emergence")
```

## Architecture: 28 modules across 10 layers + orchestrator

| Layer | Package | Key modules |
|-------|---------|-------------|
| 0 Computation | `computation/` | Davidian Hill-Power shrinkage, SGT gating, computed BFECDS |
| 1 Field | `field/` | FRACTAL PDE, FRT (reflexive path), BLC (φ-θ gate geometries), hex lattice |
| 2 Memory | `memory/` | Three-tier STM→MTM→LTM, CSBA interference, autobiography |
| 3 Tribe | `tribe/` | 11 specialists, Cross-Attention Hub, SGT-gated activation |
| 4 Metacognition | `metacognition/` | Dreaming loop, tension processing, question generation |
| 5 Executive | `executive/` | GPW, MoEGate (wave interference), TransfixionDetector (dC/dX) |
| 6 Interface | `interface/` | LLM mediator: Ollama, Claude, OpenAI, Gemini, Custom |
| 7 Sandbox | `sandbox/` | Isolated code execution (subprocess + Docker) |
| 8 Retrieval | `retrieval/` | GLNCS debiasing, multi-tier vector index, 6-retriever swarm |
| 9 Knowledge | `knowledge/` | .eris format, text extraction, ChatGPT corpus processing |
| 10 Server | `server/` | FastAPI: /chat, /vitals, /dream, /sandbox, /ingest, WebSocket |

## IP from David Pope's patent portfolio (internal license)

SGT (19/540,588), GLNCS/CSBA, Davidian Hill-Power, FRACTAL PDE, BLC, dC/dX conservation law.
