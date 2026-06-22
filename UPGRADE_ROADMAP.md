# Eris upgrade roadmap — RTX 5080 / Core Ultra 9 stack

Derived from the deep-research plan (`Upgrading_Eris_to_State_of_the_Art…md`).
This is the **tracked sequence**: dependency-ordered, each item tagged by who does
it, with an acceptance check and a live status box. Update the boxes as we go.

### Legend
- **`[code]`** — pure code I build & push into this repo (testable offline here).
- **`[machine]`** — runs on your Alienware (CUDA/GPU/NPU/model downloads/fine-tune).
  I write the code/config/instructions; you run the commands.
- Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[shelved]` parked with reason.

> **Branch note.** The `[code]` items build on the criticality monitor from the
> orchestration work, so they currently live on branch `orchestration-cip`
> alongside it. Merge that branch to ship them.

---

## Stage 0 — Foundation & cheapest "smarter" wins

| # | Item | Who | Depends on | Acceptance | Status |
|---|---|---|---|---|---|
| 0.1 | Rebuild inference on CUDA 12.8+; confirm 5080 (sm_120) recognized; enable Flash-Attention + Q8 KV-cache | `[machine]` | — | `ollama run` uses GPU; longer context fits | `[ ]` |
| 0.2 | Standardize language layer on a 14B reasoning model (Qwen3-14B *thinking* or R1-Distill-14B) at Q4_K_M/Q5 | `[machine]` + `[code]` | 0.1 | `ERIS_LOCAL_MODEL` points to it; tok/s benchmarked | `[~]` (env hook exists; model is yours to pull) |
| 0.3 | **Test-time compute: self-consistency + adaptive criticality-stop + budget-forcing hooks** | `[code]` | criticality monitor (done) | default-OFF, offline tests green, `ERIS_TTC=on` enables | `[x]` **this session** |
| 0.4 | Activate Intel NPU (OpenVINO); move Whisper STT + TTS off the GPU | `[machine]` | — | `Core().available_devices` shows `NPU`; STT/TTS latency acceptable | `[ ]` |

## Stage 1 — Capability upgrades

| # | Item | Who | Depends on | Acceptance | Status |
|---|---|---|---|---|---|
| 1.1 | Serving-route abstraction: Eris talks to Ollama **or** vLLM via one OpenAI-compatible path (`ERIS_LLM_BASE_URL`) | `[code]` | — | switch backend by env, no code change | `[x]` **this session** |
| 1.2 | vLLM (NVFP4) serving path for concurrent multi-agent calls | `[machine]` | 1.1, 0.1 | vLLM serves; Eris routes to it when >1 concurrent call | `[ ]` |
| 1.3 | Local hybrid RAG: dense + BM25 + RRF + cross-encoder reranker over Qdrant/LanceDB | `[code]` + `[machine]` | — | reranked Recall@1 beats current on a 50-query eval | `[ ]` |
| 1.4 | Durable episodic/semantic memory (mem0 or Letta) under the field-salience layer | `[code]` + `[machine]` | 1.3 | facts persist across restarts; self-edits conflicts | `[ ]` |
| 1.5 | Vision: optional Qwen3-VL-8B tool; expose tools via MCP | `[code]` + `[machine]` | — | Eris can read a screenshot/game-frame on request | `[ ]` |

## Stage 2 — "Smarter" via distillation

| # | Item | Who | Depends on | Acceptance | Status |
|---|---|---|---|---|---|
| 2.1 | Generate reasoning/agent traces on Eris's own tasks (cloud or local-32B teacher); parallelize with Ray across spare boxes | `[code]` + `[machine]` | 0.2 | a clean trace set (s1-style, small + high quality) | `[ ]` |
| 2.2 | QLoRA-distill an 8–14B student in Unsloth; export GGUF; A/B vs stock | `[machine]` | 2.1 | student ≥ stock on your eval at lower latency | `[ ]` |

## Stage 3 — Autonomy & self-improvement (experimental)

| # | Item | Who | Depends on | Acceptance | Status |
|---|---|---|---|---|---|
| 3.1 | Formalize ReAct **grounded in field state** + Reflexion self-critique; log failures to memory | `[code]` | 1.4 | loop cites field state each step; failures recalled | `[ ]` |
| 3.2 | SEAL-style self-edits: dreaming loop generates LoRA data from successful interactions; periodic QLoRA update with rollback guard | `[code]` + `[machine]` | 2.2, 3.1 | held-out eval never regresses (catastrophic-forgetting guard) | `[ ]` |
| 3.3 | (Only if dense-32B target) EAGLE-3 / draft-model speculation | `[machine]` | 0.2 | end-to-end tok/s improves (else abandon) | `[ ]` |

---

## Hardware-utilization notes (from the research, carried forward)
- **NPU = auxiliary, not primary.** STT/TTS/embeddings/reranking/small classifiers
  only — frees the whole 16GB GPU for the reasoning model. (0.4, 1.3)
- **Spare boxes (GTX 970, 3050/3060) = Python/service workers, NOT LLM shards.**
  Use **Ray** for embarrassingly-parallel work (trace generation, eval). (2.1)
- **Speculative decoding is often net-negative** on small/quantized/MoE models —
  only on a dense 32B target, measure end-to-end before keeping. (3.3)

## Done this session (`[code]`)
- **0.3 Test-time compute layer** (`eris/interface/test_time.py`): self-consistency
  with an **adaptive criticality early-stop** (reuses the Tier-1 `NoiseFloorEstimator`
  + `CriticalityMonitor` — the patent's six ideas landing at a boundary where a
  convergent signal *actually exists*: vote-stability across samples), plus
  budget-forcing hooks. Wired into the orchestrator behind `CONFIG.ttc_*` /
  `ERIS_TTC` (default OFF). New `llm_samples` counter. Offline tests.
- **1.1 Serving-route abstraction**: `ERIS_LLM_BASE_URL` swaps the primary local
  backend from Ollama to any OpenAI-compatible server (vLLM / TensorRT-LLM, NVFP4)
  with no code change. Default unset → Ollama. Your machine-side 1.2 (stand up
  vLLM) now just needs to point Eris at the URL.

## To run on your Alienware later (the `[machine]` side, ready for you)
- **0.2 model swap:** `set ERIS_LOCAL_MODEL=qwen3:14b` (or your R1-Distill GGUF tag
  in Ollama), restart Eris. Try `/think` mode for reasoning.
- **0.3 turn on TTC:** `set ERIS_TTC=on` for self-consistency on hard turns.
- **1.2 vLLM:** once vLLM serves on `:8000`, `set ERIS_LLM_BASE_URL=http://localhost:8000/v1`
  and `set ERIS_LOCAL_MODEL=<served-model-name>`; Eris routes to it automatically.
