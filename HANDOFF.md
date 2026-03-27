# ERIS ECHO v4 — BUILD SESSION HANDOFF
# =====================================
# Date: 2026-03-27
# Builder: Claude (Opus 4.6) + David Pope
# Status: ARCHITECTURE COMPLETE — ready for Alienware integration

## WHAT WAS BUILT

28 production Python modules, 109 passing tests across 4 test suites,
a quickstart script, and a FastAPI server with embedded UI.

The system implements David Pope's FRACTAL/BLECD theoretical framework
as a working cognitive architecture where:
  - Information = coupling geometry (not message property)
  - Decisions = wave interference (not binary logic)
  - Hallucination = broken coupling (dC/dX ≈ 0 + high C)
  - LLM = Broca's area (language production, not cognition)
  - Any LLM is pluggable (Ollama, Claude, OpenAI, Gemini, custom)

## EVERY MODULE AND ITS THEORETICAL SOURCE

### Layer 0: Computation
- shrinkage.py    → Davidian Hill-Power w(s;α,β,γ,δ) = [(s-δ)₊^α/((s-δ)₊^α+β)]^γ
                    NOT the Pope Filter (trade secret meta-selector).
                    BFECDS-driven parameter mapping: C→α, B→β, E→γ, D→δ.
- sgt.py          → SGT patent (19/540,588). Stateless gate + stateful wrapper with EMA.
- activations.py  → Computed BFECDS from field dynamics (Chapter 6 PDE criteria).
                    Six archetype centroids from k=6 chemistry clustering.

### Layer 1: Field
- pde.py          → FRACTAL PDE with Memory modulator (exponential kernel τ_m,
                    independent of Decay) and Attention modulator (multiplicative
                    gain field). Global observables: C(t) Kuramoto, X(t) exchange,
                    dC/dX conservation law. Regime detection: elastic/plastic/transfixed.
- frt.py          → Fractal Rolling Tokenizer: text → blake2b treelets → bit-slice
                    → SymbolicPulse(φ,θ,τ). CPU-only, microseconds, deterministic.
                    Originally built for GTX 970. System 1 reflexive pathway.
- compiler.py     → BLC: generates φ-θ seed geometries from domain-pair contradictions.
                    YES(B+S), XOR(C+D), AND(F+E), DELAY(S+F), DIODE(D+B).
                    inject_seeds() applies geometries to PDE field.
- lattice.py      → Hex logic grid. Secondary tracking layer (PDE is primary).
- propagator.py   → SLGP background thread worker.
- pulses.py       → SymbolicPulse data carrier with 9-gate blend.
- tracer.py       → Jet extraction with torsion signatures.

### Layer 2: Memory
- tiers.py        → STM (deque, 20) → MTM (JSONL, Ebbinghaus, 200) → LTM (vector, permanent).
                    SGT-gated consolidation between tiers.
- interference.py → CSBA coupling geometry when no field snapshots (NOT single cosine).
                    Per-domain elastic/plastic decomposition with Davidian shrinkage.
                    Field integral R_ij = ∫φᵢ·φⱼ·cos(θᵢ−θⱼ)dx when snapshots available.
- autobiography.py → Every interaction logged: computed BFECDS + C + X + dC/dX + regime.

### Layer 3: Tribe
- specialists.py  → 11 specialists (Logos through Kairos). SGT-gated activation.
                    CrossAttentionHub for cross-pollination. Research trigger: C>0.4 AND E>0.2.

### Layer 4: Metacognition
- dreaming.py     → SGT-gated dreaming loop: scan high-torsion autobiography entries →
                    compile contradictions → evolve field → resolve or generate question.

### Layer 5: Executive
- workspace.py    → MoEGate uses wave interference scoring (per-domain elastic/plastic
                    + Davidian shrinkage), NOT cosine. TransfixionDetector reads dC/dX
                    directly (primary: dC/dX≈0 + high C = hallucination).
                    SharedCognitiveWorkspace (GWT single-slot broadcast). GoalNetwork.

### Layer 6: Interface
- mediator.py     → LLM-agnostic: OllamaBackend, OpenAIBackend, AnthropicBackend,
                    GeminiBackend, CustomBackend. Racing via httpx + asyncio.FIRST_COMPLETED.

### Layer 7: Sandbox
- validator.py    → AST-based code safety checks. Blocks os, subprocess, socket, etc.
- executor.py     → Subprocess + Docker execution modes. Timeout + stats tracking.
                    Ported from Eve2. Allows Eris to test its own code improvements.

### Layer 8: Retrieval
- glncs_filter.py → Nullspace projector P=I-V^TV for embedding debiasing.
                    Davidian compression 1024D→64D. Makes the RAG "unimpeachable."
- vector_index.py → Multi-tier HOT/WARM/COLD with auto promote/demote.
                    Brute-force cosine for now. Upgrade: FAISS-cuVS CAGRA.
- swarm.py        → 6 specialized retrievers (semantic, domain, temporal, torsion,
                    resonance, archetype) with RRF fusion. From SuperRAG architecture.

### Layer 9: Knowledge
- descriptor.py   → .eris Knowledge Descriptor: ZIP(manifest.json, source.txt,
                    field_phi.npy, field_theta.npy, bvec.json). SHA256 integrity.
- extractor.py    → Text → chunked .eris files with computed BFECDS.
- corpus.py       → Batch processor for ChatGPT JSON exports, text directories, JSONL.

### Layer 10: Server
- app.py          → FastAPI: /chat, /vitals, /dream, /questions, /sandbox, /ingest.
                    WebSocket /ws for real-time metrics. Embedded minimal HTML UI.

### Orchestrator
- orchestrator.py → Full pipeline wiring all layers. ErisOrchestrator.process()
                    for conversations, get_vitals() for health, run_dream_cycle()
                    for metacognition.

## KEY DECISIONS MADE IN THIS SESSION

1. Davidian Hill-Power replaces ALL shrinkage everywhere. Pope Filter is trade secret.
2. Memory and Attention are independent modulators (not reducible to 6 domains).
3. BLC generates φ-θ seed geometries injected into PDE field (primary substrate).
4. Interference uses CSBA per-domain coupling geometry, NOT single cosine.
5. MoEGate uses wave interference (constructive=agreement, destructive=conflict).
6. TransfixionDetector primary signal: dC/dX ≈ 0 + high C = hallucination.
7. FRT (System 1) + PDE (System 2) = dual-process architecture.
8. LLM is Broca's area — language production only. Any LLM pluggable.
9. Information = observer-system coupling geometry (not message property).

## WHAT REMAINS FOR PRODUCTION

### On the Alienware this weekend:
1. Run: python quickstart.py (verifies all 109 tests + quick pipeline)
2. Install CuPy: pip install cupy-cuda13x (verify CUDA 13.2 first)
3. Install server deps: pip install fastapi uvicorn httpx
4. Add Ollama backend: eris.add_llm_backend(OllamaBackend(model="llama3.2"))
5. Run python quickstart.py --serve and open http://localhost:8000

### Future upgrades (not blocking deployment):
- Embedding model (ONNX Runtime BGE-M3) for semantic memory retrieval
- FAISS-cuVS with CAGRA for GPU-accelerated ANN search
- safetensors checkpoint format (replace np.savez_compressed)
- Fused ElementwiseKernel for PDE performance at 512+ grid sizes
- Specialist LLM integration (specialists currently use BFECDS math, not LLM)
- Daily compaction cycle (scheduled batch job)
- Text seeding upgrade (coupling geometry via embeddings)
- Specialist sensitivity profiles computed from PDE, not hand-assigned
- κ (understanding capacity) as measurable system parameter
- Web UI polish (the embedded minimal UI works but isn't pretty)

## FILES FROM PREVIOUS WORK NOT YET PORTED

From v3 codebase (in David's GitHub repo):
- analogy_engine.py / attractors.py — Attractor registry with Ebbinghaus decay
- harmonic_certainty.py — Weighted harmonic mean certainty (weakest-link penalty)
- research_organ.py — Self-guided web research with DDG search
- stl_parser.py — Geometric programming language (human-readable BLC)
- continuous_crawler.py — Background web learning

From other projects:
- CHIMERA v56/57 — Split-brain geometry reasoning (box embeddings, tensor grammar)
- CFC Helix — Coherent Field Computation runtime
- Eve2 full codebase — Neo4j graph memory, Kokoro TTS, ExLlamaV2 inference

These are all available for integration but none are blocking v4.0 deployment.
