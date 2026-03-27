# ERIS ECHO v4 — VALIDATION EXPERIMENTS
# Paste this into a new Claude tab along with the eris_echo_v4 codebase.

## CONTEXT

I'm David Pope, founder of Terminus IP Group LLC. I just completed a
full build of Eris Echo v4 — a cognitive architecture that processes
information through BLECD field dynamics instead of frozen LLM weights.
29 Python modules, 109 tests, all passing. The codebase is attached.

Read these files FIRST before doing anything:
- VISION_ROADMAP.md (the endgame and what's been proven)
- HANDOFF.md (every module and its theoretical source)
- README.md (quickstart)

The system uses my patent portfolio: SGT (Application 19/540,588),
Davidian Hill-Power shrinkage, GLNCS/CSBA nullspace projection,
FRACTAL PDE, BLECD Logic Compiler, dC/dX conservation law. All IP
is authorized for use here — this runs on my computer.

## WHAT I NEED

Four validation experiments, built FROM SCRATCH using the v4 codebase.
Not searching for old code. Not porting from Gemini. Fresh implementations
using the actual modules we just built (eris/field/pde.py, eris/computation/,
eris/memory/interference.py, eris/retrieval/, etc.).

### EXPERIMENT 1: dC/dX Hallucination Detector Benchmark

Take a set of questions where LLMs are known to hallucinate (factual
errors, fabricated citations, confident nonsense). Run each question
through the Eris Echo pipeline (FRACTAL PDE → compute BFECDS → measure
dC/dX + coherence + tau_rms). Compare the dC/dX signature of:
  (a) Questions the system has grounded knowledge for (from ingested docs)
  (b) Questions outside its knowledge (should show hallucination signature)

The hypothesis: dC/dX ≈ 0 + high C reliably distinguishes hallucinated
responses from grounded ones. This is the TransfixionDetector in
eris/executive/workspace.py — we just need to measure it systematically.

Deliverable: A script that runs N questions, measures field observables
for each, and produces a clear table showing dC/dX for grounded vs
ungrounded queries. Statistical test (t-test or Mann-Whitney) on the
separation. If it works, this is publishable on its own.

### EXPERIMENT 2: Field Interference Retrieval vs Cosine RAG

Head-to-head comparison. Same corpus of documents, same set of queries.
Two retrieval methods:
  (a) Standard: embed query + embed docs → cosine similarity → top-k
  (b) Ours: run query through PDE → compute BFECDS → CSBA interference
      R_ij against stored field states → top-k by interference score

Use the retrieval swarm (eris/retrieval/swarm.py) for method (b) and
a simple embedding cosine for method (a). The corpus can be small —
20-30 short documents on diverse topics. The queries should include
some that are semantically close but domain-different (to test whether
BFECDS alignment catches what cosine misses) and some that are
domain-matched but semantically distant (to test the reverse).

Deliverable: A script that runs both methods on the same query set and
reports precision@k for each. We're looking for cases where interference
retrieval finds relevant documents that cosine misses, especially when
the relevance is structural (same BLECD regime) rather than lexical.

### EXPERIMENT 3: Zero-Weight Image Classification (Rebuild)

Rebuild the cats-vs-dogs classifier from scratch using v4's field
infrastructure. NO learned weights. NO neural network. The pipeline:

  Image → grayscale → 2D statistics (edge density, symmetry, texture
  entropy, frequency spectrum) → seed a FractalField → evolve PDE →
  compute BFECDS → classify by BFECDS distance to class centroids.

Train phase: run N cat images and N dog images through the pipeline,
compute mean BFECDS centroid for each class.
Test phase: new image → PDE → BFECDS → nearest centroid = prediction.

Use a STANDARD benchmark (CIFAR-10 cats vs dogs subset, or a small
ImageNet subset). Compare against:
  - Random baseline (50%)
  - Color histogram + SVM (as a feature-engineering baseline)
  - Our BLECD field classifier

Deliverable: Accuracy numbers with confidence intervals. We don't need
to beat the SVM — we need to show that BLECD field analysis extracts
classifiable structure WITHOUT any learned parameters, and characterize
WHERE it succeeds and fails (what kinds of images are easy/hard for
field-based classification).

The ImageFrontend stub is in eris/knowledge/frontends.py. Implement it.

### EXPERIMENT 4: Field-to-Text Tiny Translator (Proof of Concept)

Use the autobiography data that the orchestrator produces. Every call to
ErisOrchestrator.process() stores a (field_state, response_text) pair.
Generate a dataset by running ~100-200 diverse prompts through the
orchestrator (even without an LLM backend — the specialist findings
provide text). Extract (phi_snapshot, theta_snapshot, bvec, text) tuples.

Then train a TINY model — not a transformer, something minimal like a
small MLP or a lookup table with interpolation — that takes a field
state (flattened phi + theta + BFECDS = ~8200 features for a 64x64 field)
and produces a text description.

We're not expecting Shakespeare. We're expecting: given a field state
with high Emergence and high Criticality, the translator outputs
something like "novel restructuring" or "phase transition detected."
Even keyword-level accuracy from field states alone — without seeing
the original text — would demonstrate that the field representation
carries semantic content that can be decoded without an LLM.

Deliverable: The translator, its training pipeline, and evaluation
showing that field states predict response characteristics (regime,
dominant domain, archetype) better than chance. If it can produce
coherent phrases, even better.

## CONSTRAINTS

- Use the v4 codebase modules directly. Don't rewrite the PDE or
  shrinkage or interference — import them.
- Use Davidian Hill-Power shrinkage wherever you'd otherwise clip or
  threshold. Use CSBA/GLNCS wherever you'd otherwise do naive cosine.
- Include checkpoint saving for any run over 30 minutes.
- No arbitrary time caps on experiments.
- Standard benchmark data where possible (CIFAR-10, known hallucination
  sets). Not toy data we generated ourselves.
- Each experiment should be a standalone script in experiments/ that
  can be run independently on my Alienware (RTX 5080, 16GB VRAM).

## PRIORITY ORDER

1 first (fastest to run, most immediately valuable if it works)
3 second (the rebuild — high impact proof of concept)
2 third (retrieval comparison — needs a corpus)
4 fourth (translator — needs autobiography data from running the system)

Let's start with Experiment 1.
