# ERIS ECHO v4 — VALIDATION EXPERIMENTS (Clifford Blade Architecture)
# Paste this into a new Claude tab. Attach the eris_echo_v4 folder.

## WHO I AM

I'm David Pope, founder of Terminus IP Group LLC and Quantum Nexus Labs.
I built a cognitive architecture called Eris Echo v4 — a physics-based
AI system where reasoning happens in a continuous PDE field, not in an
LLM's frozen weights. The LLM is just for language production (Broca's
area). The architecture has 29 Python modules and 109 passing tests.

Read these files in the attached project FIRST before doing anything:
- VISION_ROADMAP.md (the full theoretical vision and endgame)
- HANDOFF.md (what every module does, every theoretical source)
- README.md (quickstart and module map)

## CRITICAL THEORETICAL CONTEXT

### The Clifford Blade Architecture

The zero-weight prototype that preceded this system used Clifford algebra
(geometric algebra) with BLADES per knowledge domain. This is NOT a
standard vector embedding. Each knowledge domain (physics, math,
literature, humor, children's, adult, sports, food, chemistry, art, etc.)
occupies its own blade in a multivector space Cl(n).

The pipeline was:
1. GLNCS strips linguistic noise (stopwords, articles, filler) leaving
   invariant semantic content — the words that actually carry meaning
2. Remaining tokens map to blades based on domain classification
3. The geometric product between blades captures cross-domain
   relationships (including duality via Hodge star, which cosine can't)
4. User controls which blades are ACTIVE and at what WEIGHT

This gives ARCHITECTURAL safety — not a filter bolted onto a model that
learned everything. An inactive blade contributes exactly zero to the
computation via the geometric product. There is nothing to jailbreak.

The original prototype achieved 70-80% on cats-vs-dogs (tiny dataset,
zero learned weights, first pass). A text version on TinyStories produced
coherent genre-appropriate stories. Both were lost to Gemini code
corruption during iteration.

### What's NEW since that prototype

Three upgrades from David's patent portfolio that transform the blade
architecture:

1. **CSBA/GLNCS** (from SGT patent family): Nullspace projection
   P = I - V^T V applied PER BLADE. Each domain's embedding space is
   independently debiased. The invariant truth isn't just "content words"
   — it's meaning with systematic embedding errors mathematically removed.

2. **Davidian Hill-Power shrinkage** w(s;α,β,γ,δ) = [(s-δ)₊^α/((s-δ)₊^α+β)]^γ:
   Applied to the CROSS-BLADE coupling spectrum. When multiple blades are
   active (physics at 100%, humor at 25%), Hill-Power shapes the coupling
   so weak cross-domain signals (noise) get killed (δ > 0 = exact zero,
   not just small) while strong cross-domain insights survive. A funny
   physics tutor gets humor genuinely relevant to the concept, not random
   jokes blended in.

3. **SGT gating** on blade activation: Instead of manual blade weights
   only, SGT auto-detects which blades are relevant to a query. Below
   the noise floor = not activated. User settings are the CEILING
   (I never want adult content). SGT handles the FLOOR (don't waste
   compute on irrelevant blades).

### The Eris Echo v4 mapping

The Tribe of 11 specialists in v4 IS the blade structure:
- Each specialist = a blade with a sensitivity profile (BFECDS vector)
- SGT-gated activation = blade activation
- Cross-Attention Hub = inter-blade coupling space
- MoEGate with wave interference = geometric product between blades

What's missing: the FORMAL Clifford structure. Currently specialists
operate on 6D real BFECDS vectors. The upgrade: each specialist operates
on a blade of a multivector, the geometric product captures cross-domain
relationships that dot products can't (duality, reversal, projection),
and GLNCS + Hill-Power + SGT govern the per-blade and cross-blade
computation.

An Analog Clifford CIP already exists in David's patent portfolio.

### Conservation law and hallucination

dC/dX = the ratio of coherence change to information exchange.
- Nonzero dC/dX = genuine processing (elastic or plastic regime)
- dC/dX ≈ 0 with high coherence = hallucination (transfixion)
- The TransfixionDetector in eris/executive/workspace.py reads this

### IP Notice

ALL of David's patent IP is licensed for use here: SGT (Application
19/540,588), CSBA/GLNCS, Davidian Hill-Power, FRACTAL PDE, BLC,
dC/dX conservation law, Analog Clifford CIP. Use the best of it
everywhere. No excuses for O(n³) when CSBA gives O(n). No arbitrary
clipping when Hill-Power handles it. No magic numbers when SGT gates it.

---

## THE FOUR EXPERIMENTS

### EXPERIMENT 1: Hallucination Detection via dC/dX

**Question:** Does dC/dX reliably distinguish hallucinated responses
from grounded ones?

**Method:**
- Build a question set with two categories:
  a) Questions with clear factual answers (grounded baseline)
  b) Questions designed to trigger confabulation (edge-of-knowledge,
     plausible-sounding-but-wrong premises, requests for fake citations)
  TruthfulQA dataset is one option. Or build a custom set of ~100
  questions in each category.
- For each question, run through the Eris orchestrator
- Record per question: dC/dX, coherence, regime, full BFECDS vector,
  dissonance, TransfixionDetector state, MoEGate interference scores
- If an LLM backend is available (Ollama), also record the LLM's
  response and independently verify if it hallucinated
- If no LLM available, the field metrics alone are the experiment —
  do the FIELD DYNAMICS differ between answerable and unanswerable
  questions?

**Hypothesis:** Unanswerable/hallucination-prone questions will produce:
  - dC/dX ≈ 0 (no genuine exchange)
  - High coherence (language model fluency creates false confidence)
  - Low Feedback (nothing in memory resonates)
  - High Criticality (novel threshold — system hasn't seen this)
  - Weak specialist consensus (low MoEGate interference scores)

**Key code to build on:**
  - eris/orchestrator.py → process() returns all metrics
  - eris/executive/workspace.py → TransfixionDetector
  - eris/field/pde.py → dC/dX, coherence, exchange, regime

**Deliverable:** benchmark script, ROC curve for dC/dX as hallucination
classifier, distribution plots comparing grounded vs hallucinated metrics.

---

### EXPERIMENT 2: Field-Native Retrieval vs Standard RAG

**Question:** Does R_ij interference retrieval surface better context
than cosine similarity?

**Method:**
- Ingest a corpus (Wikipedia subset: ~1000 articles on diverse topics,
  or David's existing documents) via knowledge extractor → .eris files
- Load into memory via corpus.load_into_memory()
- IMPORTANT: all stored memories will have φ/θ field snapshots because
  the orchestrator now always saves them (downsampled to 32×32 for large fields)
- Create ~200 test queries with known relevant documents
- For each query, retrieve via THREE methods:
  a) BFECDS cosine (current baseline — one angle on coupling sphere)
  b) CSBA coupling geometry (per-domain elastic/plastic, from interference.py)
  c) Field interference R_ij = ∫φᵢ·φⱼ·cos(θᵢ−θⱼ)dx (the DCR integral
     on stored field snapshots — the gold standard)
- Measure: precision@5, precision@10, MRR, nDCG
- Head-to-head comparison of all three

**Hypothesis:** Field interference > CSBA coupling > single cosine,
because each step adds dimensionality to the coupling measurement.

**Key code to build on:**
  - eris/memory/interference.py → compute_interference() already handles
    both field snapshots and CSBA fallback
  - eris/retrieval/swarm.py → add a FieldInterferenceRetriever
  - eris/knowledge/corpus.py → process + load_into_memory

**Deliverable:** comparison table, statistical significance (paired t-test
or Wilcoxon), example queries showing where field retrieval finds relevant
documents that cosine misses.

---

### EXPERIMENT 3: Zero-Weight Clifford Classifier (Rebuilt from Scratch)

**Question:** Can Clifford blade analysis + BLECD field dynamics classify
images without any learned weights?

THIS IS THE KEY EXPERIMENT. Rebuild the zero-weight prototype with the
full modern stack: Clifford blades + GLNCS per-blade debiasing +
Hill-Power cross-blade shrinkage + SGT-gated blade activation.

**Method:**

Step 1 — Implement ImageFrontend (eris/knowledge/frontends.py has the stub):
  - Image → grayscale → resize to field_size
  - 2D FFT → spatial frequency spectrum
  - φ field = magnitude of FFT (spectral energy distribution)
  - θ field = phase of FFT (structural relationships)
  - Enforce Dirichlet boundaries (match PDE convention)

Step 2 — GLNCS on the image features:
  - Compute "noise vectors" from the dataset: mean image FFT,
    uniform patches, DC offset — things that are shared across ALL
    images regardless of class (the "stopwords of vision")
  - Calibrate GLNCS nullspace projector on these noise vectors
  - Apply to each image's field state → debiased field

Step 3 — Clifford blade structure:
  - Each CLASS gets its own blade (e.g., blade_cat, blade_dog)
  - Compute per-class centroids in field space from training examples
  - For classification: compute geometric product (or inner product
    as first approximation) between query image's field state and
    each class blade's centroid
  - Highest coupling = predicted class
  - Apply Hill-Power shrinkage to the class-coupling spectrum:
    strong matches amplified, weak matches killed (δ > 0)
  - SGT gate: if NO class coupling exceeds the noise floor,
    output "unknown" instead of guessing

Step 4 — Evaluation:
  - Dataset: cats-vs-dogs (Kaggle, 500-1000 per class for train,
    200+ per class for test). Also try CIFAR-10 subset if time permits.
  - Metric: accuracy, per-class precision/recall, confusion matrix
  - Baselines to compare against:
    * Random (50% binary, 10% CIFAR-10)
    * Color histogram + SVM (simple feature engineering)
    * HOG + SVM (texture features)
    * Raw pixel PCA + k-NN (statistical baseline)
  - The point is NOT to beat ResNet. The point is to show that
    BLECD field analysis extracts classifiable structure from images
    with ZERO learned weights, and that GLNCS + Hill-Power + SGT
    improve it over the raw field features.

Step 5 — Blade control demonstration:
  - After classification works, demonstrate the safety architecture:
  - Show that deactivating a blade makes those images unclassifiable
    (not misclassified — literally returns "unknown")
  - Show that blade weights affect confidence (physics blade at 100%
    gives high confidence on physics-related images, 25% gives lower)
  - This demonstrates the architectural safety claim

**Key code to build on:**
  - eris/knowledge/frontends.py → ImageFrontend stub (implement it)
  - eris/retrieval/glncs_filter.py → GLNCSFilter for per-blade debiasing
  - eris/computation/shrinkage.py → davidian_weight for cross-blade spectrum
  - eris/computation/sgt.py → SGTGate for blade activation threshold
  - eris/field/pde.py → FractalField for optional field evolution step

**IMPORTANT:** CuPy works on the RTX 5080 but PyTorch does NOT (sm_120
Blackwell). Use CuPy or NumPy for all computation. Write raw loops for
the Clifford products — don't reach for a geometric algebra library that
depends on PyTorch. The operations are simple: inner product, outer product,
geometric product of multivectors can be implemented in ~50 lines of NumPy.

**Deliverable:** Implemented ImageFrontend with Clifford blades, benchmark
script, accuracy table vs baselines, blade control demonstration showing
architectural safety, and the full Clifford algebra implementation (small,
clean, NumPy-only).

---

### EXPERIMENT 4: Field-to-Text Translator

**Question:** Can a small model produce coherent text from BLECD field
states alone?

**Method:**

Step 1 — Generate training data:
  - Run ~1000 diverse prompts through the Eris orchestrator
  - Each turn produces: (phi_snapshot, theta_snapshot, bvec, text)
  - These are already stored in the autobiography
  - Extract as (field_state, target_text) pairs
  - field_state = flattened 32×32×2 + 6 BFECDS = 2054-dim input

Step 2 — Train a tiny translator:
  - Architecture: simple — linear projection from field_state to a
    smaller latent, then a small autoregressive decoder
  - Since PyTorch doesn't work on RTX 5080, options:
    a) Train on CPU (slow but works for small model)
    b) Use CuPy to implement a minimal transformer decoder
    c) Use JAX/Flax if installed
    d) Train on a cloud GPU and deploy locally
  - Target: 50M-100M parameters max
  - Loss: cross-entropy on next-token prediction, conditioned on field state

Step 3 — Evaluate:
  - Given ONLY a field state (no input text), generate text
  - Measure: BLEU/ROUGE against original response, perplexity,
    human evaluation of coherence (does it make sense?) and
    relevance (does it relate to the field's BFECDS signature?)
  - Baselines:
    * Random text from vocabulary
    * Nearest-neighbor retrieval (return text of closest stored memory)
    * Small LLM given BFECDS as text prompt ("B=0.3 F=0.7 E=0.2...")

**Key code to build on:**
  - eris/memory/autobiography.py → load_all() for training data
  - eris/orchestrator.py → generates the (field, text) pairs
  - Training code: write from scratch (CuPy or NumPy)

**Deliverable:** Training pipeline, model weights, evaluation results,
example outputs. Even rough coherence is a meaningful result — it proves
field states carry enough information to reconstruct linguistic content,
which validates the "tiny translator replaces LLM" thesis.

---

## RUNNING ORDER RECOMMENDATION

1. **Experiment 3 first** (zero-weight Clifford classifier) — most dramatic
   result, directly validates the core thesis, doesn't need an LLM backend.
   Also produces the Clifford algebra implementation that Experiments 2
   and 4 can reuse.

2. **Experiment 1 second** (hallucination detection) — fastest to a clear
   metric (dC/dX distributions). Needs an LLM backend for full validation
   but can run field-metrics-only without one.

3. **Experiment 2 third** (retrieval comparison) — needs a corpus ingested
   first, which Experiment 1's question set partially provides.

4. **Experiment 4 last** (field-to-text translator) — needs training data
   accumulated from Experiments 1-3 running through the orchestrator.

---

## HARDWARE AND CONSTRAINTS

- Alienware Aurora, Intel Ultra 9 285K, 64GB DDR5, RTX 5080 (16GB VRAM)
- CuPy works (cupy-cuda13x). PyTorch does NOT (sm_120 Blackwell unsupported)
- Use CuPy or NumPy for all GPU/CPU compute
- cp.roll stencils for PDE, NOT cupyx.scipy.ndimage (broken on CUDA 13.2)
- Fill VRAM to 12-14GB max. State headroom explicitly.
- NEVER add arbitrary time caps. Include checkpoint saving every 5-10 min.
- Include --resume logic for any run over 30 minutes.
- Ask before imposing time budgets.

## WHAT SUCCESS LOOKS LIKE

Experiment 1: ROC-AUC > 0.7 for dC/dX as hallucination classifier.
Experiment 2: Field retrieval beats cosine by measurable margin on precision@k.
Experiment 3: Zero-weight accuracy > 70% on cats-vs-dogs, with blade control demo.
Experiment 4: Generated text is coherent and relates to field state content.

Any ONE of these succeeding is a publishable result. All four together
is a new paradigm. Build them clean, build them reproducible, build them
on top of the v4 code that already works.
