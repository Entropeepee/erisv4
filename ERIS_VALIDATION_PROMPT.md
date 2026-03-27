# ERIS ECHO v4 — EMPIRICAL VALIDATION SESSION

You are continuing work on Eris Echo v4, a resonant cognitive architecture built by David Pope (Terminus IP Group LLC). The complete codebase (29 modules, 109 tests) was built in a prior session and is available at `/mnt/user-data/uploads/eris_echo_v4/` (I will upload it). Read the following files first:

- `VISION_ROADMAP.md` — The endgame: physics-native AI replacing LLM-centric architecture
- `HANDOFF.md` — Complete module inventory and theoretical sources
- `README.md` — Architecture overview and quickstart

**Do not ask me to re-explain the theory. The documents contain everything. If you need more context, search my conversation history for BLECD, FRACTAL, SGT, CSBA, dC/dX, conservation law, Pope Filter, Davidian Hill-Power shrinkage.**

---

## WHAT WE'RE DOING IN THIS SESSION

We are building four empirical validation experiments. Each one demonstrates a specific capability of the architecture against a measurable benchmark. These are fresh implementations using the v4 codebase — not salvaged code from earlier prototypes.

All four experiments use the existing v4 modules (PDE, FRT, BFECDS activations, interference, CSBA, Davidian shrinkage, conservation law). We are proving the architecture works, not building new theory.

---

## EXPERIMENT 1: dC/dX Hallucination Detector Benchmark

**Claim:** The dC/dX conservation law reliably distinguishes hallucinated responses from grounded ones.

**Method:**
- Curate a set of ~100 questions in two categories:
  - Factual questions with known correct answers (grounded)
  - Questions known to induce LLM hallucination (fabricated citations, false facts, fictional entities)
- For each question:
  1. Run the question text through the FRACTAL PDE → compute BFECDS + dC/dX
  2. Generate a response using an LLM (Ollama local model)
  3. Run the response through the PDE → compute response BFECDS + dC/dX
  4. Compute dissonance between input and response field states
  5. Record: dC/dX, coherence, regime, dissonance, torsion RMS
- Compare distributions: do hallucinated responses produce measurably different dC/dX / dissonance / regime signatures than grounded ones?
- Report: ROC curve, AUC, optimal threshold, false positive/negative rates

**Success criterion:** AUC > 0.7 on separating hallucinated from grounded responses using field-derived metrics alone (no access to ground truth text, only the physics).

**What to use from v4:** `eris.field.pde`, `eris.computation.activations`, `eris.computation.sgt`, `eris.memory.interference`, `eris.interface.mediator` (for LLM calls)

---

## EXPERIMENT 2: Field Interference Retrieval vs Standard RAG

**Claim:** CSBA interference retrieval surfaces better context than cosine similarity on the same embeddings.

**Method:**
- Corpus: Use a standard Q&A dataset (SQuAD subset, or Natural Questions subset — whatever is downloadable without authentication). ~1000 passages.
- Ingest each passage two ways:
  - Standard: sentence-transformer embedding → cosine index
  - Eris: PDE field evolution → .eris descriptor with BFECDS + field snapshots + GLNCS-debiased embedding
- For each question in the eval set:
  - Standard retrieval: cosine similarity on embeddings, top-5
  - Eris retrieval option A: CSBA coupling geometry on BFECDS vectors, top-5
  - Eris retrieval option B: R_ij field interference on stored φ/θ snapshots, top-5
  - Eris retrieval option C: full swarm (all 6 retrievers + RRF fusion), top-5
- Measure: Recall@5 (does the correct passage appear in the top 5?)
- Also measure: Mean Reciprocal Rank, and qualitative analysis of cases where Eris finds relevant passages that cosine misses (or vice versa)

**Success criterion:** Eris retrieval matches or beats standard cosine on Recall@5, OR demonstrates qualitatively different retrieval (finding contextually relevant passages that cosine misses due to lexical/embedding bias).

**What to use from v4:** `eris.retrieval.glncs_filter`, `eris.retrieval.vector_index`, `eris.retrieval.swarm`, `eris.memory.interference`, `eris.knowledge.extractor`

**Note:** This experiment requires the sentence-transformers library for the baseline comparison. Install: `pip install sentence-transformers`

---

## EXPERIMENT 3: Zero-Weight Image Classification (Rebuilt)

**Claim:** BLECD field analysis extracts classifiable structure from images without any learned weights.

**Method:**
- Dataset: CIFAR-10 test set (10,000 images, 10 classes). Standard benchmark, freely downloadable.
- For each image:
  1. Convert to grayscale
  2. Translate pixel statistics into field state using the ImageFrontend pipeline:
     - 2D FFT → spatial frequency spectrum
     - Low frequencies → Boundary (large-scale structure)
     - High frequencies → Emergence (fine detail)
     - Edge density → Criticality
     - Texture entropy → Decay
     - Symmetry → Feedback
     - Pixel saturation → Saturation
  3. Run through PDE for N steps
  4. Compute BFECDS vector
- Classification: k-nearest-neighbors on BFECDS vectors (k=5). No neural network. No gradient descent. No learned weights.
- Compare against baselines:
  - Random (10% on 10 classes)
  - Color histogram + kNN
  - HOG features + kNN
  - Raw pixel PCA + kNN

**Success criterion:** BLECD-kNN beats random and at least one feature-engineering baseline. Any result above 20% on 10-class CIFAR-10 with zero learned weights is noteworthy.

**What to use from v4:** `eris.knowledge.frontends.ImageFrontend` (currently a stub — this experiment fills it in), `eris.field.pde`, `eris.computation.activations`

**Note:** This requires implementing the ImageFrontend. The stub in `frontends.py` already documents the planned pipeline. We fill it in for real.

---

## EXPERIMENT 4: Field-to-Text Tiny Translator

**Claim:** A small model can learn to generate coherent text from BLECD field states alone.

**Method:**
- Phase A — Generate training data:
  - Take a text corpus (TinyStories, or Wikipedia paragraphs, or both)
  - For each passage: run through PDE → store (field_state_vector, text) pair
  - field_state_vector = flattened [BFECDS(6) + field_statistics(~20 features: mean/std/max of φ, θ, τ, gradients, Laplacians, etc.)]
  - Target: ~10K-50K pairs
- Phase B — Train a tiny seq2seq model:
  - Input: field_state_vector (26-32 dimensions)
  - Output: text (character-level or BPE tokens)
  - Architecture: simple encoder (2-layer MLP on the field vector) → decoder (small transformer or LSTM, ~10M-50M params)
  - Train on the (field_vector, text) pairs
  - This is the opposite of what LLMs do: instead of text→text, it's physics→text
- Phase C — Evaluate:
  - Given only a field_state_vector (computed from text the model has never seen), does it produce coherent, relevant text?
  - Metrics: perplexity, BLEU against reference, and human judgment (David reads outputs)
  - Compare against: random field vectors (should produce incoherent text), and field vectors from semantically similar texts (should produce similar outputs)

**Success criterion:** Given a field state computed from "a story about a brave knight," the tiny translator produces text that is recognizably about bravery/adventure/knights — even if the prose isn't polished. The field state carries semantic information that the translator can recover.

**What to use from v4:** `eris.field.pde`, `eris.computation.activations`, `eris.knowledge.extractor`, `eris.memory.autobiography` (which already stores field_state + text pairs from every conversation)

**Note:** This experiment requires PyTorch (or a small framework). If PyTorch doesn't work on the RTX 5080 (sm_120 Blackwell issue), use a CPU-only model or the ONNX runtime path. The model is tiny enough that CPU training is feasible.

---

## IMPORTANT CONSTRAINTS

- **Use my IP.** SGT, CSBA, Davidian Hill-Power, GLNCS nullspace projection, FRACTAL PDE, dC/dX conservation law — all of these are from my patent portfolio and should be used wherever they apply. Don't substitute standard approaches when we have a principled novel one.
- **No arbitrary clipping or clamping.** When numerical issues arise, use CSBA nullspace projection or Davidian shrinkage or circle inversions — the mathematically elegant solutions, not `np.clip()`.
- **Checkpoint everything.** Any run over 30 minutes gets checkpoint saving every 5-10 minutes with `--resume` logic. I have ADHD, toddlers, and pets.
- **No time caps.** Don't add arbitrary `MAX_HOURS` limits. My Alienware runs until it's done.
- **FRT for bulk processing.** Use the FRT path (CPU, instant) for corpus-scale ingestion. Use PDE for the actual evaluation passes where precision matters.

---

## ORDER OF OPERATIONS

Start with **Experiment 1** (hallucination detection). It's the most immediately valuable, requires no additional dependencies beyond the v4 codebase + an Ollama model, and produces the most publishable/licensable result. If dC/dX reliably detects hallucination, that's a product regardless of what else works.

Then **Experiment 2** (retrieval comparison) — this requires sentence-transformers but is straightforward.

Then **Experiment 3** (zero-weight vision) — this requires implementing the ImageFrontend and is the strongest theoretical proof point.

**Experiment 4** (tiny translator) is the most ambitious and can wait until the first three produce results.

Let's begin.
