"""Test-time compute (TTC) — smarter without a bigger model.

Roadmap item 0.3. This is where the patent's six LLM ideas finally land on a
boundary that *has the convergent signal Eris's field lacked*: sampling the same
prompt several times and watching the **consensus answer stabilize**. As more
samples come in, the chosen (medoid) answer stops changing — that is a genuine
"more compute would not change the answer within tolerance" event, exactly the
criticality gate's premise. So self-consistency here reuses the Tier-1
`NoiseFloorEstimator` + `CriticalityMonitor` to STOP SAMPLING EARLY once the
answer has converged, instead of always paying for the full N samples.

Two capabilities:
  • `self_consistent_generate` — sample up to N responses, return the consensus
    (medoid by embedding), with an adaptive criticality early-stop. Quality lever
    (variance reduction / outlier rejection) whose cost the gate minimizes.
  • `budget_forced_generate` — s1-style minimum-thinking budget via "Wait"
    continuations. Needs a thinking-capable model to matter; approximate hook.

Both are DEFAULT OFF (`CONFIG.ttc_*` / `ERIS_TTC`). Self-consistency costs N× LLM
calls when on, so it is meant for hard/important turns, not every turn.

The estimator/monitor are created FRESH per call: each turn's sampling is an
independent computation, so a per-call noise floor is the correct scope here
(the cross-boundary shared-σ of the orchestration gates does not apply).
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Callable
import numpy as np

from eris.config import CONFIG
from eris.computation.noise_floor import NoiseFloorEstimator
from eris.computation.criticality import CriticalityMonitor, Decision


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else a


async def self_consistent_generate(
    mediator,
    prompt: str,
    system: str = "",
    *,
    min_samples: Optional[int] = None,
    max_samples: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    k: Optional[float] = None,
    patience: int = 2,
) -> Tuple[Optional[object], int]:
    """Sample the prompt up to `max_samples` times and return (consensus, n_used).

    Consensus = the medoid response (the sample whose embedding is most central),
    the open-ended analogue of self-consistency's majority vote. Stops early once
    the consensus is stable (`patience` consecutive samples) OR the criticality
    monitor judges the disagreement signal to have settled below its noise floor.

    Returns (None, 0) if no backend produced text.
    """
    min_samples = CONFIG.ttc_min_samples if min_samples is None else min_samples
    max_samples = CONFIG.ttc_max_samples if max_samples is None else max_samples
    temperature = CONFIG.ttc_temperature if temperature is None else temperature
    k = CONFIG.orch_k if k is None else k
    if embed_fn is None:
        from eris.knowledge.embeddings import get_embedding
        embed_fn = get_embedding
    max_samples = max(1, max_samples)
    min_samples = max(1, min(min_samples, max_samples))

    # Fresh, self-contained noise floor for THIS turn's sampling (warmup small so
    # the gate is responsive within a handful of samples).
    estimator = NoiseFloorEstimator(k=k, warmup=2)
    monitor = CriticalityMonitor("ttc", estimator, "self_consistency", k=k)

    samples: List[object] = []
    embs: List[np.ndarray] = []
    last_medoid_text: Optional[str] = None
    stable = 0

    gen_kw = {"temperature": temperature}
    if max_tokens is not None:
        gen_kw["max_tokens"] = max_tokens
    for _ in range(max_samples):
        resp = await mediator.generate(prompt, system, **gen_kw)
        if resp is None or not getattr(resp, "text", "").strip():
            if samples:
                break          # backend dried up; go with what we have
            continue
        samples.append(resp)
        embs.append(_unit(embed_fn(resp.text)))
        n = len(samples)

        # Current consensus (medoid): max total similarity to the others.
        E = np.vstack(embs)
        summed = E.sum(axis=0)
        medoid_idx = int(np.argmax(E @ summed))
        medoid_text = samples[medoid_idx].text
        stable = stable + 1 if medoid_text == last_medoid_text else 0
        last_medoid_text = medoid_text

        if n < min_samples:
            continue

        # Disagreement signal: 1 - mean cosine of samples to their centroid.
        centroid = _unit(summed)
        disagreement = float(1.0 - np.mean(E @ centroid))
        decision, _ = monitor.observe("ttc_disagree", disagreement, {"mode": "settle"})

        # Stop when the consensus has held (robust, deterministic) OR the gate
        # says disagreement has settled below its own floor (adaptive).
        if stable >= patience or decision == Decision.SUSPEND:
            break

    if not samples:
        return None, 0
    E = np.vstack(embs)
    medoid_idx = int(np.argmax(E @ E.sum(axis=0)))
    return samples[medoid_idx], len(samples)


async def budget_forced_generate(
    mediator,
    prompt: str,
    system: str = "",
    *,
    min_thinking_tokens: Optional[int] = None,
    max_extensions: Optional[int] = None,
    temperature: float = 0.7,
) -> Tuple[Optional[object], int]:
    """s1-style budget forcing: keep the model reasoning until it has spent a
    minimum budget, by appending a "Wait" continuation when it stops too early.

    Approximate hook: it uses response length as a proxy for "thought enough"
    (Eris's mediator exposes no thinking-token stream), and only matters with a
    thinking-capable model. Returns (response, n_calls).
    """
    min_thinking_tokens = (CONFIG.ttc_min_thinking_tokens
                           if min_thinking_tokens is None else min_thinking_tokens)
    max_extensions = (CONFIG.ttc_max_extensions
                      if max_extensions is None else max_extensions)
    resp = await mediator.generate(prompt, system, temperature=temperature)
    calls = 1
    if resp is None or min_thinking_tokens <= 0:
        return resp, calls
    text = resp.text
    # ~4 chars/token heuristic; extend while under budget.
    while calls <= max_extensions and len(text) // 4 < min_thinking_tokens:
        cont = await mediator.generate(
            f"{prompt}\n\n{text}\n\nWait, let me reconsider more carefully and continue:",
            system, temperature=temperature)
        calls += 1
        if cont is None or not cont.text.strip():
            break
        text = text + "\n" + cont.text
    resp.text = text
    return resp, calls
