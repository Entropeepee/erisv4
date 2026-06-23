"""Self-seeking topic selection — what Eris studies when nobody told her to.

The old autonomous path only cycled the three default study topics. For her to
become broadly intelligent she has to range: pull from a curated cross-domain
seed list, from the topics the user pointed her at, and from her own recent
thinking (thought-stream / conversation), while rotating away from what she just
studied so she explores instead of looping.

This is the input end of continuous study — it picks WHAT to learn; the study
engine handles read → comprehend → store.
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence
import random

# A broad, deliberately cross-domain seed list (authoritative, encyclopedic
# subjects — not the open web). She expands beyond these via her own topics and
# what she reads; these just guarantee she always has somewhere intelligent to go.
CURATED_SEEDS: List[str] = [
    # physics / dynamics
    "Statistical mechanics", "Nonlinear dynamics", "Chaos theory",
    "Renormalization group", "Phase transition", "Synchronization (physics)",
    "Kuramoto model", "Dissipative system", "Ising model", "Critical phenomena",
    # complexity / self-organization
    "Complex system", "Self-organization", "Emergence", "Self-organized criticality",
    "Network science", "Agent-based model", "Cellular automaton", "Cybernetics",
    "Autopoiesis", "Dynamical systems theory",
    # cognition / mind / neuro
    "Cognitive science", "Predictive coding", "Free energy principle",
    "Embodied cognition", "Neural oscillation", "Global workspace theory",
    "Integrated information theory", "Active inference", "Consciousness",
    "Attractor network",
    # information / computation / AI
    "Information theory", "Algorithmic information theory", "Bayesian inference",
    "Machine learning", "Representation learning", "Transformer (deep learning)",
    "Reinforcement learning", "Computational neuroscience", "Category theory",
    "Dynamical systems and computation",
    # math / logic / foundations
    "Differential geometry", "Topology", "Group theory", "Measure theory",
    "Stochastic process", "Graph theory", "Control theory", "Optimization",
    # mind / contemplative / philosophy
    "Philosophy of mind", "Phenomenology (philosophy)", "Buddhist philosophy",
    "Madhyamaka", "Process philosophy", "Systems theory", "Enactivism",
]


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def pick_topics(n: int = 1, *, recent: Optional[Iterable[str]] = None,
                extra: Optional[Sequence[str]] = None,
                rng: Optional[random.Random] = None) -> List[str]:
    """Choose `n` study topics, biased toward her own interests but ranging widely.

    Pool = the user's pointed-at topics + her own recent topics (`extra`, weighted
    by appearing first) + the curated cross-domain seeds. Excludes anything in
    `recent` (so she rotates), de-dupes, and shuffles within tiers so successive
    calls don't march down a fixed list."""
    r = rng or random
    seen = {_norm(x) for x in (recent or [])}
    out: List[str] = []

    extra_list = list(extra or [])
    r.shuffle(extra_list)                       # her own interests, randomized order
    seeds = list(CURATED_SEEDS)
    r.shuffle(seeds)

    for pool in (extra_list, seeds):            # her interests first, then the seeds
        for t in pool:
            k = _norm(t)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(t.strip())
            if len(out) >= n:
                return out
    # If everything was excluded by `recent`, fall back to the seeds ignoring recent.
    if not out:
        seeds2 = list(CURATED_SEEDS)
        r.shuffle(seeds2)
        out = seeds2[:n]
    return out[:n]
