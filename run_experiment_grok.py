#!/usr/bin/env python3
"""
ERIS v4 — Tier 5: The Grokking Experiment (falsifiable harness)
===============================================================
Implements the two experiments from ERIS_V4_REMEDIATION Tier 5, using the
instruments built in Tier 4:

  * eris.knowledge.web_reader   — dual-track reading -> field attractors in LTM
  * eris.retrieval.field_interference — R_ij resonance + resonance_vs_cosine
  * eris.knowledge.embeddings   — query/record embeddings for the cosine baseline

Experiment 5A (the GATING test): does the field carry *non-redundant* relational
structure? For held-out probes, retrieve the top neighbor two ways — embedding
COSINE vs field RESONANCE (R_ij) — and report agreement vs divergence.
  Bar: if R_ij just tracks cosine -> field is decorative for retrieval (stop
  gilding it). If R_ij surfaces real cross-domain relationships cosine misses
  -> first concrete evidence the field carries relational geometry.

Experiment 5B (the SHARP test, only meaningful if 5A passes): is grok a phase
transition? Ingest examples of ONE concept incrementally (N = 1, 2, 4, 8, ...);
after each step measure "basin width" = fraction of held-out related probes that
resolve into the attractor (high R_ij). A *sharp* jump at some N supports
grok-as-phase-transition; a smooth curve says ordinary accumulation.

Runs OFFLINE by default on a small built-in corpus (deliberately mixing
near-domain and far-domain pairs: immune-system/firewall, predator-prey/arms-race,
resonance/consensus) so it is reproducible without network. Use --online to read
real Wikipedia articles instead. Checkpoint-safe: pass --resume to continue.

Usage:
    python run_experiment_grok.py --experiment both
    python run_experiment_grok.py --experiment 5a --online --articles "Kuramoto model,Immune system,Firewall (computing)"
    python run_experiment_grok.py --experiment 5b --resume
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eris.memory.tiers import MemorySystem, MemoryRecord
from eris.knowledge.extractor import KnowledgeExtractor
from eris.knowledge.embeddings import get_embedding
from eris.knowledge import web_reader
from eris.retrieval.field_interference import (
    FieldInterferenceRetriever, field_resonance, resonance_vs_cosine,
)

CKPT = "grok_experiment_results.json"

# ── Built-in offline corpus: near-domain pairs share STRUCTURE not vocabulary ──
OFFLINE_CORPUS = {
    "Immune system": "The immune system defends an organism by detecting pathogens, "
        "distinguishing self from non-self, mounting a layered response, and retaining "
        "memory of past threats so future intrusions are neutralized faster.",
    "Firewall (computing)": "A network firewall defends a system by inspecting traffic, "
        "distinguishing trusted from untrusted packets, filtering intrusions through "
        "layered rules, and logging past attacks so future intrusions are blocked faster.",
    "Predator-prey": "In predator and prey dynamics, each adaptation by one side pressures "
        "a counter-adaptation by the other, producing an escalating cycle of offense and "
        "defense that never settles into equilibrium.",
    "Arms race": "In an arms race, each capability built by one party pressures a "
        "counter-capability by the rival, producing an escalating cycle of offense and "
        "defense that never settles into equilibrium.",
    "Resonance": "Resonance occurs when a system is driven near its natural frequency; "
        "small repeated inputs align in phase and amplify into a large coherent response.",
    "Consensus": "Consensus emerges when many agents repeatedly exchange signals and align "
        "their states in phase, amplifying small agreements into a large coherent decision.",
    "Bread baking": "Baking bread mixes flour, water, salt and yeast; fermentation produces "
        "gas that leavens the dough before the loaf is baked in a hot oven.",
    "Tax accounting": "Tax accounting records income and deductible expenses across a fiscal "
        "year to compute liabilities owed to the relevant revenue authority.",
}
# Known-related pairs (same structure, different words) — the cross-domain analogies.
RELATED_PAIRS = [("Immune system", "Firewall (computing)"),
                 ("Predator-prey", "Arms race"),
                 ("Resonance", "Consensus")]


def build_kb(extractor, memory, *, online, articles):
    titles = articles or list(OFFLINE_CORPUS.keys())
    stats = {}
    for t in titles:
        if online:
            stats[t] = web_reader.read_wikipedia(t, extractor=extractor, memory=memory)
        else:
            stats[t] = web_reader.ingest_text(OFFLINE_CORPUS.get(t, t), title=t,
                                              extractor=extractor, memory=memory)
        time.sleep(0.0 if not online else 1.0)
    return stats, titles


def _field_of(extractor, text):
    """Run the PDE on `text` and return (phi, theta) snapshot + embedding."""
    desc = extractor.extract_text(text)[0]
    return desc.phi_snapshot, desc.theta_snapshot, get_embedding(text)


def experiment_5a(extractor, memory, titles):
    """Field-resonance vs embedding-cosine top-neighbor agreement."""
    probes = []
    for t in titles:
        text = OFFLINE_CORPUS.get(t, t)
        phi, theta, emb = _field_of(extractor, text)
        probes.append({"title": t, "phi": phi, "theta": theta, "embedding": emb})
    report = resonance_vs_cosine(memory, probes)
    # Did the field rank the KNOWN cross-domain analogue highly where cosine didn't?
    retr = FieldInterferenceRetriever(memory)
    analogy_hits = 0
    for a, b in RELATED_PAIRS:
        pa = next((p for p in probes if p["title"] == a), None)
        if not pa:
            continue
        top = retr.retrieve(pa["phi"], pa["theta"], k=3)
        names = [(r.metadata or {}).get("title", "") for _, r in top]
        if b in names:
            analogy_hits += 1
    report["analogy_recall_at3"] = analogy_hits / max(1, len(RELATED_PAIRS))
    return report


def experiment_5b(extractor, memory_factory, concept_texts, probe_texts, n_grid):
    """Basin width vs N — incremental ingestion of one concept."""
    results = []
    for N in n_grid:
        mem = memory_factory()
        for txt in concept_texts[:N]:
            web_reader.ingest_text(txt, title="concept", extractor=extractor, memory=mem)
        retr = FieldInterferenceRetriever(mem)
        if mem.ltm.size == 0:
            results.append({"N": N, "basin_width": 0.0}); continue
        # threshold = mean self-resonance of stored attractors
        selfR = [field_resonance(r.phi_snapshot, r.theta_snapshot,
                                 r.phi_snapshot, r.theta_snapshot)
                 for r in mem.ltm._records]
        thresh = 0.5 * float(np.mean(selfR)) if selfR else 0.0
        resolved = 0
        for pt in probe_texts:
            phi, theta, _ = _field_of(extractor, pt)
            top = retr.retrieve(phi, theta, k=1)
            if top and top[0][0] >= thresh:
                resolved += 1
        results.append({"N": N, "basin_width": resolved / max(1, len(probe_texts)),
                        "threshold": thresh})
    # crude sharpness: max jump between consecutive basin widths
    widths = [r["basin_width"] for r in results]
    jumps = [widths[i+1]-widths[i] for i in range(len(widths)-1)]
    return {"curve": results, "max_jump": max(jumps) if jumps else 0.0,
            "verdict": "sharp (phase-transition-like)" if (jumps and max(jumps) >= 0.4)
                       else "smooth (ordinary accumulation)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["5a", "5b", "both"], default="both")
    ap.add_argument("--online", action="store_true", help="read real Wikipedia (needs network)")
    ap.add_argument("--articles", default="", help="comma-separated titles (online mode)")
    ap.add_argument("--field-size", type=int, default=16)
    ap.add_argument("--pde-steps", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = {}
    if args.resume and os.path.exists(CKPT):
        out = json.load(open(CKPT))
        print(f"[resume] loaded prior results: {list(out)}")

    articles = [a.strip() for a in args.articles.split(",") if a.strip()] or None
    extractor = KnowledgeExtractor(output_dir="grok_kb", field_size=args.field_size,
                                   pde_steps=args.pde_steps)

    def mem_factory():
        import tempfile
        return MemorySystem(data_dir=tempfile.mkdtemp())

    if args.experiment in ("5a", "both") and "5a" not in out:
        mem = mem_factory()
        stats, titles = build_kb(extractor, mem, online=args.online, articles=articles)
        print(f"[5A] knowledge base: {stats}")
        out["5a"] = experiment_5a(extractor, mem, titles)
        print(f"[5A] {json.dumps(out['5a'], indent=2)}")
        json.dump(out, open(CKPT, "w"), indent=2)

    if args.experiment in ("5b", "both") and "5b" not in out:
        # concept = 'defense via layered filtering w/ memory'; probes = related variants
        concept = [OFFLINE_CORPUS["Immune system"], OFFLINE_CORPUS["Firewall (computing)"],
                   "Antivirus software scans files, quarantines known threats, and updates "
                   "its signatures so future infections are caught faster.",
                   "A castle defends with walls, gates, watchmen and records of past sieges "
                   "so the next assault is repelled sooner.",
                   "An intrusion detection system watches for anomalies, isolates suspicious "
                   "activity, and remembers prior attacks to respond faster next time."]
        probes = ["A spam filter learns which messages are malicious and blocks future ones.",
                  "White blood cells remember a pathogen and respond faster on re-exposure.",
                  "A recipe for chocolate cake with butter, sugar, and eggs."]  # 1 unrelated
        out["5b"] = experiment_5b(extractor, mem_factory, concept, probes,
                                  n_grid=[1, 2, 4, 5])
        print(f"[5B] {json.dumps(out['5b'], indent=2)}")
        json.dump(out, open(CKPT, "w"), indent=2)

    print(f"\n[done] results -> {CKPT}")


if __name__ == "__main__":
    main()
