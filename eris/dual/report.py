"""A/B ruler over the divergence log (§4): hit@k(trad) vs hit@k(novel), the
win/loss/miss tally, and a sample of cross_domain=true rows (the field surfacing
coupled-but-lexically-distant material RAG missed). Descriptive overlap is shown
but is never the verdict.
"""
from __future__ import annotations
from typing import Optional
import json


def _load(path: str):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except (FileNotFoundError, OSError):
        pass
    return rows


def summarize(path: str) -> dict:
    rows = [r for r in _load(path) if "verdict" in r]
    n = len(rows)
    if not n:
        return {"turns": 0}

    def _succ(side):
        return [float(r[side]["arbiter"].get("success", 0.0)) for r in rows]

    def _hit(side):
        vals = [r[side]["arbiter"].get("gold_at_k") for r in rows
                if r[side]["arbiter"].get("gold_at_k") is not None]
        return (sum(vals) / len(vals)) if vals else None

    verdicts = [r["verdict"] for r in rows]
    return {
        "turns": n,
        "novel_wins": verdicts.count("novel_wins"),
        "trad_wins": verdicts.count("trad_wins"),
        "tie": verdicts.count("tie"),
        "both_miss": verdicts.count("both_miss"),
        "mean_success": {"trad": round(sum(_succ("trad")) / n, 4),
                         "novel": round(sum(_succ("novel")) / n, 4)},
        "hit_at_k": {"trad": _hit("trad"), "novel": _hit("novel")},
        "cross_domain": sum(1 for r in rows if r.get("cross_domain")),
        "mean_overlap": round(sum(r.get("overlap", 0.0) for r in rows) / n, 4),
    }


def print_report(path: str, sample: int = 5) -> dict:
    s = summarize(path)
    print("── DualPath retrieval shadow report ──")
    if not s.get("turns"):
        print("(no shadow turns logged yet)")
        return s
    print(f"turns={s['turns']}  novel_wins={s['novel_wins']}  trad_wins={s['trad_wins']}"
          f"  tie={s['tie']}  both_miss={s['both_miss']}")
    print(f"mean success: trad={s['mean_success']['trad']}  novel={s['mean_success']['novel']}")
    print(f"hit@k: trad={s['hit_at_k']['trad']}  novel={s['hit_at_k']['novel']}")
    print(f"cross_domain rows (field found what RAG missed): {s['cross_domain']}")
    print(f"mean overlap (descriptive only): {s['mean_overlap']}")
    cross = [r for r in _load(path) if r.get("cross_domain")][:sample]
    for r in cross:
        print(f"  · cross_domain: {r.get('query','')[:80]}")
    return s


if __name__ == "__main__":   # pragma: no cover
    import sys
    print_report(sys.argv[1] if len(sys.argv) > 1
                 else "eris_data/dual/divergence.jsonl")
