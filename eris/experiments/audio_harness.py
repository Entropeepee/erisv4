"""Overnight-ready audio + grounding harness/report (§3b, §5).

CLI: `prepare-audio` (ESC-50, [machine]/network), `--phase {audio-within, grounding}`,
`--resume`, `--ablation`, and `run-all` (data-prep → audio-within → grounding → report).
Datasets are class subdirs of `.npy` arrays (audio = raw mono samples; image = arrays) so
loading needs no decode deps; words come from the class names through the FRT+PDE path.
Torch-free; resumable signature cache; no time caps. Results land in `results/`.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import argparse
import hashlib
import json
import os
import time

import numpy as np

from eris.experiments.zero_weight_vision import (
    VisionConfig as FieldConfig, ZeroWeightClassifier, baseline_random,
)
from eris.experiments.zero_weight_audio import (
    compute_audio_signature, baseline_mfcc_svm, baseline_logmel_svm,
)
from eris.vision.coupling import UnknownGate
from eris.experiments import grounding as G
from eris.experiments.cross_modal import word_signature


# ── dataset loading (class subdirs of .npy) ─────────────────────────────────
def load_audio_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    xs, ys = [], []
    if not os.path.isdir(data_dir):
        return xs, ys
    for label in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, label)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".npy"):
                try:
                    xs.append(np.load(os.path.join(cdir, fn))); ys.append(label)
                except Exception:
                    continue
    return xs, ys


def _by_class(items, labels) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for it, lb in zip(items, labels):
        out.setdefault(lb, []).append(it)
    return out


# ── Phase 1: audio within-modal ─────────────────────────────────────────────
def run_audio_within(samples_by_class: Dict[str, List[np.ndarray]],
                     cfg: FieldConfig) -> dict:
    classes = [c for c in sorted(samples_by_class) if samples_by_class.get(c)]
    if len(classes) < 2:
        return {"error": "need >=2 non-empty classes", "classes": classes}
    rng = np.random.RandomState(0)
    train, test_sigs, test_y = {}, [], []
    train_samples, train_y, test_samples = [], [], []
    for c in classes:
        items = samples_by_class[c]
        idx = rng.permutation(len(items))
        split = max(1, int(0.7 * len(items)))
        train[c] = [compute_audio_signature(items[i], cfg) for i in idx[:split]]
        for i in idx[:split]:
            train_samples.append(items[i]); train_y.append(c)
        for i in idx[split:]:
            test_sigs.append(compute_audio_signature(items[i], cfg)); test_y.append(c)
            test_samples.append(items[i])
    clf = ZeroWeightClassifier(cfg).fit(train)
    gate = UnknownGate()
    correct = unknown = 0
    preds = []
    for sig, y in zip(test_sigs, test_y):
        pred, _ = clf.predict(sig, unknown_gate=gate)
        preds.append(pred)
        if pred == "unknown":
            unknown += 1
        else:
            correct += (pred == y)
    decided = max(1, len(test_y) - unknown)
    rand = baseline_random(classes, len(test_y))
    out = {
        "classes": classes, "test": len(test_y),
        "zero_weight_accuracy": round(correct / decided, 4),
        "random_baseline": round(float(np.mean([r == g for r, g in zip(rand, test_y)])), 4),
        "unknown_rate": round(unknown / max(1, len(test_y)), 4),
        # RULE 4 ablation arm (sklearn-guarded; runs on the machine)
        "mfcc_svm": _baseline_acc(baseline_mfcc_svm, train_samples, train_y, test_samples, test_y),
        "logmel_svm": _baseline_acc(baseline_logmel_svm, train_samples, train_y, test_samples, test_y),
    }
    return out


def _baseline_acc(fn, tr_x, tr_y, te_x, te_y):
    preds = fn(tr_x, tr_y, te_x)
    if preds is None:
        return "n/a (no sklearn)"
    return round(float(np.mean([p == g for p, g in zip(preds, te_y)])), 4)


# ── Phase 2: cross-modal grounding ──────────────────────────────────────────
def _fields_audio(samples_by_class, cfg) -> Dict[str, list]:
    from eris.knowledge.frontends import AudioFrontend
    af = AudioFrontend()
    return {c: [af.to_field(s, size=cfg.size) for s in items]
            for c, items in samples_by_class.items()}


def _fields_image(images_by_class, cfg) -> Dict[str, list]:
    from eris.knowledge.frontends import ImageFrontend
    im = ImageFrontend()
    return {c: [im.to_field(a, size=cfg.size) for a in items]
            for c, items in images_by_class.items()}


def _fields_word(classes, cfg, n_each: int = 1) -> Dict[str, list]:
    out = {}
    for c in classes:
        w = word_signature(c, cfg)
        out[c] = [(w.mag, w.theta)] * n_each
    return out


def _pair_report(gs, gt, gcfg, *, with_ablations: bool = True) -> dict:
    """gs, gt are DESCRIPTOR dicts ({label: [complex vec, ...]})."""
    rep = {
        "ceiling_src": round(G.within_modal_ceiling(gs, gcfg), 4),
        "unitary": {str(n): v["acc"] for n, v in
                    G.grounding_curve(gs, gt, gcfg, kind="unitary").items()},
        "shuffled": {str(n): v["acc"] for n, v in
                     G.grounding_curve(gs, gt, gcfg, kind="unitary", shuffle=True).items()},
    }
    if with_ablations:
        rep["cosine_fit"] = {str(n): v["acc"] for n, v in
                             G.grounding_curve(gs, gt, gcfg, kind="cosine").items()}
        rep["channels"] = {ch: G.grounding_curve(gs, gt, gcfg, kind="unitary",
                                                 channels=ch)[max(gcfg.Ns)]["acc"]
                           for ch in ("aligned", "tension", "both")}
    return rep


def run_grounding(audio_fields, image_fields, word_fields,
                  gcfg: Optional[G.GroundingConfig] = None) -> dict:
    gcfg = gcfg or G.GroundingConfig()
    grid = gcfg.descriptor_grid
    # descriptors (for alignment/transitivity) computed ONCE; fields kept for zero-shot.
    ga = G.descriptors_by_class(audio_fields, grid) if audio_fields else {}
    gi = G.descriptors_by_class(image_fields, grid) if image_fields else {}
    gw = G.descriptors_by_class(word_fields, grid) if word_fields else {}
    pairs = {}
    if ga and gw:
        pairs["audio->word"] = _pair_report(ga, gw, gcfg)
    if ga and gi:
        pairs["audio->image"] = _pair_report(ga, gi, gcfg)
    if gi and gw:
        pairs["image->word"] = _pair_report(gi, gw, gcfg)

    zero_shot = {}
    for name, (s, t) in {"audio->word": (audio_fields, word_fields),
                         "audio->image": (audio_fields, image_fields),
                         "image->word": (image_fields, word_fields)}.items():
        if s and t:
            # low-level probe: audio source → transient burstiness; image source → spatial
            # busyness. Separates a bouba/kiki (transient↔spiky) effect from concept transfer.
            featfn = G.transient_score if name.startswith("audio") else G.spatial_busyness
            flags, mags = G.zero_shot_items(s, t)
            acc = float(np.mean(flags)) if flags else 0.0
            probe = G.lowlevel_probe(flags, [featfn(m) for m in mags])
            zero_shot[name] = {"acc": round(acc, 4),
                               "p": round(G.permutation_null(acc, len(flags), len(s)), 5),
                               "lowlevel_probe": {k: round(v, 4) for k, v in probe.items()}}

    out = {"pairs": pairs, "zero_shot": zero_shot, "Ns": list(gcfg.Ns)}
    if ga and gi and gw:
        out["transitivity"] = G.transitivity(ga, gi, gw, gcfg)
    return out


def write_report(results: dict, out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "audio_report.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except OSError:
        pass
    return path


# ── ESC-50 prepare-audio ([machine]; network) ──────────────────────────────
def prepare_audio(out_dir: str, classes=("dog", "cat", "rain", "clock_tick"),
                  per_class: int = 40) -> dict:   # pragma: no cover
    """Fetch ESC-50, write per-class clips as .npy (mono float). NumPy only; network."""
    import io
    import urllib.request
    import zipfile
    import csv
    import wave
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    raw = urllib.request.urlopen(url, timeout=300).read()       # noqa: S310
    zf = zipfile.ZipFile(io.BytesIO(raw))
    meta = [r for r in csv.DictReader(io.TextIOWrapper(
        zf.open("ESC-50-master/meta/esc50.csv")))]
    want = {c: 0 for c in classes}
    errors = 0
    for row in meta:
        c = row["category"]
        if c in want and want[c] < per_class:
            try:
                wav = zf.open(f"ESC-50-master/audio/{row['filename']}")
                with wave.open(io.BytesIO(wav.read())) as w:
                    nch = w.getnchannels()
                    frames = w.readframes(w.getnframes())
                    x = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
                    if nch > 1:                       # de-interleave stereo → mono mean
                        x = x.reshape(-1, nch).mean(axis=1)
                if x.size == 0 or not np.isfinite(x).all():
                    errors += 1; continue
            except Exception:
                errors += 1; continue                 # one bad file never aborts the prep
            cdir = os.path.join(out_dir, c); os.makedirs(cdir, exist_ok=True)
            np.save(os.path.join(cdir, f"{want[c]:05d}.npy"), x)
            want[c] += 1
    return {"out_dir": out_dir, "counts": want, "errors": errors}


def main(argv=None):   # pragma: no cover
    ap = argparse.ArgumentParser(description="Audio + cross-modal grounding harness")
    ap.add_argument("command", choices=["prepare-audio", "audio-within", "grounding", "run-all"])
    ap.add_argument("--audio", default="results/audio")
    ap.add_argument("--images", default="results/images")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--classes", nargs="*", default=["dog", "cat", "rain", "clock_tick"])
    args = ap.parse_args(argv)
    cfg = FieldConfig(size=args.size)
    if args.command == "prepare-audio":
        print(prepare_audio(args.audio, classes=tuple(args.classes))); return
    res = {"ts": time.time(), "size": args.size}
    a_x, a_y = load_audio_dataset(args.audio)
    samples_by_class = _by_class(a_x, a_y)
    if args.command in ("audio-within", "run-all") and samples_by_class:
        res["audio_within"] = run_audio_within(samples_by_class, cfg)
        print("audio-within:", res["audio_within"])
    if args.command in ("grounding", "run-all"):
        from eris.experiments.vision_harness import load_dataset
        img_x, img_y = load_dataset(args.images)
        audio_fields = _fields_audio(samples_by_class, cfg) if samples_by_class else {}
        image_fields = _fields_image(_by_class(img_x, img_y), cfg) if img_x else {}
        classes = sorted(set(a_y) | set(img_y))
        word_fields = _fields_word(classes, cfg, n_each=max(2, len(a_x) // max(1, len(classes))))
        res["grounding"] = run_grounding(audio_fields, image_fields, word_fields)
        print("grounding pairs:", list(res["grounding"]["pairs"]))
    print("report ->", write_report(res))


if __name__ == "__main__":   # pragma: no cover
    main()
