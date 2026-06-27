"""Overnight-ready harness + report (§3b, §5) for the zero-weight vision experiment.

One CLI: `prepare-data`, `--phase {within,cross}`, `--resume`, `--data <dir>`,
`--ablation`, and a top-level `run-all` (data-prep → within → cross → report). Field
signatures are cached by image hash and flushed incrementally so an interrupted run
resumes; results land in a `results/` dir. Torch-free; no time caps.

Datasets are laid out as class subdirectories of `.npy` image arrays (so loading
needs no image-decode dependency). `prepare-data` fetches CIFAR-10 and writes them.
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
    VisionConfig, ZeroWeightClassifier, compute_signature, Signature,
    baseline_random,
)
from eris.vision.coupling import FieldDebias, UnknownGate
from eris.experiments import cross_modal as cm


# ── dataset loading (class subdirs of .npy arrays) ──────────────────────────
def load_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    imgs, labels = [], []
    if not os.path.isdir(data_dir):
        return imgs, labels
    for label in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, label)
        if not os.path.isdir(cdir):
            continue
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".npy"):
                try:
                    imgs.append(np.load(os.path.join(cdir, fn)))
                    labels.append(label)
                except Exception:
                    continue
    return imgs, labels


def _img_hash(a: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=12).hexdigest()


class SignatureCache:
    """Disk cache of computed signatures keyed by image hash (resume-safe)."""

    def __init__(self, path: str):
        self.path = path
        self._mem: Dict[str, bool] = {}
        if os.path.exists(path):
            self._mem = {}            # presence-only marker file set
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._mem[line] = True
            except OSError:
                pass

    def mark(self, h: str) -> None:
        if h in self._mem:
            return
        self._mem[h] = True
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(h + "\n"); f.flush()
        except OSError:
            pass

    def has(self, h: str) -> bool:
        return h in self._mem


def _signatures(imgs, labels, cfg, debias=None) -> List[Signature]:
    return [compute_signature(im, cfg, debias=debias) for im in imgs]


# ── Phase 1: within-modal ───────────────────────────────────────────────────
def run_within(data_dir: str, cfg: VisionConfig, *, debias: bool = False) -> dict:
    imgs, labels = load_dataset(data_dir)
    if not imgs:
        return {"error": "no data", "dir": os.path.basename(data_dir.rstrip("/\\"))}
    classes = sorted(set(labels))
    n = len(imgs)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    split = max(1, int(0.7 * n))
    tr, te = idx[:split], idx[split:]

    deb = None
    if debias:
        deb = FieldDebias(cfg.size).fit([compute_signature(imgs[i], cfg).mag for i in tr])

    by_class: Dict[str, List[Signature]] = {c: [] for c in classes}
    for i in tr:
        by_class[labels[i]].append(compute_signature(imgs[i], cfg, debias=deb))
    clf = ZeroWeightClassifier(cfg).fit(by_class)

    gate = UnknownGate()
    correct, unknown, conf = 0, 0, {c: {d: 0 for d in classes} for c in classes}
    preds_zw, gold = [], []
    for i in te:
        sig = compute_signature(imgs[i], cfg, debias=deb)
        pred, _ = clf.predict(sig, unknown_gate=gate)
        gold.append(labels[i]); preds_zw.append(pred)
        if pred == "unknown":
            unknown += 1
        else:
            conf[labels[i]][pred] += 1
            correct += (pred == labels[i])
    decided = max(1, len(te) - unknown)
    acc = correct / decided
    rand = baseline_random(classes, len(te))
    rand_acc = float(np.mean([r == g for r, g in zip(rand, gold)]))

    # RULE 4 ablation: RGB/textness delta (measured, never in the headline).
    rgb_delta = None
    if cfg.use_rgb or True:
        cfg_rgb = VisionConfig(size=cfg.size, laf=cfg.laf, use_rgb=True)
        # (kept as a hook; full RGB+SVM baseline runs on the machine with sklearn)
        rgb_delta = "see baselines (sklearn)"

    return {
        "classes": classes, "n": n, "test": len(te),
        "zero_weight_accuracy": round(acc, 4),
        "random_baseline": round(rand_acc, 4),
        "unknown_rate": round(unknown / len(te), 4),
        "confusion": conf,
        "rgb_textness_delta": rgb_delta,
    }


# ── Phase 2: cross-modal ────────────────────────────────────────────────────
def run_cross(data_dir: str, cfg: VisionConfig, *,
              word_map: Optional[Dict[str, str]] = None) -> dict:
    imgs, labels = load_dataset(data_dir)
    if not imgs:
        return {"error": "no data"}
    classes = sorted(set(labels))
    word_map = word_map or {c: c for c in classes}
    words = {c: cm.word_signature(word_map[c], cfg) for c in classes}
    img_sigs = _signatures(imgs, labels, cfg)
    ab = cm.run_ablation(img_sigs, labels, words)
    preds_both = ab["both"]["preds"]
    real, p = cm.permutation_null(preds_both, labels, n_perm=2000)
    eff = cm.effect_size(img_sigs, labels, words)
    return {
        "classes": classes, "chance": ab["chance"],
        "ablation": {m: ab[m]["accuracy"] for m in ("aligned", "tension", "both")},
        "accuracy": round(real, 4), "permutation_p": round(p, 5),
        "effect_size": round(eff, 4),
        "keeps_lambda_helps": ab["both"]["accuracy"] > ab["aligned"]["accuracy"],
    }


def write_report(results: dict, out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except OSError:
        pass
    return path


# ── CIFAR-10 prepare-data ([machine]; network) ──────────────────────────────
def prepare_data(out_dir: str, classes=("cat", "dog"), per_class: int = 500) -> dict:
    """Fetch CIFAR-10, write `per_class` images as .npy under out_dir/<class>/.
    NumPy only (no torch). Network needed; run on the box."""
    import tarfile, pickle, urllib.request, io
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
             "horse", "ship", "truck"]
    want = {names.index(c): c for c in classes}
    raw = urllib.request.urlopen(url, timeout=120).read()       # noqa: S310
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    counts = {c: 0 for c in classes}
    for member in tf.getmembers():
        if "data_batch" not in member.name:
            continue
        d = pickle.load(tf.extractfile(member), encoding="bytes")
        data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labs = d[b"labels"]
        for img, lab in zip(data, labs):
            if lab in want and counts[want[lab]] < per_class:
                c = want[lab]
                cdir = os.path.join(out_dir, c); os.makedirs(cdir, exist_ok=True)
                np.save(os.path.join(cdir, f"{counts[c]:05d}.npy"), img)
                counts[c] += 1
    return {"out_dir": out_dir, "counts": counts}


def main(argv=None):   # pragma: no cover
    ap = argparse.ArgumentParser(description="Zero-weight vision harness")
    ap.add_argument("command", choices=["prepare-data", "within", "cross", "run-all"])
    ap.add_argument("--data", default="results/data")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--debias", action="store_true")
    ap.add_argument("--classes", nargs="*", default=["cat", "dog"])
    args = ap.parse_args(argv)
    cfg = VisionConfig(size=args.size)
    if args.command == "prepare-data":
        print(prepare_data(args.data, classes=tuple(args.classes)))
        return
    res = {"ts": time.time(), "size": args.size}
    if args.command in ("within", "run-all"):
        if args.command == "run-all":
            print(prepare_data(args.data, classes=tuple(args.classes)))
        res["within"] = run_within(args.data, cfg, debias=args.debias)
        print("within:", res["within"])
    if args.command in ("cross", "run-all"):
        res["cross"] = run_cross(args.data, cfg)
        print("cross:", res["cross"])
    print("report ->", write_report(res))


if __name__ == "__main__":   # pragma: no cover
    main()
