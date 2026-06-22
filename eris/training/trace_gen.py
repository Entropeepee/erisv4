"""Distillation trace generation (roadmap 2.1) — the s1/R1-distill data step.

Harvest reasoning/answer traces from a TEACHER (a cloud model or a local 32B) on
*Eris's own* tasks, into a JSONL file you then QLoRA-distill into an 8–14B student
with Unsloth (machine-side, 2.2). The s1 lesson: a small set of high-quality
traces beats a large noisy one — so this harness is about clean, resumable
collection, not volume.

Design:
  • backend-agnostic — pass any object with `async generate(prompt, system, ...)`
    (Eris's mediator, a single cloud backend, …).
  • resumable — re-running skips prompts already in the output file, so a long
    harvest survives interruptions and you can append new tasks anytime.
  • offline-testable — works with a stub backend; no network required to verify
    the harness itself.

Output JSONL schema (one object per line), Alpaca-friendly (lean, not ChatML — an
Unsloth OOM watch-out for small cards):
  {"prompt": ..., "system": ..., "response": ..., "reasoning": ..., "ts": ...}
"""
from __future__ import annotations
from typing import Iterable, Union, List, Dict, Optional, Callable
import json
import os
import time

Task = Union[str, Dict]


def _as_task(t: Task) -> Dict:
    if isinstance(t, str):
        return {"prompt": t, "system": ""}
    return {"prompt": t.get("prompt", ""), "system": t.get("system", "")}


def _already_done(out_path: str) -> set:
    done = set()
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("prompt", ""))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return done


async def generate_traces(
    teacher,
    tasks: Iterable[Task],
    out_path: str,
    *,
    resume: bool = True,
    temperature: float = 0.7,
    on_trace: Optional[Callable[[Dict], None]] = None,
) -> int:
    """Run each task through `teacher`, appending traces to `out_path` (JSONL).

    Returns the number of NEW traces written. With `resume=True`, prompts already
    present in `out_path` are skipped, so re-runs are cheap and idempotent."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _already_done(out_path) if resume else set()
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for raw in tasks:
            task = _as_task(raw)
            prompt = task["prompt"]
            if not prompt or prompt in done:
                continue
            resp = await teacher.generate(prompt, task["system"], temperature=temperature)
            text = getattr(resp, "text", "") if resp else ""
            if not text.strip():
                continue
            trace = {
                "prompt": prompt,
                "system": task["system"],
                "response": text,
                "reasoning": getattr(resp, "reasoning", "") or "",
                "ts": time.time(),
            }
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f.flush()
            done.add(prompt)
            written += 1
            if on_trace:
                on_trace(trace)
    return written


def load_traces(path: str) -> List[Dict]:
    """Read a trace JSONL back (for inspection or conversion to a training set)."""
    out: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out
