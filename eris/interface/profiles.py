"""Per-request mode/profile selector — Fast by default, Deep on toggle, extensible.

A "model picker" like Claude's, but the cheap/instant axis is the **machinery**
(token budget, test-time compute, reasoning depth, gates, field steps), not the
model. Fast and Deep share the same Ollama model and differ only on those knobs,
so switching is instant — no reboot, no model reload. Profiles that name a
different `model` incur Ollama's load delay on first use.

Profiles come from `<data_dir>/profiles.json` (user-editable, re-read on demand);
if the file is missing or malformed we fall back to the built-in Fast + Deep.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import json
import os


@dataclass
class Profile:
    id: str = "fast"
    label: str = "Fast"
    model: str = ""           # "" => the orchestrator's default local model
    base_url: str = ""        # "" => default (Ollama / ERIS_LLM_BASE_URL)
    max_tokens: int = 1024    # caps <think> + answer length — the big latency knob
    temperature: float = 0.7
    ttc: bool = False         # test-time compute (self-consistency) — costly
    ttc_max_samples: int = 3
    reasoning: str = "low"    # low|medium|high (bounded primarily by max_tokens)
    orchestration: bool = False   # use the formalized criticality router this turn
    critic: bool = False      # run the deep calibration-critic pass before finalizing
    field_steps: int = 0      # 0 => CONFIG.pde_steps_per_input
    default: bool = False
    desc: str = ""

    def public(self) -> dict:
        """The subset the cockpit dropdown needs."""
        return {"id": self.id, "label": self.label, "desc": self.desc,
                "default": self.default, "model": self.model}


def _coerce(d: dict) -> Optional[Profile]:
    pid = str(d.get("id") or "").strip()
    if not pid:
        return None
    try:
        return Profile(
            id=pid, label=str(d.get("label", pid)),
            model=str(d.get("model", "")), base_url=str(d.get("base_url", "")),
            max_tokens=int(d.get("max_tokens", 1024)),
            temperature=float(d.get("temperature", 0.7)),
            ttc=bool(d.get("ttc", False)),
            ttc_max_samples=int(d.get("ttc_max_samples", 3)),
            reasoning=str(d.get("reasoning", "low")),
            orchestration=bool(d.get("orchestration", False)),
            critic=bool(d.get("critic", False)),
            field_steps=int(d.get("field_steps", 0)),
            default=bool(d.get("default", False)),
            desc=str(d.get("desc", "")),
        )
    except (ValueError, TypeError):
        return None


def builtin_profiles() -> List[Profile]:
    return [
        Profile(id="fast", label="Fast", default=True, max_tokens=4096,
                ttc=False, reasoning="low", orchestration=False, field_steps=30,
                desc="Quick chat. Brief reasoning, but a full (untruncated) answer."),
        Profile(id="deep", label="Deep reasoning", max_tokens=8192,
                ttc=True, ttc_max_samples=3, reasoning="high", orchestration=True,
                critic=True, field_steps=50,
                desc="Slow, thorough. Full reasoning + multi-sample + calibration "
                     "critic. For hard work."),
    ]


def reasoning_system(system: str, reasoning: str) -> str:
    """Prepend a reasoning-effort hint that gpt-oss honors (low|medium|high), so
    'Fast' gets SHORT thinking (and thus speed) instead of a truncated answer.
    Harmless for models that ignore it."""
    r = (reasoning or "").strip().lower()
    if r in ("low", "medium", "high"):
        return f"Reasoning: {r}\n\n{system}"
    return system


class ProfileStore:
    """Loads profiles from `<data_dir>/profiles.json`, falling back to built-ins.
    Re-read on demand so editing the file needs no reboot."""

    def __init__(self, data_dir: str):
        self.path = os.path.join(data_dir, "profiles.json")
        self._profiles: List[Profile] = []
        self.reload()

    def reload(self) -> None:
        profs: List[Profile] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                profs = [p for p in (_coerce(d) for d in data
                                     if isinstance(d, dict)) if p]
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            profs = []
        if not profs:
            profs = builtin_profiles()
        if not any(p.default for p in profs):
            profs[0].default = True
        self._profiles = profs

    def list(self) -> List[Profile]:
        return list(self._profiles)

    def default(self) -> Profile:
        return next((p for p in self._profiles if p.default), self._profiles[0])

    def get(self, pid: str) -> Profile:
        if not pid:
            return self.default()
        return next((p for p in self._profiles if p.id == pid), self.default())
