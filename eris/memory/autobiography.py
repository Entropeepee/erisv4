"""
Autobiography Logger
=====================

Every interaction, every dream cycle, every research finding is logged
with computed BFECDS, global observables (C, X, dC/dX), regime detection,
and tension analysis.

This creates a longitudinal record of the system's cognitive trajectory —
the self-referential dataset that can be analyzed in its own BLECD
coordinates. The BLECD framework describing its own emergence through
BLECD coordinates.

From the handoff conversation:
    "If you were to re-process [the corpus] with computed activations —
     using the Chapter 6 PDE criteria instead of LLM assignment — you'd
     have something genuinely unprecedented: a map of how a theoretical
     framework emerged over time, parameterized in its own terms."

Usage:
    from eris.memory.autobiography import Autobiography, AutobiographyEntry

    auto = Autobiography(path="autobiography.jsonl")
    auto.log_interaction(text_in, text_out, bvec_in, bvec_out,
                         coherence=0.8, exchange=0.3, dCdX=0.5,
                         regime="elastic")
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import os
import time

from eris.computation.activations import BVec, bvec_distance


@dataclass
class AutobiographyEntry:
    """A single entry in the system's autobiography.

    Captures both the content and the field-level measurements
    that characterize how the system processed it.
    """
    timestamp: float = field(default_factory=time.time)

    # Content
    input_text: str = ""
    response_text: str = ""

    # Computed field state (NOT LLM-assigned)
    input_bvec: Optional[BVec] = None
    response_bvec: Optional[BVec] = None

    # Global observables (dCdX conservation law)
    coherence: float = 0.0       # C(t) — Kuramoto order parameter
    exchange: float = 0.0        # X(t) — coherence flux
    dCdX: float = 0.0            # Conservation law ratio
    regime: str = "unknown"       # elastic | plastic | transfixed

    # Tension analysis
    dissonance: float = 0.0      # L2 distance between input and response BVecs
    dominant_domain: str = ""     # Which BLECD domain is most active
    archetype: str = ""           # Closest validated archetype

    # Source type
    source: str = "conversation"  # conversation | dream | research | metacognitive

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "input_text": self.input_text,
            "response_text": self.response_text,
            "coherence": self.coherence,
            "exchange": self.exchange,
            "dCdX": self.dCdX,
            "regime": self.regime,
            "dissonance": self.dissonance,
            "dominant_domain": self.dominant_domain,
            "archetype": self.archetype,
            "source": self.source,
        }
        if self.input_bvec:
            d["input_bvec"] = self.input_bvec.as_dict()
        if self.response_bvec:
            d["response_bvec"] = self.response_bvec.as_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AutobiographyEntry":
        entry = cls(
            timestamp=d.get("timestamp", time.time()),
            input_text=d.get("input_text", ""),
            response_text=d.get("response_text", ""),
            coherence=d.get("coherence", 0.0),
            exchange=d.get("exchange", 0.0),
            dCdX=d.get("dCdX", 0.0),
            regime=d.get("regime", "unknown"),
            dissonance=d.get("dissonance", 0.0),
            dominant_domain=d.get("dominant_domain", ""),
            archetype=d.get("archetype", ""),
            source=d.get("source", "conversation"),
        )
        if "input_bvec" in d:
            entry.input_bvec = BVec.from_dict(d["input_bvec"])
        if "response_bvec" in d:
            entry.response_bvec = BVec.from_dict(d["response_bvec"])
        return entry


class Autobiography:
    """Longitudinal record of the system's cognitive trajectory.

    Append-only JSONL file. Designed for the daily compaction cycle
    to scan, cluster by BFECDS, and consolidate to long-term memory.
    """

    def __init__(self, path: str = "autobiography.jsonl"):
        self.path = path
        self._entries_today: List[AutobiographyEntry] = []

    def log_interaction(
        self,
        input_text: str,
        response_text: str,
        input_bvec: Optional[BVec] = None,
        response_bvec: Optional[BVec] = None,
        coherence: float = 0.0,
        exchange: float = 0.0,
        dCdX: float = 0.0,
        regime: str = "unknown",
        source: str = "conversation",
    ) -> AutobiographyEntry:
        """Log one interaction with full field measurements."""
        # Compute derived fields
        dissonance = 0.0
        if input_bvec and response_bvec:
            dissonance = bvec_distance(input_bvec, response_bvec)

        dominant_domain = ""
        archetype = ""
        if response_bvec:
            arr = response_bvec.as_array()
            domain_names = ["B", "F", "E", "C", "D", "S"]
            dominant_domain = domain_names[int(arr.argmax())]
            archetype = response_bvec.archetype()

        entry = AutobiographyEntry(
            input_text=input_text,
            response_text=response_text,
            input_bvec=input_bvec,
            response_bvec=response_bvec,
            coherence=coherence,
            exchange=exchange,
            dCdX=dCdX,
            regime=regime,
            dissonance=dissonance,
            dominant_domain=dominant_domain,
            archetype=archetype,
            source=source,
        )

        self._entries_today.append(entry)

        # Append to disk
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        return entry

    def get_today(self) -> List[AutobiographyEntry]:
        """Get all entries logged in this session."""
        return list(self._entries_today)

    def get_high_torsion(self, threshold: float = 0.3,
                         include_persisted: bool = False,
                         max_persisted: int = 200) -> List[AutobiographyEntry]:
        """Get entries with high dissonance — candidates for dreaming loop.

        The metacognition loop selects high-torsion memories for re-processing.
        High dissonance = genuine information processing was happening.

        Remediation Tier 2.1: with ``include_persisted=True`` this also scans the
        most recent persisted entries from disk (deduped against this session's
        in-memory entries), so a restart no longer forgets prior-session
        tensions. The autobiography is already persisted on every write
        (append-only JSONL); this is the matching load path the dream loop uses.
        """
        entries: List[AutobiographyEntry] = list(self._entries_today)
        if include_persisted:
            seen = {(e.timestamp, e.input_text) for e in entries}
            for e in self.load_all()[-max_persisted:]:
                key = (e.timestamp, e.input_text)
                if key not in seen:
                    entries.append(e)
                    seen.add(key)
        return [e for e in entries if e.dissonance > threshold]

    def load_all(self) -> List[AutobiographyEntry]:
        """Load full autobiography from disk."""
        entries = []
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                for line in f:
                    try:
                        entries.append(AutobiographyEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return entries
