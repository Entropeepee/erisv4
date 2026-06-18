"""
.eris Knowledge Descriptor Format
====================================

A portable knowledge unit. ZIP archive containing:

    manifest.json    — Metadata: title, SHA256, timestamps, BFECDS vector
    source.txt       — Original text (lossless reconstruction)
    field_phi.npy    — FRACTAL phi field snapshot
    field_theta.npy  — FRACTAL theta field snapshot
    bvec.json        — Computed BFECDS activation vector
    graph.json       — Structural graph (optional: entities, relations)

SHA256 integrity verification on source text ensures lossless roundtrip.
The BFECDS vector is COMPUTED from field dynamics, not LLM-assigned.

From the handoff session:
    "If you were to re-process [the corpus] with computed activations —
     using the Chapter 6 PDE criteria instead of LLM assignment — you'd
     have something genuinely unprecedented: a map of how a theoretical
     framework emerged over time, parameterized in its own terms."

Usage:
    from eris.knowledge.descriptor import ErisDescriptor

    # Create
    desc = ErisDescriptor.from_text("Hello world", title="test")
    desc.save("test.eris")

    # Load
    desc = ErisDescriptor.load("test.eris")
    print(desc.source_text)
    print(desc.bvec)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import zipfile
import json
import hashlib
import io
import time
import numpy as np
from eris.config import to_numpy, xp

from eris.computation.activations import BVec


@dataclass
class ErisDescriptor:
    """A .eris knowledge unit with computed BFECDS and field state."""
    title: str = ""
    source_text: str = ""
    sha256: str = ""
    created_at: float = field(default_factory=time.time)

    # Computed from FRACTAL PDE (not LLM-assigned)
    bvec: Optional[BVec] = None
    phi_snapshot: Optional[np.ndarray] = None
    theta_snapshot: Optional[np.ndarray] = None

    # Optional structural graph
    graph: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, title: str = "",
                  field_size: int = 32, pde_steps: int = 50,
                  use_frt: bool = False) -> "ErisDescriptor":
        """Create a descriptor by running text through the FRACTAL PDE.

        This computes the BFECDS from actual field dynamics.
        """
        from eris.field.pde import FractalField

        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

        f = FractalField(size=field_size)
        f.seed_from_text(text, use_frt=use_frt)
        f.run(pde_steps)

        bvec = f.compute_bvec()
        phi = to_numpy(f.phi).copy()
        theta = to_numpy(f.theta).copy()

        return cls(
            title=title or text[:60],
            source_text=text,
            sha256=sha,
            bvec=bvec,
            phi_snapshot=phi.astype(np.float32),
            theta_snapshot=theta.astype(np.float32),
            metadata={
                "field_size": field_size,
                "pde_steps": pde_steps,
                "coherence": f.coherence,
                "exchange": f.exchange,
                "dCdX": f.dCdX,
                "regime": f.detect_regime(),
                "archetype": bvec.archetype(),
            },
        )

    def save(self, path: str) -> None:
        """Save as a .eris ZIP archive."""
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Manifest
            manifest = {
                "title": self.title,
                "sha256": self.sha256,
                "created_at": self.created_at,
                "metadata": self.metadata,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            # Source text (lossless)
            zf.writestr("source.txt", self.source_text)

            # BFECDS vector
            if self.bvec:
                zf.writestr("bvec.json", json.dumps(self.bvec.as_dict()))

            # Field snapshots as numpy arrays
            if self.phi_snapshot is not None:
                buf = io.BytesIO()
                np.save(buf, self.phi_snapshot)
                zf.writestr("field_phi.npy", buf.getvalue())

            if self.theta_snapshot is not None:
                buf = io.BytesIO()
                np.save(buf, self.theta_snapshot)
                zf.writestr("field_theta.npy", buf.getvalue())

            # Structural graph
            if self.graph:
                zf.writestr("graph.json", json.dumps(self.graph))

    @classmethod
    def load(cls, path: str) -> "ErisDescriptor":
        """Load from a .eris ZIP archive."""
        desc = cls()

        with zipfile.ZipFile(path, "r") as zf:
            # Manifest
            manifest = json.loads(zf.read("manifest.json"))
            desc.title = manifest.get("title", "")
            desc.sha256 = manifest.get("sha256", "")
            desc.created_at = manifest.get("created_at", 0.0)
            desc.metadata = manifest.get("metadata", {})

            # Source text
            desc.source_text = zf.read("source.txt").decode("utf-8")

            # Verify SHA256 integrity
            actual_sha = hashlib.sha256(desc.source_text.encode("utf-8")).hexdigest()
            if desc.sha256 and actual_sha != desc.sha256:
                raise ValueError(
                    f"SHA256 mismatch: expected {desc.sha256[:16]}..., "
                    f"got {actual_sha[:16]}... — file may be corrupted"
                )

            # BFECDS
            if "bvec.json" in zf.namelist():
                bvec_data = json.loads(zf.read("bvec.json"))
                desc.bvec = BVec.from_dict(bvec_data)

            # Field snapshots
            if "field_phi.npy" in zf.namelist():
                buf = io.BytesIO(zf.read("field_phi.npy"))
                desc.phi_snapshot = np.load(buf)

            if "field_theta.npy" in zf.namelist():
                buf = io.BytesIO(zf.read("field_theta.npy"))
                desc.theta_snapshot = np.load(buf)

            # Graph
            if "graph.json" in zf.namelist():
                desc.graph = json.loads(zf.read("graph.json"))

        return desc

    def verify_integrity(self) -> bool:
        """Verify the source text hasn't been corrupted."""
        actual = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        return actual == self.sha256
