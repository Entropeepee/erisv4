"""
Knowledge Extractor — Text/Document → .eris Files
====================================================

Converts raw text, documents, or conversation transcripts into
.eris Knowledge Descriptors with computed BFECDS activations.

Supports chunking for long documents — each chunk gets its own
BFECDS computation and field snapshot.

Usage:
    from eris.knowledge.extractor import KnowledgeExtractor

    ex = KnowledgeExtractor(output_dir="knowledge_base")
    descriptors = ex.extract_text("Long document text...", title="Paper")
    ex.extract_file("document.txt", title="My Document")
"""

from __future__ import annotations
from typing import List, Optional
import os
import re
import hashlib

from eris.knowledge.descriptor import ErisDescriptor


def chunk_text(text: str, max_chars: int = 2000,
               overlap_chars: int = 200) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Respects paragraph breaks when possible, falls back to
    sentence-level splitting, never breaks mid-word.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at paragraph boundary
        para_break = text.rfind("\n\n", start + max_chars // 2, end)
        if para_break > start:
            end = para_break + 2
        else:
            # Try sentence boundary
            sent_break = max(
                text.rfind(". ", start + max_chars // 2, end),
                text.rfind("! ", start + max_chars // 2, end),
                text.rfind("? ", start + max_chars // 2, end),
            )
            if sent_break > start:
                end = sent_break + 2
            else:
                # Try word boundary
                word_break = text.rfind(" ", start + max_chars // 2, end)
                if word_break > start:
                    end = word_break + 1

        chunks.append(text[start:end])
        start = end - overlap_chars  # Overlap for context continuity

    return chunks


class KnowledgeExtractor:
    """Convert text into .eris Knowledge Descriptors."""

    def __init__(self, output_dir: str = "knowledge_base",
                 field_size: int = 32, pde_steps: int = 50,
                 use_frt: bool = False, chunk_size: int = 2000):
        self.output_dir = output_dir
        self.field_size = field_size
        self.pde_steps = pde_steps
        self.use_frt = use_frt
        self.chunk_size = chunk_size
        os.makedirs(output_dir, exist_ok=True)

    def extract_text(self, text: str, title: str = "") -> List[ErisDescriptor]:
        """Extract knowledge from text, chunking if necessary."""
        chunks = chunk_text(text, max_chars=self.chunk_size)
        descriptors = []

        for i, chunk in enumerate(chunks):
            chunk_title = f"{title} [chunk {i+1}/{len(chunks)}]" if len(chunks) > 1 else title
            desc = ErisDescriptor.from_text(
                chunk, title=chunk_title,
                field_size=self.field_size,
                pde_steps=self.pde_steps,
                use_frt=self.use_frt,
            )

            # Save to output directory
            sha_short = desc.sha256[:12]
            filename = f"{sha_short}.eris"
            desc.save(os.path.join(self.output_dir, filename))
            descriptors.append(desc)

        return descriptors

    def extract_file(self, filepath: str, title: str = "") -> List[ErisDescriptor]:
        """Extract knowledge from a text file."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        title = title or os.path.basename(filepath)
        return self.extract_text(text, title=title)

    def extract_conversation(self, messages: List[dict],
                             title: str = "conversation") -> List[ErisDescriptor]:
        """Extract from a conversation (list of {role, content} dicts).

        Each turn becomes a separate .eris descriptor so the BFECDS
        tracks how the conversation's cognitive state evolved.
        """
        descriptors = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content.strip():
                continue

            turn_title = f"{title} [{role} turn {i+1}]"
            desc = ErisDescriptor.from_text(
                content, title=turn_title,
                field_size=self.field_size,
                pde_steps=self.pde_steps,
                use_frt=self.use_frt,
            )

            sha_short = desc.sha256[:12]
            desc.save(os.path.join(self.output_dir, f"{sha_short}.eris"))
            descriptors.append(desc)

        return descriptors
