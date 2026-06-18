"""
Corpus Processor — Batch Ingestion for Conversation Archives
==============================================================

Process large conversation corpora into .eris Knowledge Descriptors.
Supports ChatGPT JSON export, Claude DOCX exports, and plain text.

This is how Eris ingests David's 3-year ChatGPT history —
re-parameterizing every conversation in computed BFECDS coordinates
instead of LLM-assigned values.

From the handoff session:
    "You'd have your entire three-year ChatGPT history parameterized
     by computed domain vectors rather than LLM-assigned ones."

Usage:
    from eris.knowledge.corpus import CorpusProcessor

    proc = CorpusProcessor(output_dir="knowledge_base")
    stats = proc.process_chatgpt_export("conversations.json")
    print(f"Processed {stats['total_turns']} turns from {stats['conversations']} chats")
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import os
import glob
import time

from eris.knowledge.extractor import KnowledgeExtractor


class CorpusProcessor:
    """Batch process conversation archives into .eris files."""

    def __init__(self, output_dir: str = "knowledge_base",
                 field_size: int = 32, pde_steps: int = 30,
                 use_frt: bool = True):
        """
        Parameters
        ----------
        use_frt : bool
            Default True for bulk processing (speed over precision).
            Set False if you want full PDE computation (much slower).
        """
        self.extractor = KnowledgeExtractor(
            output_dir=output_dir,
            field_size=field_size,
            pde_steps=pde_steps,
            use_frt=use_frt,
        )
        self.output_dir = output_dir

    def process_chatgpt_export(self, json_path: str,
                                max_conversations: int = 0) -> Dict[str, Any]:
        """Process a ChatGPT JSON export file.

        ChatGPT exports as a JSON array of conversations, each with
        a "mapping" dict containing message nodes.
        """
        stats = {"conversations": 0, "total_turns": 0, "errors": 0,
                 "start_time": time.time()}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = data if isinstance(data, list) else [data]
        if max_conversations > 0:
            conversations = conversations[:max_conversations]

        for conv in conversations:
            try:
                title = conv.get("title", "untitled")
                messages = self._extract_chatgpt_messages(conv)

                if messages:
                    self.extractor.extract_conversation(messages, title=title)
                    stats["total_turns"] += len(messages)
                    stats["conversations"] += 1

            except Exception as e:
                stats["errors"] += 1

        stats["duration_seconds"] = time.time() - stats["start_time"]
        return stats

    def _extract_chatgpt_messages(self, conv: dict) -> List[dict]:
        """Extract messages from a ChatGPT conversation object."""
        messages = []
        mapping = conv.get("mapping", {})

        # Build ordered message list from the tree structure
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg is None:
                continue

            role = msg.get("author", {}).get("role", "unknown")
            content_parts = msg.get("content", {}).get("parts", [])
            content = " ".join(str(p) for p in content_parts if isinstance(p, str))

            if content.strip() and role in ("user", "assistant"):
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": msg.get("create_time", 0),
                })

        # Sort by timestamp
        messages.sort(key=lambda m: m.get("timestamp", 0))
        return messages

    def process_text_directory(self, dir_path: str,
                                pattern: str = "*.txt") -> Dict[str, Any]:
        """Process all text files in a directory."""
        stats = {"files": 0, "chunks": 0, "errors": 0}

        for filepath in glob.glob(os.path.join(dir_path, pattern)):
            try:
                descs = self.extractor.extract_file(filepath)
                stats["files"] += 1
                stats["chunks"] += len(descs)
            except Exception:
                stats["errors"] += 1

        return stats

    def process_jsonl_conversations(self, jsonl_path: str) -> Dict[str, Any]:
        """Process a JSONL file where each line is a conversation.

        Generic format: each line is {"title": "...", "messages": [...]}
        where messages are [{role, content}, ...].
        """
        stats = {"conversations": 0, "total_turns": 0, "errors": 0}

        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    conv = json.loads(line)
                    title = conv.get("title", "untitled")
                    messages = conv.get("messages", [])

                    if messages:
                        self.extractor.extract_conversation(messages, title=title)
                        stats["total_turns"] += len(messages)
                        stats["conversations"] += 1
                except Exception:
                    stats["errors"] += 1

        return stats

    def load_into_memory(self, memory_system,
                          max_entries: int = 0) -> Dict[str, Any]:
        """Load .eris files from knowledge base into the memory system.

        This is the bridge: corpus processor creates .eris files,
        this method loads them into LTM with field snapshots and
        BFECDS vectors so they're retrievable during conversation.

        Parameters
        ----------
        memory_system : MemorySystem
            The active memory system to load into.
        max_entries : int
            Maximum entries to load (0 = all).

        Returns
        -------
        Stats dict with counts.
        """
        from eris.knowledge.descriptor import ErisDescriptor
        from eris.memory.tiers import MemoryRecord
        import numpy as np
        from eris.config import to_numpy, xp

        stats = {"loaded": 0, "errors": 0, "skipped": 0}
        eris_dir = self.extractor.output_dir

        if not os.path.exists(eris_dir):
            return stats

        eris_files = sorted(
            f for f in os.listdir(eris_dir) if f.endswith(".eris")
        )

        if max_entries > 0:
            eris_files = eris_files[:max_entries]

        for filename in eris_files:
            try:
                path = os.path.join(eris_dir, filename)
                desc = ErisDescriptor.load(path)

                if not desc.bvec or not desc.source_text.strip():
                    stats["skipped"] += 1
                    continue

                record = MemoryRecord(
                    text=desc.source_text[:500],
                    bvec=desc.bvec,
                    source="knowledge_base",
                    phi_snapshot=desc.phi_snapshot,
                    theta_snapshot=desc.theta_snapshot,
                    metadata={
                        "title": desc.title,
                        "sha256": desc.sha256[:12],
                        "archetype": desc.bvec.archetype(),
                    },
                )
                memory_system.ltm.store(record)
                stats["loaded"] += 1

            except Exception as e:
                stats["errors"] += 1

        return stats
