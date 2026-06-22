"""Agent tool builders for the grounded ReAct loop (operationalizes Q1/Q2).

Turns this session's standalone capabilities into `Tool`s the ReAct loop can
call — without taxing the per-turn resonant path:

  • **factual_lookup** (Q1): precise BM25+dense+RRF retrieval over a read-only
    candidate pool from Eris's memory. The agent ESCALATES to this when a turn
    needs facts; resonant recall stays the default fast path. Not wired into
    `process()` — same continue/escalate discipline, applied to retrieval.
  • **remember_fact / recall_facts** (Q2): the built-in durable `LocalFactStore`.
    mem0/Letta are deferred; the seam stays in `eris.memory.durable`.

All tools are gated by `CONFIG.agent_tool_*` (default OFF; `ERIS_AGENT_TOOLS=on`).
`default_tools(orchestrator)` assembles only the enabled ones.
"""
from __future__ import annotations
from typing import List

from eris.config import CONFIG
from eris.executive.agent_loop import Tool


def factual_lookup_tool(memory, *, pool_limit: int = 400, top_k: int = 5) -> Tool:
    """Hybrid (BM25 + dense) factual lookup over a read-only memory pool."""
    from eris.retrieval.hybrid import hybrid_search, http_reranker
    from eris.knowledge.embeddings import get_embedding

    def _run(query: str) -> str:
        records = memory.all_records(limit=pool_limit)
        if not records:
            return "No records in memory."
        # Use the external reranker (NPU/iGPU) when configured; else RRF-only.
        hits = hybrid_search(query, records, query_embedding=get_embedding(query),
                             top_k=top_k, reranker=http_reranker())
        if not hits:
            return "No relevant facts found."
        return "\n".join(f"- {getattr(h, 'text', '')}" for h in hits)

    return Tool("factual_lookup",
                "Precise factual lookup over memory (exact tokens + meaning). "
                "Use when the turn needs specific facts/names/IDs, not vibes.",
                _run)


def durable_memory_tools(store) -> List[Tool]:
    """remember_fact(text) and recall_facts(query) over a DurableMemory store."""
    def _remember(text: str) -> str:
        fid = store.add(text)
        return f"Remembered (id {fid})." if fid else "Nothing to remember."

    def _recall(query: str) -> str:
        hits = store.search(query, k=5)
        if not hits:
            return "No matching facts."
        return "\n".join(f"- {h['text']}" for h in hits)

    return [
        Tool("remember_fact", "Store a durable atomic fact for later recall.", _remember),
        Tool("recall_facts", "Recall durable facts matching a query.", _recall),
    ]


def default_tools(orchestrator) -> List[Tool]:
    """Assemble the ReAct loop's default tool set from the enabled flags."""
    tools: List[Tool] = []
    if CONFIG.agent_tool_factual_lookup:
        tools.append(factual_lookup_tool(orchestrator.memory))
    if CONFIG.agent_tool_durable_memory:
        from eris.memory.durable import get_durable_memory
        if getattr(orchestrator, "_durable_memory", None) is None:
            import os
            orchestrator._durable_memory = get_durable_memory(
                os.path.join(getattr(orchestrator, "data_dir", "eris_data"),
                             "durable_facts.json"))
        tools.extend(durable_memory_tools(orchestrator._durable_memory))
    return tools
