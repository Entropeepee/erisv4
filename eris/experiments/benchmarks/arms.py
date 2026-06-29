"""The two arms. Both expose the SAME `answer_fn(prompt) -> (text, tokens)` signature so the
runner is agnostic to which is which — that interchangeability is what keeps the comparison fair.

  • Arm A (bare model): `openai_chat_arm` — one call to an OpenAI-compatible endpoint
    (Ollama / vLLM / TGI). Tokens read from the response `usage`.
  • Arm B (Eris): `eris_pipeline_arm` — ingest the provided context into a SCRATCH memory, then
    answer with the full pipeline. Wraps a user-supplied (ingest, ask) pair so it binds to the
    exact Eris entrypoint without this module importing the heavy orchestrator.

Note (from the brief): Ollama's default 4096-token context is too small for QuALITY/FRAMES — set
OLLAMA_CONTEXT_LENGTH >= 22000 before serving, or long documents are silently truncated."""
import os
import re
from typing import Callable, Tuple


def _answer_text_from_message(msg: dict) -> str:
    """Robustly pull the model's answer from a /chat/completions message — including REASONING
    models (qwen3, deepseek-r1, gpt-oss) that put their answer in `reasoning_content`/`reasoning`
    and leave `content` empty, or inline a <think>…</think> block. Without this a reasoning model
    scores 0 (empty content) and looks dumb when it actually answered."""
    text = (msg.get("content") or "").strip()
    if not text:                                  # answer landed in the reasoning channel
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    if "</think>" in text:                        # keep what FOLLOWS the think block (the answer)
        text = text.rsplit("</think>", 1)[-1].strip()
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def openai_chat_arm(base_url: str = "http://localhost:11434/v1",
                    model: str = "", api_key: str = "ollama",
                    max_tokens: int = 2048, temperature: float = 0.0,
                    system: str = "You are a careful assistant. Answer concisely and only from "
                                  "any provided source.") -> Callable[[str], Tuple[str, int]]:
    """Arm A — a bare model behind an OpenAI-compatible /chat/completions endpoint. Deterministic
    (temperature 0) for reproducible scoring. max_tokens defaults to 2048 (not 512): a reasoning
    model can spend hundreds of tokens thinking before the answer, and a small cap truncates it
    mid-thought into an empty answer. Returns (text, total_tokens)."""
    try:
        import requests
    except Exception as e:                       # pragma: no cover - environment-dependent
        raise RuntimeError("the `requests` package is required for the bare arm") from e
    url = base_url.rstrip("/") + "/chat/completions"

    def answer(prompt: str) -> Tuple[str, int]:
        body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = requests.post(url, json=body, headers=headers, timeout=300)
        r.raise_for_status()
        data = r.json()
        text = _answer_text_from_message(data["choices"][0]["message"] or {})
        tokens = int((data.get("usage") or {}).get("total_tokens", 0))
        return text, tokens

    return answer


def callable_arm(fn: Callable[[str], object]) -> Callable[[str], Tuple[str, int]]:
    """Wrap any `fn(prompt) -> str | (text, tokens) | {text, tokens}` as an answer_fn. Handy for
    tests and for binding a custom Eris entrypoint."""
    def answer(prompt: str) -> Tuple[str, int]:
        resp = fn(prompt)
        if isinstance(resp, tuple):
            return str(resp[0]), int(resp[1] or 0)
        if isinstance(resp, dict):
            return str(resp.get("text", "")), int(resp.get("tokens", 0) or 0)
        return str(resp), 0
    return answer


def eris_pipeline_arm(ingest: Callable[[str], None],
                      ask: Callable[[str], object],
                      reset: Callable[[], None] = None) -> Callable[[str], Tuple[str, int]]:
    """Arm B — the full Eris pipeline. For each question: optionally `reset()` a scratch memory,
    `ingest(context)` the provided source so Eris's RAG can retrieve it, then `ask(question)`.

    `ingest`, `ask`, `reset` bind to your live Eris (e.g. ingest → web_reader.ingest_text into a
    benchmark MemorySystem; ask → orchestrator.hive_research(question, scope='doc') or the agent
    loop). `ask` returns the answer as str, or (text, tokens) / {text, tokens} for token
    accounting (sum your tier_calls). The prompt arrives with the SOURCE block inline, so we split
    it back out to feed the ingester; if there is no source block we treat it as closed-book."""
    def answer(prompt: str) -> Tuple[str, int]:
        context, question = _split_source(prompt)
        if reset is not None:
            reset()
        if context:
            ingest(context)
        resp = ask(question or prompt)
        if isinstance(resp, tuple):
            return str(resp[0]), int(resp[1] or 0)
        if isinstance(resp, dict):
            return str(resp.get("text", "")), int(resp.get("tokens", 0) or 0)
        return str(resp), 0
    return answer


def _split_source(prompt: str) -> Tuple[str, str]:
    """Recover (context, question) from a prompt built by core.build_prompt."""
    if "=== SOURCE ===" in prompt and "=== END SOURCE ===" in prompt:
        ctx = prompt.split("=== SOURCE ===", 1)[1].split("=== END SOURCE ===", 1)[0].strip()
        tail = prompt.split("=== END SOURCE ===", 1)[1]
        q = ""
        for line in tail.splitlines():
            if line.startswith("Question:"):
                q = line[len("Question:"):].strip()
                break
        return ctx, q or tail.strip()
    return "", prompt


def default_bare_arm() -> Callable[[str], Tuple[str, int]]:
    """Arm A from env: ERIS_BENCH_BASE_URL, ERIS_BENCH_MODEL, ERIS_BENCH_API_KEY. Points at local
    Ollama by default, or any OpenAI-compatible cloud endpoint (e.g. OpenRouter:
    base_url=https://openrouter.ai/api/v1, model=qwen/qwen-2.5-72b-instruct, a real Bearer key) —
    which both SPEEDS UP the run and lets you compare LLM choices by swapping ERIS_BENCH_MODEL."""
    return openai_chat_arm(
        base_url=os.environ.get("ERIS_BENCH_BASE_URL", "http://localhost:11434/v1"),
        model=os.environ.get("ERIS_BENCH_MODEL", ""),
        api_key=os.environ.get("ERIS_BENCH_API_KEY", "ollama"))
