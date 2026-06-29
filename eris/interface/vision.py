"""Vision hook — give Eris "eyes" via an OpenAI-compatible VLM (roadmap 1.5).

Model-agnostic plumbing: it builds standard OpenAI multimodal `chat/completions`
messages (text + base64 image_url) and posts them to whatever vision server you
run locally (Qwen3-VL-8B, InternVL3-8B, Llama-3.2-Vision, … served by vLLM or
Ollama). The model choice is a machine-side download; this code doesn't care which.

Configure with env (all optional; unset = vision disabled):
  ERIS_VISION_BASE_URL  e.g. http://localhost:8000/v1
  ERIS_VISION_MODEL     e.g. qwen3-vl-8b
  ERIS_VISION_API_KEY   dummy is fine for local servers

The payload builders are pure functions (unit-tested); only `see()` touches the
network. Use as a tool in the ReAct loop (3.1) or call directly.
"""
from __future__ import annotations
from typing import List, Optional
import base64
import os
import mimetypes


def encode_image(path: str) -> str:
    """Read an image file and return an OpenAI-style `data:` URL."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_vision_messages(prompt: str, image_paths: List[str],
                          system: str = "") -> List[dict]:
    """Build OpenAI multimodal messages: one user turn with the prompt plus each
    image as an image_url content part."""
    content: List[dict] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": encode_image(p)}})
    msgs: List[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    return msgs


def is_configured() -> bool:
    return bool(os.environ.get("ERIS_VISION_BASE_URL", "").strip())


async def see(prompt: str, image_paths: List[str], *,
              base_url: Optional[str] = None, model: Optional[str] = None,
              api_key: Optional[str] = None, system: str = "",
              max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """Send images + prompt to the configured VLM; return its text answer.

    Raises RuntimeError if no vision server is configured."""
    base_url = base_url or os.environ.get("ERIS_VISION_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("No vision server configured (set ERIS_VISION_BASE_URL).")
    # Egress guard (r3 #10): the images are the owner's content (often IP). A REMOTE VLM URL would
    # ship them off-box — refuse unless explicitly consented (no in-process fallback, so we raise).
    from eris.interface.accelerators import egress_allowed
    _ok, _why = egress_allowed("vision", base_url)
    if not _ok:
        raise RuntimeError(_why)
    model = model or os.environ.get("ERIS_VISION_MODEL", "vision")
    api_key = api_key or os.environ.get("ERIS_VISION_API_KEY", "local")

    import httpx
    payload = {
        "model": model,
        "messages": build_vision_messages(prompt, image_paths, system),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"].get("content", "")
