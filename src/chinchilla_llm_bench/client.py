"""Minimal OpenAI-compatible streaming client (vLLM friendly)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx

from .prompt import estimate_tokens

TokenCallback = Callable[[str, float, str], None]


@dataclass
class ChatResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttfr: Optional[float] = None  # seconds: request start -> first token
    duration: float = 0.0  # seconds: request start -> stream end
    token_times: list[float] = field(default_factory=list)  # arrival offset per token
    text: str = ""
    error: Optional[str] = None


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    temperature: float = 0.0,
    timeout: float = 300.0,
    on_token: Optional[TokenCallback] = None,
) -> ChatResult:
    """POST ``{base_url}/chat/completions`` with ``stream=True`` and consume the SSE feed.

    ``on_token(text, t_rel, kind)`` is invoked for every content chunk, where
    ``t_rel`` is seconds since the request started and ``kind`` is
    ``"content"`` or ``"think"`` (reasoning tokens from thinking models).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    result = ChatResult()
    t0 = time.perf_counter()
    pieces: list[str] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode(errors="replace")
                    result.error = f"HTTP {resp.status_code}: {body[:200]}"
                    result.duration = time.perf_counter() - t0
                    return result
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content")
                        kind = "content"
                        if not piece:
                            piece = delta.get("reasoning_content")
                            kind = "think"
                        if piece:
                            now = time.perf_counter() - t0
                            if result.ttfr is None:
                                result.ttfr = now
                            result.token_times.append(now)
                            pieces.append(piece)
                            if on_token:
                                on_token(piece, now, kind)
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        result.prompt_tokens = int(
                            usage.get("prompt_tokens") or result.prompt_tokens
                        )
                        result.completion_tokens = int(
                            usage.get("completion_tokens") or result.completion_tokens
                        )
                        # Final usage chunk (vLLM include_usage): no more content.
                        if not choices:
                            break
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.duration = time.perf_counter() - t0
    result.text = "".join(pieces)
    if result.completion_tokens == 0:
        result.completion_tokens = len(result.token_times)
    if result.prompt_tokens == 0:
        result.prompt_tokens = estimate_tokens(prompt)
    return result


def list_models(base_url: str, timeout: float = 10.0) -> list[str]:
    """GET ``{base_url}/models`` and return the served model ids."""
    url = base_url.rstrip("/") + "/models"
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return [m.get("id", "?") for m in data.get("data", [])]


def resolve_tokenizer(base_url: str, timeout: float = 10.0) -> tuple[TokenCounter, str]:
    """Try vLLM's native ``POST /tokenize``; fall back to the offline estimate.

    Returns ``(count_tokens, source_description)``.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    def _probe() -> Optional[TokenCounter]:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(root + "/tokenize", json={"prompt": "hello world"})
                resp.raise_for_status()
                tokens = resp.json().get("tokens") or []
            if not tokens:
                return None

            def count(text: str) -> int:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(root + "/tokenize", json={"prompt": text})
                    resp.raise_for_status()
                    return len(resp.json().get("tokens") or [])

            return count
        except Exception:
            return None

    count = _probe()
    if count is not None:
        return count, "vllm /tokenize"
    return estimate_tokens, "estimate (~1.3 tok/word)"
