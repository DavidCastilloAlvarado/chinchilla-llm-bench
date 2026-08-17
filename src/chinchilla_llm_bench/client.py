"""OpenAI-compatible client built on the official ``openai`` SDK.

All endpoints we talk to are OpenAI-compatible (vLLM and friends), so we use
the official library instead of hand-rolled SSE parsing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from openai import OpenAI

from .prompt import estimate_tokens

TokenCallback = Callable[[str, float, str], None]
TokenCounter = Callable[[str], int]

_CLIENTS: dict[tuple, OpenAI] = {}
_CLIENTS_LOCK = threading.Lock()


def get_client(base_url: str, api_key: str = "dummy", timeout: float = 300.0) -> OpenAI:
    """Thread-safe cached :class:`openai.OpenAI` client for an endpoint.

    ``max_retries=0`` is deliberate: hidden retries would skew latency stats.
    """
    key = (base_url, api_key, timeout)
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key or "dummy",
                timeout=timeout,
                max_retries=0,
            )
            _CLIENTS[key] = client
    return client


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
    thinking: bool = False,
    api_key: str = "dummy",
) -> ChatResult:
    """Stream a chat completion from an OpenAI-compatible server.

    ``thinking=False`` (default) disables Qwen3-style reasoning on vLLM via
    ``chat_template_kwargs {"enable_thinking": false}`` so the model generates
    content directly — the benchmark measures real content tokens.

    ``on_token(text, t_rel, kind)`` is invoked for every streamed chunk;
    ``t_rel`` is seconds since the request started and ``kind`` is
    ``"content"`` or ``"think"`` (reasoning tokens when thinking is on).
    """
    client = get_client(base_url, api_key, timeout)
    result = ChatResult()
    pieces: list[str] = []
    t0 = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}},
        )
        for chunk in stream:
            now = time.perf_counter() - t0
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                result.prompt_tokens = int(usage.prompt_tokens or result.prompt_tokens)
                result.completion_tokens = int(
                    usage.completion_tokens or result.completion_tokens
                )
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                if usage is not None:
                    break  # final usage chunk (include_usage)
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            kind = "content"
            if not piece:
                piece = getattr(delta, "reasoning_content", None)
                kind = "think"
            if piece:
                if result.ttfr is None:
                    result.ttfr = now
                result.token_times.append(now)
                pieces.append(piece)
                if on_token is not None:
                    on_token(piece, now, kind)
    except Exception as exc:  # surface API/transport errors in the report
        result.error = f"{type(exc).__name__}: {exc}"
    result.duration = time.perf_counter() - t0
    result.text = "".join(pieces)
    if result.completion_tokens == 0:
        result.completion_tokens = len(result.token_times)
    if result.error is None and result.completion_tokens == 0:
        result.error = (
            "no tokens generated (if the model thinks, its reasoning may have "
            "eaten the whole token budget — run with thinking disabled)"
        )
    if result.prompt_tokens == 0:
        result.prompt_tokens = estimate_tokens(prompt)
    return result


def list_models(base_url: str, api_key: str = "dummy", timeout: float = 10.0) -> list[str]:
    """Return the served model ids via ``GET /models``."""
    client = get_client(base_url, api_key, timeout)
    page = client.models.list()
    items = getattr(page, "data", page)
    return [m.id for m in items]


def resolve_tokenizer(
    base_url: str, api_key: str = "dummy", timeout: float = 10.0
) -> tuple[TokenCounter, str]:
    """Try vLLM's native ``POST /v1/tokenize``; fall back to the offline estimate.

    Returns ``(count_tokens, source_description)``.
    """
    client = get_client(base_url, api_key, timeout)

    def count(text: str) -> int:
        resp = client.post(
            "/tokenize",
            body={"prompt": text, "add_special_tokens": False},
            cast_to=object,
        )
        if isinstance(resp, dict):
            ids = resp.get("input_ids", resp.get("tokens"))
        else:  # pragma: no cover - older SDKs wrap in an object
            ids = getattr(resp, "input_ids", None)
            if ids is None:
                ids = getattr(resp, "tokens", None)
        if ids is None:
            raise ValueError("no token ids in /tokenize response")
        return int(len(ids))

    try:
        count("hello world")
        return count, "vllm /tokenize"
    except Exception:
        return estimate_tokens, "estimate (~1.3 tok/word)"
