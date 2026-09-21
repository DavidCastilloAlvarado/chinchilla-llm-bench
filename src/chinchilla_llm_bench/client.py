"""OpenAI-compatible client built on the official ``openai`` SDK.

All endpoints we talk to are OpenAI-compatible (vLLM and friends), so we use
the official library instead of hand-rolled SSE parsing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from openai import APIStatusError, OpenAI

from .prompt import estimate_tokens
from .terminal import safe_terminal_text

TokenCallback = Callable[[str, float, str, int], None]  # (text, t_rel, kind, n_tokens)
TokenCounter = Callable[[str], int]

_CLIENTS: dict[tuple, OpenAI] = {}
_CLIENTS_LOCK = threading.Lock()


def get_client(
    base_url: str,
    api_key: str = "dummy",
    timeout: float = 300.0,
    headers: Mapping[str, str] | None = None,
) -> OpenAI:
    """Thread-safe cached :class:`openai.OpenAI` client for an endpoint.

    ``max_retries=0`` is deliberate: hidden retries would skew latency stats.
    """
    key = (base_url, api_key, timeout, tuple(sorted((headers or {}).items())))
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = OpenAI(
                base_url=base_url,
                api_key=api_key or "dummy",
                timeout=timeout,
                max_retries=0,
                default_headers=headers,
            )
            _CLIENTS[key] = client
    return client


@dataclass
class ChatResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttfr: Optional[float] = None  # seconds: request start -> first response chunk
    ttft: Optional[float] = None  # seconds: request start -> first generated token
    duration: float = 0.0  # seconds: request start -> stream end
    token_times: list[float] = field(default_factory=list)  # per-token arrival offsets
    text: str = ""
    error: Optional[str] = None
    http_status: Optional[int] = None


def _delta_text(delta, name: str) -> str | None:
    """Read a delta field, declared or extra (e.g. ``reasoning``).

    Reasoning tokens stream under different names depending on the server:
    ``reasoning_content`` (OpenAI-compatible / vLLM reasoning parsers) or
    ``reasoning`` (newer vLLM). Neither is declared by the SDK, so fall back
    to ``model_extra``.
    """
    value = getattr(delta, name, None)
    if value is None:
        extra = getattr(delta, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(name)
    return value


def _token_ids_of(choice) -> list[int] | None:
    """vLLM's ``return_token_ids`` extension: token IDs ride along per chunk."""
    ids = getattr(choice, "token_ids", None)
    if ids is None:
        extra = getattr(choice, "model_extra", None)
        if isinstance(extra, dict):
            ids = extra.get("token_ids")
    if isinstance(ids, list) and all(isinstance(i, int) for i in ids):
        return ids
    return None


def _finalize_stream_tokens(
    content_chunks: list[tuple[float, list[int] | None]],
    usage_completion_tokens: int | None,
) -> tuple[int, list[float]]:
    """Recover the true per-token count + timestamps from a streamed response.

    Ported from llama-benchy's signal handling:
      * every chunk carried ``token_ids`` -> exact count; a chunk with N
        tokens gets N timestamps spread over the gap since the previous one
      * else the server's ``usage`` count, interpolated over chunk times
      * else assume 1 token per chunk
    """
    if not content_chunks:
        return (usage_completion_tokens or 0, [])

    with_ids = [c for c in content_chunks if c[1] is not None]
    if len(with_ids) == len(content_chunks):
        token_times: list[float] = []
        total = 0
        last_ts: float | None = None
        for ts, ids in content_chunks:
            n = len(ids)
            total += n
            prev = last_ts if last_ts is not None else 0.0
            if n == 1:
                token_times.append(ts)
            else:
                span = max(0.0, ts - prev)
                for i in range(n):
                    token_times.append(prev + span * (i + 1) / n)
            last_ts = ts
        return total, token_times

    times = [ts for ts, _ in content_chunks]
    if usage_completion_tokens is not None and usage_completion_tokens > 0:
        n = usage_completion_tokens
        if len(times) == 1 or times[-1] <= times[0]:
            token_times = [times[0]] * n
        else:
            step = (times[-1] - times[0]) / (n - 1)
            token_times = [times[0] + step * i for i in range(n)]
        return n, token_times

    return len(content_chunks), times  # last resort: 1 token per chunk


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    temperature: float | None = None,
    timeout: float = 300.0,
    on_token: Optional[TokenCallback] = None,
    thinking: bool = False,
    api_key: str = "dummy",
    min_tokens: int | None = None,
    headers: Mapping[str, str] | None = None,
    vllm_extensions: bool = False,
    max_completion_tokens: bool = False,
    reasoning_effort: str | None = None,
) -> ChatResult:
    """Stream a chat completion from an OpenAI-compatible server.

    VLLM-only fields are sent only when ``vllm_extensions=True``. This keeps
    the default request compatible with standard OpenAI-compatible gateways.

    ``min_tokens`` (llama-benchy's ``exact_tg``) sets ``min_tokens`` and
    ``ignore_eos`` so the server generates the full budget instead of
    stopping at EOS.

    The payload requests ``return_token_ids`` (vLLM extension): each chunk
    then carries the real token IDs, so the token count — and the per-token
    timestamps used for rates — are exact instead of per-chunk guesses.

    ``on_token(text, t_rel, kind, n_tokens)`` is invoked for every streamed
    chunk; ``t_rel`` is seconds since the request started, ``kind`` is
    ``"content"`` or ``"think"`` and ``n_tokens`` is the chunk's real token
    count (1 when the server does not report token IDs).
    """
    client = get_client(base_url, api_key, timeout, headers)
    result = ChatResult()
    pieces: list[str] = []
    t0 = time.perf_counter()
    content_chunks: list[tuple[float, list[int] | None]] = []
    usage_completion_tokens: int | None = None
    try:
        request_kwargs = {}
        request_kwargs[
            "max_completion_tokens" if max_completion_tokens else "max_tokens"
        ] = max_tokens
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if vllm_extensions:
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": bool(thinking)},
                "return_token_ids": True,
            }
            if min_tokens is not None:
                extra_body["min_tokens"] = min_tokens
                extra_body["ignore_eos"] = True
            request_kwargs["extra_body"] = extra_body
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            **request_kwargs,
        )
        for chunk in stream:
            now = time.perf_counter() - t0
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                result.prompt_tokens = int(usage.prompt_tokens or result.prompt_tokens)
                completion_tokens = int(usage.completion_tokens or 0)
                if completion_tokens > 0:
                    usage_completion_tokens = completion_tokens
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                if usage is not None:
                    break  # final usage chunk (include_usage)
                continue
            if result.ttfr is None:
                result.ttfr = now  # first response chunk
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            kind = "content"
            if not piece:
                # Thinking streams: 'reasoning_content' (reasoning parsers)
                # or 'reasoning' (vLLM >= 0.26). Without this, a thinking
                # model (e.g. Qwen3 with thinking on by default) produces
                # no on_token calls at all and the live UI never updates.
                piece = _delta_text(delta, "reasoning_content") or _delta_text(
                    delta, "reasoning"
                )
                kind = "think"
            if piece:
                if result.ttft is None:
                    result.ttft = now  # first generated token
                token_ids = _token_ids_of(choices[0])
                content_chunks.append((now, token_ids))
                pieces.append(piece)
                if on_token is not None:
                    on_token(piece, now, kind, len(token_ids) if token_ids else 1)
    except Exception as exc:  # surface API/transport errors in the report
        result.error = safe_terminal_text(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, APIStatusError):
            result.http_status = exc.status_code
    result.duration = time.perf_counter() - t0
    result.text = "".join(pieces)
    total, token_times = _finalize_stream_tokens(content_chunks, usage_completion_tokens)
    result.completion_tokens = total
    result.token_times = token_times
    if result.error is None and result.completion_tokens == 0:
        result.error = (
            "no tokens generated (the model may have consumed the completion "
            "budget while reasoning; increase the budget or lower --reasoning-effort)"
        )
    if result.prompt_tokens == 0:
        result.prompt_tokens = estimate_tokens(prompt)
    return result


def measure_latency(
    base_url: str,
    api_key: str = "dummy",
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> float:
    """Mean RTT of 3 ``GET /models`` probes (llama-benchy's 'api' latency mode).

    Used to subtract network/server round-trip from TTFT -> net_ttft.
    """
    client = get_client(base_url, api_key, timeout, headers)
    samples: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        try:
            client.get("/models", cast_to=object)
            samples.append(time.perf_counter() - t0)
        except Exception:
            continue
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def list_models(
    base_url: str,
    api_key: str = "dummy",
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the served model ids via ``GET /models``."""
    client = get_client(base_url, api_key, timeout, headers)
    page = client.models.list()
    items = getattr(page, "data", page)
    return [m.id for m in items]


def resolve_tokenizer(
    base_url: str,
    api_key: str = "dummy",
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> tuple[TokenCounter, str]:
    """Try vLLM's native ``POST /v1/tokenize``; fall back to the offline estimate.

    Returns ``(count_tokens, source_description)``.
    """
    client = get_client(base_url, api_key, timeout, headers)

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
