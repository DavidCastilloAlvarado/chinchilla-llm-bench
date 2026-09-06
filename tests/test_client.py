from chinchilla_llm_bench.client import (
    _finalize_stream_tokens,
    list_models,
    measure_latency,
    resolve_tokenizer,
    stream_chat,
)

import pytest

from mock_server import MockVLLMServer


def test_list_models(mock_server):
    assert list_models(mock_server.base_url) == ["mock-model"]


def test_client_sends_custom_headers():
    server = MockVLLMServer(required_header=("X-Tenant-ID", "team-a")).start()
    try:
        headers = {"X-Tenant-ID": "team-a"}
        assert list_models(server.base_url, headers=headers) == ["mock-model"]
        result = stream_chat(server.base_url, "mock-model", "hello", 1, headers=headers)
    finally:
        server.stop()
    assert result.error is None


def test_stream_chat_counts_tokens(mock_server):
    result = stream_chat(mock_server.base_url, "mock-model", "hello", 10)
    assert result.error is None
    assert result.completion_tokens == 10
    assert result.prompt_tokens == 10  # from the usage chunk
    assert result.ttfr is not None and result.ttfr > 0
    assert result.ttft is not None and result.ttft >= result.ttfr
    assert result.duration >= result.ttfr
    # token_ids path: one real token per chunk -> 10 per-token timestamps
    assert len(result.token_times) == 10
    assert result.token_times == sorted(result.token_times)
    assert result.text.startswith("tok0")


def test_stream_chat_single_token(mock_server):
    result = stream_chat(mock_server.base_url, "mock-model", "hello", 1)
    assert result.error is None
    assert result.completion_tokens == 1
    assert result.ttfr is not None
    assert len(result.token_times) == 1


def test_stream_chat_omits_vllm_extensions_by_default(mock_server):
    result = stream_chat(mock_server.base_url, "mock-model", "hello", 1, min_tokens=1)
    assert result.error is None
    _path, body = mock_server.server.request_bodies[-1]
    assert body["max_tokens"] == 1
    assert "max_completion_tokens" not in body
    assert "chat_template_kwargs" not in body
    assert "return_token_ids" not in body
    assert "min_tokens" not in body
    assert "ignore_eos" not in body


def test_stream_chat_sends_vllm_extensions_when_enabled(mock_server):
    result = stream_chat(
        mock_server.base_url,
        "mock-model",
        "hello",
        1,
        thinking=True,
        min_tokens=1,
        vllm_extensions=True,
    )
    assert result.error is None
    _path, body = mock_server.server.request_bodies[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["return_token_ids"] is True
    assert body["min_tokens"] == 1
    assert body["ignore_eos"] is True


def test_stream_chat_uses_max_completion_tokens_when_requested(mock_server):
    result = stream_chat(
        mock_server.base_url,
        "mock-model",
        "hello",
        3,
        max_completion_tokens=True,
    )
    assert result.error is None
    _path, body = mock_server.server.request_bodies[-1]
    assert body["max_completion_tokens"] == 3
    assert "max_tokens" not in body


def test_stream_chat_sends_reasoning_effort_when_requested(mock_server):
    result = stream_chat(
        mock_server.base_url,
        "mock-model",
        "hello",
        3,
        reasoning_effort="minimal",
    )
    assert result.error is None
    _path, body = mock_server.server.request_bodies[-1]
    assert body["reasoning_effort"] == "minimal"


def test_stream_chat_no_token_ids_falls_back_to_usage(mock_server):
    """Without return_token_ids support, usage + interpolation is used."""
    server = MockVLLMServer(token_delay=0.001, token_ids=False).start()
    try:
        result = stream_chat(server.base_url, "mock-model", "hello", 10)
    finally:
        server.stop()
    assert result.error is None
    assert result.completion_tokens == 10  # from the usage chunk
    assert len(result.token_times) == 10
    assert result.token_times == sorted(result.token_times)


def test_finalize_stream_tokens_interpolates_multi_token_chunks():
    # one chunk with 1 token at t=0.1, another with 3 tokens at t=0.5
    chunks = [(0.1, [7]), (0.5, [8, 9, 10])]
    total, times = _finalize_stream_tokens(chunks, None)
    assert total == 4
    assert times[0] == 0.1
    # the 3 tokens spread evenly over the 0.1 -> 0.5 gap
    assert times[1:] == pytest.approx([0.23333333, 0.36666667, 0.5])


def test_finalize_stream_tokens_usage_fallback():
    times = [0.0, 1.0, 2.0]
    total, series = _finalize_stream_tokens([(t, None) for t in times], 4)
    assert total == 4
    assert series == [0.0, 2 / 3, 4 / 3, 2.0]


def test_finalize_stream_tokens_chunk_count_last_resort():
    total, series = _finalize_stream_tokens([(0.0, None), (1.0, None)], None)
    assert total == 2
    assert series == [0.0, 1.0]


def test_measure_latency(mock_server):
    latency = measure_latency(mock_server.base_url)
    assert latency > 0


def test_stream_chat_min_tokens_forced(mock_server):
    # mock ignores min_tokens/ignore_eos; just ensure the request still works
    result = stream_chat(mock_server.base_url, "mock-model", "hello", 5, min_tokens=5)
    assert result.error is None
    assert result.completion_tokens == 5


def test_stream_chat_on_token_callback(mock_server):
    seen = []
    result = stream_chat(
        mock_server.base_url,
        "mock-model",
        "hello",
        4,
        on_token=lambda text, t_rel, kind, n_tokens: seen.append((text, t_rel, kind, n_tokens)),
    )
    assert result.error is None
    assert len(seen) == 4
    assert all(t >= 0 for _text, t, _kind, _n in seen)
    assert all(kind == "content" for _text, _t, kind, _n in seen)
    assert all(n == 1 for _text, _t, _k, n in seen)  # mock: 1 token id per chunk


def test_stream_chat_counts_reasoning_tokens():
    server = MockVLLMServer(token_delay=0.001, reasoning_tokens=4).start()
    try:
        seen = []
        result = stream_chat(
            server.base_url,
            "mock-model",
            "hello",
            8,
            on_token=lambda text, t_rel, kind, n_tokens: seen.append((text, t_rel, kind)),
        )
    finally:
        server.stop()
    assert result.error is None
    assert result.completion_tokens == 8  # thinking + content both counted
    assert len(seen) == 8
    assert [k for _t, _r, k in seen[:4]] == ["think"] * 4
    assert [k for _t, _r, k in seen[4:]] == ["content"] * 4


def test_stream_chat_counts_reasoning_field_tokens():
    """vLLM >= 0.26 streams thinking under delta field 'reasoning' (not
    'reasoning_content'). Thinking chunks must still feed on_token — without
    this the live UI never updates for thinking models (the Qwen3 bug)."""
    server = MockVLLMServer(
        token_delay=0.001, reasoning_tokens=6, token_ids=False, reasoning_field="reasoning"
    ).start()
    try:
        seen = []
        result = stream_chat(
            server.base_url,
            "mock-model",
            "hello",
            10,
            on_token=lambda text, t_rel, kind, n_tokens: seen.append((text, t_rel, kind)),
        )
    finally:
        server.stop()
    assert result.error is None
    # every streamed token (thinking + content) must reach the callback
    assert len(seen) == 10
    assert [k for _t, _r, k in seen[:6]] == ["think"] * 6
    assert [k for _t, _r, k in seen[6:]] == ["content"] * 4
    # thinking sets ttft even before any content arrives
    assert result.ttft is not None
    assert result.ttft <= min(t for _text, t, _k in seen)
    assert result.completion_tokens == 10  # from the usage chunk
    assert len(result.token_times) == 10


def test_stream_chat_http_error():
    result = stream_chat("http://127.0.0.1:9/v1", "mock-model", "hello", 4, timeout=2)
    assert result.error is not None


def test_resolve_tokenizer_uses_vllm_endpoint(mock_server):
    count, source = resolve_tokenizer(mock_server.base_url)
    assert source == "vllm /tokenize"
    assert count("one two three four") == 4


def test_resolve_tokenizer_falls_back_to_estimate():
    count, source = resolve_tokenizer("http://127.0.0.1:9/v1", timeout=2)
    assert source.startswith("estimate")
    assert count("one two three") >= 1
