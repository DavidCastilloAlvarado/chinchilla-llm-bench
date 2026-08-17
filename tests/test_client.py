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
