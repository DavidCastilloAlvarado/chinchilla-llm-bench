from chinchilla_llm_bench.client import list_models, resolve_tokenizer, stream_chat

from mock_server import MockVLLMServer


def test_list_models(mock_server):
    assert list_models(mock_server.base_url) == ["mock-model"]


def test_stream_chat_counts_tokens(mock_server):
    result = stream_chat(mock_server.base_url, "mock-model", "hello", 10)
    assert result.error is None
    assert result.completion_tokens == 10
    assert result.prompt_tokens == 10  # from the usage chunk
    assert result.ttfr is not None and result.ttfr > 0
    assert result.duration >= result.ttfr
    assert len(result.token_times) == 10
    assert result.token_times == sorted(result.token_times)
    assert result.text.startswith("tok0")


def test_stream_chat_single_token(mock_server):
    result = stream_chat(mock_server.base_url, "mock-model", "hello", 1)
    assert result.error is None
    assert result.completion_tokens == 1
    assert result.ttfr is not None


def test_stream_chat_on_token_callback(mock_server):
    seen = []
    result = stream_chat(
        mock_server.base_url,
        "mock-model",
        "hello",
        4,
        on_token=lambda text, t_rel, kind: seen.append((text, t_rel, kind)),
    )
    assert result.error is None
    assert len(seen) == 4
    assert all(t >= 0 for _text, t, _kind in seen)
    assert all(kind == "content" for _text, _t, kind in seen)


def test_stream_chat_counts_reasoning_tokens():
    server = MockVLLMServer(token_delay=0.001, reasoning_tokens=4).start()
    try:
        seen = []
        result = stream_chat(
            server.base_url,
            "mock-model",
            "hello",
            8,
            on_token=lambda text, t_rel, kind: seen.append((text, t_rel, kind)),
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
