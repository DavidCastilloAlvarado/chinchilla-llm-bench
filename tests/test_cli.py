import pytest

from chinchilla_llm_bench.cli import build_parser, default_report_path, parse_headers
from chinchilla_llm_bench.config import BenchConfig


def test_parser_full_example():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--base-url", "http://127.0.0.1:8000/v1",
            "--model", "qwen3.8-27b-nvfp4",
            "--pp", "200",
            "--tg", "128",
            "--c", "1", "2", "3", "4",
        ]
    )
    assert args.base_url == "http://127.0.0.1:8000/v1"
    assert args.model == "qwen3.8-27b-nvfp4"
    assert args.pp == 200
    assert args.tg == 128
    assert args.c == [1, 2, 3, 4]
    assert args.n == 5
    assert args.loop is False
    assert args.vllm_extensions is False
    assert args.max_completion_tokens is False
    assert args.pp_output_tokens == 1
    assert args.reasoning_effort is None


def test_parser_accepts_max_completion_tokens():
    args = build_parser().parse_args(
        ["--base-url", "http://127.0.0.1:8000/v1", "--model", "mock-model", "--c", "1",
         "--max-completion-tokens"]
    )
    assert args.max_completion_tokens is True


def test_parser_accepts_reasoning_settings():
    args = build_parser().parse_args(
        [
            "--base-url", "http://127.0.0.1:8000/v1",
            "--model", "mock-model",
            "--c", "1",
            "--pp-output-tokens", "32",
            "--reasoning-effort", "minimal",
        ]
    )
    assert args.pp_output_tokens == 32
    assert args.reasoning_effort == "minimal"


def test_parser_accepts_repeated_headers():
    args = build_parser().parse_args(
        [
            "--base-url", "http://127.0.0.1:8000/v1",
            "--model", "mock-model",
            "--c", "1",
            "--header", "X-Tenant-ID: team-a",
            "--header", "X-Request-Source: benchmark",
        ]
    )
    assert parse_headers(args.header) == {
        "X-Tenant-ID": "team-a",
        "X-Request-Source": "benchmark",
    }


@pytest.mark.parametrize("value", ["missing-colon", ": value"])
def test_parser_rejects_invalid_headers(value):
    with pytest.raises(ValueError, match="NAME: VALUE"):
        parse_headers([value])


def test_parser_requires_c():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--base-url", "http://x/v1", "--model", "m"])


def test_config_validation():
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", pp=0)
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", pp_output_tokens=0)
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", tg=0)
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", concurrency=[])
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", concurrency=[0])
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", n=0)
    with pytest.raises(ValueError, match="vllm-extensions"):
        BenchConfig(base_url="http://x/v1", model="m", thinking=True)


def test_config_label():
    config = BenchConfig(base_url="http://x/v1", model="m", pp=200, tg=128)
    assert config.label("pp") == "pp200"
    assert config.label("tg") == "tg128"


def test_default_report_path():
    assert default_report_path("qwen3.8-27b-nvfp4") == "model_result_qwen3_8_27b_nvfp4.txt"
    assert default_report_path("a/b c") == "model_result_a_b_c.txt"
