import pytest

from chinchilla_llm_bench.cli import build_parser, default_report_path
from chinchilla_llm_bench.config import BenchConfig


def test_parser_full_example():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--base-url", "http://127.0.0.1:1235/v1",
            "--model", "qwen3.8-27b-nvfp4",
            "--pp", "200",
            "--tg", "128",
            "--c", "1", "2", "3", "4",
        ]
    )
    assert args.base_url == "http://127.0.0.1:1235/v1"
    assert args.model == "qwen3.8-27b-nvfp4"
    assert args.pp == 200
    assert args.tg == 128
    assert args.c == [1, 2, 3, 4]
    assert args.n == 5
    assert args.loop is False


def test_parser_requires_c():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--base-url", "http://x/v1", "--model", "m"])


def test_config_validation():
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", pp=0)
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", tg=0)
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", concurrency=[])
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", concurrency=[0])
    with pytest.raises(ValueError):
        BenchConfig(base_url="http://x/v1", model="m", n=0)


def test_config_label():
    config = BenchConfig(base_url="http://x/v1", model="m", pp=200, tg=128)
    assert config.label("pp") == "pp200"
    assert config.label("tg") == "tg128"


def test_default_report_path():
    assert default_report_path("qwen3.8-27b-nvfp4") == "model_result_qwen3_8_27b_nvfp4.txt"
    assert default_report_path("a/b c") == "model_result_a_b_c.txt"
