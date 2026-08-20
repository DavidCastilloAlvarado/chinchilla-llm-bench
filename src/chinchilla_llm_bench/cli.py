"""Command line interface."""
from __future__ import annotations

import argparse
import re
import sys

from rich.console import Console

from .config import BenchConfig
from .report import markdown_report, rich_report
from .runner import BenchRunner
from .ui import SwarmUI


def parse_headers(values: list[str]) -> dict[str, str]:
    """Parse repeated ``NAME: VALUE`` command-line header values."""
    headers: dict[str, str] = {}
    for value in values:
        name, separator, header_value = value.partition(":")
        name = name.strip()
        if not separator or not name:
            raise ValueError("headers must use the format 'NAME: VALUE'")
        headers[name] = header_value.strip()
    return headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chinchilla-bench",
        description="Concurrency benchmark for a vLLM server with a live agent-swarm UI.",
        epilog=(
            "example: chinchilla-bench --base-url http://127.0.0.1:8000/v1 "
            "--model qwen3.8-27b-nvfp4 --pp 200 --tg 128 --c 1 2 3 4"
        ),
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. http://host:1235/v1",
    )
    parser.add_argument("--model", required=True, help="model name served by vLLM")
    parser.add_argument(
        "--pp", type=int, default=200, help="prompt tokens for the prefill test (default 200)"
    )
    parser.add_argument(
        "--pp-output-tokens",
        type=int,
        default=1,
        help="completion allowance for prefill requests (default 1)",
    )
    parser.add_argument(
        "--tg", type=int, default=128, help="tokens to generate for the decode test (default 128)"
    )
    parser.add_argument(
        "--c",
        type=int,
        nargs="+",
        required=True,
        metavar="N",
        help="concurrency levels to sweep, e.g. --c 1 2 3 4",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="repetitions per test: each agent runs n times, total = n x c (default 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="throwaway requests before measuring, to wake a cold server (default 2)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    parser.add_argument("--seed", type=int, default=1337, help="prompt generation seed")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable Qwen3 reasoning; requires --vllm-extensions",
    )
    parser.add_argument(
        "--vllm-extensions",
        action="store_true",
        help="send vLLM-only fields for thinking, exact generation, and token IDs",
    )
    parser.add_argument(
        "--max-completion-tokens",
        action="store_true",
        help="use max_completion_tokens instead of max_tokens for newer OpenAI models",
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="LEVEL",
        default=None,
        help="provider reasoning effort, e.g. minimal, low, medium, or high",
    )
    parser.add_argument(
        "--no-exact-tg",
        dest="exact_tg",
        action="store_false",
        default=True,
        help="with --vllm-extensions, let the model stop at EOS instead of "
        "forcing the full tg budget",
    )
    parser.add_argument(
        "--api-key",
        default="dummy",
        help="API key (vLLM accepts any non-empty string; default 'dummy')",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="custom HTTP header; repeat for each header",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="repeat the whole sweep until stopped (toggle with 'l' while running)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="report file (default model_result_<model>.txt)",
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="plain progress lines instead of the swarm UI"
    )
    return parser


def default_report_path(model: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
    return f"model_result_{safe}.txt"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        config = BenchConfig(
            base_url=args.base_url,
            model=args.model,
            pp=args.pp,
            pp_output_tokens=args.pp_output_tokens,
            tg=args.tg,
            concurrency=list(args.c),
            n=args.n,
            warmup=args.warmup,
            temperature=args.temperature,
            timeout=args.timeout,
            seed=args.seed,
            loop=args.loop,
            thinking=args.thinking,
            exact_tg=args.exact_tg,
            vllm_extensions=args.vllm_extensions,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
            api_key=args.api_key,
            headers=parse_headers(args.header),
        )
    except ValueError as exc:
        console.print(f"[bold red]invalid settings:[/] {exc}")
        return 2

    ui = SwarmUI(config, quiet=args.no_ui)
    runner = BenchRunner(config, ui)
    try:
        phases = runner.run()
    except SystemExit as exc:
        if str(exc):
            console.print(f"[bold red]{exc}[/]")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/]")
        return 130

    console.print()
    console.print(config.summary())
    for warning in runner.warnings:
        console.print(f"[yellow]warning:[/] {warning}")
    if phases:
        console.print(rich_report(config, phases))
        out = args.output or default_report_path(config.model)
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(markdown_report(config, phases))
        console.print(f"[dim]report written to {out}[/]")
    else:
        console.print("[yellow]no completed phases — nothing to report[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
