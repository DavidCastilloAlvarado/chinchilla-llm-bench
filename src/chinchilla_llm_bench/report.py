"""Final report: rich box table for the console, markdown table for the file.

Columns follow llama-benchy's report layout. All token counts come from
real tokens (per-chunk ``token_ids`` when the server provides them, else
its usage chunk), never from chunk events:

* ``t/s (total)`` — aggregate tokens/second per wave (one repetition of c
  concurrent requests): prompt tokens / span or decode tokens / span
* ``t/s (req)`` — per-request tokens/second
* ``peak t/s`` — max tokens in a 1 s sliding window (per wave / per request)
* ``ttfr`` / ``net_ttft`` / ``e2e_ttft`` — prompt-processing latencies,
  where net_ttft = ttft - measured network latency
"""
from __future__ import annotations

from rich import box
from rich.table import Table

from .config import BenchConfig
from .runner import PhaseResult
from .stats import RunSummary, fmt_mean_std, fmt_ms_mean_std

COLUMNS = [
    "model",
    "test",
    "t/s (total)",
    "t/s (req)",
    "peak t/s",
    "peak t/s (req)",
    "ttfr (ms)",
    "net_ttft (ms)",
    "e2e_ttft (ms)",
]


def build_rows(config: BenchConfig, phases: list[PhaseResult]) -> list[list[str]]:
    """One row per (test, concurrency) phase.

    pp rows fill the latency columns (ttfr/net_ttft/e2e_ttft) and leave the
    peak columns empty (single token per request — nothing to peak); tg
    rows fill the throughput + peak columns.
    """
    rows: list[list[str]] = []
    for pr in phases:
        s = pr.stats
        size = config.pp if pr.test == "pp" else config.tg
        label = f"{pr.test}{size} (c{pr.concurrency})"
        if pr.test == "pp":
            rows.append(
                [
                    config.model,
                    label,
                    fmt_mean_std(*s.total_tps),
                    fmt_mean_std(*s.req_tps),
                    "",
                    "",
                    fmt_ms_mean_std(*s.ttfr),
                    fmt_ms_mean_std(*s.net_ttft),
                    fmt_ms_mean_std(*s.e2e_ttft),
                ]
            )
        else:
            rows.append(
                [
                    config.model,
                    label,
                    fmt_mean_std(*s.total_tps),
                    fmt_mean_std(*s.req_tps),
                    fmt_mean_std(*s.peak_total),
                    fmt_mean_std(*s.peak_req),
                    "",
                    "",
                    "",
                ]
            )
    return rows


def markdown_report(
    config: BenchConfig,
    phases: list[PhaseResult],
    summary: RunSummary,
) -> str:
    """Markdown table (first column left-aligned, the rest right-aligned)."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|:---" + "|---:" * (len(COLUMNS) - 1) + "|",
    ]
    for row in build_rows(config, phases):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Run summary",
            "",
            f"- Duration: {summary.duration_label}",
            f"- Max concurrency: {summary.max_concurrency}",
            (
                "- HTTP errors: "
                f"{summary.http_errors}/{summary.request_attempts} "
                f"({summary.http_error_percentage:.2f}%)"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def rich_report(
    config: BenchConfig,
    phases: list[PhaseResult],
    summary: RunSummary,
) -> Table:
    """Box-drawn table for the console."""
    table = Table(box=box.HEAVY, title=f"benchmark results · {config.model}")
    table.caption = (
        f"DURATION {summary.duration_label}  |  "
        f"MAX CONCURRENCY {summary.max_concurrency}  |  "
        f"HTTP ERRORS {summary.http_errors}/{summary.request_attempts} "
        f"({summary.http_error_percentage:.2f}%)"
    )
    table.add_column("model", justify="left", style="bold")
    for col in COLUMNS[1:]:
        table.add_column(col, justify="right")
    for row in build_rows(config, phases):
        table.add_row(*row)
    return table
