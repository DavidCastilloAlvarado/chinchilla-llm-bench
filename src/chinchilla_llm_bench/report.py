"""Final report: rich box table for the console, markdown table for the file."""
from __future__ import annotations

from rich import box
from rich.table import Table

from .config import BenchConfig
from .runner import PhaseResult
from .stats import fmt_mean_std, fmt_ms_mean_std

COLUMNS = [
    "model",
    "test",
    "t/s (total)",
    "t/s (req)",
    "peak t/s",
    "peak t/s (req)",
    "ttfr (ms)",
    "est_ppt (ms)",
    "e2e_ttft (ms)",
]


def build_rows(config: BenchConfig, phases: list[PhaseResult]) -> list[list[str]]:
    """One row per (test, concurrency) phase, matching the reference layout:

    pp rows fill ttfr/est_ppt/e2e_ttft; tg rows fill the peak columns.
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
                    fmt_ms_mean_std(*s.est_ppt),
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


def markdown_report(config: BenchConfig, phases: list[PhaseResult]) -> str:
    """Markdown table (first column left-aligned, the rest right-aligned)."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|:---" + "|---:" * (len(COLUMNS) - 1) + "|",
    ]
    for row in build_rows(config, phases):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def rich_report(config: BenchConfig, phases: list[PhaseResult]) -> Table:
    """Box-drawn table for the console."""
    table = Table(box=box.HEAVY, title=f"benchmark results · {config.model}")
    table.add_column("model", justify="left", style="bold")
    for col in COLUMNS[1:]:
        table.add_column(col, justify="right")
    for row in build_rows(config, phases):
        table.add_row(*row)
    return table
