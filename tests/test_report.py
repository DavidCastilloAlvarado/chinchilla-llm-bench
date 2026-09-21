from chinchilla_llm_bench.config import BenchConfig
from chinchilla_llm_bench.report import (
    COLUMNS,
    build_rows,
    markdown_report,
    rich_report,
)
from chinchilla_llm_bench.runner import PhaseResult
from chinchilla_llm_bench.stats import PhaseStats, RunSummary


def _config() -> BenchConfig:
    return BenchConfig(
        base_url="http://x/v1",
        model="qwen3.8-27b-nvfp4",
        pp=200,
        tg=128,
        concurrency=[1, 2],
    )


def _phase(test: str, c: int) -> PhaseResult:
    if test == "pp":
        stats = PhaseStats(
            test="pp",
            concurrency=c,
            n_ok=5,
            n_failed=0,
            total_tokens=1000,
            phase_seconds=1.0,
            total_tps=(892.88, 2.82),
            req_tps=(892.88, 2.82),
            peak_total=(0.0, 0.0),
            peak_req=(0.0, 0.0),
            ttfr=(0.14427, 0.01948),
            net_ttft=(0.04072, 0.01948),
            e2e_ttft=(0.14427, 0.01948),
        )
    else:
        stats = PhaseStats(
            test="tg",
            concurrency=c,
            n_ok=5,
            n_failed=0,
            total_tokens=640,
            phase_seconds=9.6,
            total_tps=(173.55, 12.4),
            req_tps=(66.83, 5.13),
            peak_total=(210.0, 30.0),
            peak_req=(80.0, 6.0),
            ttfr=(0.0, 0.0),
            net_ttft=(0.0, 0.0),
            e2e_ttft=(0.0, 0.0),
        )
    return PhaseResult(test, c, stats)


def _summary() -> RunSummary:
    return RunSummary(
        duration_seconds=83.45,
        max_concurrency=18,
        request_attempts=202,
        http_errors=2,
    )


def test_columns():
    assert COLUMNS == [
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


def test_build_rows_layout():
    config = _config()
    rows = build_rows(config, [_phase("pp", 1), _phase("tg", 1)])
    assert len(rows) == 2
    pp_row, tg_row = rows
    assert pp_row[0] == "qwen3.8-27b-nvfp4"
    assert pp_row[1] == "pp200 (c1)"
    assert pp_row[2] == "892.88 ± 2.82"  # t/s (total)
    assert pp_row[3] == "892.88 ± 2.82"  # t/s (req)
    assert pp_row[4] == ""  # no peak for pp (single token)
    assert pp_row[5] == ""
    assert pp_row[6] == "144.27 ± 19.48"  # ttfr
    assert pp_row[7] == "40.72 ± 19.48"  # net_ttft
    assert pp_row[8] == "144.27 ± 19.48"  # e2e_ttft
    assert tg_row[1] == "tg128 (c1)"
    assert tg_row[2] == "173.55 ± 12.40"  # t/s (total)
    assert tg_row[3] == "66.83 ± 5.13"  # t/s (req)
    assert tg_row[4] == "210.00 ± 30.00"  # peak t/s
    assert tg_row[5] == "80.00 ± 6.00"  # peak t/s (req)
    assert tg_row[6] == ""  # no latency columns for tg
    assert tg_row[7] == ""
    assert tg_row[8] == ""


def test_markdown_report_shape():
    config = _config()
    text = markdown_report(config, [_phase("pp", 1), _phase("tg", 1)], _summary())
    lines = text.strip().splitlines()
    assert len(lines) == 10
    header = lines[0].split("|")
    assert [h.strip() for h in header[1:-1]] == COLUMNS
    assert lines[1].startswith("|:---")  # first col left-aligned
    assert all(line.startswith("|") and line.endswith("|") for line in lines[:4])
    assert lines[5] == "## Run summary"
    assert lines[7] == "- Duration: 1m 23.45s"
    assert lines[8] == "- Max concurrency: 18"
    assert lines[9] == "- HTTP errors: 2/202 (0.99%)"


def test_rich_report_renders():
    config = _config()
    table = rich_report(config, [_phase("pp", 1), _phase("tg", 1)], _summary())
    from rich.console import Console

    console = Console(width=200, force_terminal=False, record=True)
    console.print(table)
    rendered = console.export_text()
    assert "pp200 (c1)" in rendered
    assert "tg128 (c1)" in rendered
    assert "892.88 ± 2.82" in rendered
    assert "DURATION 1m 23.45s" in rendered
    assert "MAX CONCURRENCY 18" in rendered
    assert "HTTP ERRORS 2/202 (0.99%)" in rendered
