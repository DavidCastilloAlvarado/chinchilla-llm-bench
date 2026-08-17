from chinchilla_llm_bench.config import BenchConfig
from chinchilla_llm_bench.report import (
    COLUMNS,
    build_rows,
    markdown_report,
    rich_report,
)
from chinchilla_llm_bench.runner import PhaseResult
from chinchilla_llm_bench.stats import PhaseStats


def _config() -> BenchConfig:
    return BenchConfig(base_url="http://x/v1", model="qwen3.8-27b-nvfp4", pp=200, tg=128, concurrency=[1, 2])


def _phase(test: str, c: int) -> PhaseResult:
    if test == "pp":
        stats = PhaseStats(
            test="pp",
            concurrency=c,
            n_ok=5,
            n_failed=0,
            total_tokens=1000,
            phase_seconds=1.0,
            req_tps=(892.88, 2.82),
            ttfr=(0.14427, 0.01948),
            est_ppt=(0.04072, 0.01948),
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
            req_tps=(66.83, 5.13),
            ttfr=(0.0, 0.0),
            est_ppt=(0.0, 0.0),
            e2e_ttft=(0.0, 0.0),
        )
    return PhaseResult(test, c, stats)


def test_columns():
    assert COLUMNS == [
        "model",
        "test",
        "t/s (req)",
        "ttfr (ms)",
        "est_ppt (ms)",
        "e2e_ttft (ms)",
    ]


def test_build_rows_layout():
    config = _config()
    rows = build_rows(config, [_phase("pp", 1), _phase("tg", 1)])
    assert len(rows) == 2
    pp_row, tg_row = rows
    assert pp_row[0] == "qwen3.8-27b-nvfp4"
    assert pp_row[1] == "pp200 (c1)"
    assert pp_row[2] == "892.88 ± 2.82"  # t/s (req)
    assert pp_row[3] == "144.27 ± 19.48"  # ttfr
    assert pp_row[4] == "40.72 ± 19.48"  # est_ppt
    assert pp_row[5] == "144.27 ± 19.48"  # e2e_ttft
    assert tg_row[1] == "tg128 (c1)"
    assert tg_row[2] == "66.83 ± 5.13"  # t/s (req)
    assert tg_row[3] == ""  # no ttfr columns for tg
    assert tg_row[4] == ""
    assert tg_row[5] == ""


def test_markdown_report_shape():
    config = _config()
    text = markdown_report(config, [_phase("pp", 1), _phase("tg", 1)])
    lines = text.strip().splitlines()
    assert len(lines) == 4
    header = lines[0].split("|")
    assert [h.strip() for h in header[1:-1]] == COLUMNS
    assert lines[1].startswith("|:---")  # first col left-aligned
    assert all(line.startswith("|") and line.endswith("|") for line in lines)


def test_rich_report_renders():
    config = _config()
    table = rich_report(config, [_phase("pp", 1), _phase("tg", 1)])
    from rich.console import Console

    console = Console(width=200, force_terminal=False, record=True)
    console.print(table)
    rendered = console.export_text()
    assert "pp200 (c1)" in rendered
    assert "tg128 (c1)" in rendered
    assert "892.88 ± 2.82" in rendered
