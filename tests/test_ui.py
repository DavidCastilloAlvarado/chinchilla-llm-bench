"""Tests for the swarm UI: MAXC metric and the scrollable grid."""
from __future__ import annotations

import io

from rich.console import Console

from chinchilla_llm_bench.agent import Agent, ROLES
from chinchilla_llm_bench.config import BenchConfig
from chinchilla_llm_bench.runner import BenchRunner
from chinchilla_llm_bench.ui import SwarmUI


def _config(**overrides) -> BenchConfig:
    base = dict(
        base_url="http://127.0.0.1:9/v1",
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[2],
        n=1,
        timeout=30.0,
    )
    base.update(overrides)
    return BenchConfig(**base)


def _agents(n: int) -> list[Agent]:
    return [Agent(i + 1, ROLES[i % len(ROLES)]) for i in range(n)]


def _render(ui: SwarmUI) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=ui.console.width, height=ui.console.height)
    console.print(ui._frame())
    return buf.getvalue()


def _swarm(n_agents: int, width: int, height: int) -> SwarmUI:
    buf = io.StringIO()
    console = Console(file=buf, width=width, height=height)
    return SwarmUI(_config(), agents=_agents(n_agents), console=console, quiet=True)


# ------------------------------------------------------------------ MAXC
def test_max_gen_concurrency_tracks_peak():
    ui = _swarm(2, 120, 20)
    assert ui.max_gen_concurrency == 0
    ui.on_gen_start()
    ui.on_gen_start()
    ui.on_gen_start()
    assert ui.max_gen_concurrency == 3
    ui.on_gen_end()
    ui.on_gen_end()
    ui.on_gen_end()
    ui.on_gen_end()  # never goes negative
    assert ui.max_gen_concurrency == 3
    # a later, smaller burst must not lower the recorded peak
    ui.on_gen_start()
    ui.on_gen_end()
    assert ui.max_gen_concurrency == 3


def test_topbar_shows_maxc():
    ui = _swarm(2, 120, 20)
    ui.on_gen_start()
    out = _render(ui)
    assert "MAXC" in out


def test_runner_maxc_counts_only_tg_generation(mock_server):
    config = _config(base_url=mock_server.base_url, concurrency=[2], tg=8, n=1)
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    runner.run()
    # Two concurrent TG requests generate at the same time -> peak of 2.
    assert ui.max_gen_concurrency == 2
    # pp phase must not have inflated it beyond the tg concurrency.
    assert ui.max_gen_concurrency <= max(config.concurrency)


# ---------------------------------------------------------------- scroll
def test_grid_windows_and_scrolls_when_too_tall():
    ui = _swarm(32, 120, 20)
    ui._grid()  # establishes window state
    assert ui._scrollable
    assert ui._visible_rows < ui._total_rows
    # at the top we see the first cards, not the last ones
    out = _render(ui)
    assert "A1" in out
    assert "A32" not in out
    ui._on_key("end")
    assert ui.scroll == ui._total_rows - ui._visible_rows
    out = _render(ui)
    assert "A1" not in out
    # the bottom cards are now visible
    last = 32
    assert f"A{last}" in out


def test_grid_no_scroll_when_it_fits():
    ui = _swarm(4, 120, 30)
    ui._grid()
    assert not ui._scrollable
    assert ui.scroll == 0
    out = _render(ui)
    assert "A1" in out
    assert "A4" in out


def test_scroll_keys_clamp_to_bounds():
    ui = _swarm(32, 120, 20)
    ui._grid()
    max_scroll = ui._total_rows - ui._visible_rows
    ui._on_key("down")
    assert ui.scroll == 1
    ui._on_key("up")
    assert ui.scroll == 0
    # cannot scroll above the top
    ui._on_key("up")
    assert ui.scroll == 0
    # cannot scroll past the bottom
    for _ in range(max_scroll + 5):
        ui._on_key("down")
    assert ui.scroll == max_scroll
    ui._on_key("home")
    assert ui.scroll == 0
    for _ in range(max_scroll):  # one pgdown per visible window, clamped at end
        ui._on_key("pgdown")
    assert ui.scroll == max_scroll
    ui._on_key("pgup")
    assert ui.scroll == max_scroll - ui._visible_rows
