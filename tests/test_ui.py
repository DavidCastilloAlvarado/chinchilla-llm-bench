"""Tests for the swarm UI: MAXC metric and the scrollable grid."""
from __future__ import annotations

import io
import threading
from types import SimpleNamespace

from rich.console import Console

from chinchilla_llm_bench.agent import Agent, ROLES
from chinchilla_llm_bench.config import BenchConfig
from chinchilla_llm_bench.runner import BenchRunner
from chinchilla_llm_bench.ui import InputWatcher, SwarmUI
from mock_server import MockVLLMServer


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
    ui.set_phase("pp10 (c3)", 3)
    ui.on_gen_start()
    ui.on_gen_start()
    ui.on_gen_start()
    assert ui.max_gen_concurrency == 3
    assert ui.generation_concurrency == (3, 3)
    ui.on_gen_end()
    ui.on_gen_end()
    ui.on_gen_end()
    ui.on_gen_end()  # never goes negative
    assert ui.max_gen_concurrency == 3
    # A later phase and smaller repetition must not reset the run-wide peak.
    ui.set_phase("tg8 (c2)", 2)
    ui.on_gen_start()
    assert ui.generation_concurrency == (1, 3)
    ui.on_gen_end()
    assert ui.generation_concurrency == (0, 3)
    assert ui.max_gen_concurrency == 3


def test_topbar_shows_concurrency_and_http_errors():
    ui = _swarm(2, 120, 20)
    ui.on_gen_start()
    ui.on_http_error()
    out = _render(ui)
    assert "CURC 1" in out
    assert "MAXC" in out
    assert "HTTP ERR 1" in out


def test_streamed_output_cannot_inject_terminal_controls():
    ui = _swarm(1, 120, 20)
    output = io.StringIO()
    ui.console = Console(
        file=output,
        force_terminal=True,
        width=120,
        height=20,
    )
    agent = ui.agents[0]
    agent.set_prompt("safe prompt", 2)
    agent.add_token("before\x1b[2Jafter\u202e")
    ui.console.print(ui._frame())
    rendered = output.getvalue()
    assert "\x1b[2J" not in rendered
    assert "\u202e" not in rendered
    assert "before [2Jafter" in rendered


def test_completed_screen_waits_for_exit():
    ui = _swarm(2, 120, 20)
    ui.quiet = False
    ui.console = Console(
        file=io.StringIO(),
        force_terminal=True,
        width=120,
        height=20,
    )
    ui._watcher = SimpleNamespace(active=True)
    exit_timer = threading.Timer(0.01, ui.request_stop)
    exit_timer.start()
    try:
        ui.wait_for_exit()
    finally:
        exit_timer.join()
    assert ui.phase == "FINISHED - PRESS Q TO EXIT"
    assert ui.stopped


def test_completed_screen_does_not_wait_for_redirected_output():
    ui = _swarm(2, 120, 20)
    ui.quiet = False
    ui._watcher = SimpleNamespace(active=True)
    ui.wait_for_exit()
    assert ui.phase == "FINISHED - PRESS Q TO EXIT"
    assert not ui.stopped


def test_completed_screen_stops_waiting_if_input_watcher_dies():
    ui = _swarm(2, 120, 20)
    ui.quiet = False
    ui.console = Console(
        file=io.StringIO(),
        force_terminal=True,
        width=120,
        height=20,
    )
    watcher = InputWatcher(lambda _key: None)
    watcher._thread = threading.Thread(target=lambda: None)
    watcher._thread.start()
    watcher._thread.join()
    assert not watcher.active
    ui._watcher = watcher
    ui.wait_for_exit()
    assert ui.phase == "FINISHED - PRESS Q TO EXIT"
    assert not ui.stopped


def test_finished_screen_freezes_live_counters_and_duration():
    ui = _swarm(2, 120, 20)
    ui._start = 100.0
    ui._finished_at = 105.0
    ui.total_tokens = 42
    ui.requests_done = 3
    ui.on_gen_start()
    ui.on_gen_end()
    ui.tracker.add(104.0, 20)
    ui.req_tracker.add(104.0)
    ui.set_phase("FINISHED - PRESS Q TO EXIT", 0)
    out = _render(ui)
    assert "FINISHED - PRESS Q TO EXIT" in out
    assert "TOK/S 0" in out
    assert "REQ/S 0.00" in out
    assert "TOK GEN 42" in out
    assert "REQ 3" in out
    assert "DURATION 5s" in out
    assert ui.run_summary.duration_seconds == 5.0
    assert ui.run_summary.duration_label == "5.00s"


def test_runner_maxc_counts_generation(mock_server):
    config = _config(base_url=mock_server.base_url, concurrency=[2], tg=8, n=1)
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    runner.run()
    # Two concurrent requests generate at the same time -> peak of 2.
    assert ui.max_gen_concurrency == 2
    assert ui.max_gen_concurrency <= max(config.concurrency)


def test_runner_maxc_updates_during_pp_phase(mock_server):
    config = _config(
        base_url=mock_server.base_url,
        concurrency=[2],
        pp_output_tokens=8,
        n=1,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    runner._preflight()
    runner._phase("pp", 2)
    assert ui.max_gen_concurrency == 2


def test_runner_counts_http_errors_across_warmup_and_phases():
    server = MockVLLMServer(chat_status=400).start()
    try:
        config = _config(
            base_url=server.base_url,
            concurrency=[1],
            warmup=1,
            n=1,
        )
        ui = SwarmUI(config, quiet=True)
        runner = BenchRunner(config, ui)
        phases = runner.run()
    finally:
        server.stop()
    assert ui.http_errors == 3  # warmup + pp + tg
    assert ui.run_summary.request_attempts == 3  # warmup + pp + tg
    assert ui.run_summary.http_error_percentage == 100.0
    assert all(phase.stats.n_failed == 1 for phase in phases)


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
