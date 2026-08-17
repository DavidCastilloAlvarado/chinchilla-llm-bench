"""End-to-end test: full sweep against the mock vLLM server."""
from chinchilla_llm_bench.config import BenchConfig
from chinchilla_llm_bench.runner import BenchRunner
from chinchilla_llm_bench.ui import SwarmUI


def _run(mock_server, n=3, concurrency=(1, 2)):
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=list(concurrency),
        n=n,
        timeout=30.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    return runner.run()


def test_full_sweep_produces_all_phases(mock_server):
    phases = _run(mock_server)
    labels = [(p.test, p.concurrency) for p in phases]
    assert labels == [("pp", 1), ("tg", 1), ("pp", 2), ("tg", 2)]


def test_phase_stats_are_sane(mock_server):
    phases = _run(mock_server)
    pp1 = phases[0]
    assert pp1.stats.n_ok == 3
    assert pp1.stats.n_failed == 0
    assert pp1.stats.total_tokens == 3 * 10  # 3 requests x 10 prompt tokens
    assert pp1.stats.total_tps[0] > 0
    assert pp1.stats.ttfr[0] > 0
    assert pp1.stats.est_ppt[0] >= 0

    tg1 = phases[1]
    assert tg1.stats.n_ok == 3
    assert tg1.stats.total_tokens == 3 * 8
    assert tg1.stats.req_tps[0] > 0
    assert tg1.stats.peak_req[0] > 0
    assert tg1.stats.peak_total[0] >= tg1.stats.peak_req[0] * 0.5


def test_agents_reused_across_phases(mock_server):
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[2],
        n=2,
        timeout=30.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    runner.run()
    assert len(runner.agents) == 2
    # after the run every agent is back to idle
    for agent in runner.agents:
        assert agent.snapshot().status == "idle"


def test_unreachable_server_raises(mock_server):
    config = BenchConfig(
        base_url="http://127.0.0.1:9/v1",
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[1],
        n=2,
        timeout=5.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    try:
        runner.run()
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_each_agent_runs_n_repetitions(mock_server):
    """--n means repetitions (llama-benchy style): with n=1, c=3 every agent
    runs exactly once, so all three agents work in a single wave."""
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[3],
        n=1,  # one repetition
        timeout=30.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    phases = runner.run()
    tg = next(p for p in phases if p.test == "tg")
    assert tg.stats.n_ok == 3  # 1 rep x 3 agents
    assert {r.agent for r in tg.requests if r.ok} == {1, 2, 3}


def test_n_reps_times_c_total(mock_server):
    """n=2, c=2 -> 4 requests total (2 full waves of 2 agents)."""
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[2],
        n=2,
        timeout=30.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    phases = runner.run()
    tg = next(p for p in phases if p.test == "tg")
    assert tg.stats.n_ok == 4
    from collections import Counter

    assert Counter(r.agent for r in tg.requests if r.ok) == {1: 2, 2: 2}
