"""End-to-end test: full sweep against the mock vLLM server."""
import pytest

from chinchilla_llm_bench.config import BenchConfig
from chinchilla_llm_bench.runner import BenchRunner
from chinchilla_llm_bench.stats import RequestStats
from chinchilla_llm_bench.ui import SwarmUI
from mock_server import MockVLLMServer


def _run(
    mock_server,
    n=3,
    concurrency=(1, 2),
    headers=None,
    max_completion_tokens=False,
    pp_output_tokens=1,
    reasoning_effort=None,
):
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=list(concurrency),
        n=n,
        timeout=30.0,
        headers=headers or {},
        max_completion_tokens=max_completion_tokens,
        pp_output_tokens=pp_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    return runner.run()


def test_full_sweep_produces_all_phases(mock_server):
    phases = _run(mock_server)
    labels = [(p.test, p.concurrency) for p in phases]
    assert labels == [("pp", 1), ("tg", 1), ("pp", 2), ("tg", 2)]


def test_full_sweep_sends_custom_headers():
    server = MockVLLMServer(required_header=("X-Tenant-ID", "team-a")).start()
    try:
        phases = _run(server, n=1, concurrency=(1,), headers={"X-Tenant-ID": "team-a"})
    finally:
        server.stop()
    assert all(phase.stats.n_failed == 0 for phase in phases)


def test_full_sweep_uses_max_completion_tokens():
    server = MockVLLMServer().start()
    try:
        phases = _run(server, n=1, concurrency=(1,), max_completion_tokens=True)
    finally:
        server.stop()
    assert all(phase.stats.n_failed == 0 for phase in phases)
    chat_bodies = [body for path, body in server.server.request_bodies if path.endswith("/chat/completions")]
    assert chat_bodies
    assert all("max_completion_tokens" in body for body in chat_bodies)
    assert all("max_tokens" not in body for body in chat_bodies)


def test_full_sweep_forwards_reasoning_settings():
    server = MockVLLMServer().start()
    try:
        phases = _run(
            server,
            n=1,
            concurrency=(1,),
            pp_output_tokens=32,
            reasoning_effort="minimal",
        )
    finally:
        server.stop()
    assert all(phase.stats.n_failed == 0 for phase in phases)
    chat_bodies = [body for path, body in server.server.request_bodies if path.endswith("/chat/completions")]
    assert chat_bodies[0]["max_tokens"] == 32
    assert all(body["reasoning_effort"] == "minimal" for body in chat_bodies)


def test_phase_stats_are_sane(mock_server):
    phases = _run(mock_server)
    pp1 = phases[0]
    assert pp1.stats.n_ok == 3
    assert pp1.stats.n_failed == 0
    assert pp1.stats.total_tokens == 3 * 10  # 3 requests x 10 prompt tokens
    assert pp1.stats.req_tps[0] > 0
    assert pp1.stats.ttfr[0] > 0
    assert pp1.stats.net_ttft[0] >= 0

    tg1 = phases[1]
    assert tg1.stats.n_ok == 3
    assert tg1.stats.total_tokens == 3 * 8
    assert tg1.stats.req_tps[0] > 0


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


def test_waves_grouped_by_repetition(mock_server):
    """n=2, c=2 -> two waves (rep 0, rep 1) of 2 requests each."""
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
    from collections import Counter

    assert Counter(r.rep for r in tg.requests if r.ok) == {0: 2, 1: 2}


def test_batch_metrics_c1_total_equals_req(mock_server):
    """c=1: the wave aggregate is the single request itself (llama-benchy)."""
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[1],
        n=2,
        timeout=30.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    phases = runner.run()
    tg = next(p for p in phases if p.test == "tg")
    assert tg.stats.total_tps == tg.stats.req_tps
    assert tg.stats.req_tps[0] > 0
    # short 8-token stream (< 1 s): peak uses the actual span -> ~ mean rate
    assert tg.stats.peak_total[0] > 0
    assert tg.stats.peak_req[0] > 0


def test_batch_metrics_c2_total_higher_than_req(mock_server):
    """c=2: the wave (total) throughput is the batch aggregate and should
    exceed the per-request rate (two decoders overlap in one wave)."""
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
    assert tg.stats.total_tps[0] > tg.stats.req_tps[0]


def test_pp_net_ttft_subtracts_latency(mock_server):
    """net_ttft = max(0, ttft - measured network latency)."""
    config = BenchConfig(
        base_url=mock_server.base_url,
        model="mock-model",
        pp=10,
        tg=8,
        concurrency=[1],
        n=2,
        timeout=30.0,
    )
    ui = SwarmUI(config, quiet=True)
    runner = BenchRunner(config, ui)
    phases = runner.run()
    pp = next(p for p in phases if p.test == "pp")
    assert runner._latency > 0  # latency probe ran
    assert 0 <= pp.stats.net_ttft[0] < pp.stats.e2e_ttft[0]
    assert pp.stats.e2e_ttft[0] > 0
    assert pp.stats.req_tps[0] > 0


def test_pp_request_rate_uses_first_token_not_first_response_chunk():
    config = BenchConfig(
        base_url="http://127.0.0.1:9/v1",
        model="mock-model",
        pp=100,
        tg=8,
        concurrency=[1],
    )
    runner = BenchRunner(config, SwarmUI(config, quiet=True))
    runner._latency = 0.1
    request = RequestStats(
        agent=1,
        test="pp",
        concurrency=1,
        rep=0,
        prompt_tokens=100,
        completion_tokens=1,
        ttfr=0.2,
        ttft=1.1,
        duration=1.2,
        token_times=[1.1],
    )
    stats = runner._summarize("pp", 1, [request], phase_seconds=1.2)
    assert stats.net_ttft == (1.0, 0.0)
    assert stats.req_tps == (100.0, 0.0)
    assert stats.total_tps == pytest.approx((100 / 1.1, 0.0))
