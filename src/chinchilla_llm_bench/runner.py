"""Orchestrates the benchmark: preflight, baseline, phases, agents, stats."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from .agent import Agent, ROLES
from .client import list_models, resolve_tokenizer, stream_chat
from .config import BenchConfig
from .prompt import ROLE_OPENERS, build_prompt, build_tg_prompt
from .stats import (
    PhaseStats,
    RequestStats,
    mean_std,
    peak_rate,
    window_rate,
)
from .ui import SwarmUI

_WINDOW_PAD = 0.05  # seconds of slack around each request's active window


@dataclass
class RequestSpec:
    test: str
    concurrency: int
    prompt: str
    prompt_tokens: int
    max_tokens: int


@dataclass
class PhaseResult:
    test: str
    concurrency: int
    stats: PhaseStats
    requests: list[RequestStats] = field(default_factory=list)


class BenchRunner:
    """Runs the full sweep: for each c in config.concurrency -> pp phase, tg phase."""

    def __init__(self, config: BenchConfig, ui: SwarmUI):
        self.config = config
        self.ui = ui
        max_c = max(config.concurrency)
        self.agents = [Agent(i + 1, ROLES[i % len(ROLES)]) for i in range(max_c)]
        self.ui.agents = self.agents
        self.phases: list[PhaseResult] = []
        self.warnings: list[str] = []
        self._count = None  # token counter (resolved in preflight)
        self._pp_prompts: dict[int, tuple[str, int]] = {}
        self._tg_prompts: dict[int, tuple[str, int]] = {}
        self._baseline_ttfr = 0.0

    # ------------------------------------------------------------------ run
    def run(self) -> list[PhaseResult]:
        self.ui.start()
        try:
            self._preflight()
            self._warmup()
            while True:
                self._baseline()
                for c in self.config.concurrency:
                    if self.ui.stopped:
                        break
                    self._phase("pp", c)
                    if self.ui.stopped:
                        break
                    self._phase("tg", c)
                if self.ui.stopped or not self.ui.loop:
                    break
        finally:
            self.ui.stop()
        return self.phases

    # ------------------------------------------------------------- preflight
    def _preflight(self) -> None:
        self.ui.set_phase("connecting…", 0)
        self.ui.log(f"== connecting to {self.config.base_url} ==")
        try:
            models = list_models(
                self.config.base_url, api_key=self.config.api_key, timeout=10
            )
        except Exception as exc:
            raise SystemExit(f"cannot reach {self.config.base_url}: {exc}") from exc
        if self.config.model not in models:
            self.warnings.append(
                f"model {self.config.model!r} not listed in /models ({', '.join(models)})"
            )
        self._count, source = resolve_tokenizer(
            self.config.base_url, api_key=self.config.api_key
        )
        # One prompt per agent: role-specific opener, padded to pp tokens.
        for agent in self.agents:
            pp_prompt = build_prompt(
                self.config.pp,
                self._count,
                seed=self.config.seed + agent.idx,
                opener=ROLE_OPENERS.get(agent.role),
            )
            self._pp_prompts[agent.idx] = (pp_prompt, self._count(pp_prompt))
            tg_prompt = build_tg_prompt(agent.role, self.config.tg, self.config.tg_prompt)
            self._tg_prompts[agent.idx] = (tg_prompt, self._count(tg_prompt))
        sample_pp = self._pp_prompts[self.agents[0].idx][1]
        self.ui.set_phase(
            f"ready · pp ~{sample_pp} tok · tokenizer: {source}", 0
        )
        self.ui.log(
            f"== ready: pp ~{sample_pp} tok, tokenizer: {source} =="
        )

    # --------------------------------------------------------------- baseline
    def _warmup(self) -> None:
        """Throwaway requests to wake a cold server (CUDA graphs, autotune).

        Without this the baseline (and the first c-level) measure wake-up
        instead of steady-state latency.
        """
        cfg = self.config
        if cfg.warmup <= 0:
            return
        self.ui.set_phase(f"warming up ({cfg.warmup} requests)", 0)
        self.ui.log(f"== warming up server ({cfg.warmup} throwaway requests) ==")
        for _ in range(cfg.warmup):
            stream_chat(
                cfg.base_url,
                cfg.model,
                "Warmup request, answer with a single word.",
                1,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
                thinking=cfg.thinking,
                api_key=cfg.api_key,
            )

    def _baseline(self) -> None:
        """Measure fixed overhead: ttfr of a 1-token prompt (max_tokens=1).

        est_ppt = ttfr - baseline, i.e. the time actually spent processing
        the pp prompt rather than network/scheduling overhead. The minimum
        of the samples is used so one slow request can't skew it.
        """
        self.ui.set_phase("calibrating (1 tok baseline)", 0)
        self.ui.log("== calibrating baseline (1-token prompt) ==")
        times: list[float] = []
        for _ in range(3):
            result = stream_chat(
                self.config.base_url,
                self.config.model,
                "Hi",
                1,
                temperature=self.config.temperature,
                timeout=self.config.timeout,
                thinking=self.config.thinking,
                api_key=self.config.api_key,
            )
            if result.error is None and result.ttfr is not None:
                times.append(result.ttfr)
        self._baseline_ttfr = min(times) if times else 0.0
        if not times:
            self.warnings.append("baseline calibration failed; est_ppt will be raw ttfr")

    # ----------------------------------------------------------------- phases
    def _phase(self, test: str, c: int) -> None:
        cfg = self.config
        max_tokens = 1 if test == "pp" else cfg.tg
        prompts = self._pp_prompts if test == "pp" else self._tg_prompts
        label = f"{cfg.label(test)} (c{c})"
        self.ui.set_phase(label, c)
        self.ui.log(
            f"== {label}: {c} agent(s) x {cfg.n} reps = {c * cfg.n} requests =="
        )

        # (t since phase start, agent idx, n_tokens) — weighted so the pp
        # test can count its prompt tokens as "processed" at ttfr time.
        arrivals: list[tuple[float, int, int]] = []
        arrivals_lock = threading.Lock()
        phase_reqs: list[RequestStats] = []
        phase_reqs_lock = threading.Lock()
        t_phase0 = time.perf_counter()

        def execute(agent: Agent, spec: RequestSpec):
            t_start = time.perf_counter() - t_phase0

            def on_token(text: str, t_rel: float, kind: str = "content"):
                self.ui.on_token()
                agent.add_token(text, kind)
                if test != "pp":
                    with arrivals_lock:
                        # t_rel is request-relative; the rate windows below are
                        # phase-relative, so convert before storing.
                        arrivals.append((t_start + t_rel, agent.idx, 1))

            result = stream_chat(
                cfg.base_url,
                cfg.model,
                spec.prompt,
                spec.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
                on_token=on_token,
                thinking=cfg.thinking,
                api_key=cfg.api_key,
            )
            t_end = time.perf_counter() - t_phase0
            if result.error is None:
                self.ui.on_request_done(result.duration, result.completion_tokens)
            rs = RequestStats(
                agent=agent.idx,
                test=test,
                concurrency=c,
                prompt_tokens=result.prompt_tokens or spec.prompt_tokens,
                completion_tokens=result.completion_tokens,
                ttfr=result.ttfr,
                duration=result.duration,
                token_times=result.token_times,
                start_offset=t_start,
                end_offset=t_end,
                error=result.error,
            )
            with phase_reqs_lock:
                phase_reqs.append(rs)
            if self.ui.quiet:
                if rs.error:
                    self.ui.log(f"  [{label}] A{rs.agent} ERROR {rs.error}")
                else:
                    self.ui.log(
                        f"  [{label}] A{rs.agent} done "
                        f"{rs.completion_tokens} tok in {rs.duration * 1000:.0f} ms"
                    )
            return result

        # Per-agent queues so every agent always sees its own role prompt.
        # --n is the number of *repetitions* (llama-benchy style): each agent
        # runs n times, so the phase issues n*c requests in n full waves of
        # c concurrent agents.
        agents = self.agents[:c]
        queues = []
        for i, agent in enumerate(agents):
            prompt, p_tokens = prompts[agent.idx]
            q: "queue.Queue[RequestSpec]" = queue.Queue()
            for _ in range(cfg.n):
                q.put(RequestSpec(test, c, prompt, p_tokens, max_tokens))
            queues.append(q)
            agent.start(q, execute)
        for q in queues:
            q.join()
        phase_seconds = time.perf_counter() - t_phase0
        for agent in agents:
            agent.join()

        stats = self._summarize(test, c, phase_reqs, arrivals, phase_seconds)
        self.phases.append(PhaseResult(test, c, stats, phase_reqs))
        self.ui.log(
            f"== {label} done in {phase_seconds:.2f}s "
            f"({stats.n_ok}/{len(phase_reqs)} ok) =="
        )

    # ------------------------------------------------------------------ stats
    def _summarize(
        self,
        test: str,
        c: int,
        phase_reqs: list[RequestStats],
        arrivals: list[tuple[float, int, int]],
        phase_seconds: float,
    ) -> PhaseStats:
        ok = [r for r in phase_reqs if r.ok]
        n_failed = len(phase_reqs) - len(ok)

        # Aggregate stream as (time, n_tokens). For pp, the "tokens" the
        # server processes are the *prompt* tokens, all of them done by the
        # ttfr moment (the model only generates 1 token, which is noise).
        stream: list[tuple[float, int]] = [(t, n) for (t, _a, n) in arrivals]
        if test == "pp":
            for r in ok:
                if r.ttfr is not None:
                    stream.append((r.start_offset + r.ttfr, r.prompt_tokens))

        # Per-request "total" rate: aggregate (all agents) tokens during the
        # request's active window. For c=1 this equals the per-request rate.
        total_rates = [
            window_rate(stream, r.start_offset - _WINDOW_PAD, r.end_offset + _WINDOW_PAD)
            for r in ok
        ]
        # Per-request own rate.
        if test == "pp":
            req_rates = [r.prompt_tokens / r.ttfr for r in ok if r.ttfr]
            ttfr_vals = [r.ttfr for r in ok if r.ttfr is not None]
            est_vals = [max(0.0, v - self._baseline_ttfr) for v in ttfr_vals]
            e2e_vals = ttfr_vals
            peak_total_vals: list[float] = []
            peak_req_vals: list[float] = []
        else:
            req_rates = [r.completion_tokens / r.duration for r in ok if r.duration > 0]
            ttfr_vals = []
            est_vals = []
            e2e_vals = []
            peak_req_vals = [peak_rate(r.token_times) for r in ok]
            peak_total_vals = []
            for r in ok:
                t0 = r.start_offset - _WINDOW_PAD
                t1 = r.end_offset + _WINDOW_PAD
                window_times = [t for (t, _n) in stream if t0 <= t <= t1]
                peak_total_vals.append(peak_rate(window_times))

        total_tokens = (
            sum(r.prompt_tokens for r in ok)
            if test == "pp"
            else sum(r.completion_tokens for r in ok)
        )

        return PhaseStats(
            test=test,
            concurrency=c,
            n_ok=len(ok),
            n_failed=n_failed,
            total_tokens=total_tokens,
            phase_seconds=phase_seconds,
            total_tps=mean_std(total_rates),
            req_tps=mean_std(req_rates),
            peak_total=mean_std(peak_total_vals),
            peak_req=mean_std(peak_req_vals),
            ttfr=mean_std(ttfr_vals),
            est_ppt=mean_std(est_vals),
            e2e_ttft=mean_std(e2e_vals),
        )
