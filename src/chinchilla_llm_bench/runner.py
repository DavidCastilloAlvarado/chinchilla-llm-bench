"""Orchestrates the benchmark: preflight, warmup, latency probe, phases."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from .agent import Agent, ROLES
from .client import list_models, measure_latency, resolve_tokenizer, stream_chat
from .config import BenchConfig
from .prompt import ROLE_OPENERS, build_prompt, build_tg_prompt
from .stats import (
    PhaseStats,
    RequestStats,
    mean_std,
    peak_rate,
)
from .ui import SwarmUI


@dataclass
class RequestSpec:
    test: str
    concurrency: int
    rep: int  # repetition (wave) index, 0-based
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
        self._latency = 0.0  # measured network RTT (seconds)

    # ------------------------------------------------------------------ run
    def run(self) -> list[PhaseResult]:
        self.ui.start()
        try:
            self._preflight()
            self._warmup()
            self._latency_probe()
            while True:
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
                self.config.base_url,
                api_key=self.config.api_key,
                timeout=10,
                headers=self.config.headers,
            )
        except Exception as exc:
            raise SystemExit(f"cannot reach {self.config.base_url}: {exc}") from exc
        if self.config.model not in models:
            self.warnings.append(
                f"model {self.config.model!r} not listed in /models ({', '.join(models)})"
            )
        self._count, source = resolve_tokenizer(
            self.config.base_url, api_key=self.config.api_key, headers=self.config.headers
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
                cfg.pp_output_tokens,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
                thinking=cfg.thinking,
                api_key=cfg.api_key,
                headers=cfg.headers,
                vllm_extensions=cfg.vllm_extensions,
                max_completion_tokens=cfg.max_completion_tokens,
                reasoning_effort=cfg.reasoning_effort,
            )

    # --------------------------------------------------------------- latency
    def _latency_probe(self) -> None:
        """Measure network RTT: mean of 3 ``GET /models`` (llama-benchy's
        'api' latency mode). est_ppt = ttfr - latency removes the round-trip
        from the prompt-processing time.
        """
        self.ui.set_phase("measuring latency", 0)
        self.ui.log("== measuring network latency (3x GET /models) ==")
        self._latency = measure_latency(
            self.config.base_url, api_key=self.config.api_key, headers=self.config.headers
        )
        self.ui.log(f"== latency: {self._latency * 1000:.2f} ms ==")

    # ----------------------------------------------------------------- phases
    def _phase(self, test: str, c: int) -> None:
        cfg = self.config
        max_tokens = cfg.pp_output_tokens if test == "pp" else cfg.tg
        prompts = self._pp_prompts if test == "pp" else self._tg_prompts
        label = f"{cfg.label(test)} (c{c})"
        self.ui.set_phase(label, c)
        self.ui.log(
            f"== {label}: {c} agent(s) x {cfg.n} reps = {c * cfg.n} requests =="
        )

        phase_reqs: list[RequestStats] = []
        phase_reqs_lock = threading.Lock()
        t_phase0 = time.perf_counter()

        def execute(agent: Agent, spec: RequestSpec):
            t_start = time.perf_counter() - t_phase0
            gen_started = False

            def on_token(
                text: str, t_rel: float, kind: str = "content", n_tokens: int = 1
            ):
                nonlocal gen_started
                # First streamed token (TTFT): the request is generating now.
                # Counted only for tg, so MAXC is "really producing tokens
                # at the same time", excluding prompt processing.
                if test == "tg" and not gen_started:
                    gen_started = True
                    self.ui.on_gen_start()
                self.ui.on_token(n_tokens)
                agent.add_token(text, kind, n_tokens)

            # llama-benchy's exact_tg: force the full budget server-side.
            min_tokens = cfg.tg if (test == "tg" and cfg.exact_tg) else None

            try:
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
                    min_tokens=min_tokens,
                    headers=cfg.headers,
                    vllm_extensions=cfg.vllm_extensions,
                    max_completion_tokens=cfg.max_completion_tokens,
                    reasoning_effort=cfg.reasoning_effort,
                )
            finally:
                if test == "tg" and gen_started:
                    self.ui.on_gen_end()
            t_end = time.perf_counter() - t_phase0
            if result.error is None:
                self.ui.on_request_done(result.duration, result.completion_tokens)
            rs = RequestStats(
                agent=agent.idx,
                test=test,
                concurrency=c,
                rep=spec.rep,
                prompt_tokens=result.prompt_tokens or spec.prompt_tokens,
                completion_tokens=result.completion_tokens,
                ttfr=result.ttfr,
                ttft=result.ttft,
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
            for rep in range(cfg.n):
                q.put(RequestSpec(test, c, rep, prompt, p_tokens, max_tokens))
            queues.append(q)
            agent.start(q, execute)
        for q in queues:
            q.join()
        phase_seconds = time.perf_counter() - t_phase0
        for agent in agents:
            agent.join()

        stats = self._summarize(test, c, phase_reqs, phase_seconds)
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
        phase_seconds: float,
    ) -> PhaseStats:
        """Aggregate per-request + per-wave stats (llama-benchy semantics).

        * per request: pp = prompt / est_ppt, tg = (N-1) / (last - first token)
        * per wave (one repetition of c concurrent requests):
          pp = sum(prompt) / (max first token - min start),
          tg = sum(N-1) / (max last token - min first token),
          peak = max tokens in a 1 s sliding window over merged token times
        """
        ok = [r for r in phase_reqs if r.ok]
        n_failed = len(phase_reqs) - len(ok)
        latency = self._latency

        # -- per-request metrics -------------------------------------------
        req_rates: list[float] = []
        ttfr_vals: list[float] = []
        est_vals: list[float] = []
        e2e_vals: list[float] = []
        for r in ok:
            if test == "pp":
                if r.ttfr is not None:
                    ttfr_vals.append(r.ttfr)
                    est = max(0.0, r.ttfr - latency)
                    est_vals.append(est)
                    if est > 0:
                        req_rates.append(r.prompt_tokens / est)
                if r.ttft is not None:
                    e2e_vals.append(r.ttft)
            else:
                n = len(r.token_times)
                if n > 1 and r.token_times[-1] > r.token_times[0]:
                    req_rates.append((n - 1) / (r.token_times[-1] - r.token_times[0]))

        # -- per-wave (batch) metrics: wave = the c requests of one rep ----
        waves: dict[int, list[RequestStats]] = {}
        for r in ok:
            waves.setdefault(r.rep, []).append(r)

        wave_totals: list[float] = []
        peak_total_vals: list[float] = []
        for rep in sorted(waves):
            batch = waves[rep]
            firsts = [
                r.first_token_offset for r in batch if r.first_token_offset is not None
            ]
            if not firsts:
                continue
            if test == "pp":
                span = max(firsts) - min(r.start_offset for r in batch)
                if span > 0:
                    wave_totals.append(sum(r.prompt_tokens for r in batch) / span)
            else:
                lasts = [r.last_token for r in batch if r.last_token is not None]
                if not lasts:
                    continue
                span = max(lasts) - min(firsts)
                if span > 0:
                    decode_tokens = sum(
                        len(r.token_times) - 1 for r in batch if r.token_times
                    )
                    if decode_tokens > 0:
                        wave_totals.append(decode_tokens / span)
            merged = [r.start_offset + t for r in batch for t in r.token_times]
            if merged:
                peak_total_vals.append(peak_rate(merged))
        peak_req_vals = [
            peak_rate(r.token_times) for r in ok if len(r.token_times) > 1
        ]

        if c > 1:
            total_tps = mean_std(wave_totals)
            req_tps = mean_std(req_rates)
        else:
            # single request: the aggregate is the request itself
            req_tps = mean_std(req_rates)
            total_tps = req_tps

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
            total_tps=total_tps,
            req_tps=req_tps,
            peak_total=mean_std(peak_total_vals),
            peak_req=mean_std(peak_req_vals),
            ttfr=mean_std(ttfr_vals),
            est_ppt=mean_std(est_vals),
            e2e_ttft=mean_std(e2e_vals),
        )
