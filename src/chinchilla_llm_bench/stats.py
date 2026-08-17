"""Metric aggregation for the final report."""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0.0 std for empty/single samples)."""
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def fmt_ms_mean_std(mean_s: float, std_s: float) -> str:
    return f"{mean_s * 1000:.2f} ± {std_s * 1000:.2f}"


def peak_rate(times: Sequence[float], window: float = 0.5) -> float:
    """Peak tokens/sec over a sliding ``window`` across sorted arrival times."""
    if len(times) < 2:
        return 0.0
    best = 0.0
    j = 0
    for i in range(1, len(times)):
        while times[i] - times[j] > window:
            j += 1
        if j >= i:
            continue
        dt = times[i] - times[j]
        if dt > 1e-9:
            best = max(best, (i - j) / dt)
    return best


def window_rate(arrivals: Sequence[tuple[float, int]], t0: float, t1: float) -> float:
    """Aggregate tokens/sec over ``[t0, t1]`` from a shared arrival stream."""
    if t1 - t0 <= 1e-9:
        return 0.0
    tokens = sum(1 for (t, _agent) in arrivals if t0 <= t <= t1)
    return tokens / (t1 - t0)


class PeakTracker:
    """Sliding-window peak over a shared token stream (thread-safe).

    Used by the top bar to show a live tokens/sec figure.
    """

    def __init__(self, window: float = 1.0):
        self.window = window
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, int]] = deque()
        self.peak = 0.0

    def add(self, t: float, tokens: int = 1) -> None:
        with self._lock:
            cum = (self._samples[-1][1] if self._samples else 0) + tokens
            self._samples.append((t, cum))
            while self._samples and self._samples[0][0] < t - self.window:
                self._samples.popleft()
            if len(self._samples) >= 2:
                t0, c0 = self._samples[0]
                t1, c1 = self._samples[-1]
                dt = t1 - t0
                if dt > 1e-9:
                    self.peak = max(self.peak, (c1 - c0) / dt)

    def rate(self, t: float, window: float | None = None) -> float:
        """Current rate over the trailing ``window`` (default: self.window)."""
        w = self.window if window is None else window
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            t1, c1 = self._samples[-1]
            t0 = c0 = None
            for ts, cs in self._samples:
                if ts >= t1 - w:
                    t0, c0 = ts, cs
                    break
            if t0 is None:
                return 0.0
            dt = t1 - t0
            return (c1 - c0) / dt if dt > 1e-9 else 0.0


@dataclass
class RequestStats:
    """Outcome of a single benchmark request."""

    agent: int
    test: str  # "pp" | "tg"
    concurrency: int
    prompt_tokens: int
    completion_tokens: int
    ttfr: float | None
    duration: float
    token_times: list[float] = field(default_factory=list)
    start_offset: float = 0.0  # seconds since phase start
    end_offset: float = 0.0  # seconds since phase start
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class PhaseStats:
    """Aggregated metrics for one (test, concurrency) cell of the report."""

    test: str
    concurrency: int
    n_ok: int
    n_failed: int
    total_tokens: int
    phase_seconds: float
    total_tps: tuple[float, float]  # (mean, std) aggregate t/s
    req_tps: tuple[float, float]  # (mean, std) per-request t/s
    peak_total: tuple[float, float]
    peak_req: tuple[float, float]
    ttfr: tuple[float, float]  # seconds
    est_ppt: tuple[float, float]  # seconds
    e2e_ttft: tuple[float, float]  # seconds
