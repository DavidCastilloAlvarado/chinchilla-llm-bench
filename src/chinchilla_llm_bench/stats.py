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


class RateTracker:
    """Trailing-window count rate over timestamps (thread-safe)."""

    def __init__(self, window: float = 1.0):
        self.window = window
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def add(self, t: float) -> None:
        with self._lock:
            self._times.append(t)
            while self._times and self._times[0] < t - self.window:
                self._times.popleft()

    def rate(self, t: float) -> float:
        """Events per second over the trailing window."""
        with self._lock:
            while self._times and self._times[0] < t - self.window:
                self._times.popleft()
            if not self._times:
                return 0.0
            return len(self._times) / self.window


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
    req_tps: tuple[float, float]  # (mean, std) per-request t/s
    ttfr: tuple[float, float]  # seconds
    est_ppt: tuple[float, float]  # seconds
    e2e_ttft: tuple[float, float]  # seconds
