"""Benchmark settings (everything the CLI exposes maps 1:1 onto this)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchConfig:
    """All tunables for a benchmark run.

    pp  -> prefill test:  prompt of `pp` tokens, max_tokens=1
    tg  -> decode test:  short prompt, generate `tg` tokens
    c   -> concurrency levels to sweep (agents in flight at once)
    """

    base_url: str
    model: str
    pp: int = 200
    tg: int = 128
    concurrency: list[int] = field(default_factory=lambda: [1])
    n: int = 5  # requests per test
    temperature: float = 0.0
    timeout: float = 300.0
    seed: int = 1337
    loop: bool = False  # repeat the whole sweep until stopped (toggle with 'l')
    tg_prompt: str = "Write a short, vivid story about a lighthouse keeper."
    warmup: int = 2  # throwaway requests to wake a cold server before measuring
    thinking: bool = False  # Qwen3-style reasoning: off by default (pure content)
    api_key: str = "dummy"  # any non-empty string for vLLM; real key for gated APIs

    def __post_init__(self) -> None:
        if self.pp <= 0:
            raise ValueError("pp must be > 0")
        if self.tg <= 0:
            raise ValueError("tg must be > 0")
        if not self.concurrency:
            raise ValueError("at least one concurrency level is required (--c 1 2 3)")
        if any(c < 1 for c in self.concurrency):
            raise ValueError("concurrency levels must be >= 1")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if not self.base_url:
            raise ValueError("base_url is required")
        if not self.model:
            raise ValueError("model is required")

    def label(self, test: str) -> str:
        """Human label for a test type, e.g. 'pp200' or 'tg128'."""
        size = self.pp if test == "pp" else self.tg
        return f"{test}{size}"

    def summary(self) -> str:
        lines = [
            "chinchilla-bench settings",
            f"  base-url : {self.base_url}",
            f"  model    : {self.model}",
            f"  pp       : {self.pp} tokens (prefill test, max_tokens=1)",
            f"  tg       : {self.tg} tokens (decode test)",
            f"  c        : {' '.join(map(str, self.concurrency))}",
            f"  n        : {self.n} requests per test",
            f"  warmup   : {self.warmup} throwaway request(s)",
            f"  temp     : {self.temperature}",
            f"  thinking : {'on' if self.thinking else 'off (content only)'}",
            f"  timeout  : {self.timeout}s",
        ]
        return "\n".join(lines)
