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
    pp_output_tokens: int = 1
    tg: int = 128
    concurrency: list[int] = field(default_factory=lambda: [1])
    n: int = 5  # repetitions per test: each agent runs n times (total = n x c)
    temperature: float | None = None
    timeout: float = 300.0
    seed: int = 1337
    loop: bool = False  # repeat the whole sweep until stopped (toggle with 'l')
    tg_prompt: str = "Write a short, vivid story about a lighthouse keeper."
    warmup: int = 2  # throwaway requests to wake a cold server before measuring
    thinking: bool = False  # Qwen3-style reasoning: off by default (pure content)
    exact_tg: bool = True  # force the full tg budget (min_tokens + ignore_eos)
    vllm_extensions: bool = False  # send vLLM-only request fields
    max_completion_tokens: bool = False  # use modern OpenAI token-limit field
    reasoning_effort: str | None = None
    api_key: str = "dummy"  # any non-empty string for vLLM; real key for gated APIs
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pp <= 0:
            raise ValueError("pp must be > 0")
        if self.pp_output_tokens <= 0:
            raise ValueError("pp_output_tokens must be > 0")
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
        if self.thinking and not self.vllm_extensions:
            raise ValueError("--thinking requires --vllm-extensions")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("reasoning_effort must not be empty")

    def label(self, test: str) -> str:
        """Human label for a test type, e.g. 'pp200' or 'tg128'."""
        size = self.pp if test == "pp" else self.tg
        return f"{test}{size}"

    def summary(self) -> str:
        lines = [
            "chinchilla-bench settings",
            f"  base-url : {self.base_url}",
            f"  model    : {self.model}",
            f"  pp       : {self.pp} tokens (prefill test)",
            f"  pp output: {self.pp_output_tokens} completion token(s)",
            f"  tg       : {self.tg} tokens (decode test)",
            f"  c        : {' '.join(map(str, self.concurrency))}",
            f"  n        : {self.n} repetitions per test (each agent runs n times)",
            f"  warmup   : {self.warmup} throwaway request(s)",
            f"  temp     : {self.temperature if self.temperature is not None else 'provider default'}",
            f"  thinking : {'on' if self.thinking else 'off (content only)'}",
            f"  vllm extensions : {'on' if self.vllm_extensions else 'off'}",
            f"  exact_tg : {'on' if self.exact_tg and self.vllm_extensions else 'off'}",
            "  token limit : "
            f"{'max_completion_tokens' if self.max_completion_tokens else 'max_tokens'}",
            f"  reasoning effort : {self.reasoning_effort or 'provider default'}",
            f"  timeout  : {self.timeout}s",
        ]
        return "\n".join(lines)
