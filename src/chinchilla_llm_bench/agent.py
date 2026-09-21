"""A single benchmark agent: one card in the swarm, one worker thread."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .terminal import safe_terminal_text

ROLES = [
    "coder",
    "researcher",
    "analyst",
    "ops",
    "writer",
    "planner",
    "tester",
    "architect",
    "reviewer",
    "data",
    "support",
    "devops",
]

STATUS_STYLES = {
    "idle": "dim",
    "queued": "yellow",
    "working": "green",
    "done": "cyan",
    "error": "red",
}


@dataclass
class AgentSnapshot:
    """Immutable copy of an agent's state, safe for the UI to render."""

    idx: int
    role: str
    status: str
    prompt: str
    output: str
    think: str
    tokens: int
    prompt_tokens: int
    detail: str


class Agent:
    """One concurrent worker. Pulls request specs from a queue and executes them.

    The agent owns its visible state (status, streamed text, token count);
    the UI renders ``snapshot()`` every frame.
    """

    def __init__(self, idx: int, role: str):
        self.idx = idx
        self.role = role
        self._lock = threading.Lock()
        self._status = "idle"
        self._prompt = ""
        self._output = ""
        self._think = ""
        self._tokens = 0
        self._prompt_tokens = 0
        self._detail = ""
        self._thread: Optional[threading.Thread] = None

    # -- state (thread-safe) --------------------------------------------------
    def snapshot(self) -> AgentSnapshot:
        with self._lock:
            return AgentSnapshot(
                self.idx,
                self.role,
                self._status,
                self._prompt,
                self._output,
                self._think,
                self._tokens,
                self._prompt_tokens,
                self._detail,
            )

    def _set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, "_" + key, value)

    def set_prompt(self, prompt: str, prompt_tokens: int) -> None:
        self._set(
            prompt=safe_terminal_text(prompt),
            prompt_tokens=prompt_tokens,
            output="",
            think="",
            tokens=0,
            detail="",
            status="working",
        )

    def add_token(self, text: str, kind: str = "content", n_tokens: int = 1) -> None:
        text = safe_terminal_text(text)
        with self._lock:
            self._tokens += n_tokens
            if kind == "think":
                self._think += text
                if len(self._think) > 4000:
                    self._think = self._think[-3000:]
            else:
                self._output += text
                if len(self._output) > 12000:
                    self._output = self._output[-9000:]

    def finish(self, ok: bool, detail: str) -> None:
        self._set(
            status="done" if ok else "error",
            detail=safe_terminal_text(detail),
        )

    # -- worker -----------------------------------------------------------------
    def start(self, q: "queue.Queue", execute: Callable[["Agent", object], object]) -> None:
        self._thread = threading.Thread(
            target=self._loop, args=(q, execute), daemon=True
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self, q: "queue.Queue", execute: Callable[["Agent", object], object]) -> None:
        while True:
            try:
                spec = q.get_nowait()
            except queue.Empty:
                break
            self.set_prompt(spec.prompt, spec.prompt_tokens)
            try:
                result = execute(self, spec)
            except Exception as exc:  # defensive: never kill the swarm
                result = _ErrorResult(
                    safe_terminal_text(f"{type(exc).__name__}: {exc}"),
                    spec.prompt_tokens,
                )
            ok = result.error is None
            if ok:
                detail = f"{result.completion_tokens} tok · {result.duration * 1000:.0f} ms"
            else:
                detail = str(result.error)[:60]
            self.finish(ok, detail)
            q.task_done()
        self._set(status="idle", detail="")


class _ErrorResult:
    """Duck-typed stand-in for ChatResult when execute() blew up."""

    def __init__(self, error: str, prompt_tokens: int):
        self.error = error
        self.completion_tokens = 0
        self.duration = 0.0
        self.ttfr = None
        self.prompt_tokens = prompt_tokens
        self.text = ""
