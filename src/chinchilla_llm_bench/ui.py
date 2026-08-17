"""Live 'agent swarm' view: a rich Live grid of per-agent cards.

Top bar:  CHINCHILLA · LLM SWARM  [1] [2] [3] [4]  ■ STOP  ✓ loop
          AGENTS LIVE · TOKENS/SEC · TOKENS TOTAL · ELAPSED
Grid:     one card per agent — status border, id + role, live prompt/output,
          token counter. Keys: q/s = stop, l = toggle loop.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from typing import Callable, Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent, AgentSnapshot, STATUS_STYLES
from .config import BenchConfig
from .stats import PeakTracker, RateTracker

_CARD_INNER_W = 22
_CARD_HEIGHT = 7  # border lines + 5 inner lines


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    """Word-wrap ``text`` into at most ``max_lines`` lines of <= width chars."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ""
        if len(lines) >= max_lines:
            break
        line = word[:width]
    if line and len(lines) < max_lines:
        lines.append(line)
    return lines[:max_lines]


class InputWatcher:
    """Reads single keys from a POSIX tty in cbreak mode (best effort)."""

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._thread: Optional[threading.Thread] = None
        self._old = None
        self._fd: Optional[int] = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            import termios
            import tty
        except ImportError:
            return
        self._fd = sys.stdin.fileno()
        try:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except termios.error:
            self._old = None
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            try:
                ch = sys.stdin.read(1)
            except Exception:
                return
            if not ch:
                return
            self._on_key(ch)

    def stop(self) -> None:
        if self._old is not None and self._fd is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None
        self._thread = None


class SwarmUI:
    """Renders the swarm. With ``quiet=True`` it degrades to plain log lines."""

    def __init__(
        self,
        config: BenchConfig,
        agents: list[Agent] | None = None,
        console: Console | None = None,
        quiet: bool = False,
    ):
        self.config = config
        self.agents: list[Agent] = agents or []
        self.console = console or Console()
        self.quiet = quiet
        self.phase = "connecting"
        self.active_c = 0
        self.total_tokens = 0
        self.requests_done = 0
        self._total_lock = threading.Lock()
        self.tracker = PeakTracker(window=1.0)
        self.req_tracker = RateTracker(window=1.0)
        self._req_rates: "deque[float]" = deque(maxlen=64)
        self._dur_lock = threading.Lock()
        self._start = time.time()
        self._live: Optional[Live] = None
        self._watcher: Optional[InputWatcher] = None
        self.loop = config.loop
        self._stop_flag = False

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if self.quiet:
            return
        self._live = Live(
            console=self.console,
            screen=True,
            refresh_per_second=12,
            vertical_overflow="hidden",
            get_renderable=self._frame,
        )
        self._live.start(refresh=True)
        self._watcher = InputWatcher(self._on_key)
        self._watcher.start()

    def stop(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    # -- control -----------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop_flag = True

    @property
    def stopped(self) -> bool:
        return self._stop_flag

    def _on_key(self, key: str) -> None:
        if key in ("q", "s"):
            self.request_stop()
        elif key == "l":
            self.loop = not self.loop

    # -- feed (called by agents/runner) -------------------------------------------
    def on_token(self) -> None:
        with self._total_lock:
            self.total_tokens += 1
        self.tracker.add(time.time())

    def on_request_done(self, duration: float, tokens: int) -> None:
        """Record a completed (ok) request for req/s and per-request tok/s stats."""
        with self._total_lock:
            self.requests_done += 1
        self.req_tracker.add(time.time())
        with self._dur_lock:
            if duration > 1e-9:
                self._req_rates.append(tokens / duration)

    @property
    def tokens_per_sec_req(self) -> float:
        """Per-request tokens/sec: mean over recent completed requests."""
        with self._dur_lock:
            if not self._req_rates:
                return 0.0
            return sum(self._req_rates) / len(self._req_rates)

    def set_phase(self, label: str, c: int) -> None:
        self.phase = label
        self.active_c = c

    def log(self, line: str) -> None:
        """Plain progress line, only used in --no-ui mode."""
        if self.quiet:
            print(line, flush=True)

    # -- rendering ------------------------------------------------------------------
    def _frame(self) -> Group:
        return Group(self._topbar(), self._grid())

    def _topbar(self) -> Table:
        t = Table.grid(padding=(0, 0))
        t.add_column(ratio=1, overflow="fold")
        t.add_column(justify="right", overflow="fold")
        t.add_row(self._topbar_left(), self._topbar_right())
        return t

    def _topbar_left(self) -> Text:
        left = Text.assemble(
            (f"{self.phase}", "bold white"),
            ("   ", ""),
        )
        for c in self.config.concurrency:
            active = c == self.active_c
            style = "bold black on green" if active else "dim"
            left.append_text(Text(f" {c} ", style=style))
            left.append(" ", "")
        left.append("■ STOP (q)  ", "bold red" if self.stopped else "dim")
        left.append("✓ loop (l)  ", "bold green" if self.loop else "dim")
        return left

    def _topbar_right(self) -> Text:
        now = time.time()
        elapsed = now - self._start
        tps_total = self.tracker.rate(now)
        tps_req = self.tokens_per_sec_req
        rps = self.req_tracker.rate(now)
        live = sum(1 for a in self.agents if a.snapshot().status == "working")
        if self.console.width < 160:
            return Text.assemble(
                (f"AGENTS {live}", "bold white"),
                (f"  TOK/S {tps_total:,.0f}", "bold yellow"),
                (f"  TOK/S·REQ {tps_req:,.0f}", "bold yellow"),
                (f"  REQ/S {rps:.2f}", "bold green"),
                (f"  TOK GEN {self.total_tokens:,}", "bold cyan"),
                (f"  REQ {self.requests_done}", "bold blue"),
                (f"  {elapsed:.0f}s", "bold magenta"),
            )
        return Text.assemble(
            (f"AGENTS LIVE {live}", "bold white"),
            (f"   TOKENS/SEC (TOTAL) {tps_total:,.0f}", "bold yellow"),
            (f"   TOKENS/SEC (REQ) {tps_req:,.0f}", "bold yellow"),
            (f"   REQ/S (TOTAL) {rps:.2f}", "bold green"),
            (f"   TOKENS GEN {self.total_tokens:,}", "bold cyan"),
            (f"   REQ DONE {self.requests_done}", "bold blue"),
            (f"   ELAPSED {elapsed:.0f}s", "bold magenta"),
        )

    def _grid(self) -> Table:
        """Lay out the agent cards to fill the whole terminal.

        Card width is fixed; the number of columns fits the terminal width
        and the card height grows to fill the remaining vertical space, so
        each card shows as much of the streamed text as possible.
        """
        width = self.console.width
        height = self.console.height
        card_w = 28  # inner 22 + 2 border + 2 padding
        gap = 2
        cols = max(1, min(8, (width - gap) // (card_w + gap)))
        n = max(1, len(self.agents))
        rows = max(1, math.ceil(n / cols))
        card_h = max(7, (height - 4) // rows)  # top bar + margins
        grid = Table.grid(padding=(0, 1))
        for _ in range(cols):
            grid.add_column(width=card_w, overflow="crop")
        cards = [self._card(a, card_h) for a in self.agents]
        for i in range(0, len(cards), cols):
            row = cards[i : i + cols]
            while len(row) < cols:
                row.append(Panel("", box=box.ROUNDED, border_style="dim", height=card_h))
            grid.add_row(*row)
        return grid

    def _card(self, agent: Agent, card_h: int) -> Panel:
        s = agent.snapshot()
        style = STATUS_STYLES.get(s.status, "dim")
        title = f"A{s.idx}  {s.role}"
        subtitle = f"{s.status} · {s.tokens} tok"
        return Panel(
            self._body(s, card_h),
            title=title,
            subtitle=subtitle,
            border_style=style,
            box=box.ROUNDED,
            height=card_h,
        )

    def _body(self, s: AgentSnapshot, card_h: int) -> Text:
        """Prompt (top), then the streamed text filling the card, tail-first.

        The most recent tokens are always visible; the buffer keeps the full
        generated text so the card scrolls as the model streams.
        """
        t = Text()
        inner = max(3, card_h - 2)
        prompt_lines = _wrap(s.prompt, _CARD_INNER_W, 2)
        for line in prompt_lines:
            t.append(line, style="grey62")
            t.append("\n")
        room = inner - len(prompt_lines)
        if room <= 0:
            return t
        if s.output:
            out_n = room if not s.think else max(1, room - 1)
            if s.think and out_n < room:
                last_think = s.think.splitlines()[-1]
                t.append(last_think[:_CARD_INNER_W] or "…", style="italic grey42")
                t.append("\n")
            for line in s.output.splitlines()[-out_n:]:
                t.append(line[:_CARD_INNER_W], style="white")
                t.append("\n")
        elif s.think:
            for line in s.think.splitlines()[-room:]:
                t.append(line[:_CARD_INNER_W], style="italic grey42")
                t.append("\n")
        elif s.detail:
            t.append(s.detail[:_CARD_INNER_W], style="cyan")
        return t
