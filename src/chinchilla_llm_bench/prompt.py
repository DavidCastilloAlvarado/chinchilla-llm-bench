"""Prompt construction for the pp (prefill) and tg (decode) phases."""
from __future__ import annotations

import random
from typing import Callable

TokenCounter = Callable[[str], int]

# Neutral, model-agnostic corpus used to pad prompts to the requested size.
_SENTENCES = [
    "The lighthouse keeper climbed the spiral stairs before dawn.",
    "Salt wind pressed against the glass of the watch room.",
    "A cargo ship sounded its horn twice in the fog.",
    "The beam swept the water in slow, patient circles.",
    "Oil lamps hissed softly on the shelf below.",
    "He logged the weather in a book with a cracked spine.",
    "The gulls argued over a piece of rope on the gallery.",
    "Somewhere below, the tide pulled the boats against their moorings.",
    "The radio crackled with a shipping forecast in a flat voice.",
    "Coffee went cold while he watched the horizon for a while.",
    "A storm was building to the northwest, slow but certain.",
    "The old keeper had kept this light for thirty-one years.",
    "Every lens panel was polished until it hummed in the dark.",
    "The fog horn woke the island once an hour, on the dot.",
    "Seagulls nested in the ironwork and left their mess everywhere.",
    "The stairs were worn smooth in the middle by decades of boots.",
    "At night the whole island was just the light and the sea.",
    "A supply boat would come on the first Tuesday of the month.",
    "He kept a log of every ship that passed within sight.",
    "The lamp room was warm enough that his breath fogged the glass.",
    "Below, the keeper's cottage sat low and stubborn against the rocks.",
    "The barometer had been falling since midday without complaint.",
    "Rope, tar, and salt were the only currencies that mattered here.",
    "He wound the clockwork mechanism twice a day, no more, no less.",
    "The sea turned the color of pewter when the light was on.",
    "A small boat had been missing since the last fog bank.",
    "The keeper's hands were mapped with scars from rope and wire.",
    "Weather reports were read aloud to no one in particular.",
    "The beam reached further than the charts ever admitted.",
    "In the logbook he wrote: all quiet, sea rough, wind north.",
]


def estimate_tokens(text: str) -> int:
    """Rough offline estimate: ~1.3 tokens per whitespace-separated word."""
    if not text:
        return 0
    return max(1, int(round(len(text.split()) * 1.3)))


def build_prompt(
    target_tokens: int,
    count_tokens: TokenCounter | None = None,
    seed: int = 1337,
) -> str:
    """Build a prompt of approximately ``target_tokens`` tokens.

    When ``count_tokens`` is provided (e.g. vLLM's ``/tokenize`` endpoint)
    the result is trimmed so the real counter reports <= target_tokens.
    Otherwise a word-count heuristic is used. Deterministic per seed.
    """
    if target_tokens <= 0:
        return ""
    count = count_tokens or estimate_tokens
    rng = random.Random(seed)
    parts: list[str] = []
    text = ""
    while count(text) < target_tokens:
        parts.append(_SENTENCES[rng.randrange(len(_SENTENCES))])
        text = " ".join(parts)
    if count(text) <= target_tokens:
        return text
    # Overshot: trim proportionally, then top up word by word.
    words = text.split()
    keep = max(1, int(len(words) * target_tokens / count(text)))
    trimmed = " ".join(words[:keep])
    for word in words[keep:]:
        candidate = f"{trimmed} {word}"
        if count(candidate) <= target_tokens:
            trimmed = candidate
        else:
            break
    return trimmed
