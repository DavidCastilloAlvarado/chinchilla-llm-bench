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


# Role-specific openers so every agent works on a task that matches its role.
ROLE_OPENERS = {
    "coder": "Review and refactor the following Python service so it is clean, typed, and free of race conditions.",
    "researcher": "Summarize the current state of research on distributed inference and its open problems.",
    "analyst": "Analyze the latency metrics of a streaming inference service and explain the outliers.",
    "ops": "Write an operations runbook for scaling and draining a production inference node safely.",
    "writer": "Write a vivid scene set on a remote island where a light warns ships of the rocks.",
    "planner": "Draw up a phased delivery plan for a multi-node model deployment with clear milestones.",
    "tester": "Design a test strategy covering edge cases, load spikes, and failure injection.",
    "architect": "Design a system architecture for a fleet of inference workers behind a gateway.",
    "reviewer": "Review the proposed design and list the concrete risks and mitigations for each.",
    "data": "Explain how to model streaming token data and choose the right pipeline and storage.",
    "support": "Draft a careful support response that resolves a flaky-connection ticket step by step.",
    "devops": "Write the CI/CD pipeline configuration for shipping a model server with canary deploys.",
}

# Short role-specific instructions for the decode (tg) test.
ROLE_TG_PROMPTS = {
    "coder": "Write a Python function that parses a semver string and returns its parts.",
    "researcher": "Explain in three short paragraphs why KV-cache memory dominates at high batch size.",
    "analyst": "Interpret this p99 latency increase and state the two most likely causes.",
    "ops": "List the exact steps to safely drain one node from a load-balanced inference pool.",
    "writer": "Write a short, vivid story about a lighthouse keeper and a storm.",
    "planner": "Outline a four-phase rollout plan for moving a model to a new GPU cluster.",
    "tester": "Describe a load test that would expose a tokenizer deadlock.",
    "architect": "Sketch the components of a multi-region inference service and their contracts.",
    "reviewer": "Give a structured review of a cache invalidation design with three concrete fixes.",
    "data": "Describe a schema for storing per-request token timings and one useful query.",
    "support": "Write a clear, empathetic reply to a user reporting intermittent 502 errors.",
    "devops": "Describe the canary deployment steps for a model server with automatic rollback.",
}


def build_prompt(
    target_tokens: int,
    count_tokens: TokenCounter | None = None,
    seed: int = 1337,
    opener: str | None = None,
) -> str:
    """Build a prompt of approximately ``target_tokens`` tokens.

    ``opener`` (e.g. a role-specific task sentence) is placed first and the
    rest is padded with filler sentences. When ``count_tokens`` is provided
    (e.g. vLLM's ``/tokenize`` endpoint) the result is trimmed so the real
    counter reports <= target_tokens. Otherwise a word-count heuristic is
    used. Deterministic per seed.
    """
    if target_tokens <= 0:
        return ""
    count = count_tokens or estimate_tokens
    rng = random.Random(seed)
    parts: list[str] = []
    if opener:
        parts.append(opener)
    text = " ".join(parts)
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
