from chinchilla_llm_bench.prompt import build_prompt, estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_positive():
    assert estimate_tokens("hello world") == 3  # 2 words * 1.3 -> 3
    assert estimate_tokens("one") >= 1


def test_build_prompt_zero_target():
    assert build_prompt(0) == ""


def test_build_prompt_reaches_target_with_counter():
    def count(text: str) -> int:
        return len(text.split())

    prompt = build_prompt(50, count_tokens=count, seed=1)
    assert 45 <= count(prompt) <= 50


def test_build_prompt_never_overshoots_with_counter():
    def count(text: str) -> int:
        return len(text.split())

    for target in (1, 5, 20, 100):
        prompt = build_prompt(target, count_tokens=count, seed=7)
        assert count(prompt) <= target


def test_build_prompt_deterministic_per_seed():
    assert build_prompt(30, seed=42) == build_prompt(30, seed=42)
    assert build_prompt(30, seed=42) != build_prompt(30, seed=43)


def test_build_prompt_fallback_estimate():
    prompt = build_prompt(40)
    assert estimate_tokens(prompt) <= 52  # within ~30% of target


def test_build_prompt_with_opener():
    from chinchilla_llm_bench.prompt import ROLE_OPENERS

    def count(text: str) -> int:
        return len(text.split())

    opener = ROLE_OPENERS["coder"]
    prompt = build_prompt(50, count_tokens=count, seed=3, opener=opener)
    assert prompt.startswith(opener)
    assert count(prompt) <= 50
    # different roles produce different prompts
    other = build_prompt(50, count_tokens=count, seed=3, opener=ROLE_OPENERS["writer"])
    assert prompt != other


def test_build_tg_prompt_demands_full_length():
    from chinchilla_llm_bench.prompt import build_tg_prompt

    prompt = build_tg_prompt("coder", 1024, "fallback task")
    assert "at least 1024 words" in prompt
    assert "do not stop" in prompt
    assert prompt.startswith("Write a Python function")


def test_build_tg_prompt_uses_fallback_for_unknown_role():
    from chinchilla_llm_bench.prompt import build_tg_prompt

    prompt = build_tg_prompt("wizard", 128, "Tell me a story.")
    assert prompt.startswith("Tell me a story.")
    assert "at least 128 words" in prompt
