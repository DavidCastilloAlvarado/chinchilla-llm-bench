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
