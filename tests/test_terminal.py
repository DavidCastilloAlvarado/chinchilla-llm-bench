from chinchilla_llm_bench.terminal import safe_terminal_text


def test_safe_terminal_text_removes_control_and_formatting_characters():
    assert safe_terminal_text("before\x1b[2J\nafter\u202e") == "before [2J after "
