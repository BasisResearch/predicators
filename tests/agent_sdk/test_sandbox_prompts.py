"""Tests for sandbox prompt construction and prose unwrapping."""
from predicators.agent_sdk.sandbox_prompts import \
    build_sandbox_system_prompt, unwrap_prose_lines


def test_unwrap_prose_lines_joins_wrapped_paragraphs() -> None:
    """Hard-wrapped prose joins into one line per paragraph; paragraph breaks
    survive."""
    text = ("First paragraph wrapped\n"
            "across two lines.\n"
            "\n"
            "Second paragraph.")
    assert unwrap_prose_lines(text) == (
        "First paragraph wrapped across two lines.\n\nSecond paragraph.")


def test_unwrap_prose_lines_preserves_structure() -> None:
    """Fenced code, headings, list items, tables, and indented lines pass
    through verbatim; bullet continuations join into the bullet."""
    text = ("## Heading\n"
            "- a bullet wrapped\n"
            "onto a second line\n"
            "- another bullet\n"
            "1. numbered item\n"
            "\n"
            "```python\n"
            "code_line_one\n"
            "code_line_two\n"
            "```\n"
            "\n"
            "  indented verbatim\n"
            "plain prose after it\n"
            "| a | table |")
    out = unwrap_prose_lines(text)
    assert "## Heading" in out.split("\n")
    assert "- a bullet wrapped onto a second line" in out.split("\n")
    assert "1. numbered item" in out.split("\n")
    assert "code_line_one\ncode_line_two" in out
    assert "  indented verbatim" in out.split("\n")
    assert "plain prose after it" in out.split("\n")
    assert "| a | table |" in out.split("\n")


def test_sandbox_system_prompt_has_no_wrapped_prose() -> None:
    """The sandbox suffix renders one line per paragraph (no manual line breaks
    mid-sentence)."""
    prompt = build_sandbox_system_prompt()
    for line in prompt.split("\n"):
        if not line or line.startswith("#"):
            continue
        # Every prose line is a full paragraph: it ends with terminal
        # punctuation rather than being wrapped mid-sentence.
        assert line.rstrip().endswith("."), line
