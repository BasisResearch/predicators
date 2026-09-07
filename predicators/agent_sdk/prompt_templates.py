"""Loader for the agent prompt templates in ``predicators/agent_sdk/prompts``.

Every prompt the agent receives (per-phase system prompts, the solve
and explore query, the synthesis learn message, the sandbox CLAUDE.md)
is authored as Markdown in that directory and composed here, so the
text can be read, diffed, and reproduced as a document rather than
reassembled from string concatenation.

Template format:

- A file is split into named sections by marker lines of the form
  ``<!-- section: name -->``.  Text before the first marker is ignored
  (it is the file's own header comment).
- ``__UPPER_SNAKE__`` placeholders are substituted at render time.
  Rendering fails loudly when a template placeholder has no value, so
  a renamed placeholder cannot silently ship as literal text.
- Prose is authored hard-wrapped; :func:`render` joins wrapped prose
  lines into one line per paragraph while preserving headings, lists,
  tables, code fences, and indented lines.
"""
import functools
import os
import re
from typing import Dict, List

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_SECTION_RE = re.compile(r"^<!-- section: ([a-z0-9_]+) -->[ \t]*$",
                         re.MULTILINE)
_PLACEHOLDER_RE = re.compile(r"__([A-Z][A-Z0-9_]*)__")
# Matches "1. ", "12) " etc. at the start of a stripped line.
_NUMBERED_ITEM_RE = re.compile(r"^\d+[.)]\s")


@functools.lru_cache(maxsize=None)
def load_sections(template: str) -> Dict[str, str]:
    """Return ``{section_name: raw_text}`` for ``prompts/<template>.md``.

    Section text keeps its internal formatting; leading and trailing
    blank lines are stripped.
    """
    path = os.path.join(_PROMPTS_DIR, f"{template}.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    matches = list(_SECTION_RE.finditer(text))
    assert matches, f"{path} declares no <!-- section: name --> markers"
    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group(1)
        assert name not in sections, f"{path}: duplicate section {name!r}"
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip("\n")
    return sections


def placeholders(text: str) -> List[str]:
    """The distinct ``__NAME__`` placeholders in ``text``, in order."""
    seen: List[str] = []
    for name in _PLACEHOLDER_RE.findall(text):
        if name not in seen:
            seen.append(name)
    return seen


def render(template: str, section: str, **values: str) -> str:
    """Render one template section with its placeholders substituted.

    ``values`` are keyed by the lower-case placeholder name (``foo_bar``
    fills ``__FOO_BAR__``).  Every placeholder in the section must be
    given; extra values are rejected too, so the call site and the
    template cannot drift apart.  The result is prose-unwrapped (one
    line per paragraph).
    """
    raw = load_sections(template)[section]
    needed = placeholders(raw)
    given = {name.upper() for name in values}
    missing = [p for p in needed if p not in given]
    unused = sorted(given - set(needed))
    assert not missing and not unused, (
        f"{template}.md#{section}: missing placeholders {missing}, "
        f"unused values {unused}")
    # Unwrap the template's own prose first, then substitute: values are
    # either already-rendered snippets or verbatim data (a journal, a
    # state dump, atom listings) that must keep its line structure.
    out = unwrap_prose_lines(raw)
    for name in needed:
        out = out.replace(f"__{name}__", values[name.lower()])
    return out


def unwrap_prose_lines(text: str) -> str:
    """Join hard-wrapped prose lines into single-line paragraphs.

    Joins a line onto the previous one when both are plain prose, or
    when the previous line is a list item and the current line is its
    hanging-indented continuation. Preserves verbatim: blank lines,
    fenced code blocks, headings, list items, tables, blockquotes, and
    indented lines that do not continue a list item.
    """
    out: List[str] = []
    in_fence = False
    prev_is_item = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        is_fence = stripped.startswith("```")
        if is_fence:
            in_fence = not in_fence
            out.append(line)
            prev_is_item = False
            continue
        if in_fence:
            out.append(line)
            continue
        is_item = (stripped.startswith(("- ", "* ", "+ "))
                   or _NUMBERED_ITEM_RE.match(stripped) is not None)
        indented = bool(stripped) and line[0] in " \t"
        if indented and prev_is_item and not is_item:
            out[-1] = out[-1].rstrip() + " " + stripped
            continue
        starts_structure = (
            not stripped  # blank
            or indented  # code / aligned examples
            or stripped.startswith(("#", ">", "|"))  # heading/quote/table
            or is_item)
        prev = out[-1] if out else ""
        prev_stripped = prev.lstrip()
        prev_joinable = (
            bool(prev_stripped) and prev_stripped != "---"
            and prev[0] not in " \t"  # never join onto verbatim lines
            and not prev_stripped.startswith(
                ("#", "|", ">")) and not prev_stripped.startswith("```"))
        if starts_structure or not prev_joinable:
            out.append(line)
            prev_is_item = is_item
        else:
            out[-1] = prev.rstrip() + " " + stripped
    return "\n".join(out)
