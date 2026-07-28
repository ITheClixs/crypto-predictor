"""The README's LaTeX must survive GitHub's markdown-then-MathJax pipeline.

GitHub renders `$...$` and `$$...$$` with a restricted MathJax build, *after* the
markdown parser has already had a pass over the text. Two things go wrong, and
both did before this test existed:

- Some macros are rejected outright. `\\operatorname` is the common one; the page
  renders a red "The following macros are not allowed" box in place of the
  equation.
- A backslash followed by ASCII punctuation is a CommonMark escape, so `\\,`,
  `\\!`, `\\;`, `\\{` and friends lose their backslash before MathJax ever sees
  them. Inside a `$$` block spanning several lines the damage is worse: markdown
  treats the interior lines as prose, so a continuation beginning with `+` turns
  into a bullet list and the equation is destroyed.

Checking the rendered page by eye does not scale and is easy to skip, so the
constraints are asserted here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

#: Macros GitHub's renderer refuses. Not exhaustive, but these are the ones a
#: mathematically written README reaches for by default.
BANNED_MACROS = ("operatorname", "def", "newcommand", "renewcommand", "require", "input")

#: CommonMark treats a backslash before any of these as an escape, which strips
#: the backslash before MathJax runs. Letters are safe, so `\quad` and `\lbrace`
#: are the workarounds for `\;` and `\{`.
ESCAPABLE_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~\\")


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _block_math(text: str) -> list[tuple[int, str]]:
    """Every `$$...$$` block with the 1-based line its opener sits on."""
    return [
        (text[: m.start()].count("\n") + 1, m.group(1))
        for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S)
    ]


def _inline_math(text: str) -> list[tuple[int, str]]:
    """Every `$...$` span, skipping `$$` blocks and fenced code."""
    without_blocks = re.sub(r"\$\$.+?\$\$", lambda m: " " * len(m.group(0)), text, flags=re.S)
    without_code = re.sub(r"```.*?```", lambda m: " " * len(m.group(0)), without_blocks, flags=re.S)
    return [
        (without_code[: m.start()].count("\n") + 1, m.group(1))
        for m in re.finditer(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", without_code)
    ]


def _all_math(text: str) -> list[tuple[int, str]]:
    return _block_math(text) + _inline_math(text)


@pytest.mark.unit
def test_readme_has_math_to_check() -> None:
    """Guards the rest of this module against silently passing on an empty match."""
    text = _readme()
    assert len(_block_math(text)) >= 20
    assert len(_inline_math(text)) >= 20


@pytest.mark.unit
def test_no_block_equation_spans_multiple_lines() -> None:
    """A multi-line `$$` block is reprocessed as markdown and comes apart."""
    offenders = [line for line, body in _block_math(_readme()) if "\n" in body]
    assert not offenders, f"multi-line $$ blocks at lines {offenders}; put each on one line"


@pytest.mark.unit
def test_no_banned_macros() -> None:
    offenders = [
        (line, macro)
        for line, body in _all_math(_readme())
        for macro in BANNED_MACROS
        if re.search(rf"\\{macro}\b", body)
    ]
    assert not offenders, f"macros GitHub rejects: {offenders}; use \\mathrm or \\text"


@pytest.mark.unit
def test_no_backslash_escapes_that_markdown_would_eat() -> None:
    offenders = [
        (line, f"\\{body[m.start() + 1]}")
        for line, body in _all_math(_readme())
        for m in re.finditer(r"\\(.)", body)
        if body[m.start() + 1] in ESCAPABLE_PUNCTUATION
    ]
    assert not offenders, (
        f"backslash-escaped punctuation inside math: {offenders}; "
        "use \\quad for spacing and \\lbrace / \\rbrace for set braces"
    )


@pytest.mark.unit
def test_no_bare_asterisk_inside_math() -> None:
    """A literal `*` inside math is read as markdown emphasis and disappears."""
    offenders = [line for line, body in _all_math(_readme()) if "*" in body]
    assert not offenders, f"bare '*' inside math at lines {offenders}; use \\ast"


@pytest.mark.unit
def test_math_delimiters_are_balanced() -> None:
    text = _readme()
    assert text.count("$$") % 2 == 0, "odd number of $$ delimiters"

    stripped = re.sub(r"\$\$.+?\$\$", "", text, flags=re.S)
    unbalanced = [i for i, line in enumerate(stripped.split("\n"), 1) if line.count("$") % 2]
    assert not unbalanced, f"unbalanced inline $ on lines {unbalanced}"


@pytest.mark.unit
def test_braces_balance_within_every_expression() -> None:
    offenders = []
    for line, body in _all_math(_readme()):
        depth = 0
        for i, char in enumerate(body):
            if i and body[i - 1] == "\\":
                continue
            depth += (char == "{") - (char == "}")
            if depth < 0:
                break
        if depth != 0:
            offenders.append(line)
    assert not offenders, f"unbalanced braces in math at lines {offenders}"
