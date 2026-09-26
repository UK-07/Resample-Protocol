"""Parsing of model responses: the final answer letter and the chain-of-thought.

The reasoning-block delimiters are a per-model property (``<think>``/``</think>``
for most models, ``<|channel>thought``/``<channel|>`` for Gemma 4): every function
takes a keyword-only ``delimiters`` defaulting to the ``<think>`` pair; callers
that know the model pass ``get_model_config(model_name)["delimiters"]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class ReasoningDelimiters:
    """The literal strings that open and close a model's reasoning block."""

    open: str = "<think>"
    close: str = "</think>"


DEFAULT_DELIMITERS = ReasoningDelimiters()


@lru_cache(maxsize=32)
def _delim_re(literal: str) -> re.Pattern:
    """Case-insensitive pattern matching *literal* verbatim (``<channel|>`` holds a metacharacter)."""
    return re.compile(re.escape(literal), re.IGNORECASE)


def parse_answer_from_response(
    response: str,
    option_letters: Optional[list[str]] = None,
    *,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
    thinking: bool = True,
    choices: Optional[list[str]] = None,
) -> Optional[str]:
    """Parse the committed answer letter of a response, or None.

    ``option_letters`` is the valid letter set (default A-D); a letter outside it
    is not parsed. The answer is read after the *real* close delimiter (the one
    followed by a commit form); under ``thinking`` a response with no close
    delimiter only counts a commit it ends with, with ``thinking=False`` the
    whole response is scanned. Commit forms in priority order: bare
    ``<answer>X</answer>``, labelled ``<answer>X) text</answer>``, JSON
    ``{"answer": "X"}``, LaTeX ``\\boxed{X}``. ``choices`` (option texts in letter
    order) makes a labelled tag whose text equals exactly one *other* option's
    text parse as None; the letter is never corrected to the text.
    """
    letters = "".join(option_letters) if option_letters else "A-D"

    end_char = _find_real_end_think_char(response, delimiters=delimiters)
    if end_char is not None:
        text = response[end_char:]
    else:
        close_match = _delim_re(delimiters.close).search(response)
        if close_match:
            text = response[close_match.end():]
        elif thinking:
            # The reasoning block never closed: a commit the response ends
            # with is a finished answer, anything else is a cut-off trace.
            return _closing_commit(
                response, letters, option_letters=option_letters, choices=choices
            )
        else:
            text = response

    match = re.search(rf"<answer>\s*([{letters}])\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Directly after the bare tag, so an explicit <answer> tag of either form
    # beats a stray JSON object or \boxed{} in the same text.
    match = re.search(_labelled_tag_pattern(letters), text, re.IGNORECASE)
    if match:
        if _labelled_contradiction(match, option_letters, choices):
            return None
        return match.group(1).upper()

    for pattern in [
        rf'\{{\s*"answer"\s*:\s*"([{letters}])"\s*\}}',
        rf"\{{\s*\"answer\"\s*:\s*'([{letters}])'\s*\}}",
        rf'"answer"\s*:\s*"([{letters}])"',
        _boxed_pattern(letters),
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # `_boxed_pattern` carries a wrapper group before the letter, so
            # take the last group that captured rather than assuming group 1.
            letter = next(g for g in reversed(match.groups()) if g)
            return letter.upper()

    return None


def _labelled_tag_pattern(letters: str) -> str:
    """Regex for a labelled ``<answer>L) text`` tag: group 1 the letter, group 2 the text.

    The text runs to the closing tag or the end of the string and may contain a
    ``<`` that does not open ``</answer>``, but never another tag.
    """
    return (
        rf"<answer>\s*\(?\s*([{letters}])\s*{_LABELLED_TAG_SEPARATOR}\s*"
        r"((?:[^<]|<(?!/?answer>))*?)\s*(?=</answer>|<answer>|$)"
    )


def _normalise_option_text(text: str) -> str:
    """Option text as compared by the contradiction check."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold().rstrip(" .;:,!?\"'")


def _labelled_contradiction(match: re.Match, option_letters, choices) -> bool:
    """True when a labelled tag's text names exactly one option other than its letter.

    ``option_letters`` gives the letter of each entry in ``choices`` (A, B, …
    by position when None). No ``choices`` → no check.
    """
    if not choices:
        return False
    text = _normalise_option_text(match.group(2))
    if not text:
        return False
    letters = list(option_letters) if option_letters else [chr(65 + i) for i in range(len(choices))]
    hits = [
        letters[i].upper() for i, option in enumerate(choices)
        if i < len(letters) and _normalise_option_text(option) == text
    ]
    return len(hits) == 1 and hits[0] != match.group(1).upper()


def _closing_commit(
    response: str,
    letters: str,
    *,
    option_letters: Optional[list[str]] = None,
    choices: Optional[list[str]] = None,
) -> Optional[str]:
    """The answer letter of a commit form that *ends* ``response``, else None.

    For a thinking-on response with no close delimiter: the same forms as the
    main parser, each anchored at the end of the text (trailing whitespace and
    one trailing period allowed). The un-braced ``"answer": "X"`` fragment is
    deliberately not accepted: it is the shape of a quoted tool output, not of
    a commit.
    """
    end = r"\s*\.?\s*$"
    # A labelled tag, closed — or left open when the response ends inside its text.
    match = re.search(
        _labelled_tag_pattern(letters) + rf"(?:</answer>)?{end}", response, re.IGNORECASE
    )
    if match:
        if _labelled_contradiction(match, option_letters, choices):
            return None
        return match.group(1).upper()
    for pattern in [
        rf"<answer>\s*([{letters}])\s*</answer>{end}",
        rf'\{{\s*"answer"\s*:\s*"([{letters}])"\s*\}}{end}',
        rf"\{{\s*\"answer\"\s*:\s*'([{letters}])'\s*\}}{end}",
        _boxed_pattern(letters) + end,
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            letter = next(g for g in reversed(match.groups()) if g)
            return letter.upper()
    return None


# Separator between the letter and the text of a labelled tag. Pickier than
# "any punctuation" so option text written alone is not read as a letter:
# a hyphen must be followed by whitespace ("D-dimer" is not option D), a period
# must not be followed by a lowercase word ("E. coli" is not option E; "C. Aspirin" is).
_LABELLED_TAG_SEPARATOR = r"(?:[):]|\.(?!\s*(?-i:[a-z]))|-(?=\s))"


def _boxed_pattern(letters: str) -> str:
    """Regex for a ``\\boxed{...}`` commit of one letter from ``letters``.

    The letter may sit inside a ``\\text{}``-style wrapper and/or parentheses.
    Two capturing groups: the optional wrapper (group 1, so a conditional can
    demand its closing brace) and the letter (group 2).
    """
    return (
        r"\\boxed\{\s*"
        r"(\\(?:text|textbf|textrm|mathrm|mathbf)\{\s*)?"  # optional wrapper
        rf"\(?\s*([{letters}])\s*\)?\s*"                  # (L) / L) / L
        r"(?(1)\})\s*\}"                                   # close wrapper, then box
    )


# The commit forms in the parser's priority order. `_ANSWER_RE` only LOCATES the
# real close delimiter (the widest letter set A-J, whatever the dataset), it never
# decides answer validity. Each form is also compiled on its own so
# `answer_commit_match` can name the alternative that matched; the forms are not
# wrapped in named groups because `_boxed_pattern`'s `(?(1)…)` conditional must
# stay group 1 of its own pattern.
COMMIT_FORMS = ("answer_tag", "labelled_tag", "json", "boxed")
_COMMIT_PATTERNS = (
    ("answer_tag", r"<answer>\s*[A-J]\s*</answer>"),
    ("labelled_tag", r"<answer>\s*\(?\s*[A-J]\s*" + _LABELLED_TAG_SEPARATOR),
    ("json", r"\{\s*[\"']answer[\"']\s*:\s*[\"'][A-J][\"']\s*\}"),
    ("boxed", _boxed_pattern("A-J")),
)
assert tuple(form for form, _ in _COMMIT_PATTERNS) == COMMIT_FORMS
_ANSWER_RE = re.compile(r"(?:" + r"|".join(pattern for _, pattern in _COMMIT_PATTERNS) + r")", re.IGNORECASE)
_COMMIT_FORM_RES = tuple((form, re.compile(pattern, re.IGNORECASE)) for form, pattern in _COMMIT_PATTERNS)


def _find_real_end_think_char(
    full_text: str,
    *,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> Optional[int]:
    """Position right after the last close delimiter that a commit form follows, or None.

    Robust to the model quoting the delimiter inside its reasoning: the quoted
    copy appears before the real one and is skipped.
    """
    target: Optional[int] = None
    for m in _delim_re(delimiters.close).finditer(full_text):
        end = m.end()
        if _ANSWER_RE.search(full_text, pos=end):
            target = end
    return target


def answer_commit_match(
    text: str,
    *,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> Optional[tuple[int, str]]:
    """``(start_char, form)`` of the first commit at or after the real close delimiter.

    ``form`` is one of :data:`COMMIT_FORMS`; None when the text has no real
    close delimiter. A commit form quoted inside the reasoning block is never
    returned.
    """
    close_end = _find_real_end_think_char(text, delimiters=delimiters)
    if close_end is None:
        return None
    match = _ANSWER_RE.search(text, pos=close_end)
    if match is None:   # cannot happen: close_end is only set when a commit follows it
        return None
    for form, pattern in _COMMIT_FORM_RES:
        if pattern.match(text, match.start()):
            return match.start(), form
    raise AssertionError("_ANSWER_RE matched but no commit form pattern did")   # pragma: no cover


def extract_cot(
    rollout_text: str,
    *,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> str:
    """The content of the thinking block, or "" when none can be located.

    Robust to a quoted close delimiter inside the reasoning; without an open
    delimiter everything up to the close counts.
    """
    open_match = _delim_re(delimiters.open).search(rollout_text)
    start = open_match.start() if open_match else -1
    content_start = open_match.end() if open_match else -1

    end_char = _find_real_end_think_char(rollout_text, delimiters=delimiters)
    if end_char is not None:
        end = end_char - len(delimiters.close)
    else:
        fallback = _delim_re(delimiters.close).search(rollout_text)
        end = fallback.start() if fallback else -1

    if start >= 0 and end > start:
        return rollout_text[content_start:end].strip()
    if end >= 0:
        return rollout_text[:end].strip()
    return ""
