"""Canonical system prompts, keyed by the effective thinking mode.

The native-thinking prompt must name no reasoning delimiter: priming the model
with its own close delimiter breaks the close-delimiter detection in
:mod:`src.lib.parsing`. Delimiters are per-model, so :func:`get_system_prompt`
enforces that invariant at runtime.
"""

from __future__ import annotations

from src.lib.parsing import DEFAULT_DELIMITERS, ReasoningDelimiters

# Must stay free of any close-delimiter literal (checked in get_system_prompt).
SYSTEM_PROMPT_NATIVE_THINKING = (
    "Once your reasoning is complete, provide your final answer letter "
    "within <answer></answer> tags."
)

SYSTEM_PROMPT_PROMPTED_THINKING_TEMPLATE = (
    "Format all the reasoning within {open}{close} tags, followed by "
    "the answer in <answer></answer> tags."
)

SYSTEM_PROMPT_PLAIN = (
    "Provide your final answer letter within <answer></answer> tags."
)


def get_prompted_thinking_prompt(
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> str:
    """Render the prompted-thinking prompt for a delimiter pair."""
    return SYSTEM_PROMPT_PROMPTED_THINKING_TEMPLATE.format(
        open=delimiters.open, close=delimiters.close
    )


def get_system_prompt(
    thinking_mode: str,
    *,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> str:
    """Return the system prompt for ``"native"``, ``"prompted"`` or ``"none"``.

    Unknown modes fall back to the prompted-thinking prompt. Raises ValueError
    if the native prompt contains the model's own close delimiter.
    """
    if thinking_mode == "native":
        if delimiters.close.lower() in SYSTEM_PROMPT_NATIVE_THINKING.lower():
            raise ValueError(
                "SYSTEM_PROMPT_NATIVE_THINKING contains the close delimiter "
                f"{delimiters.close!r}. Priming the model with its own close "
                "delimiter breaks the end-of-thinking detector every rollout "
                "consumer relies on. Reword the prompt so it names no delimiter."
            )
        return SYSTEM_PROMPT_NATIVE_THINKING
    if thinking_mode == "none":
        return SYSTEM_PROMPT_PLAIN
    return get_prompted_thinking_prompt(delimiters)
