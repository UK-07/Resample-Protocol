import unittest

from src.lib.model_utils import MODEL_CONFIGS, get_model_config
from src.lib.parsing import ReasoningDelimiters
from src.lib.prompts import (
    SYSTEM_PROMPT_NATIVE_THINKING,
    SYSTEM_PROMPT_PLAIN,
    get_prompted_thinking_prompt,
    get_system_prompt,
)


def _registered_delimiters():
    """Every distinct delimiter pair carried by the model registry."""
    return {get_model_config(name)["delimiters"] for name in MODEL_CONFIGS}


class TestSystemPromptConstants(unittest.TestCase):

    def test_native_prompt_has_no_registered_close_delimiter(self):
        for delims in _registered_delimiters():
            with self.subTest(close=delims.close):
                self.assertNotIn(
                    delims.close.lower(), SYSTEM_PROMPT_NATIVE_THINKING.lower()
                )

    def test_all_prompts_mention_answer_tags(self):
        for prompt in (
            SYSTEM_PROMPT_NATIVE_THINKING,
            get_prompted_thinking_prompt(),
            SYSTEM_PROMPT_PLAIN,
        ):
            self.assertIn("<answer>", prompt)
            self.assertIn("</answer>", prompt)

    def test_prompted_prompt_mentions_think_tags(self):
        self.assertIn("<think>", get_prompted_thinking_prompt())

    def test_plain_prompt_requests_no_reasoning(self):
        self.assertNotIn("<think>", SYSTEM_PROMPT_PLAIN)
        self.assertNotIn("reasoning", SYSTEM_PROMPT_PLAIN.lower())

    def test_prompts_are_distinct(self):
        prompts = {
            SYSTEM_PROMPT_NATIVE_THINKING,
            get_prompted_thinking_prompt(),
            SYSTEM_PROMPT_PLAIN,
        }
        self.assertEqual(len(prompts), 3)


class TestGetSystemPrompt(unittest.TestCase):

    def test_native_mode(self):
        self.assertEqual(get_system_prompt("native"), SYSTEM_PROMPT_NATIVE_THINKING)

    def test_none_mode(self):
        self.assertEqual(get_system_prompt("none"), SYSTEM_PROMPT_PLAIN)

    def test_prompted_mode(self):
        self.assertEqual(
            get_system_prompt("prompted"), get_prompted_thinking_prompt()
        )

    def test_unknown_mode_falls_back_to_prompted(self):
        self.assertEqual(
            get_system_prompt("some_future_mode"), get_prompted_thinking_prompt()
        )

    def test_empty_string_falls_back_to_prompted(self):
        self.assertEqual(get_system_prompt(""), get_prompted_thinking_prompt())

    def test_prompted_prompt_renders_custom_delimiters(self):
        delims = ReasoningDelimiters(open="<r>", close="</r>")
        rendered = get_system_prompt("prompted", delimiters=delims)
        self.assertIn("<r></r>", rendered)
        self.assertEqual(rendered, get_prompted_thinking_prompt(delims))

    def test_native_prompt_guard_raises_on_delimiter_collision(self):
        colliding = ReasoningDelimiters(open="<r>", close="answer")
        with self.assertRaises(ValueError):
            get_system_prompt("native", delimiters=colliding)

    def test_none_mode_ignores_delimiters(self):
        delims = ReasoningDelimiters(open="<|channel>thought", close="<channel|>")
        self.assertEqual(
            get_system_prompt("none", delimiters=delims), SYSTEM_PROMPT_PLAIN
        )


if __name__ == "__main__":
    unittest.main()
