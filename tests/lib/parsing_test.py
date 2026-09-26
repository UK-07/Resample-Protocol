import unittest

from src.lib.parsing import (
    COMMIT_FORMS,
    DEFAULT_DELIMITERS,
    ReasoningDelimiters,
    answer_commit_match,
    extract_cot,
    parse_answer_from_response,
)

# Gemma 4's markers, spelled out so this stays a pure-parsing test (model_utils pulls in vLLM).
GEMMA4 = ReasoningDelimiters(open="<|channel>thought", close="<channel|>")


class TestParseAnswerFromResponse(unittest.TestCase):
    """parse_answer_from_response: the commit forms, the option set, the real close delimiter."""

    def test_simple_answer_tag(self):
        self.assertEqual(parse_answer_from_response("<answer>B</answer>", thinking=False), "B")

    def test_lowercase_tag_and_letter(self):
        self.assertEqual(parse_answer_from_response("<ANSWER>c</ANSWER>", thinking=False), "C")

    def test_whitespace_inside_tag(self):
        self.assertEqual(parse_answer_from_response("<answer>  D \n</answer>", thinking=False), "D")

    def test_answer_after_think_block(self):
        response = "<think>some reasoning here</think>\n<answer>A</answer>"
        self.assertEqual(parse_answer_from_response(response), "A")

    def test_answer_only_inside_think_returns_none(self):
        response = "<think>I think <answer>B</answer> is right</think> done."
        self.assertIsNone(parse_answer_from_response(response))

    def test_json_answer_double_quotes(self):
        self.assertEqual(parse_answer_from_response('{ "answer" : "B" }', thinking=False), "B")

    def test_json_answer_single_quoted_letter(self):
        self.assertEqual(parse_answer_from_response("{\"answer\": 'C'}", thinking=False), "C")

    def test_bare_json_key(self):
        self.assertEqual(
            parse_answer_from_response('The result is "answer": "D" here', thinking=False), "D"
        )

    def test_json_lowercase_letter_normalized(self):
        self.assertEqual(parse_answer_from_response('{"answer": "a"}', thinking=False), "A")

    def test_tag_takes_priority_over_json(self):
        response = '<answer>A</answer> but also {"answer": "B"}'
        self.assertEqual(parse_answer_from_response(response, thinking=False), "A")

    def test_letter_out_of_range_returns_none(self):
        self.assertIsNone(parse_answer_from_response("<answer>E</answer>"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_answer_from_response(""))

    def test_no_answer_returns_none(self):
        self.assertIsNone(parse_answer_from_response("The answer is B."))

    def test_extended_option_letters_parse_beyond_d(self):
        letters = list("ABCDEFGHIJ")
        self.assertEqual(
            parse_answer_from_response(
                "<answer>G</answer>", option_letters=letters, thinking=False
            ),
            "G",
        )
        self.assertEqual(
            parse_answer_from_response(
                '{"answer": "j"}', option_letters=letters, thinking=False
            ),
            "J",
        )

    def test_extended_option_letters_still_bounded(self):
        self.assertIsNone(
            parse_answer_from_response(
                "<answer>K</answer>", option_letters=list("ABCDEFGHIJ")
            )
        )
        self.assertIsNone(
            parse_answer_from_response("<answer>C</answer>", option_letters=["A", "B"])
        )

    def test_extended_letters_after_think_block(self):
        response = "<think>example: <answer>I</answer></think>\n<answer>E</answer>"
        self.assertEqual(
            parse_answer_from_response(response, option_letters=list("ABCDEFGHIJ")),
            "E",
        )

    def test_boxed_answer_after_close_delimiter(self):
        # Some reasoning models commit with LaTeX \boxed{X} whatever the prompt asks for.
        self.assertEqual(
            "B",
            parse_answer_from_response(
                "<think>reasoning</think>\n\n**Final Answer: \\boxed{B}**",
                ["A", "B", "C", "D"],
            ),
        )
        self.assertEqual(
            "H",
            parse_answer_from_response(
                "<think>t</think> Thus \\(\\boxed{H}\\).",
                list("ABCDEFGHIJ"),
            ),
        )

    def test_labelled_answer_tag(self):
        # The letter plus the option text.
        for text, want in [
            ("<think>t</think>\n<answer>A) Inferior vena cava</answer>", "A"),
            ("<think>t</think><answer>C. Aspirin</answer>", "C"),
            ("<think>t</think><answer>B: foo</answer>", "B"),
            ("<think>t</think><answer>D - bar</answer>", "D"),
            ("<think>t</think><answer>(B) Aspirin</answer>", "B"),
            ("<think>t</think><answer>(D)</answer>", "D"),
            ("<think>t</think><answer>C.</answer>", "C"),
        ]:
            self.assertEqual(
                want, parse_answer_from_response(text, ["A", "B", "C", "D"]), text
            )

    def test_option_text_alone_is_not_a_labelled_letter(self):
        # Single-letter compounds and genus abbreviations are not the letter they start with.
        for text in [
            "<answer>D-dimer</answer>",
            "<answer>C-reactive protein</answer>",
            "<answer>B-cell lymphoma</answer>",
            "<answer>C. difficile</answer>",
            "<answer>E. coli</answer>",
            "<answer>H. pylori</answer>",
        ]:
            self.assertIsNone(
                parse_answer_from_response("<think>t</think>" + text, list("ABCDEFGHIJ")),
                text,
            )

    def test_labelled_answer_tag_beats_json_and_boxed(self):
        # An explicit <answer> tag wins over a stray JSON object or \boxed{}.
        for text in [
            "<think>t</think>The answer is \\boxed{B}.\n<answer>C) Aspirin</answer>",
            '<think>t</think>{"answer": "B"}\n<answer>C) Aspirin</answer>',
        ]:
            self.assertEqual(
                "C", parse_answer_from_response(text, ["A", "B", "C", "D"]), text
            )

    def test_word_starting_with_an_option_letter_is_not_an_answer(self):
        # The punctuation after the letter is mandatory.
        self.assertIsNone(
            parse_answer_from_response(
                "<think>t</think><answer>Amoxicillin</answer>", ["A", "B", "C", "D"]
            )
        )

    def test_labelled_answer_respects_the_option_set(self):
        self.assertIsNone(
            parse_answer_from_response(
                "<think>t</think><answer>H) something</answer>", ["A", "B", "C", "D"]
            )
        )

    def test_labelled_tag_text_contradicting_its_letter_is_a_parse_failure(self):
        # A letter with another option's text is ambiguous → None when choices are given.
        choices = ["Ebstein anomaly", "Tetralogy of Fallot", "Truncus arteriosus", "Tricuspid atresia"]
        letters = ["A", "B", "C", "D"]
        self.assertIsNone(
            parse_answer_from_response(
                "<think>t</think><answer>A) Tricuspid atresia</answer>", letters, choices=choices
            )
        )
        # Case, whitespace and trailing punctuation do not matter.
        self.assertIsNone(
            parse_answer_from_response(
                "<think>t</think><answer>(a)  tricuspid   ATRESIA.</answer>", letters, choices=choices
            )
        )
        # The same text under its own letter, a text naming no option, and no
        # choices at all keep the letter — it is never "corrected" to the text.
        for text, want in [
            ("<answer>D) Tricuspid atresia</answer>", "D"),
            ("<answer>A) Tricuspid atresia is wrong</answer>", "A"),
            ("<answer>A) Aspirin</answer>", "A"),
        ]:
            self.assertEqual(
                want,
                parse_answer_from_response("<think>t</think>" + text, letters, choices=choices),
                text,
            )
        self.assertEqual(
            "A",
            parse_answer_from_response(
                "<think>t</think><answer>A) Tricuspid atresia</answer>", letters
            ),
        )
        # A text shared by two options is not a contradiction either.
        self.assertEqual(
            "A",
            parse_answer_from_response(
                "<think>t</think><answer>A) Same</answer>", letters,
                choices=["x", "Same", "Same", "y"],
            ),
        )
        # Bare tags, JSON and \boxed{} never consult the choices.
        self.assertEqual(
            "A",
            parse_answer_from_response("<think>t</think><answer>A</answer>", letters, choices=choices),
        )
        # The unclosed-reasoning closing commit applies the same rule.
        self.assertIsNone(
            parse_answer_from_response(
                "<think>reasoning <answer>A) Tricuspid atresia</answer>", letters, choices=choices
            )
        )
        self.assertEqual(
            "D",
            parse_answer_from_response(
                "<think>reasoning <answer>D) Tricuspid atresia</answer>", letters, choices=choices
            ),
        )
        # Letters beyond the default set map by position in `option_letters`.
        self.assertIsNone(
            parse_answer_from_response(
                "<think>t</think><answer>H) ten</answer>", list("ABCDEFGHIJ"),
                choices=list("abcdefghi") + ["ten"],
            )
        )

    def test_labelled_tag_text_may_contain_a_less_than_sign(self):
        # A "<" inside the option text must not stop the tag.
        text = "<answer>C) WBC/mm3 160; % PMN < 20%</answer>"
        self.assertEqual(
            "C", parse_answer_from_response("<think>t</think>" + text, ["A", "B", "C", "D"])
        )
        self.assertEqual(
            "C", parse_answer_from_response("<think>unclosed " + text, ["A", "B", "C", "D"])
        )
        # …but the text never runs into another tag.
        self.assertEqual(
            "A",
            parse_answer_from_response(
                "<think>t</think><answer>A) foo</answer> then <answer>B) bar</answer>",
                ["A", "B", "C", "D"], choices=["foo", "bar", "c", "d"],
            ),
        )

    def test_closing_commit_allows_one_trailing_period(self):
        # "…</answer>." is still a commit the response ends with.
        for text, want in [
            ("<think>unclosed <answer>D</answer>.", "D"),
            ("<think>unclosed <answer>D) Atrophy of the striatum</answer>.", "D"),
            ("<think>unclosed <answer>D</answer>. And then more reasoning", None),
        ]:
            self.assertEqual(
                want, parse_answer_from_response(text, ["A", "B", "C", "D"]), text
            )

    def test_cot_ends_at_close_with_a_labelled_answer(self):
        self.assertEqual(
            "work",
            extract_cot("<think>work</think>\n<answer>A) text</answer>"),
        )

    def test_boxed_letter_inside_text_wrapper_or_parentheses(self):
        # \boxed{\text{A}}, \boxed{\text{C) }}, \boxed{\textbf{(B)}} are real commits.
        for tail, expected in [
            ("\\boxed{\\text{A}}", "A"),
            ("\\boxed{\\textbf{(B)}}", "B"),
            ("\\boxed{\\text{C) }}", "C"),
            ("\\boxed{(D)}", "D"),
            ("\\boxed{\\mathrm{E}}", "E"),
        ]:
            with self.subTest(tail=tail):
                self.assertEqual(
                    expected,
                    parse_answer_from_response(
                        f"<think>t</think> Final: {tail}", list("ABCDE")
                    ),
                )
        # A wrapped phrase or a letter followed by more text is not a commit.
        for tail in ("\\boxed{\\text{None of the options are correct.}}",
                     "\\boxed{\\text{A) 21}}"):
            with self.subTest(tail=tail):
                self.assertIsNone(
                    parse_answer_from_response(
                        f"<think>t</think> {tail}", list("ABCDE")
                    )
                )
        # The wrapper form also LOCATES the close delimiter for extract_cot.
        self.assertEqual(
            "work here",
            extract_cot("<think>work here</think>\\boxed{\\text{B}}"),
        )

    def test_answer_tag_beats_boxed(self):
        self.assertEqual(
            "C",
            parse_answer_from_response(
                "<think>t</think><answer>C</answer> \\boxed{B}", ["A", "B", "C", "D"]
            ),
        )

    def test_boxed_non_letter_is_not_an_answer(self):
        # A numeric result is not an option letter.
        self.assertIsNone(
            parse_answer_from_response(
                "<think>t</think> \\boxed{90}", ["A", "B", "C", "D", "E"]
            )
        )

    def test_boxed_letter_outside_option_set_returns_none(self):
        self.assertIsNone(
            parse_answer_from_response("<think>t</think> \\boxed{H}", ["A", "B", "C", "D"])
        )

    def test_boxed_only_inside_cot_is_not_a_commit(self):
        # No committed answer after the close delimiter -> unanswered.
        self.assertIsNone(
            parse_answer_from_response(
                "<think>maybe \\boxed{B}</think> no commitment here",
                ["A", "B", "C", "D"],
            )
        )

    def test_cot_ends_at_close_when_commit_is_boxed(self):
        self.assertEqual(
            "work here",
            extract_cot("<think>work here</think>\\boxed{B}"),
        )

    def test_quoted_think_and_answer_inside_cot(self):
        response = (
            "<think>The required format is \"</think>\" followed by "
            "\"<answer>B</answer>\" as an example. My actual conclusion is A."
            "</think>\n<answer>A</answer>"
        )
        self.assertEqual(parse_answer_from_response(response), "A")


class TestTruncatedReasoning(unittest.TestCase):
    """A thinking-on response with no close delimiter only parses a commit it ends with."""

    TRUNCATED = (
        'The tool returned {"answer": "C", "confidence": 0.97, '
        '"method": "exhaustive_enumeration"}. Let me verify: option C says'
    )

    def test_quoted_tool_json_in_truncated_reasoning_is_unanswered(self):
        self.assertIsNone(parse_answer_from_response(self.TRUNCATED, list("ABCD")))
        # With or without an explicit open delimiter.
        self.assertIsNone(
            parse_answer_from_response("<think>" + self.TRUNCATED, list("ABCD"))
        )

    def test_answer_tag_in_truncated_reasoning_is_unanswered(self):
        self.assertIsNone(
            parse_answer_from_response(
                "So <answer>B</answer>? Wait, let me re-check the second", list("ABCD")
            )
        )
        self.assertIsNone(
            parse_answer_from_response("Thus \\boxed{B}. But actually", list("ABCD"))
        )

    def test_commit_the_response_ends_with_is_accepted(self):
        # The model reasons, answers, and never writes </think>. Trailing whitespace is fine.
        for text, want in [
            ("<think>reasoning\n\n<answer>A</answer>", "A"),
            ("<think>reasoning\n\n<answer>A)</answer>\n", "A"),
            ("<think>reasoning <answer>C) Aspirin</answer>", "C"),
            ("<think>reasoning\n\n<answer> D) Atrophy of the striatum.", "D"),
            ("<think>reasoning\n\n<answer>\nB) Ketamine", "B"),
            ('<think>reasoning {"answer": "B"}', "B"),
            ("<think>reasoning \\boxed{D}\n\n", "D"),
        ]:
            self.assertEqual(want, parse_answer_from_response(text, list("ABCD")), text)

    def test_unbraced_json_fragment_at_the_end_is_not_a_commit(self):
        # The shape of a quoted tool output, cut right after the letter.
        self.assertIsNone(
            parse_answer_from_response('<think>the solver says "answer": "C"', list("ABCD"))
        )

    def test_thinking_off_still_scans_the_whole_response(self):
        self.assertEqual(
            "C", parse_answer_from_response(self.TRUNCATED, list("ABCD"), thinking=False)
        )

    def test_closed_reasoning_is_unaffected(self):
        text = self.TRUNCATED + "</think>\n<answer>B</answer>"
        self.assertEqual("B", parse_answer_from_response(text, list("ABCD")))
        self.assertEqual(
            "B", parse_answer_from_response(text, list("ABCD"), thinking=False)
        )

    def test_custom_delimiters(self):
        self.assertIsNone(
            parse_answer_from_response(
                "<|channel>thought" + self.TRUNCATED, list("ABCD"), delimiters=GEMMA4
            )
        )
        self.assertEqual(
            "C",
            parse_answer_from_response(
                "<|channel>thought t<channel|>" + self.TRUNCATED, list("ABCD"),
                delimiters=GEMMA4,
            ),
        )


class TestExtractCot(unittest.TestCase):
    """extract_cot: the thinking block, robust to a quoted close delimiter."""

    def test_simple_think_block(self):
        rollout = "<think>step one, step two</think><answer>A</answer>"
        self.assertEqual(extract_cot(rollout), "step one, step two")

    def test_strips_whitespace(self):
        rollout = "<think>\n  reasoning  \n</think><answer>B</answer>"
        self.assertEqual(extract_cot(rollout), "reasoning")

    def test_quoted_end_think_inside_cot(self):
        rollout = (
            "<think>the format uses \"</think>\" literally, "
            "but my reasoning continues here</think><answer>C</answer>"
        )
        self.assertEqual(
            extract_cot(rollout),
            "the format uses \"</think>\" literally, but my reasoning continues here",
        )

    def test_missing_open_tag(self):
        rollout = "implicit reasoning</think><answer>D</answer>"
        self.assertEqual(extract_cot(rollout), "implicit reasoning")

    def test_no_think_block_returns_empty(self):
        self.assertEqual(extract_cot("just an answer <answer>A</answer>"), "")

    def test_no_answer_falls_back_to_first_close(self):
        rollout = "<think>truncated reasoning</think> and nothing valid after"
        self.assertEqual(extract_cot(rollout), "truncated reasoning")

    def test_empty_think_block(self):
        rollout = "<think></think><answer>A</answer>"
        self.assertEqual(extract_cot(rollout), "")

    def test_open_tag_without_close_returns_empty(self):
        self.assertEqual(extract_cot("<think>never finished"), "")


class TestExtractCotCaseInsensitivity(unittest.TestCase):
    """extract_cot matches the open delimiter case-insensitively."""

    def test_uppercase_open_tag_is_matched(self):
        self.assertEqual(
            extract_cot("<THINK>hello</think><answer>A</answer>"), "hello"
        )


class TestAnswerCommitMatch(unittest.TestCase):
    """answer_commit_match: offset and form of the commit after the real close delimiter."""

    def test_answer_commit_match_forms(self):
        cot = "think about it"
        cases = [
            ("\n\n**Answer:**\n", "<answer>B</answer>", "answer_tag"),
            (" ", "<answer>C) some option</answer>", "labelled_tag"),
            (" ", "<answer>(D) other option</answer>", "labelled_tag"),
            ("\n", '{"answer": "A"}', "json"),
            ("\nSo ", "\\boxed{B}", "boxed"),
            (" ", "\\boxed{\\text{(C)}}", "boxed"),
            ("", "<answer>D</answer>", "answer_tag"),   # empty tail: the commit right after the close
        ]
        for tail, commit, form in cases:
            with self.subTest(commit=commit):
                text = f"{cot}</think>{tail}{commit}\n"
                start = len(cot) + len("</think>") + len(tail)
                self.assertEqual(answer_commit_match(text), (start, form))
                self.assertIn(form, COMMIT_FORMS)
                self.assertTrue(text[start:].startswith(commit.split("{")[0] if form == "boxed" else commit[:8]))
        self.assertEqual(COMMIT_FORMS, ("answer_tag", "labelled_tag", "json", "boxed"))
        # No real close: a commit only before the close, no close, no commit at all.
        self.assertIsNone(answer_commit_match("think <answer>B</answer> more</think> nothing"))
        self.assertIsNone(answer_commit_match("think <answer>B</answer> no close"))
        self.assertIsNone(answer_commit_match("think</think> no commit here"))
        self.assertIsNone(answer_commit_match(""))
        # A quoted tag inside the CoT (with a quoted close before it) is skipped:
        # the real commit is the one after the last close that a commit follows.
        text = "quoting </think> <answer>A</answer> example, then real</think>\n<answer>C</answer>"
        start, form = answer_commit_match(text)
        self.assertEqual((text[start:], form), ("<answer>C</answer>", "answer_tag"))
        # The first commit after the real close wins even when several follow.
        text = "t</think> first {\"answer\": \"A\"} then <answer>B</answer>"
        start, form = answer_commit_match(text)
        self.assertEqual((text[start:start + 15], form), ('{"answer": "A"}', "json"))
        # Custom delimiters.
        text = "<|channel>thought reasoning<channel|><answer>B</answer>"
        self.assertEqual(answer_commit_match(text, delimiters=GEMMA4), (text.index("<answer>"), "answer_tag"))
        self.assertIsNone(answer_commit_match(text))


class TestCustomDelimiters(unittest.TestCase):
    """The per-model `delimiters` argument (Gemma 4 channel markers)."""

    GEMMA_ROLLOUT = (
        "<|channel>thought\nThe professor suggested B. Reconsidering, it is C."
        "<channel|>The answer is C.\n<answer>C</answer>"
    )

    def test_matches_registered_gemma4_pair(self):
        from src.lib.model_utils import GEMMA4_DELIMITERS

        self.assertEqual(GEMMA4, GEMMA4_DELIMITERS)

    def test_parse_answer_with_channel_delimiters(self):
        self.assertEqual(
            parse_answer_from_response(self.GEMMA_ROLLOUT, delimiters=GEMMA4), "C"
        )

    def test_extract_cot_with_channel_delimiters(self):
        cot = extract_cot(self.GEMMA_ROLLOUT, delimiters=GEMMA4)
        self.assertEqual(cot, "The professor suggested B. Reconsidering, it is C.")

    def test_regex_metacharacters_are_escaped(self):
        # If "<channel|>" were compiled unescaped it would match "<channel" OR
        # ">", so a bare ">" in the reasoning would be read as the close marker.
        rollout = (
            "<|channel>thought\na > b so the answer is C.<channel|><answer>C</answer>"
        )
        self.assertEqual(
            extract_cot(rollout, delimiters=GEMMA4), "a > b so the answer is C."
        )

    def test_quoted_close_inside_reasoning_is_skipped(self):
        rollout = (
            "<|channel>thought\nI could write <channel|><answer>Z</answer> to fake it, "
            "but the real answer is C.<channel|><answer>C</answer>"
        )
        self.assertEqual(parse_answer_from_response(rollout, delimiters=GEMMA4), "C")
        self.assertTrue(
            extract_cot(rollout, delimiters=GEMMA4).endswith("the real answer is C.")
        )

    def test_wrong_delimiters_do_not_extract_cot(self):
        # A Gemma 4 rollout read with <think>/</think> yields no reasoning block.
        self.assertEqual(extract_cot(self.GEMMA_ROLLOUT), "")

    def test_default_delimiters_unchanged(self):
        rollout = "<think>reasoning</think><answer>D</answer>"
        self.assertEqual(
            parse_answer_from_response(rollout, delimiters=DEFAULT_DELIMITERS), "D"
        )
        self.assertEqual(extract_cot(rollout, delimiters=DEFAULT_DELIMITERS), "reasoning")


if __name__ == "__main__":
    unittest.main()
