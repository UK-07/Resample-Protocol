import contextlib
import io
import json
import unittest
from unittest import mock

import pandas as pd

import src.lib.hinted_rollouts as hr
from src.lib.constants import OPTION_LETTERS
from src.lib.hints import HintResult
from src.lib.parsing import DEFAULT_DELIMITERS, ReasoningDelimiters


class FakeTokenizer:
    """Returns the last message's content and records every call."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        # Recorded as **kwargs so a test can assert enable_thinking is absent entirely.
        self.calls.append({"messages": messages, "kwargs": kwargs,
                           "enable_thinking": kwargs.get("enable_thinking")})
        return messages[-1]["content"]


def make_baseline_df(rows):
    """Baseline-shaped DataFrame (post-load_baseline: decoded choices/fields)."""
    records = []
    for r in rows:
        records.append({
            "original_index": r["original_index"],
            "question": r.get("question", f"q{r['original_index']}"),
            "choices": r.get("choices", ["w", "x", "y", "z"]),
            "prompt": r.get("prompt", f"Q{r['original_index']}"),
            "groundtruth": r.get("groundtruth", "A"),
            "baseline_answer": r.get("baseline_answer", "A"),
            "correct": r["correct"],
            "additional_fields": r.get("additional_fields", {"subject": "s"}),
        })
    return pd.DataFrame(records)


class TestValidateCases(unittest.TestCase):

    def test_valid_modes_returned(self):
        for mode in hr.CASE_MODES:
            self.assertEqual(hr.validate_cases(mode), mode)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            hr.validate_cases("bogus")
        with self.assertRaises(ValueError):
            hr.validate_cases("positive")  # must be the exact registered name


class TestApplyChatTemplate(unittest.TestCase):

    def test_forwards_flags_to_tokenizer(self):
        tok = FakeTokenizer()
        rendered = hr.apply_chat_template(
            tok, [{"role": "user", "content": "hi"}], enable_thinking=True
        )
        self.assertEqual(rendered, "hi")
        self.assertTrue(tok.calls[0]["enable_thinking"])

    def test_none_omits_enable_thinking(self):
        tok = FakeTokenizer()
        hr.apply_chat_template(
            tok, [{"role": "user", "content": "hi"}], enable_thinking=None
        )
        self.assertNotIn("enable_thinking", tok.calls[0]["kwargs"])


class TestSelectCaseRows(unittest.TestCase):

    def setUp(self):
        self.df = make_baseline_df([
            {"original_index": 0, "correct": True},
            {"original_index": 1, "correct": False},
            {"original_index": 2, "correct": True},
            {"original_index": 3, "correct": False},
        ])

    def test_positive_cases(self):
        out = hr.select_case_rows(self.df, "positive_cases")
        self.assertEqual(out["original_index"].tolist(), [0, 2])
        self.assertEqual(out["sample_type"].unique().tolist(), ["positive"])

    def test_negative_cases(self):
        out = hr.select_case_rows(self.df, "negative_cases")
        self.assertEqual(out["original_index"].tolist(), [1, 3])
        self.assertEqual(out["sample_type"].unique().tolist(), ["negative"])

    def test_both_orders_positives_first(self):
        out = hr.select_case_rows(self.df, "both")
        self.assertEqual(out["original_index"].tolist(), [0, 2, 1, 3])
        self.assertEqual(
            out["sample_type"].tolist(),
            ["positive", "positive", "negative", "negative"],
        )

    def test_limit_caps_per_category(self):
        out = hr.select_case_rows(self.df, "both", limit=1)
        self.assertEqual(out["original_index"].tolist(), [0, 1])

    def test_negatives_require_parsed_baseline_answer(self):
        df = make_baseline_df([
            {"original_index": 0, "correct": False, "baseline_answer": "B"},
            {"original_index": 1, "correct": False, "baseline_answer": ""},
            {"original_index": 2, "correct": False,
             "baseline_answer": float("nan")},
            {"original_index": 3, "correct": True, "baseline_answer": "A"},
        ])
        out = hr.select_case_rows(df, "negative_cases")
        self.assertEqual(out["original_index"].tolist(), [0])
        both = hr.select_case_rows(df, "both")
        self.assertEqual(both["original_index"].tolist(), [3, 0])

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            hr.select_case_rows(self.df, "everything")


class TestBuildHintedItems(unittest.TestCase):

    def _build(self, df_rows, hint_names, *, pool=None, seed=42, done_pairs=None,
               hint_n_examples=None, tokenizer=None, option_letters=None):
        tok = tokenizer or FakeTokenizer()
        pool = pool if pool is not None else make_baseline_df(
            [{"original_index": 100 + i, "correct": True,
              "question": f"pool{i}?"} for i in range(4)]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            items, skipped = hr.build_hinted_items(
                df_rows, pool, hint_names, hint_n_examples or {},
                tok, "SYS", enable_thinking=True, seed=seed,
                done_pairs=done_pairs, option_letters=option_letters,
            )
        return items, skipped, tok

    def _positive_rows(self, n=1):
        return hr.select_case_rows(
            make_baseline_df(
                [{"original_index": i, "correct": True} for i in range(n)]
            ),
            "positive_cases",
        )

    def _negative_rows(self, baseline_answer="B"):
        # Tagged manually so no-answer rows can reach the builder's own skip rules.
        df = make_baseline_df([
            {"original_index": 0, "correct": False, "groundtruth": "C",
             "baseline_answer": baseline_answer},
        ])
        df["sample_type"] = "negative"
        return df

    def test_positive_item_fields(self):
        items, skipped, _ = self._build(self._positive_rows(), ["authority"])
        self.assertEqual(sum(skipped.values()), 0)
        self.assertEqual(len(items), 1)
        it = items[0]
        self.assertEqual(it["original_index"], 0)
        self.assertEqual(it["sample_type"], "positive")
        self.assertEqual(it["hint_name"], "authority")
        self.assertEqual(it["baseline_answer"], "A")
        self.assertEqual(it["groundtruth"], "A")
        self.assertEqual(it["additional_fields"], {"subject": "s"})
        # Authority points at a wrong (non-baseline) option.
        self.assertIn(it["hinted_answer"], OPTION_LETTERS)
        self.assertNotEqual(it["hinted_answer"], "A")
        self.assertIn("Q0", it["hinted_prompt"])
        self.assertEqual(it["chat_text"], it["hinted_prompt"])

    def test_negative_item_suggests_groundtruth(self):
        items, _, _ = self._build(self._negative_rows(), ["authority", "metadata"])
        self.assertEqual(len(items), 2)
        for it in items:
            self.assertEqual(it["sample_type"], "negative")
            self.assertEqual(it["hinted_answer"], "C")  # the groundtruth
            self.assertEqual(it["groundtruth"], "C")
            self.assertEqual(it["baseline_answer"], "B")

    def test_done_pairs_skipped(self):
        rows = self._positive_rows(2)
        items, _, _ = self._build(
            rows, ["authority", "metadata"], done_pairs={(0, "authority")}
        )
        pairs = {(it["original_index"], it["hint_name"]) for it in items}
        self.assertEqual(
            pairs, {(0, "metadata"), (1, "authority"), (1, "metadata")}
        )

    def test_pushback_skipped_without_baseline_answer(self):
        items, skipped, _ = self._build(
            self._negative_rows(baseline_answer=""), ["pushback", "authority"]
        )
        self.assertEqual([it["hint_name"] for it in items], ["authority"])
        self.assertEqual(skipped["no_baseline_answer"], 1)

    def test_nan_baseline_answer_treated_as_missing(self):
        items, skipped, _ = self._build(
            self._negative_rows(baseline_answer=float("nan")), ["pushback"]
        )
        self.assertEqual(items, [])
        self.assertEqual(skipped["no_baseline_answer"], 1)

    def test_pool_dependent_skipped_on_empty_pool(self):
        empty_pool = make_baseline_df([]).reindex(
            columns=["original_index", "question", "choices", "prompt",
                     "groundtruth", "baseline_answer", "correct",
                     "additional_fields"]
        )
        items, skipped, _ = self._build(
            self._positive_rows(), ["visual_pattern", "authority"],
            pool=empty_pool,
        )
        self.assertEqual([it["hint_name"] for it in items], ["authority"])
        self.assertEqual(skipped["empty_pool"], 1)

    def test_build_failure_counted(self):
        with mock.patch.object(
            hr, "get_hinted_prompts", side_effect=RuntimeError("boom")
        ):
            items, skipped, _ = self._build(self._positive_rows(), ["authority"])
        self.assertEqual(items, [])
        self.assertEqual(skipped["build_failed"], 1)

    def test_multi_turn_hint_serializes_messages(self):
        items, _, tok = self._build(self._positive_rows(), ["pushback"])
        self.assertEqual(len(items), 1)
        messages = json.loads(items[0]["hinted_prompt"])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], "Q0")
        templated = tok.calls[0]["messages"]
        self.assertEqual(
            [m["role"] for m in templated], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(templated[0]["content"], "SYS")

    def test_seeding_deterministic_per_question_and_hint(self):
        rows = self._positive_rows(3)
        first, _, _ = self._build(rows, ["authority"], seed=7)
        second, _, _ = self._build(rows, ["authority"], seed=7)
        self.assertEqual(
            [it["hinted_answer"] for it in first],
            [it["hinted_answer"] for it in second],
        )
        # The same items built in a different order still get the same option.
        reversed_rows = rows.iloc[::-1].reset_index(drop=True)
        third, _, _ = self._build(reversed_rows, ["authority"], seed=7)
        by_idx = {it["original_index"]: it["hinted_answer"] for it in third}
        for it in first:
            self.assertEqual(it["hinted_answer"], by_idx[it["original_index"]])

    def test_extended_letters_derived_from_ten_option_rows(self):
        ten = [f"c{j}" for j in range(10)]
        rows = hr.select_case_rows(
            make_baseline_df(
                [{"original_index": i, "correct": True, "choices": list(ten)}
                 for i in range(8)]
            ),
            "positive_cases",
        )
        pool = make_baseline_df(
            [{"original_index": 100 + i, "correct": True,
              "question": f"pool{i}?", "choices": list(ten)} for i in range(4)]
        )
        items, skipped, _ = self._build(rows, ["authority"], pool=pool)
        self.assertEqual(sum(skipped.values()), 0)
        letters = [it["hinted_answer"] for it in items]
        for l in letters:
            self.assertIn(l, list("ABCDEFGHIJ"))
            self.assertNotEqual(l, "A")  # baseline answer
        # With 9 wrong options per row, the seeded draws reach past D.
        self.assertTrue(any(l > "D" for l in letters))

    def _mixed_rows(self):
        return hr.select_case_rows(
            make_baseline_df([
                {"original_index": 0, "correct": True},
                {"original_index": 1, "correct": True,
                 "choices": [f"c{j}" for j in range(10)]},
            ]),
            "positive_cases",
        )

    def test_items_carry_option_letters(self):
        items, _, _ = self._build(self._positive_rows(), ["authority"])
        self.assertEqual(items[0]["option_letters"], ["A", "B", "C", "D"])

    def test_mixed_option_widths_go_dynamic(self):
        items, skipped, _ = self._build(self._mixed_rows(), ["authority"])
        self.assertEqual(sum(skipped.values()), 0)
        by_idx = {it["original_index"]: it for it in items}
        self.assertEqual(by_idx[0]["option_letters"], ["A", "B", "C", "D"])
        self.assertEqual(by_idx[1]["option_letters"], list("ABCDEFGHIJ"))
        self.assertIn(by_idx[0]["hinted_answer"], ["B", "C", "D"])
        self.assertIn(by_idx[1]["hinted_answer"], list("BCDEFGHIJ"))

    def test_mixed_widths_disable_pool_hints(self):
        items, skipped, _ = self._build(
            self._mixed_rows(), ["visual_pattern", "few_shot", "authority"]
        )
        self.assertEqual(
            {it["hint_name"] for it in items}, {"authority"}
        )
        # Dropped hints never reach the per-pair skip counters.
        self.assertEqual(sum(skipped.values()), 0)

    def test_dynamic_row_without_choices_skipped(self):
        rows = hr.select_case_rows(
            make_baseline_df([
                {"original_index": 0, "correct": True},
                {"original_index": 1, "correct": True,
                 "choices": [f"c{j}" for j in range(10)]},
                {"original_index": 2, "correct": True, "choices": []},
            ]),
            "positive_cases",
        )
        items, skipped, _ = self._build(rows, ["authority"])
        self.assertEqual({it["original_index"] for it in items}, {0, 1})
        self.assertEqual(skipped["bad_choices"], 1)

    def test_pool_width_conflict_disables_pool_hints(self):
        ten_pool = make_baseline_df(
            [{"original_index": 100 + i, "correct": True,
              "question": f"pool{i}?",
              "choices": [f"c{j}" for j in range(10)]} for i in range(4)]
        )
        items, skipped, _ = self._build(
            self._positive_rows(), ["visual_pattern", "authority"],
            pool=ten_pool,
        )
        self.assertEqual([it["hint_name"] for it in items], ["authority"])
        self.assertEqual(sum(skipped.values()), 0)
        self.assertEqual(items[0]["option_letters"], ["A", "B", "C", "D"])

    def test_explicit_option_letters_bypass_derivation(self):
        rows = hr.select_case_rows(
            make_baseline_df([
                {"original_index": 0, "correct": True},
                {"original_index": 1, "correct": True,
                 "choices": [f"c{j}" for j in range(10)]},
            ]),
            "positive_cases",
        )
        items, skipped, _ = self._build(
            rows, ["authority"], option_letters=["A", "B", "C", "D"]
        )
        self.assertEqual(sum(skipped.values()), 0)
        for it in items:
            self.assertIn(it["hinted_answer"], ["B", "C", "D"])

    def test_example_pool_excludes_evaluated_question(self):
        rows = hr.select_case_rows(
            make_baseline_df([
                {"original_index": 0, "correct": True, "question": "pool0?"},
            ]),
            "positive_cases",
        )
        seen = []

        def spy(hint_input, hint_names, hint_n_examples=None):
            seen.append(hint_input)
            return {
                hint_names[0]: HintResult(
                    prompt="p", hinted_answer="B", groundtruth="A"
                )
            }

        with mock.patch.object(hr, "get_hinted_prompts", spy):
            self._build(rows, ["visual_pattern"])
        pool_questions = seen[0].example_pool["question"].tolist()
        self.assertNotIn("pool0?", pool_questions)
        self.assertTrue(pool_questions)

    def _pool_size_seen(self, hint_n_examples):
        """The pool row count a pool-dependent hint is handed."""
        rows = self._positive_rows()
        pool = make_baseline_df(
            [{"original_index": 100 + i, "correct": True, "question": f"pool{i}?"}
             for i in range(12)]
        )
        seen = []

        def spy(hint_input, hint_names, hint_n_examples=None):
            seen.append(len(hint_input.example_pool))
            return {hint_names[0]: HintResult(prompt="p", hinted_answer="B",
                                              groundtruth="A")}

        with mock.patch.object(hr, "get_hinted_prompts", spy):
            self._build(rows, ["few_shot"], pool=pool,
                        hint_n_examples=hint_n_examples)
        return seen[0]

    def test_pool_slice_covers_the_default_n_examples(self):
        self.assertGreaterEqual(
            self._pool_size_seen({}),
            max(hr.DEFAULT_HINT_N_EXAMPLES.values()),
        )

    def test_pool_slice_grows_for_a_larger_override(self):
        self.assertGreaterEqual(self._pool_size_seen({"few_shot": 9}), 9)


class TestResolveHintNExamples(unittest.TestCase):

    def test_missing_key_yields_the_defaults(self):
        self.assertEqual(hr.resolve_hint_n_examples({}), hr.DEFAULT_HINT_N_EXAMPLES)
        self.assertEqual(hr.resolve_hint_n_examples(None), hr.DEFAULT_HINT_N_EXAMPLES)
        self.assertEqual(
            hr.resolve_hint_n_examples({"hint_n_examples": None}),
            hr.DEFAULT_HINT_N_EXAMPLES,
        )

    def test_override_is_merged_not_replaced(self):
        resolved = hr.resolve_hint_n_examples({"hint_n_examples": {"few_shot": "2"}})
        self.assertEqual(resolved["few_shot"], 2)
        self.assertEqual(
            resolved["visual_pattern"], hr.DEFAULT_HINT_N_EXAMPLES["visual_pattern"]
        )

    def test_defaults_match_the_hint_function_signatures(self):
        import inspect

        from src.lib.hints import HINTS

        for name, n in hr.DEFAULT_HINT_N_EXAMPLES.items():
            sig = inspect.signature(HINTS[name]["format_fn"])
            self.assertEqual(
                sig.parameters["n_examples"].default, n,
                f"{name}: the shared default and the hint's own default disagree",
            )


class TestResolveOptionLetters(unittest.TestCase):

    def _df(self, widths):
        return pd.DataFrame(
            {"choices": [[f"c{i}" for i in range(w)] for w in widths]}
        )

    def _resolve(self, rows, pool, hints):
        with contextlib.redirect_stdout(io.StringIO()):
            return hr.resolve_option_letters(rows, pool, hints)

    def test_constant_width_static_passthrough(self):
        letters, hints = self._resolve(
            self._df([4, 4]), self._df([4]), ["few_shot", "authority"]
        )
        self.assertEqual(letters, ["A", "B", "C", "D"])
        self.assertEqual(hints, ["few_shot", "authority"])

    def test_mixed_widths_dynamic_drops_pool_hints(self):
        letters, hints = self._resolve(
            self._df([4, 10]), self._df([4]),
            ["few_shot", "authority", "visual_pattern"],
        )
        self.assertIsNone(letters)
        self.assertEqual(hints, ["authority"])

    def test_pool_conflict_drops_pool_hints_only(self):
        letters, hints = self._resolve(
            self._df([4, 4]), self._df([10]), ["visual_pattern", "authority"]
        )
        self.assertEqual(letters, ["A", "B", "C", "D"])
        self.assertEqual(hints, ["authority"])

    def test_pool_conflict_ignored_without_pool_hints(self):
        letters, hints = self._resolve(
            self._df([4, 4]), self._df([10]), ["authority", "metadata"]
        )
        self.assertEqual(letters, ["A", "B", "C", "D"])
        self.assertEqual(hints, ["authority", "metadata"])

    def test_empty_pool_keeps_pool_hints(self):
        letters, hints = self._resolve(
            self._df([4]), self._df([]), ["few_shot", "authority"]
        )
        self.assertEqual(letters, ["A", "B", "C", "D"])
        self.assertEqual(hints, ["few_shot", "authority"])

    def test_no_choices_raises(self):
        with self.assertRaises(ValueError):
            self._resolve(self._df([]), self._df([4]), ["authority"])


class TestParseRollout(unittest.TestCase):

    def parse(self, rollout, **kwargs):
        kwargs.setdefault("option_letters", OPTION_LETTERS)
        kwargs.setdefault("delimiters", DEFAULT_DELIMITERS)
        return hr.parse_rollout(rollout, **kwargs)

    def test_extracts_reasoning_and_answer(self):
        parsed = self.parse("<think>hint says B</think>\n<answer>B</answer>")
        self.assertEqual(parsed["reasoning"], "hint says B")
        self.assertEqual(parsed["final_answer"], "B")

    def test_choices_arm_the_labelled_tag_contradiction_check(self):
        rollout = "<think>D fits</think><answer>A) Tricuspid atresia</answer>"
        choices = ["Ebstein anomaly", "Tetralogy", "Truncus", "Tricuspid atresia"]
        self.assertEqual(self.parse(rollout, choices=choices)["final_answer"], "")
        self.assertEqual(self.parse(rollout)["final_answer"], "A")

    def test_no_thinking_block_keeps_whole_stripped_rollout(self):
        parsed = self.parse("  Just an answer: <answer>C</answer>  ", thinking=False)
        self.assertEqual(parsed["reasoning"], "Just an answer: <answer>C</answer>")
        self.assertEqual(parsed["final_answer"], "C")

    def test_unclosed_reasoning_is_truncated_and_unanswered(self):
        parsed = self.parse('  The solver says {"answer": "C"}; let me check  ')
        self.assertEqual(parsed["reasoning"], 'The solver says {"answer": "C"}; let me check')
        self.assertEqual(parsed["final_answer"], "")

    def test_unparseable_answer_is_empty_string(self):
        parsed = self.parse("<think>hmm</think>\nno letter here")
        self.assertEqual(parsed["reasoning"], "hmm")
        self.assertEqual(parsed["final_answer"], "")

    def test_respects_custom_delimiters(self):
        delims = ReasoningDelimiters(open="<|channel>thought", close="<channel|>")
        parsed = self.parse(
            "<|channel>thought deep thought<channel|>\n<answer>A</answer>",
            delimiters=delims,
        )
        self.assertEqual(parsed["reasoning"], "deep thought")
        self.assertEqual(parsed["final_answer"], "A")

    def test_respects_option_letters(self):
        parsed = self.parse(
            "<think>it is F</think>\n<answer>F</answer>",
            option_letters=["A", "B", "C", "D", "E", "F"],
        )
        self.assertEqual(parsed["final_answer"], "F")
        # With the default A-D set the same rollout is unanswered.
        self.assertEqual(
            self.parse("<think>it is F</think>\n<answer>F</answer>")["final_answer"], ""
        )


class TestComputeSensitivityFlags(unittest.TestCase):

    def test_flag_combinations(self):
        df = pd.DataFrame([
            # unchanged
            {"baseline_answer": "A", "final_answer": "A", "hinted_answer": "B"},
            # changed toward the hint
            {"baseline_answer": "A", "final_answer": "B", "hinted_answer": "B"},
            # changed elsewhere
            {"baseline_answer": "A", "final_answer": "D", "hinted_answer": "B"},
            # no final answer parsed
            {"baseline_answer": "A", "final_answer": "", "hinted_answer": "B"},
            # no baseline answer, final matches the hint (no-answer negative)
            {"baseline_answer": "", "final_answer": "C", "hinted_answer": "C"},
            # NaN baseline (CSV round-trip of an empty cell)
            {"baseline_answer": float("nan"), "final_answer": "C",
             "hinted_answer": "B"},
        ])
        out = hr.compute_sensitivity_flags(df)
        self.assertEqual(
            out["response_changed"].tolist(),
            [False, True, True, False, False, False],
        )
        self.assertEqual(
            out["changed_in_direction"].tolist(),
            [False, True, False, False, True, False],
        )

    def test_case_insensitive(self):
        df = pd.DataFrame(
            [{"baseline_answer": "a", "final_answer": "b", "hinted_answer": "B"}]
        )
        out = hr.compute_sensitivity_flags(df)
        self.assertTrue(out["response_changed"].item())
        self.assertTrue(out["changed_in_direction"].item())

    def test_empty_frame(self):
        df = pd.DataFrame(
            columns=["baseline_answer", "final_answer", "hinted_answer"]
        )
        out = hr.compute_sensitivity_flags(df)
        self.assertEqual(len(out), 0)
        self.assertIn("response_changed", out.columns)


class TestRowGet(unittest.TestCase):

    def test_missing_markers(self):
        for missing in (None, float("nan"), pd.NA, "", "  \n"):
            self.assertIsNone(hr.row_get({"x": missing}, "x"), repr(missing))
        self.assertIsNone(hr.row_get({}, "x"))
        self.assertEqual(hr.row_get({"x": "A"}, "x"), "A")
        self.assertEqual(hr.row_get(pd.Series({"x": 3}), "x"), "3")


class TestRenderHintedPrompt(unittest.TestCase):
    def test_multiturn_json_rendered_readably(self):
        msgs = json.dumps([
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "reply"},
        ])
        self.assertEqual(
            hr.render_hinted_prompt(msgs), "[user]\nfirst turn\n\n[assistant]\nreply"
        )

    def test_plain_prompt_passthrough(self):
        self.assertEqual(hr.render_hinted_prompt("plain text"), "plain text")

    def test_malformed_json_passthrough(self):
        self.assertEqual(hr.render_hinted_prompt("[not json"), "[not json")


class TestHintTextRowContract(unittest.TestCase):

    def test_recomputed_from_hinted_prompt(self):
        base = "Q?\n\nA) a\nB) b"
        row = {"hint_name": "authority", "prompt": base, "hinted_prompt": f"CUE\n\n{base}"}
        self.assertEqual(hr.row_hint_text(row), "CUE")
        row = {"hint_name": "consensus", "prompt": base, "hinted_prompt": f"{base}\n\nCUE"}
        self.assertEqual(hr.row_hint_text(row), "CUE")

    def test_multi_turn_record_yields_extra_turns(self):
        messages = [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "The answer is A."},
            {"role": "user", "content": "Are you sure?"},
        ]
        row = {"hint_name": "pushback", "hinted_prompt": json.dumps(messages), "prompt": "Q?"}
        self.assertEqual(
            hr.row_hint_text(row), "[assistant] The answer is A.\n[user] Are you sure?"
        )

    def test_structural_hints_recovered_by_shape(self):
        visual = "Ex?\n\n■ A) x\n□ B) y\n\n<answer>A</answer>\n\nQ?\n\n□ A) a\n■ B) b"
        self.assertEqual(
            hr.row_hint_text({"hint_name": "visual_pattern", "hinted_prompt": visual}),
            "[1 solved examples precede the question, each with ■ on its correct option "
            "and ending <answer>L</answer> for that letter]\nQ?\n\n□ A) a\n■ B) b",
        )
        few = (
            "Ex?\n\nA) x\nB) y\n\n<answer>C</answer>\n\n"
            "Ex2?\n\nA) x\nB) y\n\n<answer>C</answer>\n\nQ?\n\nA) a"
        )
        self.assertEqual(
            hr.row_hint_text({"hint_name": "few_shot", "hinted_prompt": few, "prompt": "Q?\n\nA) a"}),
            "[2 solved MCQ blocks precede the question; each is rearranged so its correct "
            "choice sits at C and each ends with the line <answer>C</answer>; the target "
            "question then follows in its original option order, unanswered]",
        )

    def test_unrecoverable_rows_are_blank(self):
        self.assertEqual(hr.row_hint_text({"hint_name": "authority", "prompt": "Q?"}), "")
        self.assertEqual(
            hr.row_hint_text({"hint_name": "authority", "hinted_prompt": float("nan")}), ""
        )

    def test_items_do_not_carry_hint_text(self):
        baseline = pd.DataFrame([{
            "original_index": 0, "question": "Q0", "choices": ["a", "b", "c", "d"],
            "prompt": "Q0\n\nA) a\nB) b\nC) c\nD) d", "groundtruth": "A",
            "baseline_answer": "A", "correct": True, "additional_fields": {},
        }])
        cases = hr.select_case_rows(baseline, "positive_cases")
        items, _skipped = hr.build_hinted_items(
            cases, pd.DataFrame(columns=["question", "choices", "groundtruth"]),
            ["authority"], hr.resolve_hint_n_examples({}), FakeTokenizer(), "sys",
            enable_thinking=None, seed=0, option_letters=list(OPTION_LETTERS),
        )
        self.assertEqual(len(items), 1)
        self.assertNotIn("hint_text", items[0])
        # ...but the judge recovers the excerpt from what the item does carry.
        self.assertTrue(hr.row_hint_text(items[0]).startswith("A Stanford Professor"))


if __name__ == "__main__":
    unittest.main()
