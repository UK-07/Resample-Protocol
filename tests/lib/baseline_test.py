import json
import os
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from src.lib.baseline import (
    build_verified_pool,
    collect_choice_widths,
    decode_additional_fields,
    encode_additional_fields,
    letters_for_width,
    load_baseline,
    load_exclusion_list,
    load_sample_questions,
    resolve_option_letters,
    resolve_thinking,
    row_choices,
    row_option_letters,
    threshold_vote,
)


def _make_baseline_df():
    return pd.DataFrame(
        {
            "original_index": [0, 1, 2, 3, 4],
            "question": ["q0", "q1", "q2", "q3", "q4"],
            "choices": [
                json.dumps(["a0", "b0", "c0", "d0"]),
                json.dumps(["a1", "b1", "c1", "d1"]),
                json.dumps(["a2", "b2", "c2", "d2"]),
                json.dumps(["a3", "b3", "c3", "d3"]),
                None,
            ],
            "prompt": ["p0", "p1", "p2", "p3", "p4"],
            "groundtruth": ["A", "B", "C", "D", "A"],
            "baseline_answer": ["A", "C", "C", "D", "A"],
            "correct": [True, False, True, True, True],
            "additional_fields": [
                encode_additional_fields({"subject": "philosophy"}),
                encode_additional_fields({"subject": "philosophy"}),
                encode_additional_fields({"subject": "astronomy"}),
                encode_additional_fields(
                    {"subset": "gpqa_diamond", "domain": "Physics"}
                ),
                encode_additional_fields({}),
            ],
        }
    )


def _write_csv(df, tmpdir):
    path = os.path.join(tmpdir, "baseline.csv")
    df.to_csv(path, index=False)
    return path


class TestAdditionalFieldsCodec(unittest.TestCase):
    """Tests for encode_additional_fields / decode_additional_fields."""

    def test_encode_empty_dict_is_empty_string(self):
        self.assertEqual(encode_additional_fields({}), "")

    def test_encode_none_is_empty_string(self):
        self.assertEqual(encode_additional_fields(None), "")

    def test_encode_is_compact_json(self):
        encoded = encode_additional_fields({"subject": "philosophy", "k": "v"})
        self.assertNotIn(": ", encoded)
        self.assertNotIn(", ", encoded)

    def test_decode_inverts_encode(self):
        fields = {"subject": "philosophy", "domain": "Physics"}
        self.assertEqual(decode_additional_fields(encode_additional_fields(fields)), fields)

    def test_decode_empty_string_is_empty_dict(self):
        self.assertEqual(decode_additional_fields(""), {})

    def test_decode_whitespace_is_empty_dict(self):
        self.assertEqual(decode_additional_fields("   "), {})

    def test_decode_nan_is_empty_dict(self):
        self.assertEqual(decode_additional_fields(float("nan")), {})

    def test_decode_passes_through_dict(self):
        fields = {"subject": "astronomy"}
        self.assertIs(decode_additional_fields(fields), fields)

    def test_decode_preserves_unicode(self):
        fields = {"subject": "философия"}
        encoded = encode_additional_fields(fields)
        self.assertIn("философия", encoded)
        self.assertEqual(decode_additional_fields(encoded), fields)


class TestThresholdVote(unittest.TestCase):
    """Tests for threshold_vote."""

    def test_empty_list(self):
        self.assertEqual(threshold_vote([], 1), ("", 0, 0.0))

    def test_all_unparseable(self):
        self.assertEqual(threshold_vote(["", "", ""], 1), ("", 0, 0.0))

    def test_majority_accepted(self):
        accepted, top, sc = threshold_vote(["A", "B", "A"], 2)
        self.assertEqual(accepted, "A")
        self.assertEqual(top, 2)
        self.assertAlmostEqual(sc, 2 / 3)

    def test_below_threshold_rejected(self):
        accepted, top, sc = threshold_vote(["A", "B", "C"], 2)
        self.assertEqual(accepted, "")
        self.assertEqual(top, 1)
        self.assertAlmostEqual(sc, 1 / 3)

    def test_tie_breaks_lexicographically(self):
        accepted, top, _ = threshold_vote(["D", "B", "D", "B"], 2)
        self.assertEqual(accepted, "B")
        self.assertEqual(top, 2)

    def test_empty_answers_count_in_denominator(self):
        accepted, top, sc = threshold_vote(["A", "A", "", ""], 2)
        self.assertEqual(accepted, "A")
        self.assertEqual(top, 2)
        self.assertAlmostEqual(sc, 0.5)

    def test_unanimous_self_consistency(self):
        accepted, top, sc = threshold_vote(["C", "C", "C"], 3)
        self.assertEqual(accepted, "C")
        self.assertEqual(top, 3)
        self.assertAlmostEqual(sc, 1.0)


class TestLoadBaseline(unittest.TestCase):
    """Tests for load_baseline."""

    def _round_trip(self):
        df = _make_baseline_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_csv(df, tmpdir)
            return load_baseline(path)

    def test_choices_json_decoded(self):
        loaded = self._round_trip()
        self.assertEqual(loaded.loc[0, "choices"], ["a0", "b0", "c0", "d0"])
        self.assertEqual(loaded.loc[3, "choices"], ["a3", "b3", "c3", "d3"])

    def test_missing_choices_cell_becomes_empty_list(self):
        loaded = self._round_trip()
        self.assertEqual(loaded.loc[4, "choices"], [])

    def test_additional_fields_json_decoded(self):
        loaded = self._round_trip()
        self.assertEqual(loaded.loc[0, "additional_fields"], {"subject": "philosophy"})
        self.assertEqual(
            loaded.loc[3, "additional_fields"],
            {"subset": "gpqa_diamond", "domain": "Physics"},
        )

    def test_empty_additional_fields_cell_becomes_empty_dict(self):
        loaded = self._round_trip()
        self.assertEqual(loaded.loc[4, "additional_fields"], {})

    def test_correct_column_round_trips_as_bool(self):
        loaded = self._round_trip()
        self.assertTrue(bool(loaded.loc[0, "correct"]))
        self.assertFalse(bool(loaded.loc[1, "correct"]))

    def test_all_columns_present(self):
        loaded = self._round_trip()
        expected = {
            "original_index",
            "question",
            "choices",
            "prompt",
            "groundtruth",
            "baseline_answer",
            "correct",
            "additional_fields",
        }
        self.assertTrue(expected.issubset(set(loaded.columns)))

    def test_missing_choices_column_ok(self):
        df = pd.DataFrame(
            {"question": ["q0"], "correct": [True], "additional_fields": [""]}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_csv(df, tmpdir)
            loaded = load_baseline(path)
        self.assertEqual(len(loaded), 1)

    def test_missing_additional_fields_column_raises(self):
        df = pd.DataFrame(
            {"question": ["q0"], "subject": ["philosophy"], "correct": [True]}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_csv(df, tmpdir)
            with self.assertRaises(ValueError) as ctx:
                load_baseline(path)
        self.assertIn("additional_fields", str(ctx.exception))
        self.assertIn("compute_baseline.py", str(ctx.exception))


class TestBuildVerifiedPool(unittest.TestCase):
    """Tests for build_verified_pool."""

    def _pool_input(self):
        return pd.DataFrame(
            {
                "question": ["q0", "q1", "q2", "q3", "q4"],
                "correct": [True, False, True, True, False],
                "additional_fields": [
                    {"subject": "s1"},
                    {"subject": "s1"},
                    {"subject": "s2"},
                    {"subject": "s2"},
                    {"subject": "s2"},
                ],
            }
        )

    def test_only_correct_rows_kept(self):
        pool = build_verified_pool(self._pool_input())
        self.assertEqual(set(pool["question"]), {"q0", "q2", "q3"})

    def test_returns_flat_dataframe(self):
        pool = build_verified_pool(self._pool_input())
        self.assertIsInstance(pool, pd.DataFrame)
        self.assertEqual(len(pool), 3)

    def test_excluded_questions_removed(self):
        pool = build_verified_pool(self._pool_input(), excluded_questions={"q2"})
        self.assertEqual(set(pool["question"]), {"q0", "q3"})

    def test_exclusion_can_empty_the_pool(self):
        pool = build_verified_pool(
            self._pool_input(), excluded_questions={"q0", "q2", "q3"}
        )
        self.assertEqual(len(pool), 0)

    def test_no_exclusions_by_default(self):
        pool = build_verified_pool(self._pool_input())
        self.assertEqual(len(pool), 3)

    def test_empty_exclusion_set_keeps_all(self):
        pool = build_verified_pool(self._pool_input(), excluded_questions=set())
        self.assertEqual(len(pool), 3)

    def test_index_reset(self):
        pool = build_verified_pool(self._pool_input(), excluded_questions={"q0"})
        self.assertEqual(list(pool.index), list(range(len(pool))))

    def test_integer_correct_column(self):
        df = self._pool_input()
        df["correct"] = [1, 0, 1, 1, 0]
        pool = build_verified_pool(df)
        self.assertEqual(len(pool), 3)

    def test_input_not_mutated(self):
        df = self._pool_input()
        snapshot = df.copy(deep=True)
        build_verified_pool(df, excluded_questions={"q0"})
        pd.testing.assert_frame_equal(df, snapshot)


class TestLoadSampleQuestions(unittest.TestCase):
    """Tests for load_sample_questions."""

    def _write(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_decodes_choices_and_keeps_columns(self):
        path = self._write([
            {"original_index": 5, "question": "q5",
             "choices": json.dumps(["a", "b", "c", "d"]), "groundtruth": "B"},
        ])
        try:
            df = load_sample_questions(path)
        finally:
            os.remove(path)
        self.assertEqual(df.loc[0, "choices"], ["a", "b", "c", "d"])
        self.assertEqual(df.loc[0, "groundtruth"], "B")
        self.assertEqual(df.loc[0, "question"], "q5")

    def test_missing_column_raises(self):
        path = self._write([{"question": "q", "groundtruth": "A"}])  # no choices
        try:
            with self.assertRaises(ValueError):
                load_sample_questions(path)
        finally:
            os.remove(path)


class TestCollectChoiceWidths(unittest.TestCase):
    """Tests for collect_choice_widths."""

    @staticmethod
    def _df(widths):
        return pd.DataFrame(
            {"choices": [[f"c{i}" for i in range(w)] if w else None for w in widths]}
        )

    def test_widths_pooled_across_frames(self):
        self.assertEqual(
            collect_choice_widths(self._df([4, 4]), self._df([10])), {4, 10}
        )

    def test_empty_cells_and_missing_column_ignored(self):
        df = pd.DataFrame({"choices": [["a", "b"], None, []]})
        self.assertEqual(
            collect_choice_widths(df, pd.DataFrame({"question": ["q"]}), None),
            {2},
        )

    def test_no_usable_rows_is_empty_set(self):
        self.assertEqual(
            collect_choice_widths(pd.DataFrame({"choices": [None, []]})), set()
        )

    def test_undecoded_choices_raise(self):
        df = pd.DataFrame({"choices": [json.dumps(["a", "b"])]})
        with self.assertRaises(ValueError) as ctx:
            collect_choice_widths(df)
        self.assertIn("not JSON-decoded", str(ctx.exception))


class TestLettersForWidth(unittest.TestCase):
    """Tests for letters_for_width."""

    def test_four_and_ten(self):
        self.assertEqual(letters_for_width(4), ["A", "B", "C", "D"])
        self.assertEqual(letters_for_width(10), list("ABCDEFGHIJ"))

    def test_non_positive_raises(self):
        for n in (0, -1):
            with self.assertRaises(ValueError):
                letters_for_width(n)

    def test_beyond_letters_raises(self):
        with self.assertRaises(ValueError):
            letters_for_width(11)


class TestLoadExclusionList(unittest.TestCase):
    """Tests for load_exclusion_list."""

    def test_none_and_missing_return_empty(self):
        self.assertEqual(load_exclusion_list(None), set())
        self.assertEqual(load_exclusion_list("/no/such/file.json"), set())

    def test_reads_int_set(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump([3, 7, 11], f)
        try:
            self.assertEqual(load_exclusion_list(path), {3, 7, 11})
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()


class TestSidecarAndRowHelpers(unittest.TestCase):
    def test_row_option_letters_follow_the_row_width(self):
        self.assertEqual(["A", "B", "C", "D"], row_option_letters({"choices": json.dumps(list("wxyz"))}))
        self.assertEqual(list("ABCDE"), row_option_letters({"choices": json.dumps(["a", "b", "c", "d", "e"])}))
        self.assertEqual(list("ABCDEFGHIJ"), row_option_letters({"choices": json.dumps([str(i) for i in range(10)])}))

    def test_missing_or_unusable_choices_give_none(self):
        for row in ({}, {"choices": ""}, {"choices": "not json"}, {"choices": json.dumps([])}):
            self.assertIsNone(row_option_letters(row))
            self.assertIsNone(row_choices(row))

    def test_row_choices_are_the_option_texts(self):
        self.assertEqual(["w", "x", "y", "z"], row_choices({"choices": json.dumps(list("wxyz"))}))

    def test_sidecar_thinking_and_letters(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "b.meta.json"
            self.assertTrue(resolve_thinking(meta))
            self.assertIsNone(resolve_option_letters(meta))
            meta.write_text(json.dumps({"thinking": "off", "dataset": {"name": "mmlu_pro"}}))
            self.assertFalse(resolve_thinking(meta))
            self.assertEqual(list("ABCDEFGHIJ"), resolve_option_letters(meta))
            meta.write_text(json.dumps({"thinking": True, "dataset": {"name": "gpqa"}}))
            self.assertTrue(resolve_thinking(meta))
            self.assertIsNone(resolve_option_letters(meta))
