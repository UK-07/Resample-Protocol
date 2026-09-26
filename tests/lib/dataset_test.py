import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.lib.constants import EXTENDED_OPTION_LETTERS
from src.lib.dataset import (
    AQuADataset,
    CommonsenseQADataset,
    DATASET_REGISTRY,
    GPQADataset,
    MedQADataset,
    MMLUDataset,
    MMLUProDataset,
    MMLU_SPLIT_ALIASES,
    MMLU_TRAIN_SPLIT,
    _format_mcq_prompt,
    load_data,
)

EXPECTED_COLUMNS = {
    "prompt",
    "groundtruth",
    "additional_fields",
    "question",
    "choices",
    "answer_idx",
    "original_index",
}


def _mmlu_raw_df():
    return pd.DataFrame(
        {
            "question": ["q0", "q1", "q2", "q3"],
            "subject": ["philosophy", "astronomy", "philosophy", "astronomy"],
            "choices": [
                np.array(["a0", "b0", "c0", "d0"]),
                np.array(["a1", "b1", "c1", "d1"]),
                np.array(["a2", "b2", "c2", "d2"]),
                np.array(["a3", "b3", "c3", "d3"]),
            ],
            "answer": [0, 1, 2, 3],
        }
    )


def _mmlu_train_raw_df():
    df = _mmlu_raw_df()
    df["subject"] = [None, None, None, None]
    return df


def _mmlu_pro_raw_df():
    return pd.DataFrame(
        {
            # question_id is the dataset's own stable global id (not 0-based).
            "question_id": [70, 71, 72, 73],
            "question": ["pq0", "pq1", "pq2", "pq3"],
            "options": [
                np.array([f"o0_{i}" for i in range(10)]),               # full 10
                np.array([f"o1_{i}" for i in range(4)] + ["N/A"] * 6),  # padded
                np.array([f"o2_{i}" for i in range(7)]),                # short
                np.array([f"o3_{i}" for i in range(10)]),
            ],
            "answer": ["I", "B", "G", "A"],
            "answer_index": [8, 1, 6, 0],
            "cot_content": ["", "", "", ""],
            "category": ["math", "law", "math", "physics"],
            "src": ["ori_mmlu-a", "ori_mmlu-b", "stemez-c", "ori_mmlu-a"],
        }
    )


def _gpqa_raw_df():
    return pd.DataFrame(
        {
            "Question": ["gq0", "gq1", "gq2"],
            "Correct Answer": ["right0", "right1", "right2"],
            "Incorrect Answer 1": ["wrong0a", "wrong1a", "wrong2a"],
            "Incorrect Answer 2": ["wrong0b", "wrong1b", "wrong2b"],
            "Incorrect Answer 3": ["wrong0c", "wrong1c", "wrong2c"],
        }
    )


def _gpqa_raw_df_with_domains():
    df = _gpqa_raw_df()
    df["High-level domain"] = ["Physics", "Biology", "  "]
    df["Subdomain"] = ["Optics", "", "Organic Chemistry"]
    return df


def _gpqa_experts_raw_df():
    return pd.DataFrame(
        {
            "Annotator": ["expert0", "expert1"],
            "Qualifications": ["PhD", "PhD"],
        }
    )


def _medqa_raw_df():
    return pd.DataFrame(
        {
            "question": ["mq0", "mq1", "mq2"],
            "answer": ["b0", "d1", "a2"],
            "options": [
                {"A": "a0", "B": "b0", "C": "c0", "D": "d0"},
                {"A": "a1", "B": "b1", "C": "c1", "D": "d1"},
                {"A": "a2", "B": "b2", "C": "c2", "D": "d2"},
            ],
            "meta_info": ["step1", "step2&3", ""],
            "answer_idx": ["B", "D", "A"],
            "metamap_phrases": [["x"], ["y"], ["z"]],
        }
    )


def _mock_load_dataset(raw_df):
    mock = MagicMock()
    mock.return_value.to_pandas.return_value = raw_df.copy(deep=True)
    return mock


class TestMMLUDataset(unittest.TestCase):
    """MMLUDataset.load with the HF load mocked."""

    def test_schema_and_options(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, options = MMLUDataset(config="all", split="test").load()
        self.assertEqual(options, ["A", "B", "C", "D"])
        self.assertEqual(set(df.columns), EXPECTED_COLUMNS)
        self.assertEqual(len(df), 4)

    def test_groundtruth_letter_mapping(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset().load()
        self.assertEqual(list(df["groundtruth"]), ["A", "B", "C", "D"])

    def test_prompt_format(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset().load()
        self.assertEqual(
            df.iloc[0]["prompt"], "q0\n\nA) a0\nB) b0\nC) c0\nD) d0"
        )

    def test_additional_fields_contains_subject(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset().load()
        self.assertEqual(
            list(df["additional_fields"]),
            [
                {"subject": "philosophy"},
                {"subject": "astronomy"},
                {"subject": "philosophy"},
                {"subject": "astronomy"},
            ],
        )

    def test_original_index_stable_under_subject_filter(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset(subject_filter=["astronomy"]).load()
        self.assertEqual(list(df["original_index"]), [1, 3])
        self.assertEqual(list(df["question"]), ["q1", "q3"])

    def test_exclude_subjects(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset(exclude_subjects=["philosophy"]).load()
        self.assertEqual(list(df["original_index"]), [1, 3])
        subjects = {f["subject"] for f in df["additional_fields"]}
        self.assertEqual(subjects, {"astronomy"})

    def test_samples_per_subject_caps_rows(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset(samples_per_subject=1).load()
        self.assertEqual(len(df), 2)
        subjects = sorted(f["subject"] for f in df["additional_fields"])
        self.assertEqual(subjects, ["astronomy", "philosophy"])

    def test_samples_per_subject_larger_than_group_keeps_all(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset(samples_per_subject=10).load()
        self.assertEqual(len(df), 4)

    def test_samples_per_subject_deterministic(self):
        loads = []
        for _ in range(2):
            with patch(
                "src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())
            ):
                df, _ = MMLUDataset(samples_per_subject=1, seed=7).load()
            loads.append(df)
        pd.testing.assert_frame_equal(loads[0], loads[1])

    def test_df_index_equals_original_index(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset(subject_filter=["astronomy"]).load()
        self.assertEqual(list(df.index), list(df["original_index"]))

    def test_train_split_alias_normalized(self):
        mock = _mock_load_dataset(_mmlu_train_raw_df())
        with patch("src.lib.dataset.load_dataset", mock):
            with contextlib.redirect_stdout(io.StringIO()):
                MMLUDataset(config="all", split="train").load()
        mock.assert_called_once_with(
            MMLUDataset.HF_REPO, "all", split=MMLU_TRAIN_SPLIT
        )

    def test_train_split_ignores_subject_filter(self):
        mock = _mock_load_dataset(_mmlu_train_raw_df())
        stdout = io.StringIO()
        with patch("src.lib.dataset.load_dataset", mock):
            with contextlib.redirect_stdout(stdout):
                df, _ = MMLUDataset(
                    split="train", subject_filter=["philosophy"]
                ).load()
        self.assertEqual(len(df), 4)
        self.assertIn("WARNING", stdout.getvalue())

    def test_train_split_ignores_samples_per_subject(self):
        mock = _mock_load_dataset(_mmlu_train_raw_df())
        stdout = io.StringIO()
        with patch("src.lib.dataset.load_dataset", mock):
            with contextlib.redirect_stdout(stdout):
                df, _ = MMLUDataset(split="train", samples_per_subject=1).load()
        self.assertEqual(len(df), 4)
        self.assertIn("WARNING", stdout.getvalue())

    def test_train_split_rows_have_empty_additional_fields(self):
        mock = _mock_load_dataset(_mmlu_train_raw_df())
        with patch("src.lib.dataset.load_dataset", mock):
            df, _ = MMLUDataset(split="auxiliary_train").load()
        self.assertEqual(list(df["additional_fields"]), [{}, {}, {}, {}])

    def test_empty_subject_filter_returns_empty_df(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, options = MMLUDataset(subject_filter=["no_such_subject"]).load()
        self.assertEqual(len(df), 0)
        self.assertEqual(options, ["A", "B", "C", "D"])

    def test_empty_filter_with_samples_per_subject_returns_empty_df(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, options = MMLUDataset(
                subject_filter=["no_such_subject"], samples_per_subject=1
            ).load()
        self.assertEqual(len(df), 0)
        self.assertEqual(options, ["A", "B", "C", "D"])

    def _load_shuffled(self, **kwargs):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            return MMLUDataset(random_answers_order=True, **kwargs).load()

    def test_random_answers_order_choices_are_permutation(self):
        df, _ = self._load_shuffled(seed=42)
        native = _mmlu_raw_df()["choices"]
        reordered_rows = 0
        for i, (_, row) in enumerate(df.iterrows()):
            self.assertEqual(set(row["choices"]), set(native[i]))
            self.assertEqual(len(row["choices"]), 4)
            if list(row["choices"]) != list(native[i]):
                reordered_rows += 1
        self.assertGreater(reordered_rows, 0)

    def test_random_answers_order_groundtruth_tracks_correct_choice(self):
        df, options = self._load_shuffled(seed=42)
        # Native answers are [0, 1, 2, 3] → correct texts a0, b1, c2, d3.
        correct_texts = ["a0", "b1", "c2", "d3"]
        for i, (_, row) in enumerate(df.iterrows()):
            letter_idx = options.index(row["groundtruth"])
            self.assertEqual(letter_idx, row["answer_idx"])
            self.assertEqual(row["choices"][letter_idx], correct_texts[i])

    def test_random_answers_order_metadata(self):
        df, _ = self._load_shuffled(seed=42)
        native = _mmlu_raw_df()["choices"]
        self.assertEqual(
            [f["original_groundtruth"] for f in df["additional_fields"]],
            ["A", "B", "C", "D"],
        )
        for i, (_, row) in enumerate(df.iterrows()):
            order = [int(p) for p in row["additional_fields"]["answers_order"].split(",")]
            self.assertEqual(sorted(order), [0, 1, 2, 3])
            self.assertEqual([native[i][j] for j in order], list(row["choices"]))
            self.assertIn("subject", row["additional_fields"])

    def test_random_answers_order_prompt_matches_shuffled_choices(self):
        df, _ = self._load_shuffled(seed=42)
        for _, row in df.iterrows():
            self.assertEqual(
                row["prompt"], _format_mcq_prompt(row["question"], list(row["choices"]))
            )

    def test_random_answers_order_stable_under_subject_filter(self):
        full, _ = self._load_shuffled(seed=42)
        filtered, _ = self._load_shuffled(seed=42, subject_filter=["astronomy"])
        for idx in filtered["original_index"]:
            self.assertEqual(
                list(full.loc[idx, "choices"]), list(filtered.loc[idx, "choices"])
            )
            self.assertEqual(full.loc[idx, "groundtruth"], filtered.loc[idx, "groundtruth"])

    def test_random_answers_order_deterministic(self):
        df1, _ = self._load_shuffled(seed=7)
        df2, _ = self._load_shuffled(seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_random_answers_order_off_by_default(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = MMLUDataset().load()
        self.assertEqual(list(df.iloc[0]["choices"]), ["a0", "b0", "c0", "d0"])
        for fields in df["additional_fields"]:
            self.assertNotIn("answers_order", fields)
            self.assertNotIn("original_groundtruth", fields)


class TestMMLUProDataset(unittest.TestCase):
    """MMLUProDataset.load with the HF load mocked."""

    def _load(self, raw_df=None, **kwargs):
        raw = raw_df if raw_df is not None else _mmlu_pro_raw_df()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(raw)):
            return MMLUProDataset(**kwargs).load()

    def test_schema_parity_with_mmlu(self):
        pro_df, pro_options = self._load()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            mmlu_df, _ = MMLUDataset().load()
        self.assertEqual(set(pro_df.columns), set(mmlu_df.columns))
        self.assertEqual(pro_options, EXTENDED_OPTION_LETTERS)

    def test_groundtruth_beyond_d(self):
        df, _ = self._load()
        row = df.loc[70]
        self.assertEqual(row["groundtruth"], "I")
        self.assertEqual(row["answer_idx"], 8)
        self.assertEqual(len(row["choices"]), 10)

    def test_na_placeholder_options_dropped(self):
        df, _ = self._load()
        row = df.loc[71]
        self.assertEqual(row["choices"], [f"o1_{i}" for i in range(4)])
        self.assertEqual(row["groundtruth"], "B")

    def test_na_before_answer_remaps_groundtruth(self):
        raw = _mmlu_pro_raw_df()
        # A placeholder *before* the answer position: the real answer "o1_3"
        # sits at raw index 4, behind an N/A at index 2.
        raw.at[1, "options"] = np.array(["o1_0", "o1_1", "N/A", "o1_2", "o1_3"])
        raw.loc[1, "answer_index"] = 4
        df, _ = self._load(raw_df=raw)
        row = df.loc[71]
        self.assertEqual(row["choices"], ["o1_0", "o1_1", "o1_2", "o1_3"])
        self.assertEqual(row["answer_idx"], 3)
        self.assertEqual(row["groundtruth"], "D")

    def test_answer_index_outside_options_raises(self):
        raw = _mmlu_pro_raw_df()
        # 4 real options padded with N/A, but the answer points at position 5.
        raw.loc[1, "answer_index"] = 5
        with self.assertRaises(ValueError) as ctx:
            self._load(raw_df=raw)
        self.assertIn("question_id=71", str(ctx.exception))
        # And an index beyond the raw options list entirely.
        raw.loc[1, "answer_index"] = 20
        with self.assertRaises(ValueError):
            self._load(raw_df=raw)

    def test_original_index_is_question_id(self):
        df, _ = self._load()
        self.assertEqual(list(df["original_index"]), [70, 71, 72, 73])
        self.assertEqual(list(df.index), list(df["original_index"]))

    def test_additional_fields_category_and_src(self):
        df, _ = self._load()
        self.assertEqual(
            df.loc[70]["additional_fields"],
            {"category": "math", "src": "ori_mmlu-a"},
        )

    def test_category_filter(self):
        df, _ = self._load(category_filter=["math"])
        self.assertEqual(list(df["original_index"]), [70, 72])

    def test_exclude_categories(self):
        df, _ = self._load(exclude_categories=["math"])
        self.assertEqual(list(df["original_index"]), [71, 73])

    def test_samples_per_category_caps_rows(self):
        df, _ = self._load(samples_per_category=1)
        # One row per category (math capped from 2 to 1).
        self.assertEqual(len(df), 3)
        cats = sorted(f["category"] for f in df["additional_fields"])
        self.assertEqual(cats, ["law", "math", "physics"])

    def test_require_n_options_keeps_exact_width(self):
        # Widths in the fixture: 10, 4 (after N/A trim), 7, 10.
        df, _ = self._load(require_n_options=10)
        self.assertEqual(list(df["original_index"]), [70, 73])
        df4, _ = self._load(require_n_options=4)
        self.assertEqual(list(df4["original_index"]), [71])

    def test_require_n_options_runs_before_sampling(self):
        # math has one 10-option row (70) and one 7-option row (72); with the
        # width filter applied first, the per-category cap must pick row 70.
        df, _ = self._load(require_n_options=10, samples_per_category=1)
        self.assertIn(70, list(df["original_index"]))
        self.assertNotIn(72, list(df["original_index"]))

    def test_prompt_labels_only_available_options(self):
        df, _ = self._load()
        prompt = df.loc[72]["prompt"]
        self.assertIn("A) o2_0", prompt)
        self.assertIn("G) o2_6", prompt)
        self.assertNotIn("H)", prompt)

    def test_random_answers_order_choices_are_permutation(self):
        base_df, _ = self._load()
        df, _ = self._load(random_answers_order=True, seed=42)
        for idx in df.index:
            self.assertEqual(
                sorted(df.loc[idx]["choices"]), sorted(base_df.loc[idx]["choices"])
            )
            order = [int(i) for i in df.loc[idx]["additional_fields"]["answers_order"].split(",")]
            self.assertEqual(
                df.loc[idx]["choices"],
                [base_df.loc[idx]["choices"][i] for i in order],
            )

    def test_random_answers_order_groundtruth_tracks_correct_choice(self):
        base_df, _ = self._load()
        df, _ = self._load(random_answers_order=True, seed=42)
        for idx in df.index:
            correct_text = base_df.loc[idx]["choices"][base_df.loc[idx]["answer_idx"]]
            shuffled = df.loc[idx]
            self.assertEqual(shuffled["choices"][shuffled["answer_idx"]], correct_text)
            self.assertEqual(
                shuffled["groundtruth"],
                EXTENDED_OPTION_LETTERS[shuffled["answer_idx"]],
            )
            self.assertEqual(
                shuffled["additional_fields"]["original_groundtruth"],
                base_df.loc[idx]["groundtruth"],
            )

    def test_random_answers_order_deterministic(self):
        df1, _ = self._load(random_answers_order=True, seed=42)
        df2, _ = self._load(random_answers_order=True, seed=42)
        for idx in df1.index:
            self.assertEqual(df1.loc[idx]["choices"], df2.loc[idx]["choices"])

    def test_random_answers_order_off_by_default(self):
        df, _ = self._load()
        for fields in df["additional_fields"]:
            self.assertNotIn("answers_order", fields)
            self.assertNotIn("original_groundtruth", fields)

class TestGPQADataset(unittest.TestCase):
    """GPQADataset.load with the HF load mocked."""

    def _load(self, raw_df=None, **kwargs):
        raw = raw_df if raw_df is not None else _gpqa_raw_df()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(raw)):
            return GPQADataset(**kwargs).load()

    def test_schema_parity_with_mmlu(self):
        gpqa_df, gpqa_options = self._load()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            mmlu_df, mmlu_options = MMLUDataset().load()
        self.assertEqual(set(gpqa_df.columns), set(mmlu_df.columns))
        self.assertEqual(gpqa_options, mmlu_options)

    def test_groundtruth_points_at_correct_answer(self):
        df, options = self._load(seed=42)
        for _, row in df.iterrows():
            letter_idx = options.index(row["groundtruth"])
            self.assertEqual(letter_idx, row["answer_idx"])
            self.assertEqual(
                row["choices"][letter_idx], f"right{row['original_index']}"
            )

    def test_choices_are_permutation(self):
        df, _ = self._load(seed=7)
        for _, row in df.iterrows():
            i = row["original_index"]
            expected = {f"right{i}", f"wrong{i}a", f"wrong{i}b", f"wrong{i}c"}
            self.assertEqual(set(row["choices"]), expected)
            self.assertEqual(len(row["choices"]), 4)

    def test_additional_fields_subset_is_config(self):
        df, _ = self._load(config="gpqa_diamond")
        for fields in df["additional_fields"]:
            self.assertEqual(fields, {"subset": "gpqa_diamond"})

    def test_domain_labels_included_when_present(self):
        df, _ = self._load(raw_df=_gpqa_raw_df_with_domains(), config="gpqa_main")
        self.assertEqual(
            list(df["additional_fields"]),
            [
                {"subset": "gpqa_main", "domain": "Physics", "subdomain": "Optics"},
                {"subset": "gpqa_main", "domain": "Biology"},
                {"subset": "gpqa_main", "subdomain": "Organic Chemistry"},
            ],
        )

    def test_non_mcq_subset_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._load(raw_df=_gpqa_experts_raw_df(), config="gpqa_experts")
        self.assertIn("gpqa_experts", str(ctx.exception))

    def test_original_index_sequential(self):
        df, _ = self._load()
        self.assertEqual(list(df["original_index"]), [0, 1, 2])
        self.assertEqual(list(df.index), [0, 1, 2])

    def test_seed_determinism(self):
        df1, _ = self._load(seed=123)
        df2, _ = self._load(seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_can_differ(self):
        orderings = set()
        for seed in range(20):
            df, _ = self._load(seed=seed)
            orderings.add(tuple(df.iloc[0]["choices"]))
        self.assertGreater(len(orderings), 1)

    def test_prompt_format(self):
        df, _ = self._load(seed=42)
        row = df.iloc[0]
        expected = _format_mcq_prompt(row["question"], list(row["choices"]))
        self.assertEqual(row["prompt"], expected)

    def test_random_answers_order_records_permutation(self):
        df, options = self._load(seed=42, random_answers_order=True)
        for _, row in df.iterrows():
            i = row["original_index"]
            source = [f"right{i}", f"wrong{i}a", f"wrong{i}b", f"wrong{i}c"]
            order = [int(p) for p in row["additional_fields"]["answers_order"].split(",")]
            self.assertEqual(sorted(order), [0, 1, 2, 3])
            self.assertEqual([source[j] for j in order], list(row["choices"]))
            letter_idx = options.index(row["groundtruth"])
            self.assertEqual(letter_idx, row["answer_idx"])
            self.assertEqual(row["choices"][letter_idx], f"right{i}")

    def test_random_answers_order_deterministic(self):
        df1, _ = self._load(seed=5, random_answers_order=True)
        df2, _ = self._load(seed=5, random_answers_order=True)
        pd.testing.assert_frame_equal(df1, df2)

    def test_random_answers_order_off_has_no_order_key(self):
        df, _ = self._load(seed=42)
        for fields in df["additional_fields"]:
            self.assertNotIn("answers_order", fields)


class TestFormatMcqPrompt(unittest.TestCase):
    """_format_mcq_prompt layout, custom letters and overflow."""

    def test_layout(self):
        prompt = _format_mcq_prompt("Q?", ["w", "x", "y", "z"])
        self.assertEqual(prompt, "Q?\n\nA) w\nB) x\nC) y\nD) z")

    def test_extended_letters(self):
        prompt = _format_mcq_prompt(
            "Q?", ["v", "w", "x", "y", "z"], letters=EXTENDED_OPTION_LETTERS
        )
        self.assertEqual(prompt, "Q?\n\nA) v\nB) w\nC) x\nD) y\nE) z")

    def test_more_choices_than_letters_raises(self):
        with self.assertRaises(ValueError):
            _format_mcq_prompt("Q?", ["v", "w", "x", "y", "z"])


class TestLoadData(unittest.TestCase):
    """The load_data factory."""

    def test_registry_contents(self):
        self.assertIs(DATASET_REGISTRY["mmlu"], MMLUDataset)
        self.assertIs(DATASET_REGISTRY["mmlu_pro"], MMLUProDataset)
        self.assertIs(DATASET_REGISTRY["gpqa"], GPQADataset)
        self.assertIs(DATASET_REGISTRY["medqa"], MedQADataset)
        self.assertIs(DATASET_REGISTRY["aqua"], AQuADataset)
        self.assertIs(
            DATASET_REGISTRY["commonsense_qa"], CommonsenseQADataset
        )

    def test_dispatches_to_mmlu(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, options = load_data("mmlu")
        self.assertEqual(len(df), 4)
        self.assertEqual(options, ["A", "B", "C", "D"])

    def test_dispatches_to_mmlu_pro(self):
        with patch(
            "src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_pro_raw_df())
        ):
            df, options = load_data("mmlu_pro")
        self.assertEqual(len(df), 4)
        self.assertEqual(options, EXTENDED_OPTION_LETTERS)

    def test_dispatches_to_gpqa(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_gpqa_raw_df())):
            df, options = load_data("gpqa", seed=1)
        self.assertEqual(len(df), 3)
        subsets = {f["subset"] for f in df["additional_fields"]}
        self.assertEqual(subsets, {"gpqa_diamond"})

    def test_name_normalization(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = load_data("  MMLU ")
        self.assertEqual(len(df), 4)

    def test_kwargs_forwarded(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            df, _ = load_data("mmlu", subject_filter=["astronomy"])
        self.assertEqual(list(df["original_index"]), [1, 3])

    def test_dispatches_to_aqua(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_aqua_raw_df())):
            df, options = load_data("aqua", split="test")
        self.assertEqual(len(df), 3)
        self.assertEqual(options, ["A", "B", "C", "D", "E"])

    def test_dispatches_to_commonsense_qa(self):
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_csqa_raw_df())):
            df, options = load_data("commonsense_qa")
        self.assertEqual(len(df), 3)
        self.assertEqual(options, ["A", "B", "C", "D", "E"])

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            load_data("imagenet")
        self.assertIn("mmlu", str(ctx.exception))
        self.assertIn("gpqa", str(ctx.exception))


class TestSplitAliasConstants(unittest.TestCase):
    """The MMLU split alias constants."""

    def test_alias_maps_train(self):
        self.assertEqual(MMLU_SPLIT_ALIASES["train"], MMLU_TRAIN_SPLIT)
        self.assertEqual(MMLU_TRAIN_SPLIT, "auxiliary_train")



class TestMedQADataset(unittest.TestCase):
    """MedQADataset.load with the HF load mocked."""

    def _load(self, raw_df=None, **kwargs):
        raw = raw_df if raw_df is not None else _medqa_raw_df()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(raw)) as mock:
            out = MedQADataset(**kwargs).load()
        self.mock = mock
        return out

    def test_schema_parity_with_mmlu(self):
        df, options = self._load()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            mmlu_df, mmlu_options = MMLUDataset().load()
        self.assertEqual(set(df.columns), set(mmlu_df.columns))
        self.assertEqual(options, mmlu_options)
        self.assertEqual(len(df), 3)

    def test_default_split_is_test_and_no_config_arg(self):
        self._load()
        self.mock.assert_called_once_with(MedQADataset.HF_REPO, split="test")
        self._load(split="train")
        self.mock.assert_called_once_with(MedQADataset.HF_REPO, split="train")

    def test_choices_in_letter_order_and_groundtruth_from_answer_idx(self):
        raw = _medqa_raw_df()
        # Dict order deliberately scrambled — the letter keys must decide.
        raw.at[0, "options"] = {"D": "d0", "B": "b0", "A": "a0", "C": "c0"}
        df, _ = self._load(raw)
        self.assertEqual(df.loc[0, "choices"], ["a0", "b0", "c0", "d0"])
        self.assertEqual(list(df["groundtruth"]), ["B", "D", "A"])
        self.assertEqual(list(df["answer_idx"]), [1, 3, 0])
        self.assertEqual(df.loc[1, "prompt"], "mq1\n\nA) a1\nB) b1\nC) c1\nD) d1")

    def test_additional_fields_carry_the_step_when_present(self):
        df, _ = self._load()
        self.assertEqual(df.loc[0, "additional_fields"], {"step": "step1"})
        self.assertEqual(df.loc[1, "additional_fields"], {"step": "step2&3"})
        self.assertEqual(df.loc[2, "additional_fields"], {})

    def test_original_index_is_row_position_and_survives_drops(self):
        raw = _medqa_raw_df()
        raw.at[1, "options"] = {"A": "a1", "B": "", "C": "c1", "D": "d1"}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [0, 2])
        self.assertEqual(list(df.index), [0, 2])
        self.assertIn("dropped 1", out.getvalue())

    def test_bad_answer_letter_or_missing_column_raises(self):
        raw = _medqa_raw_df()
        raw.at[0, "answer_idx"] = "E"
        with self.assertRaisesRegex(ValueError, "answer_idx"):
            self._load(raw)
        with self.assertRaisesRegex(ValueError, "lacks column"):
            self._load(_medqa_raw_df().drop(columns=["options"]))

    def test_random_answers_order_shuffles_and_records(self):
        plain, _ = self._load()
        df, _ = self._load(seed=3, random_answers_order=True)
        again, _ = self._load(seed=3, random_answers_order=True)
        for i in range(3):
            self.assertEqual(sorted(df.loc[i, "choices"]), sorted(plain.loc[i, "choices"]))
            fields = df.loc[i, "additional_fields"]
            self.assertEqual(fields["original_groundtruth"], plain.loc[i, "groundtruth"])
            order = [int(x) for x in fields["answers_order"].split(",")]
            self.assertEqual(sorted(order), [0, 1, 2, 3])
            self.assertEqual(df.loc[i, "choices"], [plain.loc[i, "choices"][j] for j in order])
            # groundtruth still points at the correct text.
            correct_text = plain.loc[i, "choices"][plain.loc[i, "answer_idx"]]
            self.assertEqual(df.loc[i, "choices"][df.loc[i, "answer_idx"]], correct_text)
            self.assertEqual(df.loc[i, "groundtruth"], "ABCD"[df.loc[i, "answer_idx"]])
            self.assertEqual(list(df.loc[i, "choices"]), list(again.loc[i, "choices"]))
        self.assertNotIn("answers_order", plain.loc[0, "additional_fields"])


def _aqua_raw_df():
    return pd.DataFrame(
        {
            "question": ["aq0", "aq1", "aq2"],
            "options": [
                np.array(["A)1", "B)2", "C)3", "D)4", "E)5"]),
                # Whitespace and a multi-token text: both must survive.
                np.array(["A) 10 km", " B)20 km", "C)30 km", "D)40 km", "E)None of these"]),
                np.array(["A)x", "B)y", "C)z", "D)w", "E)v"]),
            ],
            "rationale": ["r0", "r1", "r2"],
            "correct": ["B", "E", "A"],
        }
    )


class TestAQuADataset(unittest.TestCase):
    """AQuADataset.load with the HF load mocked."""

    def _load(self, raw_df=None, **kwargs):
        raw = raw_df if raw_df is not None else _aqua_raw_df()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(raw)) as mock:
            out = AQuADataset(**kwargs).load()
        self.mock = mock
        return out

    def test_schema_parity_with_mmlu_and_five_letters(self):
        df, options = self._load()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            mmlu_df, _ = MMLUDataset().load()
        self.assertEqual(set(df.columns), set(mmlu_df.columns))
        self.assertEqual(options, ["A", "B", "C", "D", "E"])
        self.assertEqual(options, list(EXTENDED_OPTION_LETTERS[:5]))
        self.assertEqual(len(df), 3)

    def test_default_split_is_train_and_raw_config_pinned(self):
        self._load()
        self.mock.assert_called_once_with(AQuADataset.HF_REPO, "raw", split="train")
        self._load(split="test")
        self.mock.assert_called_once_with(AQuADataset.HF_REPO, "raw", split="test")

    def test_option_labels_stripped_and_prompt_labels_all_five(self):
        df, _ = self._load()
        self.assertEqual(df.loc[1, "choices"], ["10 km", "20 km", "30 km", "40 km", "None of these"])
        self.assertEqual(
            df.loc[0, "prompt"], "aq0\n\nA) 1\nB) 2\nC) 3\nD) 4\nE) 5"
        )

    def test_groundtruth_from_correct_letter(self):
        df, _ = self._load()
        self.assertEqual(list(df["groundtruth"]), ["B", "E", "A"])
        self.assertEqual(list(df["answer_idx"]), [1, 4, 0])
        # groundtruth points at the option text it labelled in the raw row.
        self.assertEqual(df.loc[1, "choices"][df.loc[1, "answer_idx"]], "None of these")

    def test_additional_fields_empty_and_rationale_not_carried(self):
        df, _ = self._load()
        self.assertEqual(list(df["additional_fields"]), [{}, {}, {}])
        self.assertNotIn("rationale", df.columns)
        for prompt in df["prompt"]:
            self.assertNotIn("r0", prompt)

    def test_malformed_options_dropped_not_relabelled(self):
        raw = _aqua_raw_df()
        # Out-of-order labels: relabelling by position would move the answer.
        raw.at[0, "options"] = np.array(["B)1", "A)2", "C)3", "D)4", "E)5"])
        # Four options only.
        raw.at[1, "options"] = np.array(["A)1", "B)2", "C)3", "D)4"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [2])
        self.assertIn("dropped 2 row(s)", out.getvalue())

    def test_repeated_same_letter_label_stripped_once(self):
        # ~1.7% of the real train rows carry the label twice ("A)A)21",
        # "C)c) 36"); an inner label naming a DIFFERENT letter is kept verbatim.
        raw = _aqua_raw_df()
        raw.at[0, "options"] = np.array(["A)A)$60", "B)b) 35", "C)C)60.6", "D)D)21", "E)E)78"])
        raw.at[1, "options"] = np.array(["A)A)11", "B)B)23", "C)C)24", "D)D)29", "E)D)36"])
        df, _ = self._load(raw)
        self.assertEqual(df.loc[0, "choices"], ["$60", "35", "60.6", "21", "78"])
        self.assertEqual(df.loc[1, "choices"], ["11", "23", "24", "29", "D)36"])
        self.assertEqual(df.loc[0, "prompt"].splitlines()[2], "A) $60")

    def test_empty_or_nonstring_option_dropped(self):
        raw = _aqua_raw_df()
        raw.at[0, "options"] = np.array(["A)", "B)2", "C)3", "D)4", "E)5"])
        raw.at[1, "options"] = None
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [2])

    def test_bad_correct_letter_dropped_with_warning(self):
        raw = _aqua_raw_df()
        raw.at[1, "correct"] = "F"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [0, 2])
        self.assertIn("'correct' letter", out.getvalue())

    def test_original_index_is_row_position_and_df_index_matches(self):
        raw = _aqua_raw_df()
        raw.at[1, "options"] = np.array(["A)1", "B)2", "C)3", "D)4"])
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [0, 2])
        self.assertEqual(list(df.index), [0, 2])

    def test_missing_column_raises(self):
        with self.assertRaisesRegex(ValueError, "lacks column"):
            self._load(_aqua_raw_df().drop(columns=["correct"]))

    def test_random_answers_order_shuffles_and_records(self):
        plain, _ = self._load()
        df, _ = self._load(seed=3, random_answers_order=True)
        again, _ = self._load(seed=3, random_answers_order=True)
        for i in range(3):
            self.assertEqual(sorted(df.loc[i, "choices"]), sorted(plain.loc[i, "choices"]))
            fields = df.loc[i, "additional_fields"]
            self.assertEqual(fields["original_groundtruth"], plain.loc[i, "groundtruth"])
            order = [int(x) for x in fields["answers_order"].split(",")]
            self.assertEqual(sorted(order), [0, 1, 2, 3, 4])
            self.assertEqual(df.loc[i, "choices"], [plain.loc[i, "choices"][j] for j in order])
            correct_text = plain.loc[i, "choices"][plain.loc[i, "answer_idx"]]
            self.assertEqual(df.loc[i, "choices"][df.loc[i, "answer_idx"]], correct_text)
            self.assertEqual(df.loc[i, "groundtruth"], "ABCDE"[df.loc[i, "answer_idx"]])
            self.assertEqual(list(df.loc[i, "choices"]), list(again.loc[i, "choices"]))
        self.assertNotIn("answers_order", plain.loc[0, "additional_fields"])


def _csqa_raw_df():
    return pd.DataFrame(
        {
            "id": ["id0", "id1", "id2"],
            "question": ["cq0", "cq1", "cq2"],
            "question_concept": ["door", "work", "  "],
            "choices": [
                {"label": np.array(["A", "B", "C", "D", "E"]),
                 "text": np.array(["a0", "b0", "c0", "d0", "e0"])},
                # Labels out of array order: the LABEL must decide the mapping.
                {"label": np.array(["C", "A", "B", "E", "D"]),
                 "text": np.array(["c1", "a1", "b1", "e1", "d1"])},
                {"label": np.array(["A", "B", "C", "D", "E"]),
                 "text": np.array(["a2", "b2", "c2", "d2", "e2"])},
            ],
            "answerKey": ["A", "E", "C"],
        }
    )


class TestCommonsenseQADataset(unittest.TestCase):
    """CommonsenseQADataset.load with the HF load mocked."""

    def _load(self, raw_df=None, **kwargs):
        raw = raw_df if raw_df is not None else _csqa_raw_df()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(raw)) as mock:
            out = CommonsenseQADataset(**kwargs).load()
        self.mock = mock
        return out

    def test_schema_parity_with_mmlu_and_five_letters(self):
        df, options = self._load()
        with patch("src.lib.dataset.load_dataset", _mock_load_dataset(_mmlu_raw_df())):
            mmlu_df, _ = MMLUDataset().load()
        self.assertEqual(set(df.columns), set(mmlu_df.columns))
        self.assertEqual(options, ["A", "B", "C", "D", "E"])
        self.assertEqual(options, list(EXTENDED_OPTION_LETTERS[:5]))
        self.assertEqual(len(df), 3)

    def test_default_split_is_validation(self):
        self._load()
        self.mock.assert_called_once_with(CommonsenseQADataset.HF_REPO, split="validation")
        self._load(split="train")
        self.mock.assert_called_once_with(CommonsenseQADataset.HF_REPO, split="train")

    def test_test_split_refused_before_any_download(self):
        with patch("src.lib.dataset.load_dataset") as mock:
            for split in ("test", " TEST "):
                with self.assertRaisesRegex(ValueError, "no gold labels"):
                    CommonsenseQADataset(split=split)
            mock.assert_not_called()

    def test_choices_taken_in_label_order(self):
        df, _ = self._load()
        # Row 1's arrays are scrambled; the labels put them back in A-E order.
        self.assertEqual(df.loc[1, "choices"], ["a1", "b1", "c1", "d1", "e1"])
        self.assertEqual(
            df.loc[0, "prompt"], "cq0\n\nA) a0\nB) b0\nC) c0\nD) d0\nE) e0"
        )

    def test_groundtruth_from_answer_key(self):
        df, _ = self._load()
        self.assertEqual(list(df["groundtruth"]), ["A", "E", "C"])
        self.assertEqual(list(df["answer_idx"]), [0, 4, 2])
        self.assertEqual(df.loc[1, "choices"][df.loc[1, "answer_idx"]], "e1")

    def test_additional_fields_carry_question_concept_when_present(self):
        df, _ = self._load()
        self.assertEqual(df.loc[0, "additional_fields"], {"question_concept": "door"})
        self.assertEqual(df.loc[1, "additional_fields"], {"question_concept": "work"})
        self.assertEqual(df.loc[2, "additional_fields"], {})

    def test_malformed_choices_dropped(self):
        raw = _csqa_raw_df()
        # Duplicate label.
        raw.at[0, "choices"] = {"label": np.array(["A", "A", "C", "D", "E"]),
                                "text": np.array(["a", "b", "c", "d", "e"])}
        # Four options.
        raw.at[1, "choices"] = {"label": np.array(["A", "B", "C", "D"]),
                                "text": np.array(["a", "b", "c", "d"])}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [2])
        self.assertIn("dropped 2 row(s)", out.getvalue())

    def test_empty_text_or_missing_cell_dropped(self):
        raw = _csqa_raw_df()
        raw.at[0, "choices"] = {"label": np.array(["A", "B", "C", "D", "E"]),
                                "text": np.array(["", "b", "c", "d", "e"])}
        raw.at[1, "choices"] = None
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [2])

    def test_blank_answer_key_dropped_with_warning(self):
        raw = _csqa_raw_df()
        raw.at[1, "answerKey"] = ""
        with contextlib.redirect_stdout(io.StringIO()) as out:
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [0, 2])
        self.assertIn("answerKey", out.getvalue())

    def test_original_index_is_row_position_and_df_index_matches(self):
        raw = _csqa_raw_df()
        raw.at[1, "answerKey"] = "Z"
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = self._load(raw)
        self.assertEqual(list(df["original_index"]), [0, 2])
        self.assertEqual(list(df.index), [0, 2])

    def test_missing_column_raises(self):
        with self.assertRaisesRegex(ValueError, "lacks column"):
            self._load(_csqa_raw_df().drop(columns=["answerKey"]))

    def test_random_answers_order_shuffles_and_records(self):
        plain, _ = self._load()
        df, _ = self._load(seed=3, random_answers_order=True)
        again, _ = self._load(seed=3, random_answers_order=True)
        for i in range(3):
            self.assertEqual(sorted(df.loc[i, "choices"]), sorted(plain.loc[i, "choices"]))
            fields = df.loc[i, "additional_fields"]
            self.assertEqual(fields["original_groundtruth"], plain.loc[i, "groundtruth"])
            order = [int(x) for x in fields["answers_order"].split(",")]
            self.assertEqual(sorted(order), [0, 1, 2, 3, 4])
            self.assertEqual(df.loc[i, "choices"], [plain.loc[i, "choices"][j] for j in order])
            correct_text = plain.loc[i, "choices"][plain.loc[i, "answer_idx"]]
            self.assertEqual(df.loc[i, "choices"][df.loc[i, "answer_idx"]], correct_text)
            self.assertEqual(df.loc[i, "groundtruth"], "ABCDE"[df.loc[i, "answer_idx"]])
            self.assertEqual(list(df.loc[i, "choices"]), list(again.loc[i, "choices"]))
        # question_concept survives alongside the shuffle metadata.
        self.assertEqual(df.loc[0, "additional_fields"]["question_concept"], "door")
        self.assertNotIn("answers_order", plain.loc[0, "additional_fields"])


if __name__ == "__main__":
    unittest.main()
