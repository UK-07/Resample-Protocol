"""MCQ dataset loaders. Every loader returns ``(df, possible_options)`` on the
baseline-CSV schema: ``prompt``, ``groundtruth``, ``additional_fields``,
``question``, ``choices``, ``answer_idx``, ``original_index``."""

from __future__ import annotations

import random
import re
from typing import Optional

import pandas as pd
from datasets import load_dataset

from src.lib.constants import EXTENDED_OPTION_LETTERS, OPTION_LETTERS

# The MMLU training split has no subject labels and exists only under the ``all`` config.
MMLU_TRAIN_SPLIT = "auxiliary_train"

MMLU_SPLIT_ALIASES: dict[str, str] = {"train": MMLU_TRAIN_SPLIT}


def _shuffled_option_order(seed: Optional[int], original_index: int, n: int) -> list[int]:
    """Per-question permutation for ``random_answers_order``, seeded on
    ``(seed, original_index)`` so it is independent of row iteration order.
    Returns ``order`` such that ``new_choices[i] == old_choices[order[i]]``."""
    rng = random.Random() if seed is None else random.Random(f"{seed}:{original_index}")
    order = list(range(n))
    rng.shuffle(order)
    return order


class MMLUDataset:
    """Loader for MMLU (cais/mmlu). ``original_index`` is the row position in the
    full combined dataset before any filtering. ``additional_fields`` carries
    ``subject`` (subject-labelled splits only) and, under ``random_answers_order``,
    ``answers_order`` / ``original_groundtruth``."""

    HF_REPO = "cais/mmlu"
    option_letters = OPTION_LETTERS
    IDX_TO_LETTER = {i: l for i, l in enumerate(OPTION_LETTERS)}

    def __init__(
        self,
        config: str = "all",
        split: str = "test",
        subject_filter: Optional[list[str]] = None,
        exclude_subjects: Optional[list[str]] = None,
        samples_per_subject: Optional[int] = None,
        seed: int = 42,
        random_answers_order: bool = False,
    ):
        self.config = config
        self.split = split
        self.subject_filter = subject_filter
        self.exclude_subjects = exclude_subjects
        self.samples_per_subject = samples_per_subject
        self.seed = seed
        self.random_answers_order = random_answers_order

    def load(self) -> tuple[pd.DataFrame, list[str]]:
        split = MMLU_SPLIT_ALIASES.get(self.split, self.split)
        is_train_split = split == MMLU_TRAIN_SPLIT

        raw = load_dataset(self.HF_REPO, self.config, split=split)
        df_raw = raw.to_pandas()

        # original_index is taken before any filtering.
        df_raw = df_raw.reset_index(drop=False).rename(columns={"index": "original_index"})

        if is_train_split:
            if self.subject_filter or self.exclude_subjects or self.samples_per_subject:
                print(
                    f"WARNING: MMLUDataset ignoring subject_filter/exclude_subjects/"
                    f"samples_per_subject for split '{split}' — it has no subject labels."
                )
        else:
            if self.subject_filter:
                df_raw = df_raw[df_raw["subject"].isin(self.subject_filter)]
            if self.exclude_subjects:
                df_raw = df_raw[~df_raw["subject"].isin(self.exclude_subjects)]
            if self.samples_per_subject is not None:
                n = int(self.samples_per_subject)
                parts = [
                    g.sample(n=min(n, len(g)), random_state=self.seed)
                    for _, g in df_raw.groupby("subject", sort=True)
                ]
                df_raw = pd.concat(parts) if parts else df_raw.iloc[0:0]

        rows = []

        for _, row in df_raw.iterrows():
            question: str = row["question"]
            choices: list[str] = list(row["choices"])
            answer_idx: int = int(row["answer"])
            subject_val = row.get("subject", "")
            subject: str = "" if pd.isna(subject_val) else str(subject_val).strip()
            additional_fields: dict = {"subject": subject} if subject else {}

            if self.random_answers_order:
                order = _shuffled_option_order(
                    self.seed, int(row["original_index"]), len(choices)
                )
                additional_fields["original_groundtruth"] = self.IDX_TO_LETTER[answer_idx]
                additional_fields["answers_order"] = ",".join(map(str, order))
                choices = [choices[i] for i in order]
                answer_idx = order.index(answer_idx)
            groundtruth: str = self.IDX_TO_LETTER[answer_idx]

            lines = [question, ""]
            for letter, choice in zip(self.option_letters, choices):
                lines.append(f"{letter}) {choice}")
            prompt = "\n".join(lines)
            rows.append({
                "prompt": prompt,
                "groundtruth": groundtruth,
                "additional_fields": additional_fields,
                "question": question,
                "choices": choices,
                "answer_idx": answer_idx,
                "original_index": int(row["original_index"]),
            })

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df.index = result_df["original_index"]
        return result_df, list(self.option_letters)


class GPQADataset:
    """Loader for GPQA (Idavidrein/gpqa; gated, needs an HF token). The correct
    and three incorrect answers are shuffled into A-D. ``additional_fields``
    carries ``subset`` plus ``domain`` / ``subdomain`` when present and, under
    ``random_answers_order``, ``answers_order`` over the raw source positions
    (``0`` = correct, ``1``-``3`` = Incorrect Answer 1-3)."""

    HF_REPO = "Idavidrein/gpqa"

    def __init__(
        self,
        config: str = "gpqa_diamond",
        split: str = "train",
        seed: Optional[int] = 42,
        random_answers_order: bool = False,
    ):
        self.config = config
        self.split = split
        self.seed = seed
        self.random_answers_order = random_answers_order

    def load(self) -> tuple[pd.DataFrame, list[str]]:
        raw = load_dataset(self.HF_REPO, self.config, split=self.split)
        df_raw = raw.to_pandas()

        # The gpqa_experts subset holds annotator metadata, not questions.
        if "Question" not in df_raw.columns:
            raise ValueError(
                f"GPQA subset '{self.config}' has no 'Question' column — it does "
                "not contain MCQ data. Use gpqa_extended, gpqa_main, or gpqa_diamond."
            )

        rng = random.Random(self.seed)
        rows = []
        option_letters = OPTION_LETTERS
        for original_index, (_, row) in enumerate(df_raw.iterrows()):
            question: str = row["Question"]
            correct: str = row["Correct Answer"]
            incorrect: list[str] = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]

            all_choices = [correct] + incorrect
            order = None
            if self.random_answers_order:
                order = _shuffled_option_order(self.seed, original_index, len(all_choices))
                all_choices = [all_choices[i] for i in order]
                # Source position 0 is the correct answer (robust to duplicate choice strings).
                correct_idx = order.index(0)
            else:
                rng.shuffle(all_choices)
                correct_idx = all_choices.index(correct)
            groundtruth = option_letters[correct_idx]
            prompt = _format_mcq_prompt(question, all_choices)

            additional_fields = {"subset": self.config}
            if order is not None:
                additional_fields["answers_order"] = ",".join(map(str, order))
            for key, col in (("domain", "High-level domain"), ("subdomain", "Subdomain")):
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    additional_fields[key] = val.strip()

            rows.append({
                "prompt": prompt,
                "groundtruth": groundtruth,
                "additional_fields": additional_fields,
                "question": question,
                "choices": all_choices,
                "answer_idx": correct_idx,
                "original_index": original_index,
            })

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df.index = result_df["original_index"]
        return result_df, list(option_letters)


class MMLUProDataset:
    """Loader for MMLU-Pro (TIGER-Lab/MMLU-Pro): up to 10 options (A-J), 14
    categories. ``original_index`` is the dataset's own ``question_id``.
    ``"N/A"`` placeholder options are dropped and ``answer_idx`` remapped.
    ``require_n_options`` keeps only rows with exactly that many real options
    (set it to 10 for a constant-width baseline). ``additional_fields`` carries
    ``category`` / ``src`` and the ``random_answers_order`` keys."""

    HF_REPO = "TIGER-Lab/MMLU-Pro"
    HF_CONFIG = "default"
    option_letters = EXTENDED_OPTION_LETTERS
    IDX_TO_LETTER = {i: l for i, l in enumerate(EXTENDED_OPTION_LETTERS)}

    def __init__(
        self,
        split: str = "test",
        category_filter: Optional[list[str]] = None,
        exclude_categories: Optional[list[str]] = None,
        samples_per_category: Optional[int] = None,
        require_n_options: Optional[int] = None,
        seed: int = 42,
        random_answers_order: bool = False,
    ):
        self.split = split
        self.category_filter = category_filter
        self.exclude_categories = exclude_categories
        self.samples_per_category = samples_per_category
        self.require_n_options = require_n_options
        self.seed = seed
        self.random_answers_order = random_answers_order

    def load(self) -> tuple[pd.DataFrame, list[str]]:
        raw = load_dataset(self.HF_REPO, self.HF_CONFIG, split=self.split)
        df_raw = raw.to_pandas()

        if self.category_filter:
            df_raw = df_raw[df_raw["category"].isin(self.category_filter)]
        if self.exclude_categories:
            df_raw = df_raw[~df_raw["category"].isin(self.exclude_categories)]
        # The width filter runs before samples_per_category so the cap samples surviving rows only.
        if self.require_n_options is not None:
            n_req = int(self.require_n_options)
            n_real = df_raw["options"].apply(
                lambda opts: sum(1 for c in opts if c != "N/A")
            )
            before = len(df_raw)
            df_raw = df_raw[n_real == n_req]
            if len(df_raw) < before:
                print(
                    f"MMLU-Pro: require_n_options={n_req} dropped "
                    f"{before - len(df_raw)} of {before} rows."
                )
        if self.samples_per_category is not None:
            n = int(self.samples_per_category)
            parts = [
                g.sample(n=min(n, len(g)), random_state=self.seed)
                for _, g in df_raw.groupby("category", sort=True)
            ]
            df_raw = pd.concat(parts) if parts else df_raw.iloc[0:0]

        rows = []
        for _, row in df_raw.iterrows():
            original_index = int(row["question_id"])
            question: str = row["question"]
            answer_idx: int = int(row["answer_index"])
            raw_options = list(row["options"])
            if not (0 <= answer_idx < len(raw_options)) or (
                raw_options[answer_idx] == "N/A"
            ):
                raise ValueError(
                    f"MMLU-Pro question_id={original_index}: answer_index "
                    f'{answer_idx} points at a missing or "N/A" option — '
                    "refusing to mislabel the groundtruth."
                )
            choices = [c for c in raw_options if c != "N/A"]
            answer_idx -= sum(1 for c in raw_options[:answer_idx] if c == "N/A")

            additional_fields: dict = {}
            for key in ("category", "src"):
                val = row.get(key, "")
                val = "" if pd.isna(val) else str(val).strip()
                if val:
                    additional_fields[key] = val

            if self.random_answers_order:
                order = _shuffled_option_order(self.seed, original_index, len(choices))
                additional_fields["original_groundtruth"] = self.IDX_TO_LETTER[answer_idx]
                additional_fields["answers_order"] = ",".join(map(str, order))
                choices = [choices[i] for i in order]
                answer_idx = order.index(answer_idx)
            groundtruth: str = self.IDX_TO_LETTER[answer_idx]
            prompt = _format_mcq_prompt(question, choices, letters=self.option_letters)

            rows.append({
                "prompt": prompt,
                "groundtruth": groundtruth,
                "additional_fields": additional_fields,
                "question": question,
                "choices": choices,
                "answer_idx": answer_idx,
                "original_index": original_index,
            })

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df.index = result_df["original_index"]
        return result_df, list(self.option_letters)


class MedQADataset:
    """Loader for MedQA USMLE 4-option (GBaker/MedQA-USMLE-4-options), splits
    ``test`` / ``train``. ``additional_fields`` carries ``step`` and the
    ``random_answers_order`` keys; rows without four non-empty A-D options are
    dropped with a warning."""

    HF_REPO = "GBaker/MedQA-USMLE-4-options"
    option_letters = OPTION_LETTERS
    IDX_TO_LETTER = {i: l for i, l in enumerate(OPTION_LETTERS)}

    def __init__(
        self,
        split: str = "test",
        seed: Optional[int] = 42,
        random_answers_order: bool = False,
    ):
        self.split = split
        self.seed = seed
        self.random_answers_order = random_answers_order

    def load(self) -> tuple[pd.DataFrame, list[str]]:
        raw = load_dataset(self.HF_REPO, split=self.split)
        df_raw = raw.to_pandas()
        required = {"question", "options", "answer_idx"}
        missing = sorted(required - set(df_raw.columns))
        if missing:
            raise ValueError(
                f"MedQA split '{self.split}' lacks column(s) {missing}; expected "
                f"{sorted(required)} (GBaker/MedQA-USMLE-4-options layout)."
            )

        rows = []
        n_dropped = 0
        for original_index, (_, row) in enumerate(df_raw.iterrows()):
            options = row["options"]
            # Options are a struct keyed A-D; take them in letter order, never dict order.
            if not isinstance(options, dict) or any(
                not isinstance(options.get(l), str) or not options[l].strip()
                for l in self.option_letters
            ):
                n_dropped += 1
                continue
            choices = [options[l] for l in self.option_letters]
            answer_letter = str(row["answer_idx"]).strip().upper()
            if answer_letter not in self.option_letters:
                raise ValueError(
                    f"MedQA row {original_index}: answer_idx {row['answer_idx']!r} "
                    f"is not one of {self.option_letters}."
                )
            answer_idx = self.option_letters.index(answer_letter)

            additional_fields: dict = {}
            step = row.get("meta_info", "")
            if isinstance(step, str) and step.strip():
                additional_fields["step"] = step.strip()
            if self.random_answers_order:
                order = _shuffled_option_order(self.seed, original_index, len(choices))
                additional_fields["original_groundtruth"] = self.IDX_TO_LETTER[answer_idx]
                additional_fields["answers_order"] = ",".join(map(str, order))
                choices = [choices[i] for i in order]
                answer_idx = order.index(answer_idx)

            rows.append({
                "prompt": _format_mcq_prompt(row["question"], choices),
                "groundtruth": self.IDX_TO_LETTER[answer_idx],
                "additional_fields": additional_fields,
                "question": row["question"],
                "choices": choices,
                "answer_idx": answer_idx,
                "original_index": original_index,
            })
        if n_dropped:
            print(
                f"WARNING: MedQADataset dropped {n_dropped} row(s) without exactly "
                f"four non-empty A-D options."
            )

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df.index = result_df["original_index"]
        return result_df, list(self.option_letters)


# AQuA options are single strings "A)21"; re.S so a multi-line text is captured whole.
_AQUA_OPTION_RE = re.compile(r"^\s*([A-Za-z])\s*\)\s*(.*)$", re.S)


def _parse_aqua_options(raw_options, letters: list[str]) -> Optional[list[str]]:
    """The option texts of one AQuA row in letter order, or None if malformed.

    A usable row has exactly ``len(letters)`` entries labelled with those letters
    in order and non-empty texts; anything else is rejected rather than
    relabelled by position (``correct`` names a letter). A label repeated inside
    the text (``"A)A)21"``) is stripped once; a different inner letter is kept."""
    try:
        options = list(raw_options)
    except TypeError:
        # A missing cell reaches us as NaN (a float), not None.
        return None
    if len(options) != len(letters):
        return None
    texts: list[str] = []
    for letter, entry in zip(letters, options):
        if not isinstance(entry, str):
            return None
        match = _AQUA_OPTION_RE.match(entry)
        if match is None or match.group(1).upper() != letter:
            return None
        text = match.group(2).strip()
        inner = _AQUA_OPTION_RE.match(text)
        if inner is not None and inner.group(1).upper() == letter:
            text = inner.group(2).strip()
        if not text:
            return None
        texts.append(text)
    return texts


class AQuADataset:
    """Loader for AQuA-RAT (deepmind/aqua_rat, ``raw`` config): five options A-E,
    splits ``train`` / ``validation`` / ``test``. ``additional_fields`` holds only
    the ``random_answers_order`` keys; ``rationale`` is never carried through
    (it is a worked solution). Malformed rows are dropped with a warning."""

    HF_REPO = "deepmind/aqua_rat"
    # The "tokenized" config carries whitespace-mangled text.
    HF_CONFIG = "raw"
    N_OPTIONS = 5
    option_letters = list(EXTENDED_OPTION_LETTERS[:N_OPTIONS])
    IDX_TO_LETTER = {i: l for i, l in enumerate(option_letters)}

    def __init__(
        self,
        split: str = "train",
        seed: Optional[int] = 42,
        random_answers_order: bool = False,
    ):
        self.split = split
        self.seed = seed
        self.random_answers_order = random_answers_order

    def load(self) -> tuple[pd.DataFrame, list[str]]:
        raw = load_dataset(self.HF_REPO, self.HF_CONFIG, split=self.split)
        df_raw = raw.to_pandas()
        required = {"question", "options", "correct"}
        missing = sorted(required - set(df_raw.columns))
        if missing:
            raise ValueError(
                f"AQuA split '{self.split}' lacks column(s) {missing}; expected "
                f"{sorted(required)} (deepmind/aqua_rat '{self.HF_CONFIG}' layout)."
            )

        rows = []
        n_dropped_options = 0
        n_dropped_answer = 0
        for original_index, (_, row) in enumerate(df_raw.iterrows()):
            choices = _parse_aqua_options(row["options"], self.option_letters)
            if choices is None:
                n_dropped_options += 1
                continue
            answer_letter = str(row["correct"]).strip().upper()
            if answer_letter not in self.option_letters:
                n_dropped_answer += 1
                continue
            answer_idx = self.option_letters.index(answer_letter)

            additional_fields: dict = {}
            if self.random_answers_order:
                order = _shuffled_option_order(self.seed, original_index, len(choices))
                additional_fields["original_groundtruth"] = self.IDX_TO_LETTER[answer_idx]
                additional_fields["answers_order"] = ",".join(map(str, order))
                choices = [choices[i] for i in order]
                answer_idx = order.index(answer_idx)

            rows.append({
                "prompt": _format_mcq_prompt(
                    row["question"], choices, self.option_letters
                ),
                "groundtruth": self.IDX_TO_LETTER[answer_idx],
                "additional_fields": additional_fields,
                "question": row["question"],
                "choices": choices,
                "answer_idx": answer_idx,
                "original_index": original_index,
            })
        if n_dropped_options or n_dropped_answer:
            print(
                f"WARNING: AQuADataset dropped {n_dropped_options} row(s) whose "
                f"options are not five 'A)…'-'E)…' entries and "
                f"{n_dropped_answer} row(s) whose 'correct' letter is outside "
                f"{self.option_letters}."
            )

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df.index = result_df["original_index"]
        return result_df, list(self.option_letters)


class CommonsenseQADataset:
    """Loader for CommonsenseQA (tau/commonsense_qa): five options A-E, splits
    ``validation`` / ``train``. The ``test`` split has no gold labels and is
    refused. ``additional_fields`` carries ``question_concept`` and the
    ``random_answers_order`` keys; malformed rows are dropped with a warning."""

    HF_REPO = "tau/commonsense_qa"
    N_OPTIONS = 5
    option_letters = list(EXTENDED_OPTION_LETTERS[:N_OPTIONS])
    IDX_TO_LETTER = {i: l for i, l in enumerate(option_letters)}
    UNLABELLED_SPLIT = "test"

    def __init__(
        self,
        split: str = "validation",
        seed: Optional[int] = 42,
        random_answers_order: bool = False,
    ):
        if str(split).strip().lower() == self.UNLABELLED_SPLIT:
            raise ValueError(
                f"CommonsenseQA split '{split}' has no gold labels (answerKey is "
                "empty on the Hub — they are held on a leaderboard), so it cannot "
                "produce a baseline. Use 'validation' (1,221 rows) or 'train' "
                "(9,741)."
            )
        self.split = split
        self.seed = seed
        self.random_answers_order = random_answers_order

    def load(self) -> tuple[pd.DataFrame, list[str]]:
        raw = load_dataset(self.HF_REPO, split=self.split)
        df_raw = raw.to_pandas()
        required = {"question", "choices", "answerKey"}
        missing = sorted(required - set(df_raw.columns))
        if missing:
            raise ValueError(
                f"CommonsenseQA split '{self.split}' lacks column(s) {missing}; "
                f"expected {sorted(required)} (tau/commonsense_qa layout)."
            )

        rows = []
        n_dropped_choices = 0
        n_dropped_answer = 0
        for original_index, (_, row) in enumerate(df_raw.iterrows()):
            choices = _parse_csqa_choices(row["choices"], self.option_letters)
            if choices is None:
                n_dropped_choices += 1
                continue
            answer_letter = str(row["answerKey"]).strip().upper()
            if answer_letter not in self.option_letters:
                n_dropped_answer += 1
                continue
            answer_idx = self.option_letters.index(answer_letter)

            additional_fields: dict = {}
            concept = row.get("question_concept", "")
            if isinstance(concept, str) and concept.strip():
                additional_fields["question_concept"] = concept.strip()
            if self.random_answers_order:
                order = _shuffled_option_order(self.seed, original_index, len(choices))
                additional_fields["original_groundtruth"] = self.IDX_TO_LETTER[answer_idx]
                additional_fields["answers_order"] = ",".join(map(str, order))
                choices = [choices[i] for i in order]
                answer_idx = order.index(answer_idx)

            rows.append({
                "prompt": _format_mcq_prompt(
                    row["question"], choices, self.option_letters
                ),
                "groundtruth": self.IDX_TO_LETTER[answer_idx],
                "additional_fields": additional_fields,
                "question": row["question"],
                "choices": choices,
                "answer_idx": answer_idx,
                "original_index": original_index,
            })
        if n_dropped_choices or n_dropped_answer:
            print(
                f"WARNING: CommonsenseQADataset dropped {n_dropped_choices} row(s) "
                f"without exactly five non-empty A-E choices and "
                f"{n_dropped_answer} row(s) whose 'answerKey' is outside "
                f"{self.option_letters}."
            )

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df.index = result_df["original_index"]
        return result_df, list(self.option_letters)


def _parse_csqa_choices(raw_choices, letters: list[str]) -> Optional[list[str]]:
    """The option texts of one CommonsenseQA row in label order, or None.

    The Hub feature is ``{"label": [...], "text": [...]}``; a usable row has
    exactly ``len(letters)`` labels covering those letters with non-empty texts.
    Anything else is rejected rather than filled in by position."""
    if not isinstance(raw_choices, dict):
        return None
    labels, texts = raw_choices.get("label"), raw_choices.get("text")
    if labels is None or texts is None:
        return None
    labels, texts = list(labels), list(texts)
    if len(labels) != len(letters) or len(texts) != len(letters):
        return None
    by_letter: dict[str, str] = {}
    for label, text in zip(labels, texts):
        if not isinstance(label, str) or not isinstance(text, str):
            return None
        key = label.strip().upper()
        if key not in letters or key in by_letter or not text.strip():
            return None
        by_letter[key] = text.strip()
    return [by_letter[l] for l in letters]


def _format_mcq_prompt(
    question: str, choices: list[str], letters: Optional[list[str]] = None
) -> str:
    """``"<question>\\n\\nA) ...\\nB) ..."`` — ``choices`` labelled in order with
    ``letters`` (default A-D); more choices than letters is an error."""
    letters = letters if letters is not None else OPTION_LETTERS
    if len(choices) > len(letters):
        raise ValueError(
            f"{len(choices)} choices but only {len(letters)} option letters."
        )
    lines = [question, ""]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}) {choice}")
    return "\n".join(lines)


DATASET_REGISTRY: dict[str, type] = {
    "mmlu": MMLUDataset,
    "mmlu_pro": MMLUProDataset,
    "gpqa": GPQADataset,
    "medqa": MedQADataset,
    "aqua": AQuADataset,
    "commonsense_qa": CommonsenseQADataset,
}


def load_data(
    dataset_name: str,
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a dataset by (case-insensitive) ``DATASET_REGISTRY`` name and return
    ``(df, possible_options)``; ``**kwargs`` go to the loader's constructor."""
    name = dataset_name.strip().lower()
    try:
        cls = DATASET_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {sorted(DATASET_REGISTRY)}."
        )
    return cls(**kwargs).load()
