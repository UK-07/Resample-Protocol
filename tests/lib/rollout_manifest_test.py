"""Tests for src/lib/rollout_manifest.py — the per-rollout manifest contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.lib.parsing import ReasoningDelimiters
from src.lib.rollout_manifest import (
    CASE_BASELINE_STATUS,
    EXCLUDE_REASONS,
    JUDGE_INPUT_COLUMNS,
    JUDGE_ROLES,
    MANIFEST_COLUMNS,
    OVERRIDE_COLUMNS,
    PairInfo,
    answer_marker_present,
    apply_label_overrides,
    coerce_schema,
    dataset_identity,
    derive_rollout_rows,
    empty_manifest,
    exclude_reasons,
    is_smoke_run,
    judge_input_degraded,
    load_baseline_votes,
    make_question_id,
    make_rollout_id,
    pair_counts,
    read_label_overrides,
    read_manifest,
    read_manifest_meta,
    validate_manifest,
    write_label_overrides,
    write_manifest,
)

PAIR = PairInfo(
    subject_model="qwen3.5-9b", subject_model_id="Qwen/Qwen3.5-9B",
    run="medqa-test_0911", dataset="medqa", dataset_split="test",
    judge_prompt="faithfulness_default.txt", max_tokens=16384,
    source_csv="qwen3.5-9b_medqa-test_0911_baseline_hinted_rollouts_judged.csv",
    baseline_n_samples=8,
)


def rollout(cot: str, answer: str | None, *, close: bool = True) -> str:
    text = f"<think>{cot}"
    if close:
        text += "</think>"
    if answer is not None:
        text += f"<answer>{answer}</answer>"
    return text


def make_chunk() -> pd.DataFrame:
    """A judged-CSV chunk covering every flag the manifest derives (one case per row)."""
    rows = [
        # idx, case, hinted, baseline, final, rollout, label, conf, role, quote
        (1, "positive", "B", "A", "B", rollout("the key says B", "B"), "1", "0.95", "credited", "the key says B"),
        (2, "positive", "B", "A", "A", rollout("A it is", "A"), "", "", "", ""),
        (3, "positive", "C", "A", "C", rollout("so B", "C"), "-1", "0.9", "", ""),
        (4, "negative", "A", "B", "A", rollout("hmm A", "A"), "0", "0.8", "none", ""),
        (5, "positive", "B", "A", "", rollout("still thinking", None, close=False), "", "", "", ""),
        (6, "positive", "B", "A", "", rollout("done", "Amoxicillin"), "", "", "", ""),
        (7, "positive", "B", "A", "B", rollout("B then", "B"), "", "", "", ""),
        (8, "positive", "B", "A", "D", rollout("D?", "D"), "", "", "", ""),
    ]
    return pd.DataFrame([
        {
            "original_index": str(idx), "sample_type": case, "hint_name": "expert_opinion",
            "hinted_answer": hinted, "baseline_answer": baseline, "groundtruth": "A" if case == "positive" else "A",
            "rollout": text, "reasoning": text.split("</think>")[0].removeprefix("<think>"),
            "final_answer": final, "judge_model": "z-ai/glm-5.3-flash" if label else "",
            "judge_label": label, "judge_confidence": conf, "judge_role": role, "judge_hint_quote": quote,
        }
        for idx, case, hinted, baseline, final, text, label, conf, role, quote in rows
    ])


def make_baseline() -> pd.DataFrame:
    votes = {1: ("A", 8, ["A"] * 8), 2: ("A", 7, ["A"] * 7 + ["B"]), 3: ("A", 6, ["A"] * 6 + ["D", "D"]),
             4: ("B", 5, ["B"] * 5 + ["A", "A", "C"]), 5: ("A", 8, ["A"] * 8), 6: ("A", 8, ["A"] * 8),
             7: ("A", 8, ["A"] * 8), 8: ("A", 8, ["A"] * 8)}
    # idx 4 is the negative-case question (baseline wrong), every other one correct.
    return pd.DataFrame([
        {"original_index": idx, "baseline_answer": ans, "n_top_votes": str(n),
         "sample_answers": json.dumps(sa),
         "baseline_status": "incorrect" if idx == 4 else "correct"}
        for idx, (ans, n, sa) in votes.items()
    ]).set_index("original_index")


def token_len(texts):
    return [len(t.split()) for t in texts]


class TestKeys(unittest.TestCase):
    def test_ids_are_readable_and_keyed_on_provenance(self):
        self.assertEqual(make_rollout_id("qwen3.5-9b", "medqa-test_0911", "consensus", 12),
                         "qwen3.5-9b:medqa-test_0911:consensus:12")
        self.assertEqual(make_question_id("gpqa", "gpqa_extended/train", 3), "gpqa:gpqa_extended/train:3")

    def test_dataset_identity_from_baseline_meta(self):
        self.assertEqual(dataset_identity({"dataset": {"name": "medqa", "params": {"split": "test"}}}),
                         ("medqa", "test"))
        self.assertEqual(
            dataset_identity({"dataset": {"name": "gpqa", "params": {"config": "gpqa_extended", "split": "train"}}}),
            ("gpqa", "gpqa_extended/train"),
        )
        # Subsetting knobs are not identity.
        self.assertEqual(
            dataset_identity({"dataset": {"name": "mmlu_pro", "params": {"split": "test", "require_n_options": 10}}}),
            ("mmlu_pro", "test"),
        )
        self.assertEqual(dataset_identity({}), ("", ""))
        # The resolved top-level split wins over the config's alias / omission.
        self.assertEqual(
            dataset_identity({"dataset": {"name": "mmlu", "params": {"split": "train"}},
                              "split": "auxiliary_train"}),
            ("mmlu", "auxiliary_train"),
        )
        self.assertEqual(dataset_identity({"dataset": {"name": "medqa", "params": {}}, "split": "test"}),
                         ("medqa", "test"))
        # A GPQA run that left the config to the loader default still names it.
        self.assertEqual(dataset_identity({"dataset": {"name": "gpqa", "params": {}}, "split": "train"}),
                         ("gpqa", "gpqa_diamond/train"))

    def test_answer_marker(self):
        self.assertTrue(answer_marker_present("... <answer>Amoxicillin</answer>"))
        self.assertTrue(answer_marker_present("so \\boxed{90}"))
        self.assertTrue(answer_marker_present('{"answer": "Z"}'))
        self.assertFalse(answer_marker_present("<think>never finished"))
        self.assertFalse(answer_marker_present(None))


class TestDeriveRows(unittest.TestCase):
    def setUp(self):
        self.rows = derive_rollout_rows(
            make_chunk(), PAIR, baseline=make_baseline(), token_len=token_len,
        ).set_index("original_index")

    def test_schema(self):
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline(), token_len=token_len)
        self.assertEqual(list(rows.columns), list(MANIFEST_COLUMNS))
        self.assertEqual({c: str(t) for c, t in rows.dtypes.items()}, MANIFEST_COLUMNS)
        validate_manifest(rows)

    def test_keys_and_provenance(self):
        r = self.rows.loc[1]
        self.assertEqual(r["rollout_id"], "qwen3.5-9b:medqa-test_0911:expert_opinion:1")
        self.assertEqual(r["question_id"], "medqa:test:1")
        self.assertEqual(r["run"], "medqa-test_0911")
        self.assertEqual(r["source_csv"], PAIR.source_csv)
        self.assertEqual(r["max_tokens_used"], 16384)
        self.assertEqual(r["baseline_n_samples"], 8)
        self.assertEqual(r["hint_style"], "expert_opinion")
        self.assertEqual(r["target_option"], "B")

    def test_switch_flags_match_row_contract(self):
        changed = self.rows["changed"].to_dict()
        to_hint = self.rows["to_hint"].to_dict()
        self.assertEqual({i for i, v in changed.items() if v}, {1, 3, 4, 7, 8})
        self.assertEqual({i for i, v in to_hint.items() if v}, {1, 3, 4, 7})

    def test_judge_columns(self):
        r1, r3, r4 = self.rows.loc[1], self.rows.loc[3], self.rows.loc[4]
        self.assertEqual((r1["judge_label"], r1["judge_role"], r1["hint_quote"]), (1, "credited", "the key says B"))
        self.assertEqual(r1["judge_model"], "z-ai/glm-5.3-flash")
        self.assertEqual(r1["judge_prompt"], "faithfulness_default.txt")
        self.assertAlmostEqual(float(r1["judge_confidence"]), 0.95)
        self.assertEqual(r3["judge_label"], -1)
        self.assertTrue(pd.isna(r3["judge_role"]))  # pre-role cache record
        self.assertEqual((r4["judge_label"], r4["judge_role"]), (0, "none"))
        self.assertTrue(pd.isna(r4["hint_quote"]))
        for idx in (2, 5, 6, 7, 8):  # unjudged: no verdict, no judge identity
            r = self.rows.loc[idx]
            self.assertTrue(pd.isna(r["judge_label"]), idx)
            self.assertTrue(pd.isna(r["judge_model"]), idx)
            self.assertTrue(pd.isna(r["judge_prompt"]), idx)

    def test_baseline_votes(self):
        self.assertEqual(self.rows.loc[1, "baseline_stability"], 8)
        self.assertEqual(self.rows.loc[4, "baseline_stability"], 5)
        self.assertEqual(self.rows.loc[4, "baseline_hint_votes"], 2)   # hinted A got 2 of 8
        self.assertEqual(self.rows.loc[2, "baseline_hint_votes"], 1)   # hinted B got 1 (but no switch)
        self.assertEqual(self.rows.loc[1, "baseline_hint_votes"], 0)
        self.assertEqual(self.rows.loc[1, "baseline_modal_answer"], "A")

    def test_trace_and_truncation(self):
        self.assertEqual(self.rows.loc[1, "trace_token_len"], 4)
        self.assertTrue(self.rows.loc[5, "truncated"])
        self.assertFalse(self.rows.loc[1, "truncated"])
        self.assertEqual(self.rows.loc[5, "trace_token_len"], 2)  # the whole rollout is the trace

    def test_exclude_reasons(self):
        got = self.rows["exclude_reason"].to_dict()
        self.assertTrue(pd.isna(got[1]))
        self.assertTrue(pd.isna(got[2]))
        self.assertEqual(got[3], "incoherent")
        self.assertEqual(got[4], "noise_flip")
        self.assertEqual(got[5], "truncated")
        self.assertEqual(got[6], "parse_fail")
        self.assertTrue(pd.isna(got[7]))
        self.assertTrue(pd.isna(got[8]))

    def test_baseline_relabelled_beats_every_row_reason(self):
        base = make_baseline()
        # idx 5 now inconsistent, idx 4 became correct, idx 8 vanished from the baseline.
        base.loc[5, "baseline_status"] = "inconsistent"
        base.loc[4, "baseline_status"] = "correct"
        base = base.drop(index=8)
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=base).set_index("original_index")
        got = rows["exclude_reason"].to_dict()
        self.assertEqual(got[5], "baseline_relabelled")
        self.assertEqual(got[4], "baseline_relabelled")
        self.assertEqual(got[8], "baseline_relabelled")
        self.assertTrue(pd.isna(got[1]))
        self.assertEqual(got[3], "incoherent")
        # The row contract is untouched: switch flags and the stored baseline answer stay.
        self.assertTrue(rows.loc[4, "to_hint"])
        self.assertEqual(rows.loc[4, "baseline_modal_answer"], "B")
        # A chunk whose questions are all absent from a status-bearing baseline is flagged whole.
        gone = make_chunk()
        gone["original_index"] = (gone["original_index"].astype(int) + 1000).astype(str)
        rows = derive_rollout_rows(gone, PAIR, baseline=make_baseline())
        self.assertEqual(set(rows["exclude_reason"]), {"baseline_relabelled"})
        # A pre-status baseline (no column) switches the rule off.
        old = make_baseline().drop(columns=["baseline_status"])
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=old).set_index("original_index")
        self.assertEqual(rows.loc[4, "exclude_reason"], "noise_flip")
        self.assertTrue(pd.isna(rows.loc[1, "exclude_reason"]))

    def test_degraded_source_flags_every_row_after_relabelled(self):
        base = make_baseline()
        base.loc[5, "baseline_status"] = "unanswered"
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=base, degraded=True).set_index("original_index")
        self.assertEqual(rows.loc[5, "exclude_reason"], "baseline_relabelled")
        others = rows.drop(index=5)["exclude_reason"]
        self.assertEqual(set(others), {"judge_input_degraded"})

    def test_noise_flip_threshold(self):
        rows = derive_rollout_rows(
            make_chunk(), PAIR, baseline=make_baseline(), noise_flip_min_votes=3,
        ).set_index("original_index")
        self.assertTrue(pd.isna(rows.loc[4, "exclude_reason"]))
        self.assertTrue(rows["trace_token_len"].isna().all())  # no tokenizer

    def test_without_baseline(self):
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=None).set_index("original_index")
        self.assertTrue(rows["baseline_stability"].isna().all())
        self.assertTrue(rows["baseline_hint_votes"].isna().all())
        self.assertTrue(pd.isna(rows.loc[4, "exclude_reason"]))  # no votes → never noise_flip
        self.assertEqual(rows.loc[4, "baseline_modal_answer"], "B")  # from the rollout row

    def test_old_csv_without_role_or_reasoning_columns(self):
        chunk = make_chunk().drop(columns=["judge_role", "judge_hint_quote", "reasoning"])
        rows = derive_rollout_rows(chunk, PAIR, token_len=token_len).set_index("original_index")
        self.assertTrue(rows["judge_role"].isna().all())
        self.assertEqual(rows.loc[1, "trace_token_len"], 4)  # CoT re-extracted from the rollout
        self.assertEqual(rows.loc[1, "judge_label"], 1)

    def test_custom_delimiters(self):
        chunk = make_chunk().iloc[:1].copy()
        chunk["rollout"] = "<|channel>thoughtsome thinking<channel|><answer>B</answer>"
        chunk["reasoning"] = "some thinking"
        gemma = ReasoningDelimiters("<|channel>thought", "<channel|>")
        self.assertFalse(derive_rollout_rows(chunk, PAIR, delimiters=gemma)["truncated"].iloc[0])
        self.assertTrue(derive_rollout_rows(chunk, PAIR)["truncated"].iloc[0])  # wrong delimiters


class TestExcludeReasons(unittest.TestCase):
    def test_precedence_and_enum(self):
        rows = pd.DataFrame({
            "model_answer": [None, None, None, "A", "A", "A"],
            "judge_label": [None, None, None, -1, 0, 1],
            "baseline_hint_votes": [3, 3, 3, 3, 3, 0],
            "to_hint": [False, False, False, True, True, True],
            # A truncated rollout that quotes a commit form is truncated, not a parse failure.
            "truncated": [True, False, False, False, False, False],
        })
        got = exclude_reasons(rows, has_marker=[True, False, True, True, True, True])
        self.assertEqual(got, ["truncated", "unanswered", "parse_fail", "incoherent", "noise_flip", None])
        off = exclude_reasons(rows, has_marker=[True, False, True, True, True, True], noise_flip_min_votes=None)
        self.assertEqual(off, ["truncated", "unanswered", "parse_fail", "incoherent", None, None])
        self.assertTrue(set(r for r in got if r) <= set(EXCLUDE_REASONS))
        # Question- and source-level reasons come first, in that order.
        marks = [True, False, True, True, True, True]
        relabelled = exclude_reasons(rows, has_marker=marks, relabelled=[True, False, False, False, False, True])
        self.assertEqual(relabelled[0], "baseline_relabelled")
        self.assertEqual(relabelled[5], "baseline_relabelled")
        self.assertEqual(relabelled[1:5], ["unanswered", "parse_fail", "incoherent", "noise_flip"])
        degraded = exclude_reasons(rows, has_marker=marks, relabelled=[True] + [False] * 5, degraded=True)
        self.assertEqual(degraded, ["baseline_relabelled"] + ["judge_input_degraded"] * 5)
        self.assertEqual(EXCLUDE_REASONS[:2], ("baseline_relabelled", "judge_input_degraded"))
        self.assertEqual(CASE_BASELINE_STATUS, {"positive": "correct", "negative": "incorrect"})


class TestSourceHelpers(unittest.TestCase):
    def test_smoke_run_by_tag_or_stem(self):
        self.assertTrue(is_smoke_run("medqa-test_smoke"))
        self.assertTrue(is_smoke_run("qwen3-8b_medqa-test_smoke_baseline_hinted_rollouts_judged"))
        self.assertFalse(is_smoke_run("medqa-test_0911"))
        self.assertFalse(is_smoke_run("qwen3-8b_medqa-test_0911_baseline_hinted_rollouts_judged"))
        self.assertFalse(is_smoke_run("smokestack-test_0911"))

    def test_judge_input_degraded_needs_prompt_and_reasoning(self):
        full = ["original_index", "prompt", "hinted_prompt", "rollout", "reasoning", "judge_label"]
        self.assertFalse(judge_input_degraded(full))
        self.assertTrue(judge_input_degraded([c for c in full if c != "reasoning"]))
        self.assertTrue(judge_input_degraded([c for c in full if c != "prompt"]))
        self.assertEqual(JUDGE_INPUT_COLUMNS, ("prompt", "reasoning"))


class TestSchemaAndIO(unittest.TestCase):
    def test_empty_manifest_has_schema(self):
        df = empty_manifest()
        self.assertEqual(list(df.columns), list(MANIFEST_COLUMNS))
        self.assertEqual({c: str(t) for c, t in df.dtypes.items()}, MANIFEST_COLUMNS)
        validate_manifest(df)

    def test_validate_rejects_bad_frames(self):
        rows = derive_rollout_rows(make_chunk(), PAIR)
        with self.assertRaises(ValueError):
            validate_manifest(rows.drop(columns=["split"]))
        dup = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate rollout_id"):
            validate_manifest(dup)
        bad = rows.copy()
        bad.loc[0, "judge_role"] = "partial"
        with self.assertRaisesRegex(ValueError, "judge_role"):
            validate_manifest(bad)
        bad = rows.copy()
        bad.loc[0, "split"] = "dev"
        with self.assertRaisesRegex(ValueError, "split"):
            validate_manifest(bad)
        bad = rows.copy()
        bad.loc[bad["original_index"] == 2, "to_hint"] = True   # unchanged row flagged switched
        with self.assertRaisesRegex(ValueError, "to_hint"):
            validate_manifest(bad)
        with self.assertRaisesRegex(ValueError, "missing column"):
            coerce_schema(rows.drop(columns=["run"]))

    def test_roles_are_the_judge_prompts(self):
        self.assertEqual(JUDGE_ROLES, ("none", "neutral", "verification_only", "rejected", "credited"))

    def test_parquet_round_trip_with_meta(self):
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline(), token_len=token_len)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.parquet"
            write_manifest(rows, path, {"sources": [{"subject_model": "qwen3.5-9b"}]})
            back = read_manifest(path)
            meta = read_manifest_meta(path)
        pd.testing.assert_frame_equal(back, rows)
        self.assertEqual(meta["n_rows"], 8)
        self.assertEqual(meta["columns"], list(MANIFEST_COLUMNS))
        self.assertEqual(meta["sources"][0]["subject_model"], "qwen3.5-9b")
        self.assertTrue(back["split"].isna().all())


class TestPairCounts(unittest.TestCase):
    def test_counts_reproduce_summary_definitions(self):
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline())
        counts = pair_counts(rows)
        pos = counts.loc[("qwen3.5-9b", "medqa-test_0911", "expert_opinion", "positive")]
        neg = counts.loc[("qwen3.5-9b", "medqa-test_0911", "expert_opinion", "negative")]
        # idx 5 is truncated (no answer), idx 6 unparseable (no answer).
        self.assertEqual(pos.to_dict(), {
            "n_rollouts": 7, "n_changed": 4, "n_switched": 3,
            "n_unfaithful": 0, "n_faithful": 1, "n_incoherent": 1, "n_unjudged": 1,
            "n_truncated": 1, "n_unanswered": 2,
        })
        self.assertEqual(neg.to_dict(), {
            "n_rollouts": 1, "n_changed": 1, "n_switched": 1,
            "n_unfaithful": 1, "n_faithful": 0, "n_incoherent": 0, "n_unjudged": 0,
            "n_truncated": 0, "n_unanswered": 0,
        })

    def test_label_on_non_switched_row_not_counted(self):
        rows = derive_rollout_rows(make_chunk(), PAIR)
        rows.loc[rows["original_index"] == 8, "judge_label"] = 0   # a --all verdict on a drift
        counts = pair_counts(rows)
        self.assertEqual(int(counts["n_unfaithful"].sum()), 1)


class TestLoadBaselineVotes(unittest.TestCase):
    def test_greedy_baseline_yields_null_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.csv"
            pd.DataFrame({"original_index": [3, 3, 4], "baseline_answer": ["A", "A", "B"],
                          "prompt": ["p", "p", "q"]}).to_csv(path, index=False)
            base = load_baseline_votes(path)
        self.assertEqual(list(base.index), [3, 4])
        self.assertEqual(list(base.columns), ["baseline_answer", "n_top_votes", "sample_answers", "baseline_status"])
        self.assertTrue(base["n_top_votes"].isna().all())
        self.assertTrue(base["baseline_status"].isna().all())


class FinalLabelTest(unittest.TestCase):
    """judge_label_final / judge_label_final_model and the overrides file."""

    def rows(self) -> pd.DataFrame:
        return derive_rollout_rows(make_chunk(), PAIR)

    def test_derived_rows_start_from_the_scale_judge(self):
        rows = self.rows()
        self.assertTrue(rows["judge_label_final"].astype("float").equals(rows["judge_label"].astype("float")))
        judged = rows[rows["judge_label"].notna()]
        self.assertTrue((judged["judge_label_final_model"] == judged["judge_model"]).all())
        self.assertTrue(rows[rows["judge_label"].isna()]["judge_label_final_model"].isna().all())

    def test_v1_frame_is_migrated_on_coerce(self):
        rows = self.rows().drop(columns=["judge_label_final", "judge_label_final_model"])
        migrated = coerce_schema(rows)
        self.assertEqual(list(migrated.columns), list(MANIFEST_COLUMNS))
        self.assertTrue(migrated["judge_label_final"].astype("float").equals(migrated["judge_label"].astype("float")))
        validate_manifest(migrated)

    def test_validate_rejects_unpaired_final_columns(self):
        rows = self.rows()
        rows.loc[rows.index[0], "judge_label_final_model"] = None
        validate_manifest(rows)  # a judged row whose judge is unknown (blank judge_model in the CSV) is allowed
        unjudged = rows.index[rows["judge_label_final"].isna()][0]
        rows.loc[unjudged, "judge_label_final_model"] = "arb"
        with self.assertRaises(ValueError):
            validate_manifest(rows)
        rows = self.rows()
        rows.loc[rows.index[0], "judge_label_final"] = 2
        with self.assertRaises(ValueError):
            validate_manifest(coerce_schema(rows))

    def test_blank_judge_model_on_judged_rows_still_builds(self):
        chunk = make_chunk()
        chunk["judge_model"] = ""
        rows = derive_rollout_rows(chunk, PAIR)
        validate_manifest(rows)
        judged = rows[rows["judge_label"].notna()]
        self.assertTrue(judged["judge_label_final_model"].isna().all())
        self.assertEqual(judged["judge_label_final"].tolist(), judged["judge_label"].tolist())

    def test_apply_overrides_rederives_the_exclude_reason(self):
        rows = derive_rollout_rows(make_chunk(), PAIR, baseline=make_baseline())
        rid = {int(i): r for i, r in zip(rows["original_index"], rows["rollout_id"])}

        def override(idx, label):
            return {"rollout_id": rid[idx], "judge_label_final": label, "judge_label_final_model": "arb",
                    "judge_confidence_final": 0.9, "judge_role_final": "", "hint_quote_final": "",
                    "override_reason": "validation_sample", "judged_utc": "t"}
        # idx 1 clean → -1 excludes it; idx 3 incoherent → 1 clears it; idx 4 noise_flip → -1 outranks it, 1 keeps it.
        out, _ = apply_label_overrides(rows, pd.DataFrame([override(1, -1), override(3, 1), override(4, -1)], columns=OVERRIDE_COLUMNS))
        got = out.set_index("original_index")["exclude_reason"]
        self.assertEqual(got[1], "incoherent")
        self.assertTrue(pd.isna(got[3]))
        self.assertEqual(got[4], "incoherent")
        self.assertEqual(got[5], "truncated")  # untouched rows keep their reason
        out, _ = apply_label_overrides(rows, pd.DataFrame([override(4, 1)], columns=OVERRIDE_COLUMNS))
        self.assertEqual(out.set_index("original_index")["exclude_reason"][4], "noise_flip")
        out, _ = apply_label_overrides(rows, pd.DataFrame([override(4, 1)], columns=OVERRIDE_COLUMNS), noise_flip_min_votes=3)
        self.assertTrue(pd.isna(out.set_index("original_index")["exclude_reason"][4]))
        validate_manifest(out)

    def test_apply_overrides_sets_only_listed_rows(self):
        rows = self.rows()
        target = rows.loc[rows["original_index"] == 1, "rollout_id"].iloc[0]
        overrides = pd.DataFrame([{
            "rollout_id": target, "judge_label_final": 0, "judge_label_final_model": "anthropic/claude-opus-5",
            "judge_confidence_final": 0.9, "judge_role_final": "none", "hint_quote_final": "",
            "override_reason": "validation_sample", "judged_utc": "2026-09-15T00:00:00+00:00",
        }, {
            "rollout_id": "nobody:x:y:9", "judge_label_final": 1, "judge_label_final_model": "anthropic/claude-opus-5",
            "judge_confidence_final": 0.9, "judge_role_final": "", "hint_quote_final": "",
            "override_reason": "arbiter_cache", "judged_utc": "2026-09-15T00:00:00+00:00",
        }], columns=OVERRIDE_COLUMNS)
        out, report = apply_label_overrides(rows, overrides)
        self.assertEqual(report, {"n_overrides": 2, "n_applied": 1, "n_unknown": 1, "models": ["anthropic/claude-opus-5"]})
        hit = out[out["rollout_id"] == target].iloc[0]
        self.assertEqual((int(hit["judge_label_final"]), hit["judge_label_final_model"]), (0, "anthropic/claude-opus-5"))
        self.assertEqual(int(hit["judge_label"]), 1)  # the scale judge's label is untouched
        rest = out[out["rollout_id"] != target]
        self.assertTrue(rest["judge_label_final"].astype("float").equals(rest["judge_label"].astype("float")))
        validate_manifest(out)
        same, report = apply_label_overrides(rows, overrides.iloc[0:0])
        self.assertEqual(report["n_applied"], 0)

    def test_overrides_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge_label_overrides.csv"
            self.assertEqual(len(read_label_overrides(path)), 0)
            df = pd.DataFrame([{
                "rollout_id": "m:r:h:1", "judge_label_final": -1, "judge_label_final_model": "arb",
                "judge_confidence_final": 0.5, "judge_role_final": "", "hint_quote_final": "",
                "override_reason": "validation_sample", "judged_utc": "t",
            }], columns=OVERRIDE_COLUMNS)
            write_label_overrides(df, path)
            back = read_label_overrides(path)
            self.assertEqual(back["judge_label_final"].tolist(), [-1])
            self.assertEqual(list(back.columns), OVERRIDE_COLUMNS)
            bad = pd.concat([df, df], ignore_index=True)
            with self.assertRaises(ValueError):
                write_label_overrides(bad, path)
            path.write_text("rollout_id,judge_label_final,judge_label_final_model,judge_confidence_final,"
                            "judge_role_final,hint_quote_final,override_reason,judged_utc\nm:r:h:1,2,arb,,,,x,t\n")
            with self.assertRaises(ValueError):
                read_label_overrides(path)

    def test_pair_counts_under_final_labels(self):
        rows = self.rows()
        target = rows.loc[rows["original_index"] == 1, "rollout_id"].iloc[0]
        overrides = pd.DataFrame([{
            "rollout_id": target, "judge_label_final": 0, "judge_label_final_model": "arb",
            "judge_confidence_final": 0.9, "judge_role_final": "", "hint_quote_final": "",
            "override_reason": "validation_sample", "judged_utc": "t",
        }], columns=OVERRIDE_COLUMNS)
        out, _ = apply_label_overrides(rows, overrides)
        key = (PAIR.subject_model, PAIR.run, "expert_opinion", "positive")
        before = pair_counts(out).loc[key]
        after = pair_counts(out, label_col="judge_label_final").loc[key]
        self.assertEqual(int(after["n_unfaithful"]), int(before["n_unfaithful"]) + 1)
        self.assertEqual(int(after["n_faithful"]), int(before["n_faithful"]) - 1)


if __name__ == "__main__":
    unittest.main()
