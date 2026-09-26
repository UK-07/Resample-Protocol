import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.lib.llm_judge_verb import (
    JudgeSetupError,
    FAITHFULNESS_PROMPT,
    HINT_TEXT_UNAVAILABLE,
    judge_cache_path,
    judge_case_record,
    judge_verdict,
    load_judged_records,
    load_prompt_template,
    render_prompt,
    run_judge_batch,
    template_style,
    validate_judgement,
    JUDGE_ROLES,
    ALL_PROMPT_PLACEHOLDERS,
    PROMPT_PLACEHOLDERS,
)
from src.lib.llm_judge_verb import build_judge_prompt, prompt_template_hash


FAITHFUL_PAYLOAD = {
    "reasoning": "The CoT defers to the hint explicitly.",
    "label": 1,
    "confidence": 0.9,
}

UNFAITHFUL_PAYLOAD = {
    "reasoning": "The CoT never mentions the hint and rationalizes A.",
    "label": 0,
    "confidence": 0.85,
}


def make_response(text):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason="stop"
            )
        ]
    )


def make_empty_response():
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None), finish_reason="length"
            )
        ]
    )


def make_choiceless_response():
    return SimpleNamespace(choices=[])


def make_client(responses):
    client = MagicMock()
    client.chat.completions.create.side_effect = list(responses)
    return client


class TestValidateJudgement(unittest.TestCase):
    def test_valid_labels(self):
        no_audit = {"hint_role": None, "hint_quote": None}
        self.assertEqual(validate_judgement(FAITHFUL_PAYLOAD), {**FAITHFUL_PAYLOAD, **no_audit})
        self.assertEqual(validate_judgement(UNFAITHFUL_PAYLOAD), {**UNFAITHFUL_PAYLOAD, **no_audit})

    def test_audit_keys_kept(self):
        parsed = dict(
            FAITHFUL_PAYLOAD, trace_conclusion="A",
            hint_quote="the professor says A", hint_role=" Credited",
        )
        out = validate_judgement(parsed)
        self.assertEqual(out["hint_role"], "credited")
        self.assertEqual(out["hint_quote"], "the professor says A")
        self.assertNotIn("trace_conclusion", out)
        for role in JUDGE_ROLES:
            self.assertEqual(validate_judgement(dict(parsed, hint_role=role))["hint_role"], role)
        self.assertEqual(
            validate_judgement(dict(parsed, hint_role="verification-only"))["hint_role"],
            "verification_only",
        )

    def test_bad_audit_keys_never_reject(self):
        for role in ("partial", 3, None, ""):
            out = validate_judgement(dict(UNFAITHFUL_PAYLOAD, hint_role=role))
            self.assertIsNotNone(out)
            self.assertIsNone(out["hint_role"])
        for quote in (None, "", "null", "None", 7):
            out = validate_judgement(dict(UNFAITHFUL_PAYLOAD, hint_quote=quote))
            self.assertIsNotNone(out)
            self.assertIsNone(out["hint_quote"])

    def test_string_label_coerced(self):
        parsed = dict(FAITHFUL_PAYLOAD, label="1")
        self.assertEqual(validate_judgement(parsed)["label"], 1)

    def test_invalid_label_rejected(self):
        self.assertIsNone(validate_judgement(dict(FAITHFUL_PAYLOAD, label=2)))
        self.assertIsNone(validate_judgement(dict(FAITHFUL_PAYLOAD, label=-2)))

    def test_incoherent_label_accepted(self):
        self.assertEqual(validate_judgement(dict(FAITHFUL_PAYLOAD, label=-1))["label"], -1)
        self.assertIsNone(validate_judgement(dict(FAITHFUL_PAYLOAD, label=True)))
        self.assertIsNone(validate_judgement(dict(FAITHFUL_PAYLOAD, label=None)))

    def test_missing_or_bad_confidence_rejected(self):
        payload = {"reasoning": "r", "label": 1}
        self.assertIsNone(validate_judgement(payload))
        self.assertIsNone(validate_judgement(dict(payload, confidence=1.5)))
        self.assertIsNone(validate_judgement(dict(payload, confidence=-0.1)))

    def test_non_dict_rejected(self):
        self.assertIsNone(validate_judgement(None))
        self.assertIsNone(validate_judgement([1]))


class TestLoadPromptTemplate(unittest.TestCase):
    def test_valid_template_loaded(self):
        import tempfile
        from pathlib import Path

        template = (
            "{baseline_prompt} {hint_description} {target_option} {model_answer} {reasoning_trace}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text(template)
            self.assertEqual(load_prompt_template(path), template)

    def test_missing_placeholder_raises(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text("{baseline_prompt} {target_option} {model_answer} {reasoning_trace}")
            with self.assertRaises(ValueError) as ctx:
                load_prompt_template(path)
            self.assertIn("hint_description", str(ctx.exception))

    def test_unescaped_literal_braces_raise(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text(
                "{baseline_prompt} {hint_description} {target_option} "
                '{model_answer} {reasoning_trace} respond {"label": 1}'
            )
            with self.assertRaises(ValueError) as ctx:
                load_prompt_template(path)
            self.assertIn("doubled", str(ctx.exception))

    def test_default_prompt_is_valid(self):
        for placeholder in (
            "{baseline_prompt}", "{hint_description}", "{hint_text}",
            "{target_option}", "{model_answer}", "{reasoning_trace}",
        ):
            self.assertIn(placeholder, FAITHFULNESS_PROMPT)


class TestShippedPromptTemplates(unittest.TestCase):
    """Every shipped template loads; the default file mirrors the built-in prompt."""

    TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "configs" / "llm_judge_prompts"
    FROZEN = {"faithfulness_v1_default.txt"}

    def test_every_template_loads(self):
        paths = sorted(self.TEMPLATE_DIR.glob("*.txt"))
        self.assertTrue(paths, f"no templates found under {self.TEMPLATE_DIR}")
        for path in paths:
            with self.subTest(template=path.name):
                template = load_prompt_template(path)
                for placeholder in PROMPT_PLACEHOLDERS:
                    self.assertIn("{%s}" % placeholder, template)
                if path.name not in self.FROZEN:
                    self.assertIn("{hint_text}", template)

    def test_default_file_mirrors_builtin_prompt(self):
        path = self.TEMPLATE_DIR / "faithfulness_default.txt"
        self.assertEqual(path.read_text(), FAITHFULNESS_PROMPT)


def make_api_error(cls, status: int, message: str = "boom"):
    """A real openai.APIStatusError subclass instance (needs httpx objects)."""
    import httpx

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, request=request, json={"error": message})
    return cls(message, response=response, body=None)


def make_transient_error(message: str = "429"):
    import httpx
    import openai

    return make_api_error(openai.RateLimitError, 429, message)


class TestJudgeErrorClassification(unittest.TestCase):
    """Misconfiguration must abort with JudgeSetupError, never be retried or swallowed."""

    def test_missing_api_key_is_setup_error(self):
        import os

        from src.lib import llm_judge_verb as mod

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
             patch.object(mod, "_client", None):
            with self.assertRaises(JudgeSetupError):
                mod._get_client()

    def test_batch_record_aborts_on_setup_error(self):
        import openai

        client = MagicMock()
        client.chat.completions.create.side_effect = make_api_error(
            openai.PermissionDeniedError, 403, "forbidden"
        )
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            with self.assertRaises(JudgeSetupError):
                judge_case_record(make_batch_row(), "prov/model", retries=3)
        self.assertEqual(client.chat.completions.create.call_count, 1)


def make_batch_row(original_index=0, hint_name="expert_opinion", **overrides):
    row = {
        "original_index": original_index,
        "hint_name": hint_name,
        "rollout": "<think>Because the hint says A.</think>\n<answer>A</answer>",
        "prompt": "BASELINE_PROMPT_TEXT",
        "hinted_prompt": "HINTED_PROMPT_TEXT",
        "baseline_answer": "C",
        "hinted_answer": "A",
    }
    row.update(overrides)
    return row


def make_usage_response(text):
    response = make_response(text)
    response.usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, model_extra={"cost": 0.002}
    )
    return response


def dollar_template(body: str = "") -> str:
    """A minimal valid ${name}-style template, plus optional extra body."""
    slots = "".join("<%s>${%s}</%s>\n" % (p, p, p) for p in PROMPT_PLACEHOLDERS)
    return slots + body


class TestTemplateStyles(unittest.TestCase):
    """Judge prompts may use ``{name}`` or ``${name}`` slots."""

    def test_style_detected_per_template(self):
        self.assertEqual(template_style("hello {name}"), "brace")
        self.assertEqual(template_style("hello ${name}"), "dollar")
        self.assertEqual(template_style("no placeholders at all"), "brace")

    def test_render_prompt_fills_both_styles(self):
        self.assertEqual(render_prompt("a {x} b", x="V"), "a V b")
        self.assertEqual(render_prompt("a ${x} b", x="V"), "a V b")

    def test_dollar_template_needs_no_brace_escaping(self):
        template = dollar_template('respond with {"label": 1, "confidence": 0.9}')
        rendered = render_prompt(template, **{p: "v" for p in ALL_PROMPT_PLACEHOLDERS})
        self.assertIn('{"label": 1, "confidence": 0.9}', rendered)

    def test_dollar_template_with_raw_json_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text(dollar_template('respond {"label": 1}'))
            template = load_prompt_template(path)
            self.assertEqual(template_style(template), "dollar")

    def test_dollar_template_missing_placeholder_raises(self):
        body = "".join(
            "${%s}" % p for p in PROMPT_PLACEHOLDERS if p != "hint_description"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text(body)
            with self.assertRaises(ValueError) as ctx:
                load_prompt_template(path)
            self.assertIn("${hint_description}", str(ctx.exception))

    def test_dollar_template_unknown_placeholder_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text(dollar_template("${nonsense}"))
            with self.assertRaises(ValueError) as ctx:
                load_prompt_template(path)
            self.assertIn("${nonsense}", str(ctx.exception))

    def test_render_prompt_raises_on_unsupplied_dollar_slot(self):
        with self.assertRaises(KeyError):
            render_prompt("a ${missing} b", x="V")

    def test_brace_template_unaffected_by_dollar_support(self):
        template = "cost is $5 for {baseline_prompt}"
        self.assertEqual(template_style(template), "brace")
        self.assertEqual(
            render_prompt(template, baseline_prompt="BP"), "cost is $5 for BP"
        )

    def test_builtin_prompt_is_brace_style(self):
        self.assertEqual(template_style(FAITHFULNESS_PROMPT), "brace")

    def test_cache_identity_is_the_raw_text_hash(self):
        import hashlib

        for text in ("some template", FAITHFULNESS_PROMPT):
            expected = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
            self.assertEqual(prompt_template_hash(text), expected)
        self.assertEqual(
            prompt_template_hash(None), prompt_template_hash(FAITHFULNESS_PROMPT)
        )


class TestBuildJudgePrompt(unittest.TestCase):
    """`build_judge_prompt` must render exactly what the batch path sends."""

    def test_matches_the_prompt_judge_case_record_sends(self):
        row = make_batch_row()
        client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            judge_case_record(row, "prov/model")
        sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(build_judge_prompt(row), sent)

    def test_renders_a_dollar_style_custom_template(self):
        prompt = build_judge_prompt(
            make_batch_row(), prompt_template=dollar_template()
        )
        self.assertIn("<baseline_prompt>BASELINE_PROMPT_TEXT</baseline_prompt>", prompt)
        self.assertIn("<target_option>A</target_option>", prompt)

    def test_calls_no_api(self):
        with patch(
            "src.lib.llm_judge_verb._get_client",
            side_effect=AssertionError("no API call expected"),
        ):
            build_judge_prompt(make_batch_row())


class TestJudgeVerdict(unittest.TestCase):
    def rec(self, **overrides):
        base = {"label": 0, "confidence": 0.9, "error": None}
        base.update(overrides)
        return base

    def test_unfaithful_and_faithful(self):
        self.assertEqual(judge_verdict(self.rec(label=0)), "unfaithful")
        self.assertEqual(judge_verdict(self.rec(label=1)), "faithful")
        self.assertEqual(judge_verdict(self.rec(label=-1)), "incoherent")

    def test_invalid_label_and_missing_label(self):
        self.assertEqual(judge_verdict(self.rec(label=2)), "invalid")
        self.assertEqual(judge_verdict(self.rec(label=None)), "invalid")
        self.assertEqual(judge_verdict({"h_score": 0, "error": None}), "invalid")

    def test_error_and_missing_record(self):
        self.assertEqual(judge_verdict(self.rec(error="boom")), "error")
        self.assertEqual(judge_verdict(None), "error")


class TestJudgeCaseRecord(unittest.TestCase):
    def test_success_with_usage_accounting(self):
        client = make_client([make_usage_response(json.dumps(UNFAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(make_batch_row(), "prov/model")
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["label"], 0)
        self.assertEqual(rec["confidence"], 0.85)
        self.assertEqual(rec["reasoning"], UNFAITHFUL_PAYLOAD["reasoning"])
        self.assertIn("hint_role", rec)   # audit keys always present on a verdict
        self.assertIsNone(rec["hint_role"])
        self.assertIsNone(rec["hint_quote"])
        self.assertEqual(rec["judge_model"], "prov/model")
        self.assertEqual(rec["prompt_tokens"], 10)
        self.assertEqual(rec["completion_tokens"], 5)
        self.assertAlmostEqual(rec["cost_usd"], 0.002)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("<baseline_prompt>\nBASELINE_PROMPT_TEXT\n</baseline_prompt>", prompt)
        self.assertIn("<target_option>\nA\n</target_option>", prompt)
        self.assertIn("<hint_text>\nHINTED_PROMPT_TEXT\n</hint_text>", prompt)
        self.assertIn("expert", prompt)

    def test_audit_keys_stored_in_record(self):
        payload = dict(FAITHFUL_PAYLOAD, hint_role="credited", hint_quote="the key lists A")
        client = make_client([make_usage_response(json.dumps(payload))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(make_batch_row(), "prov/model")
        self.assertEqual(rec["hint_role"], "credited")
        self.assertEqual(rec["hint_quote"], "the key lists A")

    def test_falls_back_to_hinted_prompt_without_prompt_column(self):
        row = make_batch_row()
        del row["prompt"]
        client = make_client([make_usage_response(json.dumps(UNFAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(row, "prov/model")
        self.assertIsNone(rec["error"])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("<baseline_prompt>\nHINTED_PROMPT_TEXT\n</baseline_prompt>", prompt)

    def test_prefers_stored_reasoning_column(self):
        row = make_batch_row(reasoning="STORED_COT_TEXT")
        client = make_client([make_usage_response(json.dumps(UNFAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(row, "prov/model")
        self.assertIsNone(rec["error"])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("<reasoning_trace>\nSTORED_COT_TEXT\n</reasoning_trace>", prompt)

    def test_extracts_cot_when_reasoning_column_missing_or_blank(self):
        for row in (make_batch_row(), make_batch_row(reasoning=""), make_batch_row(reasoning=float("nan"))):
            client = make_client([make_usage_response(json.dumps(UNFAITHFUL_PAYLOAD))])
            with patch("src.lib.llm_judge_verb._get_client", return_value=client):
                judge_case_record(row, "prov/model")
            prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
            self.assertIn(
                "<reasoning_trace>\nBecause the hint says A.\n</reasoning_trace>", prompt
            )

    def test_api_errors_back_off_and_land_in_error_field(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = make_transient_error("429")
        with patch("src.lib.llm_judge_verb._get_client", return_value=client), \
             patch("src.lib.llm_judge_verb.time.sleep") as sleep:
            rec = judge_case_record(make_batch_row(), "prov/model", retries=3)
        self.assertIn("429", rec["error"])
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(sleep.call_count, 3)

    def test_unparseable_response_retries_then_errors(self):
        client = make_client([make_response("not json")] * 2)
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(make_batch_row(), "prov/model", retries=2)
        self.assertIn("invalid/unparseable", rec["error"])


class TestExtractResponseJson(unittest.TestCase):

    PAYLOAD = {"reasoning": "ok", "label": 1, "confidence": 0.9}

    def test_plain_and_fenced(self):
        from src.lib.llm_judge_verb import extract_response_json

        raw = json.dumps(self.PAYLOAD)
        self.assertEqual(extract_response_json(raw), self.PAYLOAD)
        self.assertEqual(extract_response_json(f"```json\n{raw}\n```"), self.PAYLOAD)
        self.assertEqual(extract_response_json(f"```\n{raw}\n```"), self.PAYLOAD)

    def test_preamble_and_trailing_prose(self):
        from src.lib.llm_judge_verb import extract_response_json

        raw = json.dumps(self.PAYLOAD)
        self.assertEqual(
            extract_response_json(f"Sure, here is my verdict:\n{raw}\nHope that helps."),
            self.PAYLOAD,
        )

    def test_brace_inside_quoted_reasoning(self):
        from src.lib.llm_judge_verb import extract_response_json

        payload = dict(self.PAYLOAD, reasoning='the trace says "}" and "{x}" early')
        self.assertEqual(extract_response_json(json.dumps(payload) + " tail"), payload)

    def test_escaped_quote_and_brace(self):
        from src.lib.llm_judge_verb import extract_response_json

        payload = dict(self.PAYLOAD, reasoning='model wrote \\"}\\" then quit')
        self.assertEqual(extract_response_json(json.dumps(payload)), payload)

    def test_stray_brace_in_preamble_is_skipped(self):
        from src.lib.llm_judge_verb import extract_response_json

        raw = json.dumps(self.PAYLOAD)
        self.assertEqual(extract_response_json(f"Note {{not json}} then {raw}"), self.PAYLOAD)

    def test_non_object_json_is_ignored(self):
        from src.lib.llm_judge_verb import extract_response_json

        self.assertIsNone(extract_response_json("[1, 2, 3]"))
        self.assertIsNone(extract_response_json("42"))
        self.assertIsNone(extract_response_json("no json here"))
        self.assertIsNone(extract_response_json('{"unterminated": '))

    def test_first_object_wins(self):
        from src.lib.llm_judge_verb import extract_response_json

        first = json.dumps(self.PAYLOAD)
        second = json.dumps(dict(self.PAYLOAD, label=0))
        self.assertEqual(extract_response_json(f"{first}\n{second}"), self.PAYLOAD)


class TestHintDescriptionContext(unittest.TestCase):

    def test_positive_changed_row_is_bare_description(self):
        from src.lib.hints import HINTS
        from src.lib.llm_judge_verb import hint_description_for

        desc = hint_description_for(
            "authority", sample_type="positive", changed_to_hint=True
        )
        self.assertEqual(desc, HINTS["authority"]["description"])
        self.assertNotIn("NOTE:", desc)

    def test_negative_case_gets_correct_option_note(self):
        from src.lib.llm_judge_verb import hint_description_for

        desc = hint_description_for("authority", sample_type="negative")
        self.assertIn("actually correct", desc)

    def test_unchanged_row_gets_no_change_note(self):
        from src.lib.llm_judge_verb import hint_description_for

        desc = hint_description_for("authority", changed_to_hint=False)
        self.assertIn("did NOT match", desc)

    def test_unknown_hint_falls_back_to_name(self):
        from src.lib.llm_judge_verb import hint_description_for

        self.assertEqual(hint_description_for("no_such_hint"), "no_such_hint")


class TestJudgeBatchCache(unittest.TestCase):
    def write_cache(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def cached_rec(self, original_index, hint_name, error=None):
        return {
            "original_index": original_index,
            "hint_name": hint_name,
            "judge_model": "prov/model",
            "label": 0,
            "confidence": 0.9,
            "reasoning": "cached",
            "error": error,
        }

    def test_cache_path_has_binary_suffix_and_prompt_hash(self):
        from pathlib import Path

        from src.lib.llm_judge_verb import prompt_template_hash

        path = judge_cache_path(Path("/tmp/cache"), "prov/model:free")
        self.assertEqual(
            path.name, f"prov_model_free_binary_{prompt_template_hash(None)}.jsonl"
        )
        self.assertTrue(path.name.startswith("prov_model_free_binary_"))

    def test_cache_path_differs_per_prompt_template(self):
        from pathlib import Path

        custom = "{baseline_prompt}{hint_description}{target_option}{model_answer}{reasoning_trace}"
        default_path = judge_cache_path(Path("/c"), "prov/model")
        custom_path = judge_cache_path(Path("/c"), "prov/model", custom)
        self.assertNotEqual(default_path, custom_path)
        self.assertEqual(custom_path, judge_cache_path(Path("/c"), "prov/model", custom))

    def test_load_keeps_last_record_and_skips_truncated_lines(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prov_model_binary.jsonl"
            self.write_cache(path, [
                self.cached_rec(0, "expert_opinion", error="boom"),
                self.cached_rec(0, "expert_opinion"),
            ])
            with path.open("a") as f:
                f.write('{"original_index": 1, "hint_')  # interrupted write
            records = load_judged_records(path)
            self.assertEqual(len(records), 1)
            self.assertIsNone(records[(0, "expert_opinion")]["error"])

    def test_batch_judges_missing_and_errored_only(self):
        import pandas as pd
        import tempfile
        from pathlib import Path

        rows = pd.DataFrame([
            make_batch_row(0, "expert_opinion"),   # cached ok -> skipped
            make_batch_row(1, "expert_opinion"),   # cached error -> retried
            make_batch_row(2, "expert_opinion"),   # missing -> judged
        ])
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            path = judge_cache_path(cache_dir, "prov/model")
            self.write_cache(path, [
                self.cached_rec(0, "expert_opinion"),
                self.cached_rec(1, "expert_opinion", error="boom"),
            ])
            client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))] * 2)
            with patch("src.lib.llm_judge_verb._get_client", return_value=client):
                records = run_judge_batch(
                    rows, "prov/model", cache_dir, max_workers=1, progress=None
                )
            self.assertEqual(client.chat.completions.create.call_count, 2)
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r["error"] is None for r in records.values()))

            # Second run: everything cached, no API calls.
            client2 = make_client([])
            with patch("src.lib.llm_judge_verb._get_client", return_value=client2):
                records = run_judge_batch(
                    rows, "prov/model", cache_dir, max_workers=1, progress=None
                )
            self.assertEqual(client2.chat.completions.create.call_count, 0)
            self.assertEqual(len(records), 3)

    def test_retry_errors_false_keeps_cached_errors(self):
        import pandas as pd
        import tempfile
        from pathlib import Path

        rows = pd.DataFrame([
            make_batch_row(0, "expert_opinion"),   # cached ok -> skipped
            make_batch_row(1, "expert_opinion"),   # cached error -> kept
            make_batch_row(2, "expert_opinion"),   # missing -> judged
        ])
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            path = judge_cache_path(cache_dir, "prov/model")
            self.write_cache(path, [
                self.cached_rec(0, "expert_opinion"),
                self.cached_rec(1, "expert_opinion", error="boom"),
            ])
            client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))])
            lines = []
            with patch("src.lib.llm_judge_verb._get_client", return_value=client):
                records = run_judge_batch(
                    rows, "prov/model", cache_dir, max_workers=1,
                    progress=lines.append, retry_errors=False,
                )
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(records[(1, "expert_opinion")]["error"], "boom")
            self.assertIsNone(records[(2, "expert_opinion")]["error"])
            self.assertIn("1 errored kept", lines[0])


class TestContentAddressedCache(unittest.TestCase):
    """A record is served only when its ``input_sha`` matches the row's current inputs."""

    def write_cache(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_sha_tracks_every_judge_input_and_nothing_else(self):
        from src.lib.llm_judge_verb import judge_input_sha

        base = judge_input_sha(make_batch_row())
        self.assertEqual(len(base), 16)
        self.assertEqual(base, judge_input_sha(make_batch_row()))
        # Unrelated columns and the row's key do not change the hash.
        self.assertEqual(base, judge_input_sha(make_batch_row(original_index=99, extra="x")))
        # Every rendered input does.
        for change in (
            {"prompt": "OTHER BASELINE"},
            {"hinted_prompt": "OTHER HINTED"},        # → hint excerpt
            {"hinted_answer": "B"},
            {"final_answer": "B"},
            {"rollout": "<think>Other trace.</think>\n<answer>A</answer>"},
            {"sample_type": "negative"},
        ):
            self.assertNotEqual(base, judge_input_sha(make_batch_row(**change)), change)
        # A `reasoning` column wins over the rollout, like the judge prompt.
        with_col = judge_input_sha(make_batch_row(reasoning="Because the hint says A."))
        self.assertEqual(with_col, base)
        self.assertNotEqual(base, judge_input_sha(make_batch_row(reasoning="other")))

    def test_record_carries_sha_and_budget(self):
        import tempfile
        from pathlib import Path

        from src.lib.llm_judge_verb import judge_case_record, judge_input_sha

        row = make_batch_row()
        client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(row, "prov/model", max_tokens=4096)
        self.assertEqual(rec["input_sha"], judge_input_sha(row))
        self.assertEqual(rec["max_tokens"], 4096)
        # An error record hashes the same input so the retry recognises it.
        client = make_client([make_choiceless_response()] * 5)
        with patch("src.lib.llm_judge_verb._get_client", return_value=client), \
                patch("src.lib.llm_judge_verb.time.sleep"):
            rec = judge_case_record(row, "prov/model", retries=1)
        self.assertIsNotNone(rec["error"])
        self.assertEqual(rec["input_sha"], judge_input_sha(row))

    def test_length_failure_retries_once_with_double_budget(self):
        from src.lib.llm_judge_verb import JUDGE_MAX_TOKENS_CAP, judge_case_record

        client = make_client([make_empty_response(), make_response(json.dumps(FAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(make_batch_row(), "prov/model", max_tokens=8192)
        budgets = [c.kwargs["max_tokens"] for c in client.chat.completions.create.call_args_list]
        self.assertEqual(budgets, [8192, 16384])
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["max_tokens"], 16384)

        # At the cap a second `length` gives up instead of burning retries.
        client = make_client([make_empty_response()] * 5)
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(make_batch_row(), "prov/model", max_tokens=JUDGE_MAX_TOKENS_CAP)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertIn("finish_reason=length", rec["error"])

    def test_non_json_response_body_is_retried_not_raised(self):
        from src.lib.llm_judge_verb import judge_case_record

        bad_body = json.JSONDecodeError("Expecting value", "\n\n\n", 3)
        client = make_client([bad_body, make_response(json.dumps(FAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client), \
                patch("src.lib.llm_judge_verb.time.sleep"):
            rec = judge_case_record(make_batch_row(), "prov/model")
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["label"], FAITHFUL_PAYLOAD["label"])

        client = make_client([bad_body] * 3)
        with patch("src.lib.llm_judge_verb._get_client", return_value=client), \
                patch("src.lib.llm_judge_verb.time.sleep"):
            rec = judge_case_record(make_batch_row(), "prov/model", retries=3)
        self.assertIn("JSONDecodeError", rec["error"])

    def test_changed_input_is_rejudged_and_new_record_wins(self):
        import pandas as pd
        import tempfile
        from pathlib import Path

        from src.lib.llm_judge_verb import judge_input_sha

        old_row = make_batch_row(0, "expert_opinion", final_answer="A")
        new_row = make_batch_row(0, "expert_opinion", final_answer="B", hinted_answer="B")
        stale = dict(
            original_index=0, hint_name="expert_opinion", judge_model="prov/model",
            label=0, confidence=0.9, reasoning="stale", error=None,
            input_sha=judge_input_sha(old_row),
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            path = judge_cache_path(cache_dir, "prov/model")
            self.write_cache(path, [stale])
            client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))])
            lines = []
            with patch("src.lib.llm_judge_verb._get_client", return_value=client):
                records = run_judge_batch(
                    pd.DataFrame([new_row]), "prov/model", cache_dir,
                    max_workers=1, progress=lines.append,
                )
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual(records[(0, "expert_opinion")]["label"], 1)
            self.assertEqual(records[(0, "expert_opinion")]["input_sha"], judge_input_sha(new_row))
            self.assertIn("1 stale (input changed)", lines[0])
            self.assertEqual(load_judged_records(path)[(0, "expert_opinion")]["reasoning"],
                             FAITHFUL_PAYLOAD["reasoning"])
            client2 = make_client([])
            with patch("src.lib.llm_judge_verb._get_client", return_value=client2):
                run_judge_batch(pd.DataFrame([new_row]), "prov/model", cache_dir,
                                max_workers=1, progress=None)
            self.assertEqual(client2.chat.completions.create.call_count, 0)

    def test_legacy_records_policy(self):
        import pandas as pd
        import tempfile
        from pathlib import Path

        from src.lib.llm_judge_verb import cache_hit

        legacy = dict(
            original_index=0, hint_name="expert_opinion", judge_model="prov/model",
            label=0, confidence=0.9, reasoning="legacy", error=None,
        )
        row = make_batch_row(0, "expert_opinion")
        self.assertTrue(cache_hit(legacy, row))
        self.assertTrue(cache_hit(legacy, row, legacy_records="trust"))
        self.assertFalse(cache_hit(legacy, row, legacy_records="rejudge"))
        self.assertFalse(cache_hit(None, row))
        with self.assertRaises(ValueError):
            cache_hit(legacy, row, legacy_records="maybe")
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            self.write_cache(judge_cache_path(cache_dir, "prov/model"), [legacy])
            client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))])
            with patch("src.lib.llm_judge_verb._get_client", return_value=client):
                run_judge_batch(pd.DataFrame([row]), "prov/model", cache_dir,
                                max_workers=1, progress=None)  # trust: no call
                self.assertEqual(client.chat.completions.create.call_count, 0)
                run_judge_batch(pd.DataFrame([row]), "prov/model", cache_dir,
                                max_workers=1, progress=None, legacy_records="rejudge")
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_batch_passes_the_budget_to_every_call(self):
        import pandas as pd
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            client = make_client([make_response(json.dumps(FAITHFUL_PAYLOAD))])
            with patch("src.lib.llm_judge_verb._get_client", return_value=client):
                run_judge_batch(pd.DataFrame([make_batch_row()]), "prov/model", Path(tmp),
                                max_workers=1, progress=None, judge_max_tokens=12345)
            self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 12345)


class TestCheckJudgeModel(unittest.TestCase):

    def test_served_request_returns_true(self):
        from src.lib.llm_judge_verb import check_judge_model

        client = make_client([make_response("OK")])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            self.assertTrue(check_judge_model("prov/model"))
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "prov/model")
        self.assertEqual(kwargs["temperature"], 0)
        self.assertGreaterEqual(kwargs["max_tokens"], 16)

    def test_unknown_model_raises_setup_error(self):
        import openai

        from src.lib.llm_judge_verb import check_judge_model

        client = MagicMock()
        client.chat.completions.create.side_effect = make_api_error(
            openai.NotFoundError, 404, "no such model"
        )
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            with self.assertRaises(JudgeSetupError) as ctx:
                check_judge_model("prov/typo")
        self.assertIn("prov/typo", str(ctx.exception))
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_transient_error_returns_false(self):
        import io
        from contextlib import redirect_stdout

        from src.lib.llm_judge_verb import check_judge_model

        client = MagicMock()
        client.chat.completions.create.side_effect = make_transient_error()
        with patch("src.lib.llm_judge_verb._get_client", return_value=client), \
             redirect_stdout(io.StringIO()):
            self.assertFalse(check_judge_model("prov/model"))

    def test_missing_key_raises_setup_error(self):
        import os

        from src.lib import llm_judge_verb as mod

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}), \
             patch.object(mod, "_client", None):
            with self.assertRaises(JudgeSetupError):
                mod.check_judge_model("prov/model")


class TestHintTextInput(unittest.TestCase):
    """The judge's ``{hint_text}`` input is recomputed from the row per call."""

    def _prompt_for(self, row):
        client = make_client([make_usage_response(json.dumps(UNFAITHFUL_PAYLOAD))])
        with patch("src.lib.llm_judge_verb._get_client", return_value=client):
            rec = judge_case_record(row, "prov/model")
        self.assertIsNone(rec["error"])
        return client.chat.completions.create.call_args.kwargs["messages"][0]["content"]

    def test_recovered_from_hinted_prompt(self):
        row = make_batch_row(hinted_prompt="BASELINE_PROMPT_TEXT\n\nEXPERT SAYS A")
        self.assertNotIn("hint_text", row)
        prompt = self._prompt_for(row)
        self.assertIn("<hint_text>\nEXPERT SAYS A\n</hint_text>", prompt)

    def test_unrecoverable_row_renders_unavailable_marker(self):
        prompt = self._prompt_for(make_batch_row(hinted_prompt=""))
        self.assertIn(f"<hint_text>\n{HINT_TEXT_UNAVAILABLE}\n</hint_text>", prompt)
        self.assertNotIn("<hint_text>\n\n</hint_text>", prompt)

    def test_template_without_hint_text_still_loads(self):
        body = "".join("<%s>{%s}</%s>\n" % (p, p, p) for p in PROMPT_PLACEHOLDERS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.txt"
            path.write_text(body)
            template = load_prompt_template(path)
            self.assertNotIn("{hint_text}", template)
            rendered = template.format(
                **{p: "v" for p in PROMPT_PLACEHOLDERS}, hint_text="ignored"
            )
            self.assertNotIn("ignored", rendered)


if __name__ == "__main__":
    unittest.main()
