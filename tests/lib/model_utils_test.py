import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import src.lib.model_utils as mu
from src.lib.model_utils import (
    GEMMA4_DELIMITERS,
    MODEL_CONFIGS,
    MODEL_CONFIG_DEFAULTS,
    build_chat_messages,
    coerce_thinking,
    generate_rollouts_batch,
    get_model_config,
    get_model_short_name,
    get_thinking_config,
    load_model_hf,
    load_model_vLLM,
    template_kwargs,
)
from src.lib.parsing import DEFAULT_DELIMITERS, ReasoningDelimiters
from src.lib.prompts import (
    SYSTEM_PROMPT_NATIVE_THINKING,
    SYSTEM_PROMPT_PLAIN,
    get_prompted_thinking_prompt,
)


def fake_vllm_output(texts):
    return SimpleNamespace(outputs=[SimpleNamespace(text=t) for t in texts])


def make_generation_mocks(outputs_per_prompt):
    model = MagicMock()
    model.generate.return_value = [fake_vllm_output(ts) for ts in outputs_per_prompt]
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = (
        lambda messages, **kwargs: f"templated:{messages[-1]['content']}"
    )
    return model, tokenizer


class TestModelConfigs(unittest.TestCase):
    """The MODEL_CONFIGS registry and its lookup helpers."""

    def test_every_config_has_required_keys(self):
        for name, cfg in MODEL_CONFIGS.items():
            self.assertIn("short_name", cfg)
            self.assertIn("thinking_mode", cfg)

    def test_thinking_mode_values_are_valid(self):
        for cfg in MODEL_CONFIGS.values():
            self.assertIn(cfg["thinking_mode"], {"native", "prompted"})

    def test_short_names_are_filesystem_safe(self):
        for cfg in MODEL_CONFIGS.values():
            short = cfg["short_name"]
            self.assertNotIn("/", short)
            self.assertNotIn(" ", short)
            self.assertEqual(short, short.lower())

    def test_short_names_are_unique(self):
        shorts = [cfg["short_name"] for cfg in MODEL_CONFIGS.values()]
        self.assertEqual(len(shorts), len(set(shorts)))

    def test_get_model_config_known_model(self):
        cfg = get_model_config("Qwen/Qwen3.5-27B")
        self.assertEqual(cfg["short_name"], "qwen3.5-27b")
        self.assertEqual(cfg["thinking_mode"], "native")

    def test_get_model_config_unknown_model_fallback(self):
        cfg = get_model_config("someorg/Some-New-Model")
        self.assertEqual(cfg["short_name"], "some-new-model")
        self.assertEqual(cfg["thinking_mode"], "prompted")

    def test_get_model_short_name_known_models(self):
        self.assertEqual(get_model_short_name("Qwen/Qwen3-8B"), "qwen3-8b")
        self.assertEqual(get_model_short_name("google/gemma-3-27b-it"), "gemma3-27b-it")

    def test_get_model_short_name_unknown_model(self):
        self.assertEqual(get_model_short_name("meta-llama/Llama-3-8B"), "llama-3-8b")

    def test_get_model_config_fills_defaults(self):
        for name in ["Qwen/Qwen3.5-9B", "someorg/Some-New-Model"]:
            cfg = get_model_config(name)
            for key in MODEL_CONFIG_DEFAULTS:
                self.assertIn(key, cfg, f"{name} missing {key}")

    def test_lookup_is_case_insensitive(self):
        cfg = get_model_config("google/gemma-4-31b-it")
        self.assertEqual(cfg["short_name"], "gemma4-31b-it")
        self.assertEqual(cfg["thinking_mode"], "native")
        self.assertEqual(cfg["delimiters"], GEMMA4_DELIMITERS)

    def test_delimiters_are_reasoning_delimiters(self):
        for name in MODEL_CONFIGS:
            self.assertIsInstance(
                get_model_config(name)["delimiters"], ReasoningDelimiters
            )

    def test_gemma4_entries_use_channel_delimiters(self):
        for name in ["google/gemma-4-12B-it", "google/gemma-4-31B-it"]:
            cfg = get_model_config(name)
            self.assertEqual(cfg["thinking_mode"], "native")
            self.assertEqual(cfg["delimiters"].open, "<|channel>thought")
            self.assertEqual(cfg["delimiters"].close, "<channel|>")

    def test_new_models_are_registered(self):
        for name in [
            "Qwen/Qwen3.5-9B",
            "google/gemma-4-12B-it",
            "google/gemma-4-31B-it",
            "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            "allenai/Olmo-3-7B-Think",
        ]:
            self.assertIn(name, MODEL_CONFIGS)

    def test_olmo3_think_entry(self):
        cfg = get_model_config("allenai/Olmo-3-7B-Think")
        self.assertEqual(cfg["short_name"], "olmo3-7b-think")
        self.assertEqual(cfg["thinking_mode"], "native")
        self.assertEqual(cfg["delimiters"].open, "<think>")
        self.assertEqual(cfg["delimiters"].close, "</think>")
        self.assertEqual((cfg["n_layers"], cfg["hidden_size"]), (32, 4096))
        self.assertEqual(cfg["vllm_extra_kwargs"], {})
        self.assertFalse(cfg["supports_thinking_off"])

    def test_medqa_comparison_models_registered(self):
        expected = {
            "Qwen/Qwen3-8B": ("qwen3-8b", "native", 36, 4096, True),
            "google/gemma-3-4b-it": ("gemma3-4b-it", "prompted", 34, 2560, True),
            "NousResearch/Meta-Llama-3.1-8B-Instruct": (
                "llama3.1-8b-instruct", "prompted", 32, 4096, True),
        }
        for name, (short, mode, layers, hidden, off_ok) in expected.items():
            self.assertIn(name, MODEL_CONFIGS)
            cfg = get_model_config(name)
            self.assertEqual(cfg["short_name"], short)
            self.assertEqual(cfg["thinking_mode"], mode, name)
            self.assertEqual((cfg["n_layers"], cfg["hidden_size"]), (layers, hidden), name)
            self.assertEqual(cfg["supports_thinking_off"], off_ok, name)

    def test_prompted_models_get_the_tag_naming_prompt(self):
        # A prompted model must be told the delimiters; a native one must not be.
        for name in ("google/gemma-3-4b-it",
                     "NousResearch/Meta-Llama-3.1-8B-Instruct"):
            cfg = get_thinking_config(name, "on")
            self.assertEqual(cfg.mode, "prompted", name)
            self.assertIsNone(cfg.enable_thinking, name)
            self.assertIn(cfg.delimiters.close, cfg.system_prompt, name)
        native = get_thinking_config("Qwen/Qwen3-8B", "on")
        self.assertEqual(native.mode, "native")
        self.assertIs(native.enable_thinking, True)
        self.assertNotIn(native.delimiters.close, native.system_prompt)

    def test_nemotron_nano_entry(self):
        cfg = get_model_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        self.assertEqual(cfg["short_name"], "nemotron-nano-9b-v2")
        self.assertEqual(cfg["thinking_mode"], "native")
        self.assertEqual(cfg["delimiters"].open, "<think>")
        self.assertEqual(cfg["delimiters"].close, "</think>")
        self.assertEqual((cfg["n_layers"], cfg["hidden_size"]), (56, 4480))
        self.assertEqual(cfg["vllm_extra_kwargs"], {})
        self.assertFalse(cfg["supports_thinking_off"])

    def test_supports_thinking_off_defaults_true(self):
        opted_out = {
            n for n in MODEL_CONFIGS
            if not get_model_config(n)["supports_thinking_off"]
        }
        self.assertEqual(
            opted_out,
            {"nvidia/NVIDIA-Nemotron-Nano-9B-v2", "allenai/Olmo-3-7B-Think"},
        )
        # Opting out only makes sense for a native reasoner.
        for name in opted_out:
            self.assertEqual(get_model_config(name)["thinking_mode"], "native", name)
        self.assertTrue(get_model_config("Qwen/Qwen3.5-9B")["supports_thinking_off"])
        self.assertTrue(get_model_config("some/unknown-model")["supports_thinking_off"])

    def test_gdn_kwarg_only_on_qwen(self):
        for name in MODEL_CONFIGS:
            has_gdn = "gdn_prefill_backend" in get_model_config(name)["vllm_extra_kwargs"]
            self.assertEqual(has_gdn, name.startswith("Qwen/"), name)


class TestGetThinkingConfig(unittest.TestCase):
    """Thinking on/off resolution."""

    def test_native_model_thinking_on(self):
        cfg = get_thinking_config("Qwen/Qwen3.5-9B", "on")
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.mode, "native")
        self.assertIs(cfg.enable_thinking, True)
        self.assertEqual(cfg.system_prompt, SYSTEM_PROMPT_NATIVE_THINKING)
        self.assertEqual(cfg.delimiters, DEFAULT_DELIMITERS)

    def test_native_model_thinking_off(self):
        cfg = get_thinking_config("Qwen/Qwen3.5-9B", "off")
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.mode, "none")
        self.assertIs(cfg.enable_thinking, False)
        self.assertEqual(cfg.system_prompt, SYSTEM_PROMPT_PLAIN)

    def test_thinking_off_refused_when_unsupported(self):
        with self.assertRaisesRegex(ValueError, "cannot run with thinking off"):
            get_thinking_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "off")
        with self.assertRaisesRegex(ValueError, "cannot run with thinking off"):
            get_thinking_config("allenai/Olmo-3-7B-Think", "off")
        # Thinking ON is unaffected.
        cfg = get_thinking_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "on")
        self.assertEqual(cfg.mode, "native")
        self.assertIs(cfg.enable_thinking, True)

    def test_prompted_model_thinking_on(self):
        cfg = get_thinking_config("google/gemma-3-27b-it", "on")
        self.assertEqual(cfg.mode, "prompted")
        self.assertIsNone(cfg.enable_thinking)
        self.assertEqual(cfg.system_prompt, get_prompted_thinking_prompt())

    def test_prompted_model_thinking_off(self):
        cfg = get_thinking_config("google/gemma-3-27b-it", "off")
        self.assertEqual(cfg.mode, "none")
        self.assertIsNone(cfg.enable_thinking)
        self.assertEqual(cfg.system_prompt, SYSTEM_PROMPT_PLAIN)

    def test_default_follows_registry(self):
        self.assertTrue(get_thinking_config("Qwen/Qwen3.5-9B").enabled)
        self.assertTrue(get_thinking_config("google/gemma-4-31B-it").enabled)

    def test_gemma4_prompt_uses_channel_delimiters(self):
        cfg = get_thinking_config("google/gemma-4-31B-it", "on")
        self.assertEqual(cfg.delimiters, GEMMA4_DELIMITERS)
        self.assertIs(cfg.enable_thinking, True)

    def test_unknown_model_defaults_to_prompted(self):
        cfg = get_thinking_config("someorg/Some-New-Model")
        self.assertEqual(cfg.mode, "prompted")
        self.assertIsNone(cfg.enable_thinking)
        self.assertEqual(cfg.delimiters, DEFAULT_DELIMITERS)


class TestCoerceThinking(unittest.TestCase):
    """Normalization of the `thinking` config/CLI value."""

    def test_none_and_empty_are_unset(self):
        self.assertIsNone(coerce_thinking(None))
        self.assertIsNone(coerce_thinking(""))
        self.assertIsNone(coerce_thinking("   "))

    def test_truthy_spellings(self):
        for v in ["on", "ON", "true", "yes", "1", "enabled"]:
            self.assertIs(coerce_thinking(v), True, v)

    def test_falsy_spellings(self):
        for v in ["off", "OFF", "false", "no", "0", "disabled"]:
            self.assertIs(coerce_thinking(v), False, v)

    def test_yaml_booleans_pass_through(self):
        self.assertIs(coerce_thinking(True), True)
        self.assertIs(coerce_thinking(False), False)

    def test_unrecognized_value_raises(self):
        with self.assertRaises(ValueError):
            coerce_thinking("maybe")


class TestTemplateKwargs(unittest.TestCase):
    """The apply_chat_template kwarg helper."""

    def test_none_omits_kwarg(self):
        self.assertEqual(template_kwargs(None), {})

    def test_bool_forwards_kwarg(self):
        self.assertEqual(template_kwargs(True), {"enable_thinking": True})
        self.assertEqual(template_kwargs(False), {"enable_thinking": False})


class TestBuildChatMessages(unittest.TestCase):
    """Chat-message list construction."""

    def test_user_only(self):
        msgs = build_chat_messages("hello")
        self.assertEqual(msgs, [{"role": "user", "content": "hello"}])

    def test_with_system_prompt(self):
        msgs = build_chat_messages("hello", system_prompt="be brief")
        self.assertEqual(
            msgs,
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
            ],
        )


class TestLoadModelVLLM(unittest.TestCase):
    """The vLLM loader with the LLM constructor mocked."""

    def test_constructs_llm_with_expected_kwargs(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("Qwen/Qwen3-8B")
        kwargs = mock_llm.call_args.kwargs
        self.assertEqual(kwargs["model"], "Qwen/Qwen3-8B")
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertEqual(kwargs["max_model_len"], 16384)
        self.assertEqual(kwargs["gpu_memory_utilization"], 0.9)
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["tensor_parallel_size"], 1)

    def test_returns_llm_and_its_tokenizer(self):
        with patch.object(mu, "LLM") as mock_llm:
            llm, tokenizer = load_model_vLLM("Qwen/Qwen3-8B")
        self.assertIs(llm, mock_llm.return_value)
        self.assertIs(tokenizer, mock_llm.return_value.get_tokenizer.return_value)

    def test_tensor_parallel_size_forwarded(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("Qwen/Qwen3-8B", tensor_parallel_size=4)
        self.assertEqual(mock_llm.call_args.kwargs["tensor_parallel_size"], 4)

    def test_qwen_gets_gdn_prefill_backend(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("Qwen/Qwen3.5-9B")
        self.assertEqual(
            mock_llm.call_args.kwargs["gdn_prefill_backend"], "triton"
        )

    def test_gemma_omits_gdn_prefill_backend(self):
        for name in ["google/gemma-4-31B-it", "google/gemma-3-27b-it"]:
            with patch.object(mu, "LLM") as mock_llm:
                load_model_vLLM(name)
            self.assertNotIn("gdn_prefill_backend", mock_llm.call_args.kwargs, name)

    def test_unknown_model_omits_gdn_prefill_backend(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("someorg/Some-New-Model")
        self.assertNotIn("gdn_prefill_backend", mock_llm.call_args.kwargs)

    def test_explicit_gdn_backend_overrides_registry(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("Qwen/Qwen3-8B", gdn_prefill_backend="flashinfer")
        self.assertEqual(
            mock_llm.call_args.kwargs["gdn_prefill_backend"], "flashinfer"
        )

    def test_max_num_seqs_forwarded(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("Qwen/Qwen3-8B", max_num_seqs=256)
        self.assertEqual(mock_llm.call_args.kwargs["max_num_seqs"], 256)

    def test_max_num_seqs_omitted_by_default(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM("Qwen/Qwen3-8B")
        self.assertNotIn("max_num_seqs", mock_llm.call_args.kwargs)

    def test_extra_kwargs_merge_over_registry(self):
        with patch.object(mu, "LLM") as mock_llm:
            load_model_vLLM(
                "Qwen/Qwen3-8B",
                extra_kwargs={"gdn_prefill_backend": "flashinfer", "seed": 7},
            )
        kwargs = mock_llm.call_args.kwargs
        self.assertEqual(kwargs["gdn_prefill_backend"], "flashinfer")
        self.assertEqual(kwargs["seed"], 7)

    def test_heterogeneous_config_error_is_reraised_with_remedy(self):
        # The real class is absent from the pinned transformers, so the guard matches on the name.
        class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
            pass

        original = AmbiguousGlobalPerLayerAttributeError(
            "'head_dim' is a per-layer attribute and may vary across layers."
        )
        with patch.object(mu, "LLM", side_effect=original):
            with self.assertRaises(RuntimeError) as ctx:
                load_model_vLLM("google/gemma-4-12B-it")
        message = str(ctx.exception)
        self.assertIn("transformers", message)
        self.assertIn("5.15", message)
        self.assertIn("google/gemma-4-12B-it", message)
        self.assertIs(ctx.exception.__cause__, original)

    def test_unrelated_runtime_error_propagates(self):
        original = RuntimeError("CUDA out of memory")
        with patch.object(mu, "LLM", side_effect=original):
            with self.assertRaises(RuntimeError) as ctx:
                load_model_vLLM("Qwen/Qwen3-8B")
        self.assertIs(ctx.exception, original)


class TestLoadModelHF(unittest.TestCase):
    """The HuggingFace loader with from_pretrained mocked."""

    def _load(self, model_name="Qwen/Qwen3-8B", **kwargs):
        with patch.object(mu, "AutoTokenizer") as mock_tok, patch.object(
            mu, "AutoModelForCausalLM"
        ) as mock_model:
            result = load_model_hf(model_name, **kwargs)
        return result, mock_tok, mock_model

    def test_trust_remote_code_follows_registry(self):
        _, mock_tok, mock_model = self._load("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        self.assertFalse(mock_tok.from_pretrained.call_args.kwargs["trust_remote_code"])
        self.assertFalse(mock_model.from_pretrained.call_args.kwargs["trust_remote_code"])
        _, mock_tok, mock_model = self._load("nvidia/NVIDIA-Nemotron-Nano-9B-v2", trust_remote_code=True)
        self.assertTrue(mock_model.from_pretrained.call_args.kwargs["trust_remote_code"])
        self.assertTrue(mu.get_model_config("Qwen/Qwen3-8B")["hf_trust_remote_code"])

    def test_loads_tokenizer_and_model(self):
        (model, tokenizer), mock_tok, mock_model = self._load()
        self.assertIs(tokenizer, mock_tok.from_pretrained.return_value)
        self.assertIs(model, mock_model.from_pretrained.return_value)
        tok_args, tok_kwargs = mock_tok.from_pretrained.call_args
        self.assertEqual(tok_args[0], "Qwen/Qwen3-8B")
        self.assertTrue(tok_kwargs["trust_remote_code"])
        model_args, model_kwargs = mock_model.from_pretrained.call_args
        self.assertEqual(model_args[0], "Qwen/Qwen3-8B")
        self.assertTrue(model_kwargs["trust_remote_code"])
        self.assertEqual(model_kwargs["device_map"], "auto")
        self.assertEqual(model_kwargs["attn_implementation"], "sdpa")

    def test_default_dtype_is_bfloat16(self):
        _, _, mock_model = self._load()
        self.assertEqual(
            mock_model.from_pretrained.call_args.kwargs["torch_dtype"], torch.bfloat16
        )

    def test_dtype_mapping(self):
        _, _, mock_model = self._load(dtype="half")
        self.assertEqual(
            mock_model.from_pretrained.call_args.kwargs["torch_dtype"], torch.float16
        )
        _, _, mock_model = self._load(dtype="float")
        self.assertEqual(
            mock_model.from_pretrained.call_args.kwargs["torch_dtype"], torch.float32
        )

    def test_model_put_in_eval_mode(self):
        (model, _), _, _ = self._load()
        model.eval.assert_called_once()

    def test_generation_config_sampling_params_cleared(self):
        (model, _), _, _ = self._load()
        self.assertIsNone(model.generation_config.temperature)
        self.assertIsNone(model.generation_config.top_p)
        self.assertIsNone(model.generation_config.top_k)

    def test_model_kwargs_reach_from_pretrained(self):
        _, mock_tok, mock_model = self._load(model_kwargs={"chunk_size": 64, "attn_implementation": "eager"})
        kwargs = mock_model.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["chunk_size"], 64)
        self.assertEqual(kwargs["attn_implementation"], "eager")   # an override wins over the loader's own
        self.assertNotIn("chunk_size", mock_tok.from_pretrained.call_args.kwargs)
        _, _, mock_model = self._load(model_kwargs=None)
        self.assertNotIn("chunk_size", mock_model.from_pretrained.call_args.kwargs)

    def test_forward_patch_applied_for_nemotron(self):
        self.assertEqual(
            get_model_config("nvidia/NVIDIA-Nemotron-Nano-9B-v2")["hf_forward_patch"],
            "nemotron_h_memory_efficient",
        )
        patcher = MagicMock(return_value=28)
        with patch.dict(mu._HF_FORWARD_PATCHES, {"nemotron_h_memory_efficient": patcher}):
            (model, _), _, _ = self._load("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        patcher.assert_called_once_with(model)

    def test_forward_patch_not_applied_for_qwen(self):
        self.assertIsNone(get_model_config("Qwen/Qwen3-8B")["hf_forward_patch"])
        patcher = MagicMock(return_value=1)
        with patch.dict(mu._HF_FORWARD_PATCHES, {"nemotron_h_memory_efficient": patcher}):
            self._load("Qwen/Qwen3-8B")
        patcher.assert_not_called()

    def test_unknown_forward_patch_name_raises(self):
        entry = dict(MODEL_CONFIGS["nvidia/NVIDIA-Nemotron-Nano-9B-v2"], hf_forward_patch="no_such_patch")
        with patch.dict(MODEL_CONFIGS, {"nvidia/NVIDIA-Nemotron-Nano-9B-v2": entry}):
            with self.assertRaisesRegex(ValueError, "no_such_patch"):
                self._load("nvidia/NVIDIA-Nemotron-Nano-9B-v2")


class TestLoadModelHFAutoClassFallback(unittest.TestCase):
    """Auto-class selection in the HF loader, with the name→class resolver stubbed."""

    def _resolver(self, classes):
        return lambda name: classes.get(name)

    def _load(self, classes, model_name="google/gemma-4-31B-it", **kwargs):
        with patch.object(mu, "AutoTokenizer"), patch.object(
            mu, "_resolve_auto_class", self._resolver(classes)
        ):
            return load_model_hf(model_name, **kwargs)

    def test_falls_through_to_next_class(self):
        causal, fallback = MagicMock(), MagicMock()
        causal.from_pretrained.side_effect = ValueError(
            "Unrecognized configuration class"
        )
        model, _ = self._load(
            {
                "AutoModelForCausalLM": causal,
                "AutoModelForImageTextToText": fallback,
            }
        )
        self.assertIs(model, fallback.from_pretrained.return_value)
        causal.from_pretrained.assert_called_once()

    def test_missing_class_is_skipped(self):
        fallback = MagicMock()
        model, _ = self._load(
            {
                "AutoModelForCausalLM": None,
                "AutoModelForImageTextToText": fallback,
            }
        )
        self.assertIs(model, fallback.from_pretrained.return_value)

    def test_exhausted_chain_raises_with_all_attempts(self):
        causal = MagicMock()
        causal.from_pretrained.side_effect = ValueError("nope")
        with patch.object(mu, "_HF_AUTO_CLASS_CHAIN", ("AutoModelForCausalLM",)):
            with self.assertRaises(RuntimeError) as ctx:
                self._load({"AutoModelForCausalLM": causal})
        message = str(ctx.exception)
        self.assertIn("AutoModelForCausalLM", message)
        self.assertIn("nope", message)
        self.assertIn("hf_auto_class", message)

    def test_registry_auto_class_short_circuits(self):
        causal, fallback = MagicMock(), MagicMock()
        entry = {
            **MODEL_CONFIGS["google/gemma-4-31B-it"],
            "hf_auto_class": "AutoModelForImageTextToText",
        }
        with patch.dict(MODEL_CONFIGS, {"google/gemma-4-31B-it": entry}):
            model, _ = self._load(
                {
                    "AutoModelForCausalLM": causal,
                    "AutoModelForImageTextToText": fallback,
                }
            )
        self.assertIs(model, fallback.from_pretrained.return_value)
        causal.from_pretrained.assert_not_called()

    def test_explicit_auto_class_argument_wins(self):
        causal, fallback = MagicMock(), MagicMock()
        model, _ = self._load(
            {
                "AutoModelForCausalLM": causal,
                "AutoModelForImageTextToText": fallback,
            },
            model_name="Qwen/Qwen3-8B",
            auto_class="AutoModelForImageTextToText",
        )
        self.assertIs(model, fallback.from_pretrained.return_value)
        causal.from_pretrained.assert_not_called()


class TestGenerationHelpers(unittest.TestCase):
    """The vLLM generation wrapper with model and SamplingParams mocked."""

    def test_generate_applies_chat_template_with_thinking(self):
        model, tokenizer = make_generation_mocks([["r"]])
        with patch.object(mu, "SamplingParams"):
            generate_rollouts_batch(model, tokenizer, [[{"role": "user", "content": "hi"}]])
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        self.assertFalse(kwargs["tokenize"])
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertTrue(kwargs["enable_thinking"])
        prompts = model.generate.call_args.args[0]
        self.assertEqual(prompts, ["templated:hi"])

    def test_generate_disable_thinking_forwarded(self):
        model, tokenizer = make_generation_mocks([["r"]])
        with patch.object(mu, "SamplingParams"):
            generate_rollouts_batch(
                model,
                tokenizer,
                [[{"role": "user", "content": "hi"}]],
                enable_thinking=False,
            )
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        self.assertFalse(kwargs["enable_thinking"])

    def test_sampling_params_forwarded(self):
        model, tokenizer = make_generation_mocks([["a", "b"]])
        with patch.object(mu, "SamplingParams") as mock_sp:
            generate_rollouts_batch(
                model,
                tokenizer,
                [[{"role": "user", "content": "hi"}]],
                n=2,
                max_tokens=64,
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                seed=42,
            )
        kwargs = mock_sp.call_args.kwargs
        self.assertEqual(kwargs["n"], 2)
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(kwargs["top_p"], 0.9)
        self.assertEqual(kwargs["top_k"], 40)
        self.assertEqual(kwargs["seed"], 42)
        # Unseeded by default.
        with patch.object(mu, "SamplingParams") as mock_sp:
            generate_rollouts_batch(model, tokenizer, [[{"role": "user", "content": "hi"}]], n=2)
        self.assertIsNone(mock_sp.call_args.kwargs["seed"])

    def test_generate_rollouts_batch_shape(self):
        model, tokenizer = make_generation_mocks([["a1", "a2"], ["b1", "b2"]])
        batch = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
        ]
        with patch.object(mu, "SamplingParams"):
            out = generate_rollouts_batch(model, tokenizer, batch, n=2)
        self.assertEqual(out, [["a1", "a2"], ["b1", "b2"]])


class FakeSpecialTokenizer:
    """A tokenizer double exposing only the special-token vocabulary."""

    def __init__(self, specials, added=()):
        self.all_special_tokens = list(specials)
        self.added_tokens_decoder = {
            i: SimpleNamespace(content=c, special=True) for i, c in enumerate(added)
        }
        self.eos_token = specials[0] if specials else None

    def apply_chat_template(self, messages, **kwargs):
        return f"templated:{messages[-1]['content']}"


GEMMA4_SPECIALS = ["<eos>", "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<bos>"]
QWEN_SPECIALS = ["<|im_end|>", "<|endoftext|>"]


class TestSpecialTokenDelimiters(unittest.TestCase):
    """Gemma 4's reasoning delimiters are special tokens vLLM's default decoding strips."""

    def test_gemma4_delimiters_are_special(self):
        tok = FakeSpecialTokenizer(GEMMA4_SPECIALS)
        self.assertTrue(mu.delimiters_are_special_tokens(tok, mu.GEMMA4_DELIMITERS))
        self.assertFalse(mu.delimiters_are_special_tokens(tok, DEFAULT_DELIMITERS))
        self.assertFalse(mu.delimiters_are_special_tokens(FakeSpecialTokenizer(QWEN_SPECIALS)))
        self.assertFalse(mu.delimiters_are_special_tokens(FakeSpecialTokenizer([])))

    def test_added_tokens_decoder_counts(self):
        tok = FakeSpecialTokenizer(["<eos>"], added=["<channel|>"])
        self.assertTrue(mu.delimiters_are_special_tokens(tok, mu.GEMMA4_DELIMITERS))

    def test_sampling_kwargs(self):
        self.assertEqual(
            mu.sampling_kwargs_for(FakeSpecialTokenizer(GEMMA4_SPECIALS), mu.GEMMA4_DELIMITERS),
            {"skip_special_tokens": False},
        )
        self.assertEqual(mu.sampling_kwargs_for(FakeSpecialTokenizer(QWEN_SPECIALS)), {})

    def test_strip_terminal_special_tokens(self):
        tok = FakeSpecialTokenizer(GEMMA4_SPECIALS)
        text = "<|channel>thought\nhmm<channel|>\n<answer>C</answer><turn|>\n<eos>"
        self.assertEqual(
            mu.strip_terminal_special_tokens(text, tok, mu.GEMMA4_DELIMITERS),
            "<|channel>thought\nhmm<channel|>\n<answer>C</answer>",
        )
        # A delimiter at the very end is never stripped (a truncated block).
        self.assertEqual(
            mu.strip_terminal_special_tokens("<|channel>thought\nx<channel|>", tok, mu.GEMMA4_DELIMITERS),
            "<|channel>thought\nx<channel|>",
        )
        # Nothing to strip → identical text, for every model.
        plain = "<think>x</think><answer>A</answer>"
        self.assertEqual(mu.strip_terminal_special_tokens(plain, FakeSpecialTokenizer(QWEN_SPECIALS)), plain)
        self.assertEqual(mu.strip_terminal_special_tokens(plain, FakeSpecialTokenizer([])), plain)
        self.assertEqual(
            mu.strip_terminal_special_tokens(plain + "<|im_end|>", FakeSpecialTokenizer(QWEN_SPECIALS)), plain
        )

    def test_generate_batch_keeps_delimiters_and_strips_terminal(self):
        model, _ = make_generation_mocks([["<|channel>thought\nx<channel|><answer>B</answer><turn|><eos>"]])
        tok = FakeSpecialTokenizer(GEMMA4_SPECIALS)
        with patch.object(mu, "SamplingParams") as mock_sp:
            out = generate_rollouts_batch(
                model, tok, [[{"role": "user", "content": "q"}]], delimiters=mu.GEMMA4_DELIMITERS
            )
        self.assertFalse(mock_sp.call_args.kwargs["skip_special_tokens"])
        self.assertEqual(out, [["<|channel>thought\nx<channel|><answer>B</answer>"]])
        # A <think> model: no kwarg, text untouched.
        model, _ = make_generation_mocks([["<think>x</think><answer>B</answer>"]])
        with patch.object(mu, "SamplingParams") as mock_sp:
            out = generate_rollouts_batch(model, FakeSpecialTokenizer(QWEN_SPECIALS), [[{"role": "user", "content": "q"}]])
        self.assertNotIn("skip_special_tokens", mock_sp.call_args.kwargs)
        self.assertEqual(out, [["<think>x</think><answer>B</answer>"]])


if __name__ == "__main__":
    unittest.main()
