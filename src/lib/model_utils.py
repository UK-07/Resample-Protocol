"""Model registry, thinking-mode resolution and the vLLM / HF loaders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# vLLM is optional at import time (GPU-free consumers only need the registry);
# the names stay module attributes so tests can patch them.
try:
    import vllm
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - depends on the machine, not the code
    vllm = None
    LLM = None
    SamplingParams = None
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lib.nemotron_h_patch import HF_FORWARD_PATCHES as _HF_FORWARD_PATCHES
from src.lib.parsing import DEFAULT_DELIMITERS, ReasoningDelimiters
from src.lib.prompts import get_system_prompt

# Gemma 4 emits its reasoning inside a "thought" channel, not <think> tags.
GEMMA4_DELIMITERS = ReasoningDelimiters(open="<|channel>thought", close="<channel|>")

# vLLM kwarg for the gated-delta-net prefill path (Qwen 3.x); per-model because
# non-GDN architectures may reject it.
_GDN_KWARGS = {"gdn_prefill_backend": "triton"}

# Every field a consumer may read; `get_model_config` merges an entry over these.
MODEL_CONFIG_DEFAULTS: dict = {
    # Filesystem-safe name used throughout output paths. Required per entry.
    "short_name": None,
    # Capability: "native" (emits its own reasoning block when
    # enable_thinking=True) or "prompted" (CoT elicited by the system prompt;
    # the tokenizer has no enable_thinking kwarg).
    "thinking_mode": "prompted",
    # Run default when a config omits `thinking`.
    "thinking_default": True,
    # Literal strings wrapping the reasoning block in generated text.
    "delimiters": DEFAULT_DELIMITERS,
    # Extra kwargs merged into the vLLM `LLM(...)` call.
    "vllm_extra_kwargs": {},
    # transformers auto-class name; None → walk _HF_AUTO_CLASS_CHAIN.
    "hf_auto_class": None,
    # Dotted path to the decoder block stack; None → walk LAYER_STACK_PATHS.
    "layer_stack_path": None,
    # False picks transformers' native implementation over the checkpoint's
    # remote modeling code.
    "hf_trust_remote_code": True,
    # Name of a forward-pass patch `load_model_hf` applies after loading (see
    # `_HF_FORWARD_PATCHES`); None → the class's own forward.
    "hf_forward_patch": None,
    # Optional shape hints, validated against the loaded model at runtime.
    "n_layers": None,
    "hidden_size": None,
    # False when the chat template ignores `enable_thinking` and always opens a
    # reasoning block; `get_thinking_config` then refuses a thinking-off run.
    "supports_thinking_off": True,
}

MODEL_CONFIGS: dict[str, dict] = {
    "Qwen/Qwen3.5-27B": {
        "short_name": "qwen3.5-27b",
        "thinking_mode": "native",
        "vllm_extra_kwargs": _GDN_KWARGS,
    },
    "Qwen/Qwen3.6-27B": {
        "short_name": "qwen3.6-27b",
        "thinking_mode": "native",
        "vllm_extra_kwargs": _GDN_KWARGS,
        "n_layers": 64,
        "hidden_size": 5120,
    },
    "Qwen/Qwen3-8B": {
        "short_name": "qwen3-8b",
        "thinking_mode": "native",
        # Dense Qwen3; the GDN kwarg is a plain EngineArgs field, unused here.
        "vllm_extra_kwargs": _GDN_KWARGS,
        "n_layers": 36,
        "hidden_size": 4096,
    },
    "Qwen/Qwen3.5-9B": {
        "short_name": "qwen3.5-9b",
        "thinking_mode": "native",
        "vllm_extra_kwargs": _GDN_KWARGS,
        "n_layers": 32,
        "hidden_size": 4096,
    },
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": {
        "short_name": "nemotron-nano-9b-v2",
        # The template pre-injects "<think>", so rollouts carry only "</think>".
        "thinking_mode": "native",
        # The template ignores `enable_thinking` (it reads a /think or /no_think
        # directive from the text instead).
        "supports_thinking_off": False,
        # Hybrid Mamba-2 / attention stack; takes no GDN kwarg.
        "n_layers": 56,
        "hidden_size": 4480,
        # The remote modeling code hard-requires mamba-ssm; transformers'
        # native `nemotron_h` class falls back to a pure-torch SSD path.
        "hf_trust_remote_code": False,
        "hf_forward_patch": "nemotron_h_memory_efficient",
    },
    "allenai/Olmo-3-7B-Think": {
        "short_name": "olmo3-7b-think",
        # The template pre-injects "<think>", so rollouts carry only "</think>".
        "thinking_mode": "native",
        # No non-thinking mode (allenai/Olmo-3-7B-Instruct is the separate
        # non-reasoning checkpoint).
        "supports_thinking_off": False,
        "n_layers": 32,
        "hidden_size": 4096,
    },
    "NousResearch/Meta-Llama-3.1-8B-Instruct": {
        "short_name": "llama3.1-8b-instruct",
        # Ungated re-upload of meta-llama/Llama-3.1-8B-Instruct.
        "thinking_mode": "prompted",
        "n_layers": 32,
        "hidden_size": 4096,
    },
    "google/gemma-3-4b-it": {
        "short_name": "gemma3-4b-it",
        "thinking_mode": "prompted",
        "n_layers": 34,
        "hidden_size": 2560,
    },
    "google/gemma-3-27b-it": {
        "short_name": "gemma3-27b-it",
        "thinking_mode": "prompted",
    },
    "google/gemma-4-12B-it": {
        "short_name": "gemma4-12b-it",
        # Native reasoner whose checkpoint defaults to thinking off; the
        # pipeline passes enable_thinking explicitly.
        "thinking_mode": "native",
        "delimiters": GEMMA4_DELIMITERS,
        "n_layers": 48,      # text_config.num_hidden_layers
        "hidden_size": 3840,  # text_config.hidden_size
    },
    "google/gemma-4-31B-it": {
        "short_name": "gemma4-31b-it",
        "thinking_mode": "native",
        "delimiters": GEMMA4_DELIMITERS,
        "n_layers": 60,
        "hidden_size": 5376,
    },
}

# Case-insensitive lookup index: HF ids differ only in capitalization between
# sources, and an exact match would fall through to the prompted default.
_CANONICAL_KEYS: dict[str, str] = {k.lower(): k for k in MODEL_CONFIGS}


def get_model_config(model_name: str) -> dict:
    """Return the defaults-filled config for *model_name* (case-insensitive).

    Unregistered models get a derived short name plus MODEL_CONFIG_DEFAULTS.
    """
    key = _CANONICAL_KEYS.get(model_name.lower())
    if key is None:
        return {
            **MODEL_CONFIG_DEFAULTS,
            "short_name": model_name.split("/")[-1].lower(),
        }
    return {**MODEL_CONFIG_DEFAULTS, **MODEL_CONFIGS[key]}


def get_model_short_name(model_name: str) -> str:
    """Return a filesystem-safe short name for *model_name* (e.g. 'qwen3.5-27b')."""
    return get_model_config(model_name)["short_name"]


@dataclass(frozen=True)
class ThinkingConfig:
    """Resolved reasoning settings for a run."""

    enabled: bool
    # Effective run mode: "native", "prompted", or "none" (thinking off).
    mode: str
    system_prompt: str
    # apply_chat_template kwarg; None means omit it.
    enable_thinking: bool | None
    delimiters: ReasoningDelimiters


def coerce_thinking(value) -> bool | None:
    """Normalize a config/CLI ``thinking`` value to True/False/None (unset)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"on", "true", "yes", "y", "1", "enabled", "enable"}:
        return True
    if text in {"off", "false", "no", "n", "0", "disabled", "disable"}:
        return False
    raise ValueError(
        f"Unrecognized thinking value {value!r}. Use on/off (or true/false)."
    )


def get_thinking_config(model_name: str, thinking=None) -> ThinkingConfig:
    """Resolve the run's thinking settings for *model_name*.

    Precedence: explicit ``thinking`` > the model's ``thinking_default`` > True.
    Thinking-off resolves to mode "none" and the plain system prompt for both
    native and prompted models. Raises ValueError when thinking is off for a
    model whose ``supports_thinking_off`` is False.
    """
    cfg = get_model_config(model_name)
    resolved = coerce_thinking(thinking)
    if resolved is None:
        resolved = bool(cfg["thinking_default"])

    capability = cfg["thinking_mode"]
    delimiters = cfg["delimiters"]

    if not resolved:
        if not cfg["supports_thinking_off"]:
            raise ValueError(
                f"{model_name} cannot run with thinking off: its chat template "
                "ignores the enable_thinking kwarg and always opens a reasoning "
                "block, so the run would still produce a CoT while the plain "
                "answer-only system prompt told every consumer there is none. "
                "Run with thinking on. Real thinking-off support is model-specific\n"
                "(Nemotron: a '/no_think' directive in the system text; Olmo 3: the\n"
                "separate allenai/Olmo-3-7B-Instruct checkpoint)."
            )
        mode = "none"
        enable_thinking = False if capability == "native" else None
    else:
        mode = capability
        enable_thinking = True if capability == "native" else None

    return ThinkingConfig(
        enabled=resolved,
        mode=mode,
        system_prompt=get_system_prompt(mode, delimiters=delimiters),
        enable_thinking=enable_thinking,
        delimiters=delimiters,
    )


def template_kwargs(enable_thinking: bool | None) -> dict:
    """Chat-template kwargs for an ``enable_thinking`` setting; None omits the kwarg."""
    return {} if enable_thinking is None else {"enable_thinking": enable_thinking}


def build_chat_messages(prompt: str, system_prompt: str | None = None) -> list[dict]:
    """Build a chat-template message list from a user prompt."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def load_tokenizer(model_name: str):
    """The model's HF tokenizer alone (prompt building without a model load)."""
    return AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(get_model_config(model_name)["hf_trust_remote_code"])
    )


def load_model_vLLM(
    model_name: str,
    *,
    dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] = "bfloat16",
    max_model_len: int = 16384,
    gpu_memory_utilization: float = 0.9,
    gdn_prefill_backend: Literal["flashinfer", "triton"] | None = None,
    tensor_parallel_size: int = 1,
    max_num_seqs: int | None = None,
    extra_kwargs: dict | None = None,
) -> tuple[LLM, object]:
    """Load a model with vLLM and return (llm, tokenizer).

    Architecture-specific kwargs come from the registry's ``vllm_extra_kwargs``;
    ``gdn_prefill_backend=`` / ``extra_kwargs=`` override them per call.
    ``max_num_seqs`` caps concurrent decode sequences (None = vLLM's default).
    """
    if LLM is None:
        raise ImportError(
            "vLLM is not installed, so no model can be loaded. This path needs a "
            "GPU box with the full environment (`uv sync`); the GPU-free entry "
            "points do not reach here."
        )
    kwargs = dict(get_model_config(model_name)["vllm_extra_kwargs"])
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    if gdn_prefill_backend is not None:
        kwargs["gdn_prefill_backend"] = gdn_prefill_backend
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs

    try:
        llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs,
        )
    except RuntimeError as exc:
        _raise_if_heterogeneous_config(exc, model_name)
        raise
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def _raise_if_heterogeneous_config(exc: RuntimeError, model_name: str) -> None:
    """Re-raise transformers' heterogeneous-config error with the version-pin remedy.

    Matched by class name: the class does not exist in the pinned transformers
    versions, so it cannot be imported. Other exceptions are left alone.
    """
    if type(exc).__name__ != "AmbiguousGlobalPerLayerAttributeError":
        return
    raise RuntimeError(
        f"vLLM {getattr(vllm, '__version__', 'unknown')} cannot read the config of {model_name!r} under "
        f"transformers {transformers.__version__}: {exc}\n"
        "transformers >=5.15 expresses Gemma 4's dual head widths as a heterogeneous "
        "`per_layer_config` (full-attention layers use global_head_dim=512, sliding "
        "layers head_dim=256) and drops `global_head_dim`/`num_global_key_value_heads`; "
        "vLLM still reads all three as plain config attributes. Silencing this would "
        "mis-size the KV cache rather than fix it.\n"
        "Fix: keep transformers pinned to '>=5.14.1,<5.15' in pyproject.toml, then "
        "`uv lock && uv sync`."
    ) from exc


# Auto-classes tried in order when the registry names none; AutoModelForCausalLM
# does not cover the *ForConditionalGeneration wrappers Gemma 4 ships as.
_HF_AUTO_CLASS_CHAIN = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModel",
)


def _resolve_auto_class(name: str):
    """Look up a transformers auto-class by name (module globals win, so tests can patch)."""
    return globals().get(name) or getattr(transformers, name, None)


def neutralize_sampling_config(model) -> None:
    """Clear sampling knobs from `model.generation_config` (silences the do_sample=False warning)."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    for attr in ("temperature", "top_p", "top_k", "typical_p", "epsilon_cutoff", "eta_cutoff"):
        if hasattr(gc, attr):
            setattr(gc, attr, None)


def get_text_config(hf_config):
    """Return the text sub-config of a multimodal HF config, else the config itself."""
    return getattr(hf_config, "text_config", None) or hf_config


def load_model_hf(
    model_name: str,
    *,
    dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] = "bfloat16",
    device_map: str | dict = "auto",
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa",
    trust_remote_code: bool | None = None,
    auto_class: str | None = None,
    model_kwargs: dict | None = None,
) -> tuple[object, object]:
    """Load a model with HuggingFace transformers and return (model, tokenizer).

    The auto-class is `auto_class`, else the registry's ``hf_auto_class``, else
    _HF_AUTO_CLASS_CHAIN walked until one accepts the checkpoint. ``model_kwargs``
    are merged into ``from_pretrained`` after the loader's own. The model is put
    in eval mode and the registry's ``hf_forward_patch`` (if any) applied.
    """

    dtype_map = {
        "auto": "auto",
        "half": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    if trust_remote_code is None:
        trust_remote_code = bool(get_model_config(model_name)["hf_trust_remote_code"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    chain = (auto_class,) if auto_class else None
    if chain is None:
        registered = get_model_config(model_name)["hf_auto_class"]
        chain = (registered,) if registered else _HF_AUTO_CLASS_CHAIN

    load_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    load_kwargs.update(dict(model_kwargs or {}))
    model = None
    attempts: list[str] = []
    for cls_name in chain:
        cls = _resolve_auto_class(cls_name)
        if cls is None:
            attempts.append(f"{cls_name}: not available in this transformers version")
            continue
        try:
            model = cls.from_pretrained(model_name, **load_kwargs)
            break
        except (ValueError, KeyError, TypeError) as exc:
            attempts.append(f"{cls_name}: {type(exc).__name__}: {exc}")
    if model is None:
        detail = "\n".join(f"  - {a}" for a in attempts)
        raise RuntimeError(
            f"Could not load {model_name!r} with any transformers auto-class:\n"
            f"{detail}\n"
            "Set 'hf_auto_class' on its MODEL_CONFIGS entry to the correct class."
        )

    model.eval()
    neutralize_sampling_config(model)
    _apply_forward_patch(model, model_name)
    return model, tokenizer


def _apply_forward_patch(model, model_name: str) -> str | None:
    """Apply the registry's ``hf_forward_patch`` to ``model``; return its name or None."""
    patch_name = get_model_config(model_name)["hf_forward_patch"]
    if not patch_name:
        return None
    try:
        patcher = _HF_FORWARD_PATCHES[patch_name]
    except KeyError:
        raise ValueError(
            f"{model_name}: unknown hf_forward_patch {patch_name!r}; "
            f"known: {sorted(_HF_FORWARD_PATCHES)}"
        ) from None
    n_modules = patcher(model)
    print(f"[load_model_hf] applied forward patch {patch_name!r} to {n_modules} module(s)")
    return patch_name


# Special-token reasoning delimiters (Gemma 4): vLLM's default
# ``skip_special_tokens=True`` would strip them, so generation keeps every special
# token and the terminal EOS / end-of-turn ones are stripped explicitly.


def _special_tokens(tokenizer) -> list[str]:
    """Every special token string the tokenizer knows (a fake tokenizer may lack either source)."""
    tokens = list(getattr(tokenizer, "all_special_tokens", None) or [])
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for added in decoder.values():
        if getattr(added, "special", False) and added.content not in tokens:
            tokens.append(added.content)
    return tokens


def delimiters_are_special_tokens(
    tokenizer, delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS
) -> bool:
    """True when either reasoning delimiter is, or begins with, a special token of the tokenizer."""
    specials = _special_tokens(tokenizer)
    return any(
        delimiter == token or delimiter.startswith(token)
        for delimiter in (delimiters.open, delimiters.close)
        for token in specials
        if token
    )


def sampling_kwargs_for(
    tokenizer, delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS
) -> dict:
    """``{"skip_special_tokens": False}`` when the delimiters are special tokens, else ``{}``."""
    if delimiters_are_special_tokens(tokenizer, delimiters):
        return {"skip_special_tokens": False}
    return {}


def strip_terminal_special_tokens(
    text: str, tokenizer, delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS
) -> str:
    """Strip trailing special tokens (and whitespace before them) other than the delimiters.

    A no-op when nothing matches, so it is safe on every model's output.
    """
    keep = {delimiters.open, delimiters.close}
    terminal = [
        token for token in _special_tokens(tokenizer)
        if token and not any(kept == token or kept.startswith(token) for kept in keep)
    ]
    if not terminal:
        return text
    stripped = text
    while True:
        candidate = stripped.rstrip()
        hit = next((token for token in terminal if candidate.endswith(token)), None)
        if hit is None:
            return stripped
        stripped = candidate[: -len(hit)]


def _generate_batch(
    model: LLM,
    tokenizer,
    batch_messages: list[list[dict]],
    *,
    n: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int | None = None,
    enable_thinking: bool | None = True,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> list[list[str]]:
    """Generate ``n`` texts per conversation; returns a list aligned with ``batch_messages``.

    ``seed`` is vLLM's per-request sampling seed (None = unseeded);
    ``enable_thinking=None`` omits the kwarg; special-token delimiters are kept
    and the terminal special tokens stripped.
    """
    chat_kwargs = template_kwargs(enable_thinking)
    prompt_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        for messages in batch_messages
    ]
    sampling_params = SamplingParams(
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        **sampling_kwargs_for(tokenizer, delimiters),
    )
    outputs = model.generate(prompt_texts, sampling_params)
    return [
        [strip_terminal_special_tokens(o.text, tokenizer, delimiters) for o in out.outputs]
        for out in outputs
    ]


def generate_rollouts_batch(
    model: LLM,
    tokenizer,
    batch_messages: list[list[dict]],
    *,
    n: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int | None = None,
    enable_thinking: bool | None = True,
    delimiters: ReasoningDelimiters = DEFAULT_DELIMITERS,
) -> list[list[str]]:
    """Generate n rollouts per conversation for a batch of conversations."""
    return _generate_batch(
        model, tokenizer, batch_messages,
        n=n, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k, seed=seed, enable_thinking=enable_thinking, delimiters=delimiters,
    )
