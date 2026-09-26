"""Text features for the TF-IDF probe: spaCy normalisation of the CoT, a per-dataset cache, the vectorizer.

spaCy is imported only inside :func:`load_nlp` / :func:`spacy_version` and torch only inside
:func:`to_torch_sparse`, so the module imports on a machine without either.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.lib.paths import resolve_data_path
from src.lib.probe_datasets import NAME_RE, dataset_sidecars, umask_mode

__all__ = [
    "CACHE_COLUMNS",
    "reasoning_sha",
    "DEFAULT_CACHE_DIR",
    "DISABLED_PIPES",
    "FINGERPRINT_CHARS",
    "MISSING_MODEL_MESSAGE",
    "SMALL_CORPUS_ROWS",
    "TOKEN_PATTERN",
    "TextPipelineConfig",
    "VectorizerConfig",
    "cache_path",
    "dataset_name_for",
    "fit_vectorizer",
    "load_nlp",
    "load_vectorizer",
    "n_features",
    "normalise_texts",
    "normalised_corpus",
    "save_vectorizer",
    "spacy_version",
    "to_torch_sparse",
    "transform",
]

DEFAULT_CACHE_DIR = "${DATA_ROOT}/probe_datasets/text_cache"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
#: Pipeline components not needed for lemmas (tagger + attribute_ruler + lemmatizer stay).
DISABLED_PIPES = ("parser", "ner")
#: Below this many texts spaCy runs in-process: forking workers costs more than it saves.
SMALL_CORPUS_ROWS = 200
FINGERPRINT_CHARS = 16
#: The vectorizer splits the normalised text on whitespace only — the pipeline already tokenised.
TOKEN_PATTERN = r"\S+"
CACHE_COLUMNS = ("rollout_id", "reasoning_sha", "text")
MISSING_MODEL_MESSAGE = "run: uv sync (en_core_web_sm is a pinned dependency)"
#: Settings of :class:`TextPipelineConfig` that do not change the normalised text.
_EXECUTION_KEYS = ("n_process", "batch_size", "cache_dir")


def _from_dict(cls, cfg: dict | None, block: str):
    cfg = {} if cfg is None else dict(cfg)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"{block}: unknown keys {unknown}; allowed: {sorted(known)}")
    return cls(**cfg)


@dataclass(frozen=True)
class TextPipelineConfig:
    """The ``probe.text:`` block — how a CoT becomes a token string.

    ``fingerprint`` covers every setting but ``n_process`` / ``batch_size`` / ``cache_dir``.
    """

    lowercase: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    spacy_model: str = DEFAULT_SPACY_MODEL
    min_token_chars: int = 2
    keep_numbers: bool = False
    n_process: int = 4
    batch_size: int = 64
    cache_dir: str = DEFAULT_CACHE_DIR

    def __post_init__(self):
        for name in ("lowercase", "remove_stopwords", "lemmatize", "keep_numbers"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"text.{name} must be a bool, got {getattr(self, name)!r}")
        if not isinstance(self.spacy_model, str) or not self.spacy_model:
            raise ValueError(f"text.spacy_model must be a non-empty string, got {self.spacy_model!r}")
        for name, low in (("min_token_chars", 0), ("n_process", 1), ("batch_size", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"text.{name} must be an int >= {low}, got {value!r}")
        if not isinstance(self.cache_dir, str) or not self.cache_dir:
            raise ValueError(f"text.cache_dir must be a non-empty string, got {self.cache_dir!r}")

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "TextPipelineConfig":
        """Parse the YAML block; an unknown key is an error."""
        return _from_dict(cls, cfg, "probe.text")

    def to_dict(self) -> dict:
        return asdict(self)

    def output_settings(self) -> dict:
        """The settings that change the normalised text (everything but the execution keys)."""
        return {k: v for k, v in asdict(self).items() if k not in _EXECUTION_KEYS}

    def fingerprint(self, *, nlp=None, model_version: str | None = None, spacy_ver: str | None = None) -> str:
        """Short sha256 hex of the output settings + the spaCy version + the model version.

        The model version comes from ``nlp.meta["version"]``, from ``model_version``, else from
        loading the model; ``spacy_ver`` defaults to :func:`spacy_version`.
        """
        if model_version is None:
            if nlp is None:
                nlp = load_nlp(self)
            _check_nlp(nlp, self)
            model_version = str(nlp.meta["version"])
        if spacy_ver is None:
            spacy_ver = spacy_version()
        payload = {"settings": self.output_settings(), "spacy_version": spacy_ver, "model_version": model_version}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:FINGERPRINT_CHARS]

    def resolved_cache_dir(self) -> Path:
        return resolve_data_path(self.cache_dir)


@dataclass(frozen=True)
class VectorizerConfig:
    """The ``probe.vectorizer:`` block — the ``TfidfVectorizer`` knobs."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 5
    max_df: float = 1.0
    max_features: int | None = 100000
    sublinear_tf: bool = True
    norm: str = "l2"
    binary: bool = False

    def __post_init__(self):
        rng = self.ngram_range
        if (not isinstance(rng, (tuple, list)) or len(rng) != 2
                or not all(isinstance(n, int) and not isinstance(n, bool) for n in rng) or not 1 <= rng[0] <= rng[1]):
            raise ValueError(f"vectorizer.ngram_range must be [lo, hi] with 1 <= lo <= hi, got {rng!r}")
        object.__setattr__(self, "ngram_range", (int(rng[0]), int(rng[1])))
        for name in ("min_df", "max_df"):
            value = getattr(self, name)
            ok = (not isinstance(value, bool)
                  and ((isinstance(value, int) and value >= 1) or (isinstance(value, float) and 0.0 <= value <= 1.0)))
            if not ok:  # sklearn's contract: an int count >= 1, or a float share in [0, 1]
                raise ValueError(f"vectorizer.{name} must be an int >= 1 or a float in [0, 1], got {value!r}")
        if self.max_features is not None and (isinstance(self.max_features, bool)
                                              or not isinstance(self.max_features, int) or self.max_features < 1):
            raise ValueError(f"vectorizer.max_features must be null or an int >= 1, got {self.max_features!r}")
        for name in ("sublinear_tf", "binary"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"vectorizer.{name} must be a bool, got {getattr(self, name)!r}")
        if self.norm not in ("l1", "l2", None):
            raise ValueError(f"vectorizer.norm must be 'l1', 'l2' or null, got {self.norm!r}")

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "VectorizerConfig":
        """Parse the YAML block (``ngram_range`` as a two-element list); an unknown key is an error."""
        return _from_dict(cls, cfg, "probe.vectorizer")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ngram_range"] = list(self.ngram_range)
        return d


def _import_spacy():
    """The lazily imported ``spacy`` module — the seam the tests patch."""
    import spacy

    return spacy


def spacy_version() -> str:
    return str(_import_spacy().__version__)


def load_nlp(cfg: TextPipelineConfig):
    """``spacy.load(cfg.spacy_model, disable=DISABLED_PIPES)``; a missing model is a ``RuntimeError``."""
    spacy = _import_spacy()
    try:
        return spacy.load(cfg.spacy_model, disable=list(DISABLED_PIPES))
    except OSError as exc:
        raise RuntimeError(MISSING_MODEL_MESSAGE) from exc


def _check_nlp(nlp, cfg: TextPipelineConfig) -> None:
    """A caller-supplied pipeline must be ``cfg.spacy_model`` — its lemmas go into a cache keyed on that name."""
    meta = getattr(nlp, "meta", {})
    if "lang" in meta and "name" in meta:
        loaded = f"{meta['lang']}_{meta['name']}"
        if loaded != cfg.spacy_model:
            raise ValueError(f"nlp is {loaded!r} but text.spacy_model is {cfg.spacy_model!r}")


def _tokens(doc, cfg: TextPipelineConfig) -> list[str]:
    out = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if cfg.remove_stopwords and token.is_stop:
            continue
        if token.like_num and not cfg.keep_numbers:
            continue
        text = token.lemma_ if cfg.lemmatize else token.text
        if cfg.lowercase:
            text = text.lower()
        text = text.strip()
        if len(text) < cfg.min_token_chars or not text:
            continue
        out.append(text)
    return out


def _effective_n_process(cfg: TextPipelineConfig, n_texts: int) -> int:
    return 1 if n_texts < SMALL_CORPUS_ROWS else cfg.n_process


def normalise_texts(texts: list[str], cfg: TextPipelineConfig, *, nlp=None) -> list[str]:
    """One space-joined token string per input text, through the pipeline ``cfg`` describes.

    ``nlp`` defaults to :func:`load_nlp`; its ``max_length`` is raised to the longest input + 1.
    ``n_process > 1`` uses ``nlp.pipe``'s own multiprocessing (each worker holds a whole
    ``batch_size`` batch of ``Doc`` objects); below :data:`SMALL_CORPUS_ROWS` texts the call
    runs in-process regardless.
    """
    texts = list(texts)
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(f"texts[{i}] is {type(t).__name__}, expected str")
    if not texts:
        return []
    if nlp is None:
        nlp = load_nlp(cfg)
    _check_nlp(nlp, cfg)
    longest = max(len(t) for t in texts)
    if nlp.max_length < longest + 1:
        nlp.max_length = longest + 1
    docs = nlp.pipe(texts, batch_size=cfg.batch_size, n_process=_effective_n_process(cfg, len(texts)))
    return [" ".join(_tokens(doc, cfg)) for doc in docs]


def dataset_name_for(dataset: str | os.PathLike | dict) -> str:
    """The dataset name a cache file is keyed on: a bare ``NAME_RE`` name, ``read_dataset``'s meta
    (its ``name``), or the dataset's parquet path (the name from its ``.meta.json`` sidecar)."""
    if isinstance(dataset, dict):
        name = dataset.get("name")
        if not name:
            raise ValueError("dataset meta carries no 'name'")
        return str(name)
    text = os.fspath(dataset)
    if NAME_RE.match(text) and os.sep not in text and not text.endswith(".parquet"):
        return text
    parquet = resolve_data_path(text)
    meta_file, _ = dataset_sidecars(parquet)
    if not meta_file.exists():
        raise FileNotFoundError(f"{parquet}: sidecar {meta_file.name} is missing — not a dataset written by write_dataset")
    meta = json.loads(meta_file.read_text())
    name = meta.get("name")
    if not name:
        raise ValueError(f"{meta_file}: sidecar carries no 'name' — not a dataset written by write_dataset")
    if not NAME_RE.match(str(name)):
        raise ValueError(f"{meta_file}: dataset name {name!r} does not match {NAME_RE.pattern}")
    return str(name)


def cache_path(dataset_name: str, fingerprint: str, cfg: TextPipelineConfig) -> Path:
    """``<cache_dir>/<dataset name>__<fingerprint>.parquet``."""
    if not NAME_RE.match(dataset_name):
        raise ValueError(f"dataset name must match {NAME_RE.pattern}, got {dataset_name!r}")
    return cfg.resolved_cache_dir() / f"{dataset_name}__{fingerprint}.parquet"


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp, index=False)
        os.chmod(tmp, umask_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def reasoning_sha(text) -> str:
    """sha256 (hex) of one ``reasoning`` string — the cache's per-row content key."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="string") for c in CACHE_COLUMNS})
    cached = pd.read_parquet(path)
    missing = [c for c in CACHE_COLUMNS if c not in cached.columns]
    if missing:
        raise ValueError(f"{path}: cache lacks columns {missing}")
    cached = cached[list(CACHE_COLUMNS)].astype("string")
    if cached["rollout_id"].duplicated().any():
        raise ValueError(f"{path}: cache holds duplicate rollout_ids")
    if cached["rollout_id"].isna().any() or cached["text"].isna().any():
        raise ValueError(f"{path}: cache holds null rollout_ids or texts")
    return cached


def _log(log, message: str) -> None:
    if log is not None:
        log(message)


def normalised_corpus(dataset: str | os.PathLike | dict, rows: pd.DataFrame, cfg: TextPipelineConfig, *,
                      nlp=None, log=None) -> pd.Series:
    """The normalised ``reasoning`` of ``rows``, cached per dataset and pipeline fingerprint.

    ``rows`` carries a unique ``rollout_id`` and ``reasoning``. A cached row is served only when
    its ``reasoning_sha`` matches the row's current text; the missing or changed ids are
    normalised and the union rewritten atomically. Returns a ``text`` series indexed by
    ``rollout_id`` in the order of ``rows``.
    """
    for column in ("rollout_id", "reasoning"):
        if column not in rows.columns:
            raise ValueError(f"rows lack the {column!r} column")
    ids = rows["rollout_id"].astype("string")
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("rows' rollout_id must be unique and non-null")
    if nlp is None:
        nlp = load_nlp(cfg)
    name = dataset_name_for(dataset)
    path = cache_path(name, cfg.fingerprint(nlp=nlp), cfg)
    cached = _read_cache(path)
    shas = rows["reasoning"].map(reasoning_sha)
    have = dict(zip(cached["rollout_id"], cached["reasoning_sha"]))
    hit = [have.get(i) == h for i, h in zip(ids, shas)]
    todo = rows.loc[[not h for h in hit]]
    if len(todo):
        cached = cached[~cached["rollout_id"].isin(set(todo["rollout_id"].astype("string")))]
    _log(log, f"text cache {path}: {len(cached)} cached, {len(todo)} of {len(rows)} rows to normalise")
    if len(todo):
        texts = todo["reasoning"].tolist()
        for i, t in zip(todo["rollout_id"].tolist(), texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"rollout {i}: blank reasoning cannot be normalised")
        fresh = pd.DataFrame({"rollout_id": todo["rollout_id"].astype("string").to_numpy(),
                              "reasoning_sha": pd.array([reasoning_sha(t) for t in texts], dtype="string"),
                              "text": pd.array(normalise_texts(texts, cfg, nlp=nlp), dtype="string")})
        cached = pd.concat([cached, fresh], ignore_index=True)
        _atomic_write_parquet(cached, path)
        _log(log, f"text cache {path}: wrote {len(cached)} rows")
    series = cached.set_index("rollout_id")["text"].reindex(ids.to_numpy())
    series.index.name = "rollout_id"
    series.name = "text"
    return series.astype("string")


def fit_vectorizer(train_texts, cfg: VectorizerConfig) -> TfidfVectorizer:
    """A ``TfidfVectorizer`` fitted on the fold's training texts only; the input is already
    tokenised and cased, so it only splits on whitespace (``dtype`` float32)."""
    train_texts = list(train_texts)
    if not train_texts:
        raise ValueError("fit_vectorizer needs at least one training text")
    vectorizer = TfidfVectorizer(
        analyzer="word", token_pattern=TOKEN_PATTERN, lowercase=False, dtype=np.float32,
        ngram_range=cfg.ngram_range, min_df=cfg.min_df, max_df=cfg.max_df, max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf, norm=cfg.norm, binary=cfg.binary,
    )
    try:
        vectorizer.fit(train_texts)
    except ValueError as exc:
        # sklearn's own reason ("empty vocabulary", "no terms remain", "max_df corresponds to ...") kept
        raise ValueError(f"the training texts cannot be vectorised under {cfg.to_dict()}: {exc}") from exc
    return vectorizer


def n_features(vectorizer: TfidfVectorizer) -> int:
    """The fitted vocabulary size — ``TfidfLogReg(n_features=...)``."""
    return len(vectorizer.vocabulary_)


def transform(vectorizer: TfidfVectorizer, texts) -> sp.csr_matrix:
    """``[n_texts, n_features]`` float32 CSR matrix of TF-IDF features."""
    X = vectorizer.transform(list(texts))
    return sp.csr_matrix(X, dtype=np.float32)


def to_torch_sparse(csr: sp.csr_matrix):
    """A ``torch.sparse_csr`` float32 tensor with the same shape and values (torch imported here only)."""
    import torch

    csr = sp.csr_matrix(csr, dtype=np.float32)
    csr.sort_indices()
    with warnings.catch_warnings():
        # torch's one-time beta / invariant-check notices; the indices come straight from scipy
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
        return torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)), torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(csr.data.astype(np.float32)), size=tuple(csr.shape),
        )


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | os.PathLike) -> Path:
    """``joblib.dump`` the fitted vectorizer to ``path`` (written atomically); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(vectorizer, tmp)
        os.chmod(tmp, umask_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def load_vectorizer(path: str | os.PathLike) -> TfidfVectorizer:
    """The vectorizer :func:`save_vectorizer` wrote."""
    vectorizer = joblib.load(path)
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError(f"{path} holds a {type(vectorizer).__name__}, not a TfidfVectorizer")
    return vectorizer
