"""Tests for src/lib/text_features.py (spaCy mocked)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src.lib import text_features as tf
from src.lib.text_features import (
    MISSING_MODEL_MESSAGE,
    SMALL_CORPUS_ROWS,
    TextPipelineConfig,
    VectorizerConfig,
    cache_path,
    fit_vectorizer,
    load_vectorizer,
    n_features,
    normalise_texts,
    normalised_corpus,
    save_vectorizer,
    to_torch_sparse,
    transform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

LEMMAS = {"cats": "cat", "running": "run", "were": "be", "bridges": "bridge", "Answers": "Answer", "is": "be"}
STOP_WORDS = {"the", "a", "were", "is", "of"}


class FakeToken:
    """A spaCy-token stand-in built from a whitespace token by :class:`FakeNLP`."""

    def __init__(self, text: str):
        self.text = text
        self.lemma_ = LEMMAS.get(text, text)
        self.is_space = text.isspace()
        self.is_punct = bool(text) and all(not c.isalnum() for c in text)
        self.is_stop = text.lower() in STOP_WORDS
        self.like_num = text.replace(".", "", 1).isdigit()


def patch_spacy(**kwargs):
    """Patch the spaCy import seam (never sys.modules: evicting pandas' pyarrow module breaks later tests)."""
    return mock.patch.object(tf, "_import_spacy", return_value=fake_spacy(**kwargs))


class FakeNLP:
    """A fake ``nlp``: whitespace tokens, table lemmas, ``meta``/``max_length`` and a call log."""

    def __init__(self, version: str = "3.8.0", name: str = "core_web_sm"):
        self.meta = {"version": version, "lang": "en", "name": name}
        self.max_length = 100
        self.calls: list[dict] = []

    def pipe(self, texts, *, batch_size, n_process):
        texts = list(texts)
        self.calls.append({"n": len(texts), "batch_size": batch_size, "n_process": n_process})
        for text in texts:
            yield [FakeToken(t) for t in text.split(" ") if t != ""]


def fake_spacy(*, version: str = "3.8.16", load=None) -> types.ModuleType:
    module = types.ModuleType("spacy")
    module.__version__ = version
    module.load = load if load is not None else mock.Mock(side_effect=AssertionError("spacy.load must not be called"))
    return module


def cfg(**overrides) -> TextPipelineConfig:
    return TextPipelineConfig(**overrides)


class TestPipeline(unittest.TestCase):
    """normalise_texts and TextPipelineConfig: flags, max_length, in-process threshold, fingerprint, errors."""

    def test_pipeline_flags_each_toggle(self):
        nlp = FakeNLP()
        text = "The cats were running , across 42 bridges ! Answers of X"
        self.assertEqual(normalise_texts([text], cfg(), nlp=nlp), ["cat run across bridge answer"])
        self.assertEqual(normalise_texts([text], cfg(lemmatize=False), nlp=nlp), ["cats running across bridges answers"])
        self.assertEqual(normalise_texts([text], cfg(lowercase=False), nlp=nlp), ["cat run across bridge Answer"])
        self.assertEqual(normalise_texts([text], cfg(remove_stopwords=False), nlp=nlp),
                         ["the cat be run across bridge answer of"])
        self.assertEqual(normalise_texts([text], cfg(keep_numbers=True), nlp=nlp), ["cat run across 42 bridge answer"])
        self.assertEqual(normalise_texts([text], cfg(min_token_chars=1), nlp=nlp), ["cat run across bridge answer x"])
        self.assertEqual(normalise_texts([text], cfg(min_token_chars=4), nlp=nlp), ["across bridge answer"])
        self.assertEqual(normalise_texts([], cfg(), nlp=nlp), [])
        self.assertEqual(normalise_texts(["", "! !"], cfg(), nlp=nlp), ["", ""])
        with self.assertRaises(TypeError):
            normalise_texts([None], cfg(), nlp=nlp)
        with self.assertRaisesRegex(ValueError, "en_core_web_md.*en_core_web_sm"):
            normalise_texts([text], cfg(), nlp=FakeNLP(name="core_web_md"))
        with self.assertRaisesRegex(ValueError, "spacy_model"):
            cfg().fingerprint(nlp=FakeNLP(name="core_web_md"), spacy_ver="3.8.16")

    def test_max_length_raised(self):
        nlp = FakeNLP()
        nlp.max_length = 10
        normalise_texts(["a" * 50, "b" * 20], cfg(), nlp=nlp)
        self.assertEqual(nlp.max_length, 51)
        nlp.max_length = 10_000
        normalise_texts(["a" * 50], cfg(), nlp=nlp)
        self.assertEqual(nlp.max_length, 10_000)

    def test_small_corpus_runs_in_process(self):
        nlp = FakeNLP()
        normalise_texts(["x y"] * (SMALL_CORPUS_ROWS - 1), cfg(n_process=4, batch_size=7), nlp=nlp)
        self.assertEqual(nlp.calls[-1], {"n": SMALL_CORPUS_ROWS - 1, "batch_size": 7, "n_process": 1})
        normalise_texts(["x y"] * SMALL_CORPUS_ROWS, cfg(n_process=4, batch_size=7), nlp=nlp)
        self.assertEqual(nlp.calls[-1], {"n": SMALL_CORPUS_ROWS, "batch_size": 7, "n_process": 4})

    def test_config_rejects_unknown_keys_and_bad_values(self):
        with self.assertRaisesRegex(ValueError, "unknown keys \\['stem'\\]"):
            TextPipelineConfig.from_dict({"stem": True})
        self.assertEqual(TextPipelineConfig.from_dict(None), TextPipelineConfig())
        self.assertEqual(TextPipelineConfig.from_dict({"n_process": 2}).n_process, 2)
        for bad in ({"lowercase": "yes"}, {"n_process": 0}, {"batch_size": 0}, {"min_token_chars": -1},
                    {"spacy_model": ""}, {"cache_dir": ""}):
            with self.assertRaises(ValueError, msg=bad):
                TextPipelineConfig.from_dict(bad)

    def test_fingerprint_changes_with_settings_and_model_version(self):
        with patch_spacy(version="3.8.16"):
            base = cfg().fingerprint(nlp=FakeNLP("3.8.0"))
            self.assertEqual(len(base), 16)
            int(base, 16)
            self.assertEqual(base, cfg().fingerprint(nlp=FakeNLP("3.8.0")))
            self.assertEqual(base, cfg().fingerprint(model_version="3.8.0"))
            self.assertEqual(base, cfg().fingerprint(model_version="3.8.0", spacy_ver="3.8.16"))
            seen = {base}
            for change in ({"lowercase": False}, {"remove_stopwords": False}, {"lemmatize": False},
                           {"spacy_model": "en_core_web_md"}, {"min_token_chars": 3}, {"keep_numbers": True}):
                name = change.get("spacy_model", "en_core_web_sm").removeprefix("en_")
                fp = cfg(**change).fingerprint(nlp=FakeNLP("3.8.0", name=name))
                self.assertNotIn(fp, seen, change)
                seen.add(fp)
            self.assertNotEqual(base, cfg().fingerprint(nlp=FakeNLP("3.9.0")))
        with patch_spacy(version="3.9.1"):
            self.assertNotEqual(base, cfg().fingerprint(nlp=FakeNLP("3.8.0")))

    def test_fingerprint_ignores_process_settings(self):
        base = cfg().fingerprint(model_version="3.8.0", spacy_ver="3.8.16")
        for change in ({"n_process": 1}, {"batch_size": 1}, {"cache_dir": "/elsewhere"}):
            self.assertEqual(base, cfg(**change).fingerprint(model_version="3.8.0", spacy_ver="3.8.16"), change)

    def test_missing_model_message(self):
        loader = mock.Mock(side_effect=OSError("[E050] Can't find model 'en_core_web_sm'"))
        with patch_spacy(load=loader):
            with self.assertRaises(RuntimeError) as ctx:
                normalise_texts(["x"], cfg())
        self.assertEqual(str(ctx.exception), MISSING_MODEL_MESSAGE)
        self.assertEqual(MISSING_MODEL_MESSAGE, "run: uv sync (en_core_web_sm is a pinned dependency)")
        self.assertIsInstance(ctx.exception.__cause__, OSError)
        loader.assert_called_once_with("en_core_web_sm", disable=["parser", "ner"])


class TestCache(unittest.TestCase):
    """normalised_corpus and its cache file: partial recompute, atomic write, changed reasoning, bad rows."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = str(Path(self.tmp.name) / "text_cache")
        self.patch = patch_spacy(version="3.8.16")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def rows(self, ids, texts=None):
        texts = texts if texts is not None else [f"The cats {i} were running" for i in ids]
        return pd.DataFrame({"rollout_id": ids, "reasoning": texts, "label": 0})

    def test_cache_computes_only_missing_ids_and_preserves_order(self):
        nlp = FakeNLP()
        c = cfg(cache_dir=self.cache_dir)
        first = normalised_corpus("ds_a", self.rows(["m:r:h:1", "m:r:h:2"]), c, nlp=nlp)
        self.assertEqual(list(first.index), ["m:r:h:1", "m:r:h:2"])
        self.assertEqual(first.tolist(), ["cat m:r:h:1 run", "cat m:r:h:2 run"])
        self.assertEqual([call["n"] for call in nlp.calls], [2])

        second = normalised_corpus("ds_a", self.rows(["m:r:h:3", "m:r:h:1", "m:r:h:4"]), c, nlp=nlp)
        self.assertEqual([call["n"] for call in nlp.calls], [2, 2])
        self.assertEqual(list(second.index), ["m:r:h:3", "m:r:h:1", "m:r:h:4"])
        self.assertEqual(second.tolist(), ["cat m:r:h:3 run", "cat m:r:h:1 run", "cat m:r:h:4 run"])
        self.assertEqual(second.name, "text")
        self.assertEqual(second.index.name, "rollout_id")

        third = normalised_corpus("ds_a", self.rows(["m:r:h:4", "m:r:h:2"]), c, nlp=nlp)
        self.assertEqual([call["n"] for call in nlp.calls], [2, 2])
        self.assertEqual(third.tolist(), ["cat m:r:h:4 run", "cat m:r:h:2 run"])

        stored = pd.read_parquet(cache_path("ds_a", c.fingerprint(nlp=nlp), c))
        self.assertEqual(sorted(stored["rollout_id"]), ["m:r:h:1", "m:r:h:2", "m:r:h:3", "m:r:h:4"])

    def test_cache_written_atomically_and_reused(self):
        nlp = FakeNLP("3.8.0")
        c = cfg(cache_dir=self.cache_dir)
        log = []
        normalised_corpus("ds_b", self.rows(["q:a", "q:b"]), c, nlp=nlp, log=log.append)
        path = cache_path("ds_b", c.fingerprint(nlp=nlp), c)
        self.assertEqual(path, Path(self.cache_dir) / f"ds_b__{c.fingerprint(nlp=nlp)}.parquet")
        self.assertTrue(path.exists())
        self.assertEqual(sorted(os.listdir(self.cache_dir)), [path.name])  # no temp file left behind
        self.assertEqual(list(pd.read_parquet(path).columns), ["rollout_id", "reasoning_sha", "text"])
        self.assertTrue(any("2 of 2 rows to normalise" in m for m in log))

        fresh = FakeNLP("3.8.0")
        again = normalised_corpus("ds_b", self.rows(["q:b", "q:a"]), c, nlp=fresh)
        self.assertEqual(fresh.calls, [])
        self.assertEqual(again.tolist(), ["cat q:b run", "cat q:a run"])

        other = FakeNLP("3.9.0")
        normalised_corpus("ds_b", self.rows(["q:a"]), c, nlp=other)
        self.assertEqual([call["n"] for call in other.calls], [1])
        self.assertEqual(len(os.listdir(self.cache_dir)), 2)

        # the name from a dataset parquet's sidecar, or from read_dataset's meta
        parquet = Path(self.tmp.name) / "ds_b.parquet"
        parquet.touch()
        (Path(self.tmp.name) / "ds_b.meta.json").write_text('{"name": "ds_b"}')
        self.assertEqual(tf.dataset_name_for(parquet), "ds_b")
        self.assertEqual(tf.dataset_name_for({"name": "ds_b"}), "ds_b")
        self.assertEqual(tf.dataset_name_for("ds_b"), "ds_b")
        via_path = normalised_corpus(parquet, self.rows(["q:a", "q:b"]), c, nlp=fresh)
        self.assertEqual(fresh.calls, [])
        self.assertEqual(via_path.tolist(), ["cat q:a run", "cat q:b run"])
        with self.assertRaises(FileNotFoundError):
            tf.dataset_name_for(Path(self.tmp.name) / "nope.parquet")
        (Path(self.tmp.name) / "ds_b.meta.json").write_text('{"fingerprint": "abc"}')
        with self.assertRaisesRegex(ValueError, "no 'name'"):
            tf.dataset_name_for(parquet)

    def test_cache_rejects_bad_rows(self):
        nlp = FakeNLP()
        c = cfg(cache_dir=self.cache_dir)
        with self.assertRaisesRegex(ValueError, "reasoning"):
            normalised_corpus("ds_c", self.rows(["a"]).drop(columns=["reasoning"]), c, nlp=nlp)
        with self.assertRaisesRegex(ValueError, "unique"):
            normalised_corpus("ds_c", self.rows(["a", "a"]), c, nlp=nlp)
        with self.assertRaisesRegex(ValueError, "blank reasoning"):
            normalised_corpus("ds_c", self.rows(["a"], ["  "]), c, nlp=nlp)
        normalised_corpus("ds_c", self.rows(["a"]), c, nlp=nlp)
        path = cache_path("ds_c", c.fingerprint(nlp=nlp), c)
        pd.DataFrame({"rollout_id": ["a", "a"], "reasoning_sha": ["s", "s"], "text": ["x", "y"]}).to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalised_corpus("ds_c", self.rows(["a"]), c, nlp=nlp)
        pd.DataFrame({"rollout_id": ["a", "b"], "reasoning_sha": ["s", "s"], "text": ["x", None]}).to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "null"):
            normalised_corpus("ds_c", self.rows(["a"]), c, nlp=nlp)
        pd.DataFrame({"rollout_id": ["a"]}).to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "lacks columns"):
            normalised_corpus("ds_c", self.rows(["a"]), c, nlp=nlp)

    def test_cache_recomputes_changed_reasoning(self):
        nlp = FakeNLP()
        c = cfg(cache_dir=self.cache_dir)
        first = normalised_corpus("ds_d", self.rows(["a", "b"], ["one text", "two text"]), c, nlp=nlp)
        # Same ids, one CoT re-parsed: only that row is normalised again and its text changes.
        log = []
        second = normalised_corpus("ds_d", self.rows(["a", "b"], ["one text", "two words changed"]), c,
                                   nlp=nlp, log=log.append)
        self.assertEqual(first["a"], second["a"])
        self.assertNotEqual(first["b"], second["b"])
        self.assertTrue(any("1 of 2 rows to normalise" in m for m in log))
        cached = pd.read_parquet(cache_path("ds_d", c.fingerprint(nlp=nlp), c))
        self.assertEqual(len(cached), 2)
        self.assertEqual(set(cached["rollout_id"]), {"a", "b"})


class TestVectorizer(unittest.TestCase):
    """fit_vectorizer / transform / to_torch_sparse / save + load and VectorizerConfig validation."""

    TRAIN = ["hint say answer", "answer key say", "hint answer key", "say answer hint", "answer answer"]
    TEST = ["zebra answer hint", "zebra zebra"]

    def test_vectorizer_fit_on_train_only_vocabulary(self):
        v = fit_vectorizer(self.TRAIN, VectorizerConfig(min_df=1, ngram_range=(1, 1)))
        self.assertEqual(sorted(v.vocabulary_), ["answer", "hint", "key", "say"])
        self.assertNotIn("zebra", v.vocabulary_)
        X = transform(v, self.TEST)
        self.assertEqual(X.shape, (2, 4))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(X[1].nnz, 0)  # a test row made only of unseen tokens is all zeros
        self.assertEqual(n_features(v), 4)

        v2 = fit_vectorizer(self.TRAIN, VectorizerConfig(min_df=3, ngram_range=(1, 1)))
        self.assertEqual(sorted(v2.vocabulary_), ["answer", "hint", "say"])
        v3 = fit_vectorizer(self.TRAIN, VectorizerConfig(min_df=2, ngram_range=(1, 2)))
        self.assertIn("say answer", v3.vocabulary_)
        with self.assertRaises(ValueError):
            fit_vectorizer([], VectorizerConfig())
        with self.assertRaisesRegex(ValueError, "cannot be vectorised.*no terms remain"):
            fit_vectorizer(["a b", "c d", "e f", "g h", "i j", "k l"], VectorizerConfig(min_df=5))
        with self.assertRaisesRegex(ValueError, "cannot be vectorised.*empty vocabulary"):
            fit_vectorizer(["   ", " "], VectorizerConfig(min_df=1))
        with self.assertRaisesRegex(ValueError, "cannot be vectorised.*max_df"):
            fit_vectorizer(["a b"], VectorizerConfig(min_df=5))

    def test_vectorizer_config_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "unknown keys \\['stemmer'\\]"):
            VectorizerConfig.from_dict({"stemmer": "porter"})
        self.assertEqual(VectorizerConfig.from_dict(None), VectorizerConfig())
        c = VectorizerConfig.from_dict({"ngram_range": [1, 3], "max_features": None})
        self.assertEqual(c.ngram_range, (1, 3))
        self.assertIsNone(c.max_features)
        self.assertEqual(c.to_dict()["ngram_range"], [1, 3])
        self.assertEqual(VectorizerConfig.from_dict({"min_df": 0.01, "max_df": 3}).min_df, 0.01)
        for bad in ({"ngram_range": [2, 1]}, {"ngram_range": [1]}, {"ngram_range": [0, 1]}, {"norm": "l3"},
                    {"max_features": 0}, {"min_df": -1}, {"min_df": 0}, {"min_df": 1.5}, {"max_df": 2.0},
                    {"max_df": True}, {"sublinear_tf": "yes"}):
            with self.assertRaises(ValueError, msg=bad):
                VectorizerConfig.from_dict(bad)

    def test_vectorizer_settings_reach_sklearn(self):
        c = VectorizerConfig(ngram_range=(1, 3), min_df=2, max_df=0.9, max_features=7, sublinear_tf=False,
                             norm="l1", binary=True)
        v = fit_vectorizer(self.TRAIN, c)
        params = v.get_params()
        self.assertFalse(params["lowercase"])
        self.assertEqual(params["token_pattern"], r"\S+")
        self.assertEqual(params["analyzer"], "word")
        self.assertIs(params["dtype"], np.float32)
        for key, value in c.to_dict().items():
            self.assertEqual(params[key], tuple(value) if key == "ngram_range" else value, key)
        self.assertIn("Hint", fit_vectorizer(["Hint hint"], VectorizerConfig(min_df=1)).vocabulary_)

    def test_to_torch_sparse_roundtrip(self):
        import torch

        v = fit_vectorizer(self.TRAIN, VectorizerConfig(min_df=1))
        X = transform(v, self.TRAIN + self.TEST)
        T = to_torch_sparse(X)
        self.assertEqual(T.layout, torch.sparse_csr)
        self.assertEqual(tuple(T.shape), X.shape)
        self.assertEqual(T.dtype, torch.float32)
        np.testing.assert_allclose(T.to_dense().numpy(), X.toarray(), rtol=1e-6)
        empty = to_torch_sparse(transform(v, ["zebra"]))
        self.assertEqual(tuple(empty.shape), (1, X.shape[1]))
        self.assertEqual(empty.to_dense().abs().sum().item(), 0.0)

    def test_save_load_vectorizer(self):
        v = fit_vectorizer(self.TRAIN, VectorizerConfig(min_df=1))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_vectorizer(v, Path(tmp) / "fold" / "vectorizer.joblib")
            self.assertEqual(os.listdir(Path(tmp) / "fold"), ["vectorizer.joblib"])
            loaded = load_vectorizer(path)
            self.assertEqual(loaded.vocabulary_, v.vocabulary_)
            np.testing.assert_array_equal(transform(loaded, self.TEST).toarray(), transform(v, self.TEST).toarray())
            bad = Path(tmp) / "bad.joblib"
            import joblib

            joblib.dump({"not": "a vectorizer"}, bad)
            with self.assertRaises(TypeError):
                load_vectorizer(bad)


class TestImports(unittest.TestCase):
    """The module imports with spacy and torch blocked; only their consumers raise ImportError."""

    def test_module_imports_without_spacy_or_torch(self):
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('spacy', 'torch'):\n"
            "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "from src.lib import text_features as tf\n"
            "assert 'spacy' not in sys.modules and 'torch' not in sys.modules\n"
            "v = tf.fit_vectorizer(['a b', 'a c'], tf.VectorizerConfig(min_df=1))\n"
            "X = tf.transform(v, ['a'])\n"
            "tf.TextPipelineConfig.from_dict({'n_process': 1}).fingerprint(model_version='3.8.0', spacy_ver='x')\n"
            "for fn in (lambda: tf.to_torch_sparse(X), lambda: tf.load_nlp(tf.TextPipelineConfig()), tf.spacy_version):\n"
            "    try:\n        fn()\n    except ImportError:\n        pass\n"
            "    else:\n        raise SystemExit('expected ImportError')\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
