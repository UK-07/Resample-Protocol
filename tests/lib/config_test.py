import os
import tempfile
import unittest

import yaml

from src.lib.config import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    LEGACY_TOP_K,
    LEGACY_TOP_P,
    SamplingConfig,
    get_sampling_config,
    InferenceConfig,
    get_inference_config,
    load_config,
)


class TestLoadConfig(unittest.TestCase):
    """load_config on plain YAML files."""

    def _write(self, tmpdir, content):
        path = os.path.join(tmpdir, "config.yaml")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_loads_flat_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, "model: qwen\nbatch_size: 8\n")
            config = load_config(path)
        self.assertEqual(config, {"model": "qwen", "batch_size": 8})

    def test_loads_nested_structures(self):
        content = (
            "dataset:\n"
            "  name: mmlu\n"
            "  subjects:\n"
            "    - philosophy\n"
            "    - astronomy\n"
            "temperature: 0.7\n"
            "greedy: true\n"
            "seed: null\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, content)
            config = load_config(path)
        self.assertEqual(
            config,
            {
                "dataset": {"name": "mmlu", "subjects": ["philosophy", "astronomy"]},
                "temperature": 0.7,
                "greedy": True,
                "seed": None,
            },
        )

    def test_accepts_path_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, "key: value\n")
            config = load_config(str(path))
        self.assertEqual(config["key"], "value")

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "nope.yaml")
            with self.assertRaises(FileNotFoundError):
                load_config(missing)

    def test_invalid_yaml_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, "key: [unclosed\n  - broken: {\n")
            with self.assertRaises(yaml.YAMLError):
                load_config(path)

    def test_does_not_execute_arbitrary_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(
                tmpdir, "obj: !!python/object/apply:os.system ['echo hi']\n"
            )
            with self.assertRaises(yaml.YAMLError):
                load_config(path)


class TestLoadConfigExtends(unittest.TestCase):
    """Single-level `extends:` inheritance in load_config."""

    def _write(self, tmpdir, name, content):
        path = os.path.join(tmpdir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_child_overrides_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write(tmpdir, "base.yaml", "model_name: qwen\nseed: 42\n")
            child = self._write(
                tmpdir, "child.yaml", "extends: base.yaml\nseed: 7\ndataset: mmlu\n"
            )
            config = load_config(child)
        self.assertEqual(
            config, {"model_name": "qwen", "seed": 7, "dataset": "mmlu"}
        )
        self.assertNotIn("extends", config)

    def test_deep_merges_nested_dicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write(
                tmpdir,
                "base.yaml",
                "inference:\n  max_model_len: 16384\n  max_tokens: 8192\n",
            )
            child = self._write(
                tmpdir,
                "child.yaml",
                "extends: base.yaml\ninference:\n  max_model_len: 4096\n",
            )
            config = load_config(child)
        # max_model_len overridden, max_tokens inherited from base.
        self.assertEqual(
            config["inference"], {"max_model_len": 4096, "max_tokens": 8192}
        )

    def test_missing_base_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            child = self._write(tmpdir, "child.yaml", "extends: nope.yaml\n")
            with self.assertRaises(FileNotFoundError):
                load_config(child)

    def test_two_level_extends_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write(tmpdir, "root.yaml", "seed: 1\n")
            self._write(tmpdir, "base.yaml", "extends: root.yaml\nseed: 2\n")
            child = self._write(tmpdir, "child.yaml", "extends: base.yaml\n")
            with self.assertRaises(ValueError):
                load_config(child)

    def test_no_extends_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir, "child.yaml", "model_name: qwen\nseed: 42\n")
            config = load_config(path)
        self.assertEqual(config, {"model_name": "qwen", "seed": 42})


class TestGetInferenceConfig(unittest.TestCase):
    """get_inference_config and get_sampling_config."""

    def test_reads_nested_block(self):
        cfg = {
            "inference": {
                "max_model_len": 4096,
                "max_tokens": 1024,
                "gpu_memory_utilization": 0.8,
                "tensor_parallel_size": 2,
            }
        }
        self.assertEqual(
            get_inference_config(cfg),
            InferenceConfig(
                max_model_len=4096,
                max_tokens=1024,
                gpu_memory_utilization=0.8,
                tensor_parallel_size=2,
            ),
        )

    def test_falls_back_to_flat_keys(self):
        cfg = {"max_tokens": 2048, "gpu_memory_utilization": 0.95}
        inf = get_inference_config(cfg)
        self.assertEqual(inf.max_tokens, 2048)
        self.assertEqual(inf.gpu_memory_utilization, 0.95)
        # Unspecified keys fall through to defaults.
        self.assertEqual(inf.max_model_len, 16384)
        self.assertEqual(inf.tensor_parallel_size, 1)

    def test_max_new_tokens_alias(self):
        self.assertEqual(get_inference_config({"max_new_tokens": 512}).max_tokens, 512)

    def test_block_plus_flat_raises(self):
        cfg = {"max_tokens": 2048, "inference": {"max_tokens": 8192}}
        with self.assertRaises(ValueError):
            get_inference_config(cfg)

    def test_block_plus_flat_alias_raises(self):
        # The nested block plus the max_new_tokens alias is the same conflict.
        cfg = {"max_new_tokens": 2048, "inference": {"max_tokens": 8192}}
        with self.assertRaises(ValueError):
            get_inference_config(cfg)

    def test_null_inference_knob_falls_back(self):
        cfg = {"inference": {"max_tokens": None}}
        self.assertEqual(get_inference_config(cfg).max_tokens, 8192)

    def test_null_top_level_is_unset(self):
        # An explicit-null top-level key is not a conflict and not a value.
        cfg = {"max_tokens": None, "inference": {"max_tokens": 4096}}
        self.assertEqual(get_inference_config(cfg).max_tokens, 4096)

    def test_defaults_override(self):
        inf = get_inference_config({}, defaults=InferenceConfig(max_tokens=2048))
        self.assertEqual(inf.max_tokens, 2048)
        # An explicitly-set knob still wins over the caller default.
        inf = get_inference_config(
            {"inference": {"max_tokens": 512}}, defaults=InferenceConfig(max_tokens=2048)
        )
        self.assertEqual(inf.max_tokens, 512)

    def test_defaults_when_empty(self):
        self.assertEqual(get_inference_config({}), InferenceConfig())

    def test_sampling_config_defaults_and_validation(self):
        self.assertEqual(get_sampling_config({}), SamplingConfig(DEFAULT_TOP_P, DEFAULT_TOP_K))
        self.assertEqual(get_sampling_config({"top_p": None, "top_k": None}).top_p, DEFAULT_TOP_P)
        legacy = get_sampling_config({"top_p": LEGACY_TOP_P, "top_k": LEGACY_TOP_K})
        self.assertEqual((legacy.top_p, legacy.top_k), (1.0, -1))
        self.assertEqual(get_sampling_config({"top_p": "0.9", "top_k": "5"}), SamplingConfig(0.9, 5))
        for bad in ({"top_p": 0.0}, {"top_p": 1.5}, {"top_k": 0}, {"top_k": -2},
                    {"top_k": 2.5}, {"top_p": True}, {"top_k": "x"}):
            with self.assertRaises(ValueError):
                get_sampling_config(bad)

    def test_max_num_seqs_optional(self):
        self.assertIsNone(get_inference_config({}).max_num_seqs)
        cfg = {"inference": {"max_num_seqs": 256}}
        self.assertEqual(get_inference_config(cfg).max_num_seqs, 256)
        # Explicit null stays unset (no int(None) crash).
        cfg = {"inference": {"max_num_seqs": None}}
        self.assertIsNone(get_inference_config(cfg).max_num_seqs)


if __name__ == "__main__":
    unittest.main()
