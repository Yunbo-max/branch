"""CPU-only checkpoint preflight tests; fake safetensors headers, NO actual model weights."""
from __future__ import annotations
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from checkpoint_preflight import (inspect_local_checkpoint, main, safetensors_key_names)


def fake_checkpoint(root: Path, *, extra_d=False, no_tokenizer=False,
                    model_type="gated_deltanet", layers=2):
    config = {
        "model_type": model_type,
        "architectures": ["GatedDeltaNetForCausalLM"],
        "num_hidden_layers": layers,
        "max_position_embeddings": 2048,
        "hidden_size": 1024,
        "head_dim": 256,
        "num_heads": 4,
        "attn": None,
    }
    (root / "config.json").write_text(json.dumps(config))
    if not no_tokenizer:
        (root / "tokenizer.json").write_text("{}")
    names = [f"model.layers.{i}.attn.q_proj.weight" for i in range(layers)]
    if extra_d:
        names.extend(f"model.layers.{i}.attn.D" for i in range(layers))
    head = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in names:
        head[name] = {
            "dtype": "F32", "shape": [1],
            "data_offsets": [offset, offset + 4]
        }
        offset += 4
    raw = json.dumps(head).encode()
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"\x00" * offset
    )
    return root / "model.safetensors"


class CheckpointPreflightTests(unittest.TestCase):
    def test_compatible_shape_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = fake_checkpoint(root)
            result = inspect_local_checkpoint(root)
            self.assertEqual(result["status"], "preflight_passed_not_loaded")
            self.assertEqual(result["num_hidden_layers"], 2)
            self.assertEqual(result["layers_found"], 2)
            self.assertEqual(result["weight_shard_count"], 1)
            self.assertEqual(result["weight_tensor_count"], 2)
            self.assertEqual(len(safetensors_key_names(weights)), 2)
            self.assertIsNone(result["single_next_token_labels"])
            self.assertTrue(any("NOT a successful FLA forward" in x for x in result["limitations"]))

    def test_m_a_p_fork_d_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_checkpoint(root, extra_d=True)
            result = inspect_local_checkpoint(root)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("attn.D tensors", " ".join(result["issues"]))
            self.assertEqual(main(["--model-dir", str(root)]), 2)

    def test_bad_sha_and_correct_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = fake_checkpoint(root)
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            ok = inspect_local_checkpoint(root, expected_weight_sha256=sha)
            self.assertEqual(ok["status"], "preflight_passed_not_loaded")
            self.assertEqual(ok["weight_sha256"], sha)
            wrong = inspect_local_checkpoint(root, expected_weight_sha256="0" * 64)
            self.assertEqual(wrong["status"], "blocked")
            self.assertIn("mismatch", " ".join(wrong["issues"]))

    def test_missing_tokenizer_and_wrong_model_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_checkpoint(root, no_tokenizer=True, model_type="mamba")
            result = inspect_local_checkpoint(root)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("model_type", " ".join(result["issues"]))
            self.assertIn("missing tokenizer", " ".join(result["issues"]))

    def test_missing_and_invalid_weight_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"model_type":"gated_deltanet"}')
            with self.assertRaisesRegex(ValueError, "no .safetensors"):
                inspect_local_checkpoint(root)
            weight = root / "model.safetensors"
            weight.write_bytes(struct.pack("<Q", 123) + b"{}")
            with self.assertRaisesRegex(ValueError, "truncated"):
                safetensors_key_names(weight)

    def test_sharded_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = fake_checkpoint(root)
            (root / "more.safetensors").write_bytes(weight.read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate tensor names"):
                inspect_local_checkpoint(root)

    def test_cli_saves_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoint"
            root.mkdir()
            fake_checkpoint(root)
            out = Path(tmp) / "receipt.json"
            code = main(["--model-dir", str(root), "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertTrue(out.is_file())
            self.assertEqual(json.loads(out.read_text())["schema"], "gdn-checkpoint-preflight.v1")
            with self.assertRaises(FileExistsError):
                main(["--model-dir", str(root), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
