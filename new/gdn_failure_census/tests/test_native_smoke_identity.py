"""Regression tests for native-smoke identity/receipt validation; NO model inference.

Only the GPU model loader/forward is replaced. Probe construction, option
encoding, checkpoint fingerprinting and native-smoke control flow are real.
The in-memory test reports are not benchmark artifacts or GPU-success evidence.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import MockTokenizer, ModelInfo
from native_smoke import audit_scores, run_native

MODEL = "puigde/gated-deltanet-360M-15B-slimpajama"
PIN = "1d1a7bf99323601635f3d63f67c97614f9c2ec10"


def valid_score():
    return {
        "prediction": "A", "log_prob_choice": {x: -math.log(4) for x in "ABCD"},
        "log_prob_full": {x: -math.log(4) for x in "ABCD"},
        "latency_ms": 0.0, "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }


class LoaderBoundaryDouble:
    """Replace the unavailable CUDA dependency, never the identity resolver."""
    def __init__(self, model_id, *, tokenizer_id=None, revision=None,
                 tokenizer_revision=None, device="cuda", dtype="bf16",
                 allow_extrapolation=False):
        self.tok = MockTokenizer()
        effective_tokenizer = tokenizer_id or model_id
        effective_revision = (tokenizer_revision if effective_tokenizer != model_id
                              else tokenizer_revision or revision)
        self.info = ModelInfo(
            model_id, revision, effective_tokenizer, effective_revision,
            2048, 0, "UNIT-TEST-DOUBLE-NOT-NATIVE", device, dtype, {})
        if allow_extrapolation:
            raise AssertionError("native smoke must not opt into extrapolation")

    def score(self, token_ids, choices):
        if len(token_ids) != 2048 or set(choices) != set("ABCD"):
            raise AssertionError("real probe/choice plumbing changed unexpectedly")
        return valid_score()


class NativeSmokeIdentityTests(unittest.TestCase):
    def call_native(self, model=MODEL, **kwargs):
        with patch("native_smoke.PretrainedCausalLM", LoaderBoundaryDouble):
            return run_native(model, length=2048, **kwargs)

    def test_default_checkpoint_and_tokenizer_are_immutable(self):
        report = self.call_native()
        self.assertEqual(report["model_info"]["revision"], PIN)
        self.assertEqual(report["model_info"]["tokenizer_revision"], PIN)

    def test_explicit_checkpoint_revision_also_pins_implicit_tokenizer(self):
        revision = "a" * 40
        report = self.call_native(revision=revision)
        self.assertEqual(report["model_info"]["revision"], revision)
        self.assertEqual(report["model_info"]["tokenizer_revision"], revision)

    def test_separate_tokenizer_revision_does_not_replace_model_pin(self):
        report = self.call_native(tokenizer_id="test/separate-tokenizer",
                                  tokenizer_revision="b" * 40)
        self.assertEqual(report["model_info"]["revision"], PIN)
        self.assertEqual(report["model_info"]["tokenizer_id"], "test/separate-tokenizer")
        self.assertEqual(report["model_info"]["tokenizer_revision"], "b" * 40)

    def test_report_retains_resolved_input_identity(self):
        report = self.call_native()
        self.assertEqual(report.get("input_provenance"), {
            "model_id": MODEL, "model_revision": PIN,
            "tokenizer_id": MODEL, "tokenizer_revision": PIN,
        })

    def test_local_checkpoint_has_byte_fingerprint_not_a_hub_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text('{}', encoding="utf-8")
            weights = root / "model.safetensors"
            weights.write_bytes(b"test-only-weights-v1")
            before = self.call_native(folder)
            identity_before = before.get("input_provenance", {})
            self.assertTrue(identity_before.get("model_revision", "").startswith("local-sha256:"))
            self.assertEqual(identity_before["tokenizer_revision"], identity_before["model_revision"])
            self.assertIsNone(before["model_info"]["revision"])
            self.assertIsNone(before["model_info"]["tokenizer_revision"])
            weights.write_bytes(b"test-only-weights-v2")
            after = self.call_native(folder)
            self.assertNotEqual(identity_before["model_revision"],
                                after["input_provenance"]["model_revision"])

    def test_empty_local_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "empty local checkpoint"):
                self.call_native(folder)


class NativeSmokeReceiptTests(unittest.TestCase):
    def test_valid_zero_latency_receipt_is_accepted(self):
        audit_scores(valid_score())

    def test_empty_prediction_is_not_a_choice(self):
        score = valid_score()
        score["prediction"] = ""
        with self.assertRaises(ValueError):
            audit_scores(score)

    def test_multiletter_substring_is_not_a_choice(self):
        score = valid_score()
        score["prediction"] = "AB"
        with self.assertRaises(ValueError):
            audit_scores(score)

    def test_nonfinite_latency_is_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                score = valid_score()
                score["latency_ms"] = value
                with self.assertRaises(ValueError):
                    audit_scores(score)

    def test_boolean_latency_is_not_a_numeric_measurement(self):
        score = valid_score()
        score["latency_ms"] = True
        with self.assertRaises(ValueError):
            audit_scores(score)

    def test_negative_latency_is_rejected(self):
        score = valid_score()
        score["latency_ms"] = -1.0
        with self.assertRaises(ValueError):
            audit_scores(score)


if __name__ == "__main__":
    unittest.main()
