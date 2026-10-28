"""Merge integrity tests using clearly synthetic test fixtures, never model scores."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merge_runs import MergeError, merge_runs, validate_shard


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")


def make_shard(path, seed=1, *, status="completed", error=False):
    path.mkdir()
    manifest = {
        "schema": "gdn-long-context-census.v2", "dry_run": False, "backend": "hf",
        "config": {"model": "test/model", "tokenizer": "test/tokenizer", "revision": "pin",
                   "dtype": "bf16", "seed_start": seed, "seeds": 1, "device": "cuda:0",
                   "lengths": [512], "positions": [.5], "events": [8], "conditions": ["plain"],
                   "event_placement": "early", "preedit_probes": False},
        "resolved": {"model_id": "test/model", "tokenizer_id": "test/tokenizer",
                     "model_revision": "a"*40, "tokenizer_revision": "b"*40},
        "runtime": {"python": "3.11", "torch": "fixture-version", "cuda": "fixture-cuda", "platform": "host-one", "gpu_name": "GPU-one"},
        "code_fingerprint": {"run.py": "c"*64},
        "loaded_model": {"model_id": "test/model", "revision": "a"*40,
                         "tokenizer_id": "test/tokenizer", "tokenizer_revision": "b"*40,
                         "configured_context": 2048, "parameters": 100,
                         "backend": "TestCausalLM", "device": "cuda:0", "dtype": "bf16",
                         "last_logits_policy": {"logits_to_keep": 1}, "model_type": "gated_deltanet",
                         "attention_config": None, "allow_neg_eigval": False},
        "planned_cases": 1, "execution": {"status": status},
    }
    case = {"probe_id": f"plain-seed{seed}", "pair_id": f"seed{seed}", "seed": seed,
            "prompt_hash": str(seed)*64, "answer": "A", "condition": "plain", "actual_length": 512}
    result = dict(case, status="error", error_type="OutOfMemoryError", error="test fixture") if error else dict(case, status="ok", correct=1, prediction="A")
    result["evidence_type"] = "pretrained_lm"
    write_json(path/"manifest.json", manifest)
    write_rows(path/"cases.jsonl", [case])
    write_rows(path/"results.jsonl", [result])
    return manifest, case, result


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_success_preserves_errors_and_provenance(self):
        first = self.root/"first"
        second = self.root/"second"
        make_shard(first)
        manifest, _, _ = make_shard(second, seed=20, error=True)
        manifest["config"]["device"] = "cuda:1"
        manifest["loaded_model"]["device"] = "cuda:1"
        manifest["runtime"].update(platform="different-host", gpu_name="GPU-two")
        write_json(second/"manifest.json", manifest)
        out = self.root/"combined"
        merged = merge_runs([first, second], out)
        self.assertEqual(merged["execution"], {"status": "completed", "attempted_cases": 2, "ok_cases": 1, "error_cases": 1, "missing_cases": 0})
        self.assertFalse(merged["merge"]["partial_data"])
        self.assertIsNone(merged["config"]["seed_start"])
        self.assertEqual(merged["config"]["seeds"], 2)
        self.assertEqual(len(merged["merge"]["sources"]), 2)
        self.assertEqual(len(merged["merge"]["sources"][0]["sha256"]["manifest.json"]), 64)
        _, cases, results = validate_shard(out)
        self.assertEqual(len(cases), 2)
        self.assertEqual(results[1]["error_type"], "OutOfMemoryError")

    def test_incomplete_source_remains_partial_and_preserves_plan(self):
        shard = self.root/"partial"
        make_shard(shard, status="running")
        (shard/"results.jsonl").unlink()
        merged = merge_runs([shard], self.root/"out")
        self.assertEqual(merged["execution"]["status"], "partial")
        self.assertTrue(merged["merge"]["partial_data"])
        self.assertEqual(merged["execution"]["missing_cases"], 1)
        self.assertEqual(merged["merge"]["sources"][0]["sha256"]["results.jsonl"], None)

    def test_different_experiment_or_provenance_rejected(self):
        first, second = self.root/"first", self.root/"second"
        make_shard(first)
        original, _, _ = make_shard(second, seed=2)
        modifications = [
            ("config", "model", "another-model"), ("config", "dtype", "fp32"),
            ("config", "lengths", [1024]), ("config", "preedit_probes", True),
            ("config", "conditions", ["collision"]), ("resolved", "model_revision", "d"*40),
            ("resolved", "tokenizer_revision", "e"*40), ("runtime", "python", "different"),
            ("runtime", "torch", "different"), ("runtime", "cuda", "different"),
            ("code_fingerprint", "run.py", "f"*64),
            ("loaded_model", "configured_context", 4096),
            ("loaded_model", "attention_config", {"layers": [1, 2]}),
            ("loaded_model", "allow_neg_eigval", True),
        ]
        for section, field, value in modifications:
            with self.subTest(section=section, field=field):
                modified = copy.deepcopy(original)
                modified[section][field] = value
                write_json(second/"manifest.json", modified)
                with self.assertRaises(MergeError):
                    merge_runs([first, second], self.root/"out")
                self.assertFalse((self.root/"out").exists())

    def test_duplicate_planned_probe_rejected_even_if_results_identical(self):
        first, second = self.root/"first", self.root/"second"
        make_shard(first)
        make_shard(second)
        with self.assertRaisesRegex(MergeError, "overlapping planned"):
            merge_runs([first, second], self.root/"out")
        with self.assertRaisesRegex(MergeError, "more than once"):
            merge_runs([first, first], self.root/"out")

    def test_prompt_hash_or_gold_mismatch_rejected(self):
        shard = self.root/"first"
        _, _, original = make_shard(shard)
        for field, value in (("prompt_hash", "wrong"), ("answer", "B")):
            result = dict(original, **{field: value})
            write_rows(shard/"results.jsonl", [result])
            with self.subTest(field=field), self.assertRaisesRegex(MergeError, field):
                validate_shard(shard)

    def test_mixed_duplicate_error_and_ok_rejected(self):
        shard = self.root/"first"
        _, case, result = make_shard(shard)
        error = dict(case, status="error", error_type="RuntimeError", evidence_type="pretrained_lm")
        write_rows(shard/"results.jsonl", [result, error])
        with self.assertRaisesRegex(MergeError, "duplicate/overlapping"):
            validate_shard(shard)
        write_rows(shard/"results.jsonl", [dict(result, error_type="RuntimeError")])
        with self.assertRaisesRegex(MergeError, "mixes ok and error"):
            validate_shard(shard)

    def test_mock_dry_run_v1_or_missing_identity_rejected(self):
        shard = self.root/"first"
        original, _, _ = make_shard(shard)
        for updates in ({"schema": "gdn-long-context-census.v1"}, {"backend": "mock"},
                        {"backend": "oracle"}, {"dry_run": True}, {"code_fingerprint": {}},
                        {"resolved": {}}, {"execution": {"status": "unknown"}}):
            write_json(shard/"manifest.json", dict(original, **updates))
            with self.subTest(updates=updates), self.assertRaises(MergeError):
                validate_shard(shard)

    def test_false_completion_unknown_result_and_malformed_json_rejected(self):
        shard = self.root/"first"
        manifest, _, result = make_shard(shard)
        write_rows(shard/"results.jsonl", [])
        with self.assertRaisesRegex(MergeError, "completed status"):
            validate_shard(shard)
        write_rows(shard/"results.jsonl", [dict(result, probe_id="unplanned")])
        with self.assertRaisesRegex(MergeError, "not in cases"):
            validate_shard(shard)
        (shard/"results.jsonl").write_text('{"probe_id":', encoding="utf-8")
        with self.assertRaisesRegex(MergeError, "invalid JSON"):
            validate_shard(shard)

    def test_existing_destination_never_overwritten(self):
        shard, out = self.root/"first", self.root/"existing"
        make_shard(shard)
        out.mkdir()
        marker = out/"keep"
        marker.write_text("unchanged")
        with self.assertRaisesRegex(MergeError, "already exists"):
            merge_runs([shard], out)
        self.assertEqual(marker.read_text(), "unchanged")

    def test_mislabelled_mock_rows_and_loaded_identity_rejected(self):
        shard = self.root/"first"
        manifest, case, result = make_shard(shard)
        for bad in (dict(result, evidence_type="plumbing_mock"),
                    dict(case, status="error", error_type="RuntimeError", evidence_type="plumbing_mock")):
            write_rows(shard/"results.jsonl", [bad])
            with self.assertRaisesRegex(MergeError, "pretrained_lm"):
                validate_shard(shard)
        write_rows(shard/"results.jsonl", [result])
        manifest["loaded_model"]["revision"] = "different-model-pin"
        write_json(shard/"manifest.json", manifest)
        with self.assertRaisesRegex(MergeError, "checkpoint identity"):
            validate_shard(shard)


if __name__ == "__main__":
    unittest.main()
