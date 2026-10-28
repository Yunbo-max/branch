"""CPU integration tests of the real CLI, using explicitly NON-SCIENTIFIC mocks.

These tests establish planning, persistence and provenance behavior. They do
not establish native HF/FLA compatibility or any GDN accuracy/performance.
"""
import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as census_run
from backend import MockEvaluator, MockTokenizer
from merge_runs import MergeError, validate_shard
from provenance import resolve_revision
from spec import CONDITIONS, make_probe


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


class RunnerV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, name="run", *extra):
        return ["--backend", "mock", "--out", str(self.root/name),
                "--lengths", "4096", "--positions", "0.2", "--events", "8",
                "--seeds", "2", "--log-every", "1000", *extra]

    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return census_run.main(args)

    def test_actual_main_and_identical_resume_do_not_repeat_results(self):
        args = self.args()
        self.assertEqual(self.invoke(args), 0)
        out = self.root/"run"
        manifest = read_json(out/"manifest.json")
        plan, results = read_rows(out/"cases.jsonl"), read_rows(out/"results.jsonl")
        self.assertEqual(manifest["schema"], "gdn-long-context-census.v2")
        self.assertEqual(manifest["planned_cases"], 2*len(CONDITIONS))
        self.assertEqual(manifest["execution"]["status"], "completed")
        self.assertEqual({row["evidence_type"] for row in results}, {"plumbing_mock"})
        self.assertTrue(all("token_ids" not in row for row in plan))
        before = (out/"results.jsonl").read_bytes()
        with patch.object(census_run, "MockEvaluator", side_effect=AssertionError("resume must not load/evaluate")):
            self.assertEqual(self.invoke(args+["--resume"]), 0)
        self.assertEqual(before, (out/"results.jsonl").read_bytes())
        with self.assertRaisesRegex(MergeError, "mock"):
            validate_shard(out)

    def test_dry_run_has_no_evaluator_or_results(self):
        with patch.object(census_run, "MockEvaluator", side_effect=AssertionError("dry run loaded evaluator")):
            self.assertEqual(self.invoke(self.args("dry", "--dry-run")), 0)
        out = self.root/"dry"
        self.assertFalse((out/"results.jsonl").exists())
        self.assertEqual(read_json(out/"manifest.json")["execution"]["status"], "dry_run")
        with self.assertRaisesRegex(ValueError, "dry-run"):
            self.invoke(self.args("dry", "--resume"))

    def test_sample_and_token_limits_keep_whole_matched_groups(self):
        self.invoke(self.args("samples", "--max-samples", str(len(CONDITIONS)+1)))
        manifest = read_json(self.root/"samples"/"manifest.json")
        self.assertEqual(manifest["planned_cases"], len(CONDITIONS))
        self.assertEqual(manifest["planning_stop_reason"], "max_samples_whole_groups")
        self.assertEqual(len({row["pair_id"] for row in read_rows(self.root/"samples"/"cases.jsonl")}), 1)
        cap = 4096*len(CONDITIONS)+4096
        self.invoke(self.args("tokens", "--max-input-tokens", str(cap)))
        manifest = read_json(self.root/"tokens"/"manifest.json")
        self.assertEqual(manifest["planned_cases"], len(CONDITIONS))
        self.assertLessEqual(manifest["planned_input_tokens"], cap)
        self.assertEqual(manifest["planning_stop_reason"], "max_input_tokens_whole_groups")

    def test_planned_forward_tokens_equal_executed_with_preedit_and_no_target(self):
        calls = []
        class CountingEvaluator(MockEvaluator):
            def score(self, token_ids, choices):
                calls.append(len(token_ids))
                return super().score(token_ids, choices)
        with patch.object(census_run, "MockEvaluator", CountingEvaluator):
            self.invoke(self.args("preedit", "--preedit-probes"))
        out = self.root/"preedit"
        manifest = read_json(out/"manifest.json")
        results = read_rows(out/"results.jsonl")
        self.assertEqual(manifest["planned_input_tokens"], sum(calls))
        for row in results:
            if row["condition"] == "no_target":
                self.assertFalse(row["answerable"])
                self.assertIsNone(row["preedit_prompt_tokens"])

    def test_infeasible_length_drops_entire_group(self):
        self.invoke(self.args("invalid", "--lengths", "512,4096", "--seeds", "1"))
        manifest = read_json(self.root/"invalid"/"manifest.json")
        plan = read_rows(self.root/"invalid"/"cases.jsonl")
        self.assertEqual(len(plan), len(CONDITIONS))
        self.assertEqual({row["target_length"] for row in plan}, {4096})
        self.assertEqual(manifest["skipped_invalid_grid_cases"], len(CONDITIONS))
        self.assertEqual({row["condition"] for row in plan}, set(CONDITIONS))

    def test_config_plan_result_and_code_mutations_refuse_resume(self):
        args = self.args()
        self.invoke(args)
        out = self.root/"run"
        with self.assertRaisesRegex(ValueError, "config"):
            self.invoke(args+["--resume", "--dtype", "fp32"])
        with patch.object(census_run, "code_fingerprint", return_value="modified-code"):
            with self.assertRaisesRegex(ValueError, "inference code"):
                self.invoke(args+["--resume"])
        original_plan = (out/"cases.jsonl").read_bytes()
        (out/"cases.jsonl").write_bytes(original_plan+b"\n")
        with self.assertRaisesRegex(ValueError, "plan changed"):
            self.invoke(args+["--resume"])
        (out/"cases.jsonl").write_bytes(original_plan)
        results = read_rows(out/"results.jsonl")
        results[0]["prompt_hash"] = "modified-prompt"
        write_rows(out/"results.jsonl", results)
        with self.assertRaisesRegex(ValueError, "metadata/prompt"):
            self.invoke(args+["--resume"])

    def test_resume_rejects_wrong_evidence_type_and_invalid_scored_row(self):
        args = self.args()
        self.invoke(args)
        result_path = self.root/"run"/"results.jsonl"
        original = read_rows(result_path)
        for field, value in (("evidence_type", "pretrained_lm"), ("correct", 7)):
            rows = [dict(row) for row in original]
            rows[0][field] = value
            write_rows(result_path, rows)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.invoke(args+["--resume"])

    def test_record_errors_survive_resume_without_retry(self):
        class FailingEvaluator(MockEvaluator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.calls = 0
            def score(self, token_ids, choices):
                self.calls += 1
                if self.calls % 2 == 0:
                    raise RuntimeError("intentional integration-test failure")
                return super().score(token_ids, choices)
        args = self.args("errors", "--error-policy", "record")
        with patch.object(census_run, "MockEvaluator", FailingEvaluator):
            self.assertEqual(self.invoke(args), 0)
        path = self.root/"errors"/"results.jsonl"
        rows = read_rows(path)
        self.assertEqual(sum(row["status"] == "error" for row in rows), len(CONDITIONS))
        before = path.read_bytes()
        with patch.object(census_run, "MockEvaluator", side_effect=AssertionError("must not retry failures")):
            self.assertEqual(self.invoke(args+["--resume"]), 0)
        self.assertEqual(before, path.read_bytes())

    def test_resume_reconciles_status_after_final_case_failure(self):
        class LastFailure(MockEvaluator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.calls = 0
            def score(self, token_ids, choices):
                self.calls += 1
                if self.calls == len(CONDITIONS):
                    raise RuntimeError("final-case test failure")
                return super().score(token_ids, choices)
        args = self.args("last", "--seeds", "1")
        with patch.object(census_run, "MockEvaluator", LastFailure):
            with self.assertRaisesRegex(RuntimeError, "final-case"):
                self.invoke(args)
        out = self.root/"last"
        self.assertEqual(read_json(out/"manifest.json")["execution"]["status"], "failed")
        self.assertEqual(self.invoke(args+["--resume"]), 0)
        manifest = read_json(out/"manifest.json")
        self.assertEqual(manifest["execution"]["status"], "completed")
        self.assertEqual(sum(row["status"] == "error" for row in read_rows(out/"results.jsonl")), 1)

    def test_wall_budget_stops_between_groups_and_preserves_partial_state(self):
        # First group starts at t=0; the next starts after the cumulative cap.
        with patch.object(census_run.time, "perf_counter", side_effect=[0.0, 0.0, 10.0, 10.0]):
            code = self.invoke(self.args("wall", "--max-runtime-seconds", "5"))
        out = self.root/"wall"
        manifest = read_json(out/"manifest.json")
        results = read_rows(out/"results.jsonl")
        self.assertEqual(code, 75)
        self.assertEqual(manifest["execution"]["status"], "partial")
        self.assertEqual(manifest["execution"]["stop_reason"], "cumulative_wall_budget")
        self.assertEqual(len(results), len(CONDITIONS))
        self.assertEqual(len({row["pair_id"] for row in results}), 1)

    def test_nonfinite_wall_limits_are_rejected(self):
        for bad in ("nan", "inf", "-inf"):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.invoke(self.args("invalid-time", "--max-runtime-seconds", bad))

    def test_token_payloads_are_streamed_not_kept_for_entire_grid(self):
        alive, maximum, created = 0, 0, 0
        original = census_run.make_probe
        def released():
            nonlocal alive
            alive -= 1
        def tracked(*args, **kwargs):
            nonlocal alive, maximum, created
            probe = original(*args, **kwargs)
            alive += 1
            created += 1
            maximum = max(maximum, alive)
            weakref.finalize(probe, released)
            return probe
        with patch.object(census_run, "make_probe", tracked):
            self.invoke(self.args("stream", "--seeds", "12"))
        gc.collect()
        self.assertGreaterEqual(created, 12*len(CONDITIONS))
        self.assertLessEqual(maximum, 3*len(CONDITIONS)+3)
        self.assertEqual(alive, 0)

    def test_no_target_removes_record_and_local_weight_changes_change_identity(self):
        tok = MockTokenizer()
        absent = make_probe(tok, condition="no_target", length=4096, seed=1000,
                            needle_fraction=.2, num_events=8)
        present = make_probe(tok, condition="unrelated", length=4096, seed=1000,
                             needle_fraction=.2, num_events=8)
        target_line = f"Record {absent.key}: {absent.initial_answer}."
        self.assertNotIn(target_line, tok.decode(absent.token_ids))
        self.assertIn(target_line, tok.decode(present.token_ids))
        self.assertEqual(absent.needle_token, present.needle_token)
        self.assertEqual(absent.actual_length, present.actual_length)
        checkpoint = self.root/"local_checkpoint"
        checkpoint.mkdir()
        (checkpoint/"config.json").write_text('{}')
        weights = checkpoint/"model.safetensors"
        weights.write_bytes(b"test-only-placeholder-v1")
        first = resolve_revision(str(checkpoint), None)
        weights.write_bytes(b"test-only-placeholder-v2")
        second = resolve_revision(str(checkpoint), None)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("local-sha256:"))


if __name__ == "__main__":
    unittest.main()
