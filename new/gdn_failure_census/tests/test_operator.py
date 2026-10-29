"""Independent identities and boundary checks for the gated-delta reference."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from operator_probe import reference_sequence, run_probe, update_state
import operator_probe as operator_module


class GatedDeltaOperatorTests(unittest.TestCase):
    def test_decay_precedes_residual_and_input_is_not_mutated(self):
        initial = np.array([[2.0, 0.0], [0.0, 0.0]])
        output = update_state(initial, [1, 0], [1, 0], alpha=0.5, beta=1)
        np.testing.assert_allclose(output, [[1, 0], [0, 0]], atol=0, rtol=0)
        np.testing.assert_array_equal(initial, [[2, 0], [0, 0]])

    def test_reference_matches_independent_matrix_transition(self):
        rng = np.random.default_rng(22)
        keys = rng.standard_normal((17, 5))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        values = rng.standard_normal((17, 3))
        queries = rng.standard_normal((17, 5))
        alphas, betas = rng.uniform(0, 1, (2, 17))
        initial = rng.standard_normal((5, 3))
        outputs, final = reference_sequence(keys, values, alphas, betas, queries, initial)
        independent = initial.copy()
        expected_outputs = []
        for key, value, query, a, b in zip(keys, values, queries, alphas, betas):
            transition = a * (np.eye(5) - b * np.outer(key, key))
            independent = transition @ independent + b * np.outer(key, value)
            expected_outputs.append(query @ independent)
        np.testing.assert_allclose(final, independent, atol=2e-15, rtol=1e-12)
        np.testing.assert_allclose(outputs, expected_outputs, atol=2e-15, rtol=1e-12)

    def test_no_write_closed_form(self):
        rng = np.random.default_rng(5)
        initial = rng.standard_normal((4, 3))
        keys = rng.standard_normal((20, 4))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True)
        values = rng.standard_normal((20, 3))
        gates = np.linspace(0.6, 1.0, 20)
        _, final = reference_sequence(keys, values, gates, 0, initial_state=initial)
        np.testing.assert_allclose(final, np.prod(gates) * initial, atol=1e-15)

    def test_collinear_zero_value_decay_is_alpha_times_one_minus_beta(self):
        initial = np.array([[1.0, 0.0], [0.0, 0.0]])
        length, alpha, beta = 23, 0.97, 0.15
        keys = np.tile([1.0, 0.0], (length, 1))
        values = np.zeros((length, 2))
        _, final = reference_sequence(keys, values, alpha, beta, initial_state=initial)
        np.testing.assert_allclose(final, (alpha * (1 - beta)) ** length * initial,
                                   atol=1e-15, rtol=1e-12)

    def test_orthogonal_writes_preserve_unrelated_association(self):
        initial = np.array([[1.0, -2.0], [0.0, 0.0]])
        keys = np.tile([0.0, 1.0], (50, 1))
        rng = np.random.default_rng(6)
        values = rng.standard_normal((50, 2))
        _, final = reference_sequence(keys, values, 1, 0.7, initial_state=initial)
        np.testing.assert_allclose(final[0], initial[0], atol=0, rtol=0)

    def test_beta_one_exactly_overwrites_unit_key(self):
        rng = np.random.default_rng(4)
        key = rng.standard_normal(8)
        key /= np.linalg.norm(key)
        value = rng.standard_normal(3)
        state = rng.standard_normal((8, 3))
        final = update_state(state, key, value, alpha=0.82, beta=1)
        np.testing.assert_allclose(key @ final, value, atol=1e-14, rtol=1e-14)

    def test_float32_preserved_and_empty_sequence_allowed(self):
        output = update_state(np.zeros((2, 3), dtype=np.float32), [1, 0], [1, 2, 3], 1, 1)
        self.assertEqual(output.dtype, np.dtype("float32"))
        outputs, state = reference_sequence(np.empty((0, 2)), np.empty((0, 3)), dtype="float32")
        self.assertEqual(outputs.shape, (0, 3))
        self.assertEqual(state.shape, (2, 3))
        self.assertEqual(state.dtype, np.dtype("float32"))

    def test_reject_invalid_gate_shape_key_and_state(self):
        state = np.zeros((2, 2))
        for invalid in (-0.1, 1.1, np.nan, np.inf, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                update_state(state, [1, 0], [0, 1], invalid, 1)
        with self.assertRaises(ValueError):
            update_state(state, [2, 0], [0, 1], 1, 1)
        with self.assertRaises(ValueError):
            update_state(np.zeros((2, 2), dtype=int), [1, 0], [0, 1], 1, 1)
        with self.assertRaises(ValueError):
            reference_sequence([[1, 0], [0, 1]], [[1, 0], [0, 1]], alpha=[1])

    def test_probe_analytic_form_and_json_serialization(self):
        report = run_probe({"dimensions": [4], "value_dim": 3,
                            "alphas": [1, 0.9], "betas": [0, 0.25, 1],
                            "gaps": [0, 1, 7], "correlations": [0, 0.4, 1],
                            "loads": [1, 8], "seeds": [0, 3],
                            "dtypes": ["float64", "float32"]})
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["trained_model_evaluated"])
        rows = report["rows"]
        self.assertEqual(report["row_count"], len(rows))
        float64_errors = [r["max_abs_analytic_error"] for r in rows
                          if r["dtype"] == "float64" and "max_abs_analytic_error" in r]
        float32_errors = [r["max_abs_analytic_error"] for r in rows
                          if r["dtype"] == "float32" and "max_abs_analytic_error" in r]
        self.assertLess(max(float64_errors), 1e-12)
        self.assertLess(max(float32_errors), 3e-6)
        full_orthogonal = [r for r in rows if r["probe"] == "orthogonal_interference"
                           and r["alpha"] == 1]
        self.assertTrue(all(r["retention_amplitude"] == 1 for r in full_orthogonal))
        beta_zero_capacity = [r for r in rows if r["probe"] == "capacity" and r["beta"] == 0]
        self.assertTrue(all(r["classification_accuracy"] == 0 for r in beta_zero_capacity))
        self.assertTrue(all(r["mean_cosine_nonzero"] is None for r in beta_zero_capacity))

    def test_probe_is_reproducible_and_rejects_typos(self):
        config = {"dimensions": [2], "value_dim": 2, "alphas": [1],
                  "betas": [0.5], "gaps": [0, 2], "correlations": [0.2],
                  "loads": [3], "seeds": [7], "dtypes": ["float64"]}
        self.assertEqual(run_probe(config), run_probe(config))
        for bad in ({"dimension": [2]}, {"dimensions": [1]}, {"seeds": [-1]},
                    {"gaps": [1.5]}, {"dtypes": ["float16"]}, {"loads": []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_probe(bad)

    def test_standalone_cli_protects_existing_reports(self):
        config = {"dimensions": [2], "value_dim": 2, "alphas": [1],
                  "betas": [1], "gaps": [0, 2], "correlations": [0.4],
                  "loads": [2], "seeds": [0]}
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            output_path = Path(temp) / "report.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            command = [sys.executable, str(Path(operator_module.__file__).resolve()),
                       "--config", str(config_path), "--out", str(output_path)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            original_bytes = output_path.read_bytes()
            self.assertFalse(json.loads(original_bytes)["trained_model_evaluated"])
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output_path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
