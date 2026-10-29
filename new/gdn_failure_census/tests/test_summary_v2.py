"""No-model tests for independent-seed inference and explicit coverage."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize import summarize_records
from plot import build_plot_metadata


def rec(seed, condition="plain", *, suffix="", length=512, correct=1, **extra):
    pair = f"s{seed}-{length}-{suffix}"
    value = {"status": "ok", "probe_id": pair + "/" + condition,
             "pair_id": pair, "condition": condition, "seed": seed,
             "actual_length": length, "needle_fraction": .1, "num_events": 8,
             "correct": correct, "margin": .2, "gold_conditional_nll": .3,
             "latency_ms": 10., "is_context_extrapolated": False}
    value.update(extra)
    return value


class SummaryTests(unittest.TestCase):
    def summarize(self, rows, **kw):
        return summarize_records(rows, bootstrap_rounds=200, **kw)

    def test_long_minimum_never_becomes_short_control(self):
        out = self.summarize([rec(i, length=8192) for i in range(30)])
        ctrl = out["short_control"]
        self.assertEqual(ctrl["min_tested_length"], 8192)
        self.assertIsNone(ctrl["evaluated_control_length"])
        self.assertEqual(ctrl["competence_gate"], "INSUFFICIENT_SHORT_CONTROL")

    def test_extrapolated_short_cannot_pass(self):
        out = self.summarize([rec(i, is_context_extrapolated=True) for i in range(25)])
        self.assertEqual(out["short_control"]["plain_n"], 0)

    def test_smoke_is_provisional_and_twenty_seeds_can_pass(self):
        self.assertEqual(self.summarize([rec(1), rec(2)])["short_control"]["competence_gate"],
                         "PROVISIONAL_SHORT_CONTROL")
        self.assertEqual(self.summarize([rec(i) for i in range(20)])["short_control"]["competence_gate"],
                         "PASS")

    def test_repeating_one_seed_does_not_create_independent_evidence(self):
        rows = [rec(7, suffix=str(i)) for i in range(40)]
        out = self.summarize(rows)
        self.assertEqual(out["short_control"]["n_independent_seeds"], 1)
        self.assertEqual(out["short_control"]["competence_gate"], "PROVISIONAL_SHORT_CONTROL")
        self.assertIsNone(out["groups"][0]["accuracy_ci_low"])

    def test_missing_seed_disables_inferential_ci(self):
        rows = [rec(i) for i in range(30)]
        for row in rows:
            del row["seed"]
        out = self.summarize(rows)
        self.assertEqual(out["short_control"]["n_independent_seeds"], 0)
        self.assertEqual(out["short_control"]["competence_gate"], "PROVISIONAL_SHORT_CONTROL")
        self.assertIsNone(out["groups"][0]["accuracy_ci_high"])

    def test_bootstrap_keeps_seed_replicates_together(self):
        small = [rec(0, correct=0), rec(1, correct=1)]
        repeated = [rec(seed, suffix=str(i), correct=seed) for seed in (0, 1) for i in range(25)]
        a, b = (self.summarize(x)["groups"][0] for x in (small, repeated))
        self.assertEqual(a["accuracy_ci_low"], b["accuracy_ci_low"])
        self.assertEqual(a["accuracy_ci_high"], b["accuracy_ci_high"])
        self.assertEqual(b["n_independent_seeds"], 2)

    def test_paired_effect_and_incomplete_coverage(self):
        rows = []
        for i in range(3):
            for cond in ("plain", "unrelated", "collision"):
                rows.append(rec(i, cond, correct=int(cond != "collision")))
        rows[-1].update(status="error", error_type="OutOfMemoryError", error="CUDA out of memory")
        out = self.summarize(rows, planned_cases=12)
        self.assertEqual(out["n_ok"], 8)
        self.assertEqual(out["n_error"], 1)
        self.assertEqual(out["n_oom"], 1)
        self.assertEqual(out["n_missing"], 3)
        self.assertEqual(out["pair_coverage"]["incomplete_or_failed_triplets"], 1)
        self.assertEqual(out["paired"][0]["n_triplets"], 2)
        self.assertEqual(out["paired"][0]["unrelated_vs_collision_mean"], -1)
        self.assertEqual(out["paired"][0]["conditional_collision_failure_mean"], 1)

    def test_no_unrelated_success_has_no_conditional_failure_rate(self):
        rows = [rec(i, c, correct=0) for i in range(2) for c in ("plain", "unrelated", "collision")]
        out = self.summarize(rows)
        self.assertIsNone(out["paired"][0]["conditional_collision_failure_mean"])

    def test_duplicate_probe_and_duplicate_pair_condition_rejected(self):
        a = rec(1)
        with self.assertRaisesRegex(ValueError, "duplicate probe_id"):
            self.summarize([a, dict(a)])
        b = dict(a, probe_id="another-id")
        with self.assertRaisesRegex(ValueError, "duplicate pair_id/condition"):
            self.summarize([a, b])

    def test_outside_plan_and_mismatched_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the case plan"):
            self.summarize([rec(1)], expected_probe_ids=["unknown"])
        a, b = rec(1), rec(1, "collision")
        b["seed"] = 2
        with self.assertRaisesRegex(ValueError, "mismatched paired seed"):
            self.summarize([a, b])

    def test_revision_latest_is_not_same_fact_loss(self):
        rows = [rec(1, "revision_latest", correct=0, preedit_correct=1,
                    initial_answer="A", answer="B"),
                rec(1, "revision_earliest", correct=0, preedit_correct=1,
                    initial_answer="A", answer="A")]
        out = self.summarize(rows)
        by_condition = {r["condition"]: r for r in out["preedit"]}
        self.assertIsNone(by_condition["revision_latest"]["lost_after_edit_given_preedit_correct"])
        self.assertFalse(by_condition["revision_latest"]["prepost_same_target"])
        self.assertEqual(by_condition["revision_earliest"]["lost_after_edit_given_preedit_correct"], 1)
        self.assertEqual(out["revisions"][0]["both_correct"], 0)

    def test_all_errors_still_has_coverage_and_tables(self):
        out = self.summarize([rec(1, status="unsupported")], planned_cases=4)
        self.assertEqual(out["n_ok"], 0)
        self.assertEqual(out["n_unsupported"], 1)
        self.assertEqual(out["n_missing"], 3)
        for key in ("groups", "paired", "preedit", "revisions"):
            self.assertEqual(out[key], [])
        self.assertEqual(out["chance_accuracy"], .25)

    def test_mock_rows_never_enter_scientific_tables_or_gate(self):
        rows = [rec(i, c, evidence_type="plumbing_mock")
                for i in range(25) for c in ("plain", "unrelated", "collision")]
        out = self.summarize(rows, planned_cases=75)
        self.assertEqual(out["n_mock"], 75)
        self.assertEqual(out["n_mock_ok"], 75)
        self.assertEqual(out["n_ok"], 0)
        self.assertEqual(out["n_missing"], 0)
        self.assertTrue(out["diagnostic_only"])
        self.assertEqual(out["short_control"]["competence_gate"], "DIAGNOSTIC_ONLY")
        self.assertEqual(out["groups"], [])
        self.assertEqual(out["paired"], [])

    def test_mixed_mock_and_real_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mixed evidence types"):
            self.summarize([rec(1, evidence_type="plumbing_mock"),
                            rec(2, evidence_type="pretrained_lm")])

    def test_revision_label_change_is_stratified(self):
        rows = []
        for i, latest in enumerate(("A", "B")):
            for condition in ("revision_latest", "revision_earliest"):
                rows.append(rec(i, condition, correct=int(latest == "A"),
                                initial_answer="A", latest_answer=latest))
        group = self.summarize(rows)["revisions"][0]
        self.assertEqual(group["n_changed_target_labels"], 1)
        self.assertEqual(group["n_unchanged_target_labels"], 1)
        self.assertEqual(group["changed_target_latest_accuracy"], 0)
        self.assertEqual(group["unchanged_target_latest_accuracy"], 1)

    def test_no_target_has_separate_above_chance_check(self):
        rows = [rec(i, "no_target", correct=1) for i in range(20)]
        out = self.summarize(rows)
        control = out["no_target_control"]
        self.assertEqual(control["n"], 20)
        self.assertEqual(control["n_seeds"], 20)
        self.assertEqual(control["chance_accuracy"], .25)
        self.assertEqual(control["flag"], "ABOVE_CHANCE")
        self.assertFalse(control["answerable_control"])
        self.assertFalse(out["groups"][0]["answerable_control"])
        self.assertEqual(out["short_control"]["competence_gate"], "INSUFFICIENT_SHORT_CONTROL")

    def test_no_target_small_sample_stays_provisional(self):
        out = self.summarize([rec(1, "no_target"), rec(2, "no_target")])
        self.assertEqual(out["no_target_control"]["flag"], "PROVISIONAL")
        self.assertEqual(self.summarize([rec(1)])["no_target_control"]["flag"], "NOT_MEASURED")
        rows = [rec(i, "no_target", correct=int(i % 4 == 0)) for i in range(24)]
        self.assertEqual(self.summarize(rows)["no_target_control"]["flag"], "NO_CLEAR_ABOVE_CHANCE")

    def test_full_vocabulary_loss_has_its_own_denominator(self):
        rows = [rec(0, gold_full_nll=4.), rec(1, gold_full_nll=6.), rec(2)]
        group = self.summarize(rows)["groups"][0]
        self.assertEqual(group["gold_full_nll"], 5.)
        self.assertEqual(group["n_gold_full_nll"], 2)
        self.assertAlmostEqual(group["conditional_nll"], .3)

    def test_plot_metadata_exposes_length_dependent_cell_selection(self):
        payload = {"groups": [
            {"condition": "plain", "length": 512, "events": 8,
             "needle_region": "early", "n": 20},
            {"condition": "plain", "length": 4096, "events": 32,
             "needle_region": "early", "n": 20},
        ]}
        meta = build_plot_metadata(payload)
        self.assertTrue(meta["descriptive_only"])
        self.assertFalse(meta["length_curve_is_causal"])
        self.assertTrue(meta["cell_composition_varies_by_length"]["plain"])
        self.assertEqual(len(meta["coverage"]), 2)


if __name__ == "__main__":
    unittest.main()
