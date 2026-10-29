"""No-GPU verification for the strict one-forward smoke result validator."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_smoke import audit_scores


def mock_result():
    return {
        "prediction": "A",
        "log_prob_choice": dict(zip("ABCD", [math.log(0.4), math.log(0.3),
                                             math.log(0.2), math.log(0.1)])),
        "latency_ms": 17.0,
    }


class NativeSmokeAuditTests(unittest.TestCase):
    def test_valid_scoring_passes(self):
        audit_scores(mock_result())

    def test_reject_missing_candidate(self):
        r = mock_result()
        r["log_prob_choice"].pop("D")
        with self.assertRaisesRegex(ValueError, "four"):
            audit_scores(r)

    def test_reject_nan_or_bad_normalization(self):
        r = mock_result()
        r["log_prob_choice"]["B"] = math.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit_scores(r)
        r = mock_result()
        r["log_prob_choice"] = {c: math.log(.05) for c in "ABCD"}
        with self.assertRaisesRegex(ValueError, "sum to one"):
            audit_scores(r)

    def test_reject_invalid_model_result(self):
        r = mock_result()
        r["prediction"] = "E"
        with self.assertRaisesRegex(ValueError, "invalid choice"):
            audit_scores(r)
        r = mock_result()
        r["latency_ms"] = -1
        with self.assertRaisesRegex(ValueError, "timing"):
            audit_scores(r)


if __name__ == "__main__":
    unittest.main()
