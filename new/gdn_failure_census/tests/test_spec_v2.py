"""Scientific-control regression checks; pure stdlib and no model downloads."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from spec import CONDITIONS, LETTERS, make_probe


class CharTokenizerWithBOS:
    """Makes special-token accounting observable, without using HF packages."""
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return ([self.bos_token_id] if add_special_tokens else []) + [ord(c) + 10 for c in text]

    def decode(self, token_ids):
        return "".join(chr(x - 10) for x in token_ids if x != self.bos_token_id)


def records(tokenizer, probe):
    text = tokenizer.decode(probe.token_ids)
    database = text.split("Database:\n", 1)[1].rsplit("\nQuestion:", 1)[0]
    return re.findall(r"Record (REC-[A-Z]+-\d+): ([ABCD])\.", database)


class SpecV2Tests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizerWithBOS()

    def make(self, condition, **kwargs):
        defaults = dict(length=8192, seed=17, needle_fraction=0.35,
                        num_events=16, event_placement="early")
        defaults.update(kwargs)
        return make_probe(self.tokenizer, condition=condition, **defaults)

    def test_exact_budgets_with_bos_and_repeatability(self):
        for condition in CONDITIONS:
            for placement in ("early", "late"):
                with self.subTest(condition=condition, placement=placement):
                    first = self.make(condition, length=4096, event_placement=placement)
                    second = self.make(condition, length=4096, event_placement=placement)
                    self.assertEqual(first, second)
                    self.assertEqual(len(first.token_ids), 4096)
                    self.assertEqual(first.actual_length, first.target_length)
                    self.assertEqual(first.token_ids.count(self.tokenizer.bos_token_id), 1)
                    self.assertEqual(first.token_ids[0], self.tokenizer.bos_token_id)
                    self.assertEqual(first.needle_token, round(0.35 * 4096))
                    self.assertEqual(first.schema_version, 2)
                    self.assertEqual(first.requested_needle_fraction, 0.35)
                    self.assertEqual(first.event_placement, placement)
                    self.assertNotIn("token_ids", first.to_meta())

    def test_collision_changes_keys_but_not_intervening_value_sequence(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                unrelated = self.make("unrelated", seed=seed, num_events=128)
                collision = self.make("collision", seed=seed, num_events=128)
                unrelated_records = records(self.tokenizer, unrelated)
                collision_records = records(self.tokenizer, collision)
                self.assertEqual(unrelated_records[0], collision_records[0])
                self.assertEqual(unrelated_records[0], (unrelated.key, unrelated.answer))
                unrelated_values = [value for _, value in unrelated_records[1:]]
                collision_values = [value for _, value in collision_records[1:]]
                self.assertEqual(unrelated_values, collision_values)
                # Guard against the original missing-label shortcut. A target
                # value must remain possible among irrelevant record values.
                self.assertEqual(set(unrelated_values), set(LETTERS))
                self.assertIn(unrelated.answer, unrelated_values)
                self.assertEqual(len(unrelated_records), 129)
                self.assertEqual(len(collision_records), 129)
                self.assertEqual(unrelated.needle_token, collision.needle_token)
                self.assertEqual(unrelated.pair_id, collision.pair_id)
                self.assertEqual(unrelated.question_suffix, collision.question_suffix)
                self.assertTrue(all(key != collision.key for key, _ in collision_records[1:]))
                self.assertEqual(len({key for key, _ in collision_records}), 129)
                target_prefix = collision.key.rsplit("-", 1)[0]
                self.assertTrue(all(key.rsplit("-", 1)[0] == target_prefix
                                    for key, _ in collision_records[1:]))

    def test_no_target_removes_answer_record_and_keeps_paired_events(self):
        for placement in ("early", "late"):
            for seed in (3, 19):
                with self.subTest(placement=placement, seed=seed):
                    present = self.make("unrelated", seed=seed, event_placement=placement)
                    absent = self.make("no_target", seed=seed, event_placement=placement)
                    present_records = records(self.tokenizer, present)
                    absent_records = records(self.tokenizer, absent)
                    self.assertEqual(absent_records, present_records[1:])
                    self.assertFalse(any(key == absent.key for key, _ in absent_records))
                    self.assertEqual(absent.answer, present.answer)
                    self.assertEqual(absent.pair_id, present.pair_id)
                    self.assertEqual(absent.needle_token, present.needle_token)
                    self.assertEqual(absent.event_token, present.event_token)
                    self.assertEqual(absent.question_suffix, present.question_suffix)
                    absent_text = self.tokenizer.decode(absent.token_ids)
                    self.assertEqual(absent_text.count(absent.key), 1)  # Query only.
                    self.assertEqual(absent.actual_length, present.actual_length)
                    self.assertEqual(absent.token_ids[:absent.needle_token],
                                     present.token_ids[:present.needle_token])
                    self.assertEqual(absent.token_ids[absent.event_token:],
                                     present.token_ids[present.event_token:])

    def test_revision_queries_share_record_history_and_target_location(self):
        for placement in ("early", "late"):
            for seed in (0, 7, 91):
                with self.subTest(placement=placement, seed=seed):
                    latest = self.make("revision_latest", seed=seed, event_placement=placement)
                    earliest = self.make("revision_earliest", seed=seed, event_placement=placement)
                    latest_history = records(self.tokenizer, latest)
                    earliest_history = records(self.tokenizer, earliest)
                    self.assertEqual(latest_history, earliest_history)
                    self.assertEqual(len(latest_history), latest.num_events + 1)
                    self.assertTrue(all(key == latest.key for key, _ in latest_history))
                    self.assertEqual(earliest.answer, earliest_history[0][1])
                    self.assertEqual(latest.answer, latest_history[-1][1])
                    self.assertEqual(latest.initial_answer, earliest.initial_answer)
                    self.assertEqual(latest.latest_answer, earliest.latest_answer)
                    self.assertEqual(latest.needle_token, earliest.needle_token)
                    self.assertEqual(latest.pair_id, earliest.pair_id)
                    self.assertEqual(latest.actual_length, earliest.actual_length)
                    self.assertNotEqual(latest.question_suffix, earliest.question_suffix)

    def test_revision_values_do_not_systematically_exclude_original_status(self):
        for seed in (0, 3, 17, 31):
            with self.subTest(seed=seed):
                probe = self.make("revision_earliest", seed=seed, num_events=128)
                revised_values = [value for _, value in records(self.tokenizer, probe)[1:]]
                self.assertEqual(set(revised_values), set(LETTERS))
                self.assertIn(probe.initial_answer, revised_values)

    def test_no_revision_means_both_temporal_queries_have_initial_answer(self):
        earliest = self.make("revision_earliest", num_events=0)
        latest = self.make("revision_latest", num_events=0)
        self.assertEqual(earliest.answer, latest.answer)
        self.assertIsNone(earliest.event_token)
        self.assertIsNone(latest.event_token)
        self.assertEqual(records(self.tokenizer, earliest), records(self.tokenizer, latest))

    def test_late_events_start_at_a_record_boundary(self):
        for condition in ("unrelated", "collision", "revision_earliest"):
            probe = self.make(condition, event_placement="late", length=4097)
            text = self.tokenizer.decode(probe.token_ids)
            history = records(self.tokenizer, probe)
            first_event = f"Record {history[1][0]}: {history[1][1]}."
            # Search after the target to distinguish repeated-key revisions.
            target = f"Record {history[0][0]}: {history[0][1]}."
            offset = text.index(first_event, text.index(target) + len(target))
            self.assertEqual(text[offset - 1], "\n")

    def test_event_count_cap_rejects_unbounded_finite_pool_requests(self):
        for count in (-1, 8001, 8999, 10 ** 9):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.make("collision", num_events=count, length=1_000_000)

    def test_largest_allowed_collision_load_has_unique_non_target_keys(self):
        probe = self.make("collision", num_events=8000, length=400_000,
                          needle_fraction=0.12, seed=5)
        history = records(self.tokenizer, probe)
        self.assertEqual(len(history), 8001)
        self.assertEqual(len(Counter(key for key, _ in history)), 8001)
        self.assertEqual(probe.actual_length, 400_000)

    def test_too_short_is_rejected_without_removing_support(self):
        for condition in ("unrelated", "collision", "revision_latest", "two_hop"):
            with self.subTest(condition=condition), self.assertRaisesRegex(ValueError, "required text"):
                self.make(condition, length=128, num_events=32)


if __name__ == "__main__":
    unittest.main()
