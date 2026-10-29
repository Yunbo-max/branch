"""Deterministic, token-budgeted long-context associative memory probes.

The gold answer never appears in the question. The fact itself does appear in
the context by construction. Input is a list of token IDs (not a decoded string)
so no model/tokenizer can silently truncate a 16K/32K prompt.

This is a diagnostic suite, NOT the official RULER/NoLiMa/BABILong benchmark.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Protocol, Sequence

LETTERS = ("A", "B", "C", "D")
CONDITIONS = (
    "plain", "unrelated", "collision", "revision_latest", "revision_earliest",
    "alias", "two_hop", "no_target",
)
MEMOS = (
    "Field note: the service team completed the routine checklist and filed its report.\n",
    "Inventory memo: maintenance staff verified equipment before starting the shift.\n",
    "Archive note: routine weekly inspections were completed without exceptional findings.\n",
    "Transit bulletin: several teams reviewed documentation and scheduled further meetings.\n",
    "Office log: all standard forms were collected during the afternoon inspection.\n",
    "Process report: the operator confirmed that the calibration workflow was followed.\n",
)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Sequence[int]: ...


@dataclass(frozen=True)
class Probe:
    probe_id: str
    pair_id: str
    condition: str
    target_length: int
    actual_length: int
    needle_fraction: float
    needle_token: int
    event_token: int | None
    num_events: int
    answer: str
    initial_answer: str
    latest_answer: str
    key: str
    seed: int
    question_suffix: str
    token_ids: tuple[int, ...]
    prompt_hash: str
    requested_needle_fraction: float
    event_placement: str
    answerable: bool
    schema_version: int = 2

    def to_meta(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items() if k != "token_ids"
        }


def _enc(tok: TokenizerLike, text: str) -> list[int]:
    xs = list(tok.encode(text, add_special_tokens=False))
    if not xs:
        raise ValueError("tokenizer encoded a nonempty string as zero tokens")
    return xs


def _pad(tok: TokenizerLike, count: int, seed: int) -> list[int]:
    if count < 0:
        raise ValueError("negative filler budget")
    if count == 0:
        return []
    rng = random.Random(seed)
    cached = [_enc(tok, x) for x in MEMOS]
    out: list[int] = []
    while len(out) < count:
        out.extend(cached[rng.randrange(len(cached))])
    return out[:count]


def _prefix(seed: int) -> str:
    rng = random.Random(seed ^ 0x6A31C)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(5))


def _different_letter(letter: str, index: int) -> str:
    return [x for x in LETTERS if x != letter][index % 3]


def make_probe(
    tokenizer: TokenizerLike,
    *,
    condition: str,
    length: int,
    seed: int,
    needle_fraction: float = 0.12,
    num_events: int = 8,
    event_placement: str = "early",
) -> Probe:
    """Construct EXACTLY length input tokens, with deterministic paired conditions.

    All conditions for one (length, seed, position, events) share the SAME
    initial key/value and the SAME prefix filler. The only intervention changes
    later record keys or the query's type. The final suffix filler is adjusted
    to make all prompt token lengths exactly equal, not silently truncated.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if length < 128 or not 0.02 <= needle_fraction <= 0.95:
        raise ValueError("length >= 128 and position fraction in [0.02,0.95] required")
    if not 0 <= num_events <= 8000 or event_placement not in ("early", "late"):
        raise ValueError("invalid event count/placement")

    rng = random.Random(seed)
    # Separate value stream from key generation AND the target answer.
    # Excluding the target value would let a model solve the task by finding
    # the missing label, without remembering the requested record at all.
    value_rng = random.Random(seed ^ 0x714E2B)
    name = _prefix(seed)
    key = f"REC-{name}-{rng.randrange(1000, 9999)}"
    alias = f"ALIAS-{_prefix(seed ^ 0x13AC)}"
    bridge = f"REF-{_prefix(seed ^ 0x4023)}"
    # Contiguous seed blocks of four balance the marginal answer distribution.
    answer = LETTERS[seed % 4]
    latest = answer

    header = (
        "You are reading a database log. A status is exactly one of A, B, C, D.\n"
        "Example:\n"
        "Record DEMO-100: B.\n"
        "Question: What is the status of record DEMO-100?\n"
        "Answer: B\n"
        "Example:\n"
        "Record DEMO-200: D.\n"
        "Question: What is the status of record DEMO-200?\n"
        "Answer: D\n"
        "Database:\n"
    )
    target_text = f"\nRecord {key}: {answer}.\n"
    if condition == "alias":
        target_text += f"Alias {alias} refers to record {key}.\n"
    if condition == "two_hop":
        target_text += f"Reference {bridge} refers to record {key}.\n"
        target_text += f"Alias {alias} refers to reference {bridge}.\n"

    events: list[str] = []
    seen = {key}
    if condition != "plain":
        for j in range(num_events):
            if condition.startswith("revision_"):
                event_key = key
                event_value = LETTERS[value_rng.randrange(4)]
                latest = event_value
            elif condition == "collision":
                # Same 5-letter prefix, different exact identifier.
                while True:
                    event_key = f"REC-{name}-{rng.randrange(1000, 9999)}"
                    if event_key not in seen:
                        break
                seen.add(event_key)
                event_value = LETTERS[value_rng.randrange(4)]
            else:
                while True:
                    event_key = f"REC-{_prefix(seed + 1091 + j)}-{rng.randrange(1000, 9999)}"
                    if event_key not in seen:
                        break
                seen.add(event_key)
                event_value = LETTERS[value_rng.randrange(4)]
            events.append(f"Record {event_key}: {event_value}.\n")

    if condition == "alias":
        question = f"\nQuestion: What is the status of the record named by alias {alias}?\nAnswer:"
    elif condition == "two_hop":
        question = f"\nQuestion: What is the status of the record reached from alias {alias}?\nAnswer:"
    elif condition == "revision_earliest":
        question = f"\nQuestion: What was the EARLIEST recorded status of {key}?\nAnswer:"
    elif condition == "revision_latest":
        question = f"\nQuestion: What was the MOST RECENT recorded status of {key}?\nAnswer:"
    else:
        question = f"\nQuestion: What is the status of record {key}?\nAnswer:"

    gold = answer if condition != "revision_latest" else latest
    h, target, ev, q = (
        list(tokenizer.encode(header, add_special_tokens=True)), _enc(tokenizer, target_text),
        _enc(tokenizer, "\n" + "".join(events)) if events else [],
        _enc(tokenizer, question)
    )
    if condition == "no_target":
        # Match the unrelated condition in every token-budget component but
        # replace the answer-bearing record with neutral tokens. Its hidden
        # gold is used ONLY as an information-leakage/chance control.
        target = _pad(tokenizer, len(target), seed ^ 0xBED5)
    fixed = len(h) + len(target) + len(ev) + len(q)
    available = length - fixed
    if available < 0:
        raise ValueError(
            f"{condition}: requested {length} tokens, but required text is {fixed}; "
            "reduce num_events or use a longer target length"
        )
    # Place the needle according to an absolute token position, with a
    # conservative post-needle slot for all non-filler interventions.
    # Absolute token offset must not depend on query wording. Otherwise alias
    # and revision suffixes move the target and the full matched suite vanishes.
    wanted = round(needle_fraction * length)
    n_before = min(available, max(0, wanted - len(h)))
    n_after = available - n_before
    before = _pad(tokenizer, n_before, seed ^ 0xDAA1)
    if event_placement == "early":
        post_first, post_last = [], _pad(tokenizer, n_after, seed ^ 0xAF34)
    else:
        post_first, post_last = _pad(tokenizer, n_after, seed ^ 0xAF34), []

    needle_idx = len(h) + len(before)
    ev_idx = None
    if event_placement == "early":
        if ev:
            ev_idx = needle_idx + len(target)
        tokens = h + before + target + ev + post_last + q
    else:
        if ev:
            ev_idx = needle_idx + len(target) + len(post_first)
        tokens = h + before + target + post_first + ev + q

    if len(tokens) != length:
        raise AssertionError(f"length budget violated: {len(tokens)} vs {length}")
    pair = json.dumps(
        [length, seed, needle_fraction, num_events, event_placement],
        separators=(",", ":")
    )
    pair_id = hashlib.sha256(pair.encode()).hexdigest()[:16]
    probe_id = f"{condition}-{pair_id}"
    prompt_hash = hashlib.sha256(
        ",".join(str(t) for t in tokens).encode()
    ).hexdigest()
    return Probe(
        probe_id=probe_id, pair_id=pair_id, condition=condition,
        target_length=length, actual_length=len(tokens),
        needle_fraction=needle_idx / length, needle_token=needle_idx,
        event_token=ev_idx, num_events=num_events,
        answer=gold, initial_answer=answer, latest_answer=latest,
        key=key, seed=seed, question_suffix=question,
        token_ids=tuple(tokens), prompt_hash=prompt_hash,
        requested_needle_fraction=needle_fraction,
        event_placement=event_placement,
        answerable=condition != "no_target",
    )


def suite_size(
    lengths: list[int], seeds: list[int], positions: list[float],
    event_counts: list[int], conditions: list[str],
) -> int:
    return (
        len(lengths) * len(seeds) * len(positions) *
        len(event_counts) * len(conditions)
    )

