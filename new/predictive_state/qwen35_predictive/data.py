"""Question-only data with deterministic, disjoint prompt identities."""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(file)
    records, seen = [], set()
    with file.open(encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            item = json.loads(raw)
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str) \
                    or not item["prompt"].strip():
                raise ValueError("one nonempty prompt string per JSONL record required")
            key = str(item.get("id", len(records)))
            if key in seen:
                raise ValueError(f"duplicate prompt id: {key}")
            seen.add(key)
            # Gold answers and task metadata NEVER enter teacher/student contexts.
            records.append({"id": key, "prompt": item["prompt"]})
    if not records:
        raise ValueError("empty question file")
    return records


def shuffled_questions(records: list[dict], seed: int) -> list[dict]:
    copy = list(records)
    random.Random(seed).shuffle(copy)
    return copy


def source_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_seed(base: int, record_id: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{base}|{record_id}|{index}".encode()).digest()[:8], "big"
    ) % (2**63 - 1)
