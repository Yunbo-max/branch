#!/usr/bin/env python3
"""Merge compatible, nonoverlapping real-model census shards without rescoring.

Inputs must use schema v2.  The merge preserves error rows, exact case metadata,
and source file hashes.  It does not turn an interrupted run into a completed
experiment, combine different checkpoints, or count a retried probe twice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


SCHEMA = "gdn-long-context-census.v2"
SHARD_CONFIG_FIELDS = frozenset({"seed_start", "seeds", "device"})
# Compare package/runtime versions and Python exactly, but allow the same
# software stack on another physical GPU or host. CUDA/driver versions remain.
RUNTIME_HOST_FIELDS = frozenset({
    "platform", "hostname", "machine", "node", "processor", "device", "devices",
    "gpu_name", "gpu_names", "gpu_count", "gpu_uuid", "gpu_uuids",
    "gpu_memory_bytes", "gpu_total_memory_bytes", "gpu_device_index",
})
EXECUTION_STATES = frozenset({"completed", "partial", "failed", "running"})


class MergeError(ValueError):
    """A shard is invalid or cannot be compared with the other shards."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MergeError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise MergeError(f"nonfinite JSON number: {value}")


def _parse_json(text: str, label: str) -> dict[str, Any]:
    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    except (ValueError, TypeError) as exc:
        raise MergeError(f"invalid JSON at {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise MergeError(f"expected a JSON object at {label}")
    return result


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    rows.append(_parse_json(line, f"{path}:{line_number}"))
    except (OSError, UnicodeError) as exc:
        raise MergeError(f"cannot read {path}: {exc}") from exc
    return rows


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MergeError(f"{label} must be a nonempty string")
    return value


def _runtime_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _runtime_identity(item) for key, item in value.items()
                if key not in RUNTIME_HOST_FIELDS}
    if isinstance(value, list):
        return [_runtime_identity(item) for item in value]
    return value


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": {key: value for key, value in manifest["config"].items()
                   if key not in SHARD_CONFIG_FIELDS},
        "resolved": manifest["resolved"],
        "code_fingerprint": manifest["code_fingerprint"],
        "runtime": _runtime_identity(manifest["runtime"]),
        "backend": manifest["backend"],
        "loaded_model": {key: value for key, value in manifest["loaded_model"].items()
                         if key != "device"},
    }


def validate_shard(directory: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one shard, returning its manifest, cases, and attempted results."""
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    try:
        manifest = _parse_json(manifest_path.read_text(encoding="utf-8"), str(manifest_path))
    except (OSError, UnicodeError) as exc:
        raise MergeError(f"cannot read {manifest_path}: {exc}") from exc
    if manifest.get("schema") != SCHEMA:
        raise MergeError(f"{directory}: only {SCHEMA} is accepted")
    if manifest.get("dry_run") is not False:
        raise MergeError(f"{directory}: real evaluation requires dry_run=false")
    backend = _nonempty_text(manifest.get("backend"), f"{directory}: backend")
    if any(word in backend.lower() for word in ("mock", "oracle", "dummy", "dry_run", "dry-run")):
        raise MergeError(f"{directory}: mock/oracle/dry-run results cannot be merged as model evidence")
    for name in ("config", "resolved", "runtime"):
        if not isinstance(manifest.get(name), dict) or not manifest[name]:
            raise MergeError(f"{directory}: missing nonempty {name}")
    for name in ("model_revision", "tokenizer_revision", "model_id", "tokenizer_id"):
        _nonempty_text(manifest["resolved"].get(name), f"{directory}: resolved.{name}")
    if not manifest.get("code_fingerprint"):
        raise MergeError(f"{directory}: missing code_fingerprint")
    info = manifest.get("loaded_model")
    if not isinstance(info, dict) or not info:
        raise MergeError(f"{directory}: real evaluation requires loaded_model metadata")
    for key, resolved_key in (("model_id", "model_id"), ("tokenizer_id", "tokenizer_id"),
                              ("revision", "model_revision"), ("tokenizer_revision", "tokenizer_revision")):
        wanted = manifest["resolved"][resolved_key]
        if key.endswith("revision") and wanted.startswith("local-sha256:"):
            wanted = None
        if key not in info or info[key] != wanted:
            raise MergeError(f"{directory}: loaded_model.{key} disagrees with resolved checkpoint identity")
    model_backend = _nonempty_text(info.get("backend"), f"{directory}: loaded_model.backend")
    if any(word in model_backend.lower() for word in ("mock", "oracle", "dummy")):
        raise MergeError(f"{directory}: loaded_model cannot be a mock/oracle")
    if info.get("dtype") != manifest["config"].get("dtype"):
        raise MergeError(f"{directory}: loaded_model.dtype disagrees with config")
    if "python" not in manifest["runtime"]:
        raise MergeError(f"{directory}: runtime.python is required")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("status") not in EXECUTION_STATES:
        raise MergeError(f"{directory}: invalid execution.status")
    planned = manifest.get("planned_cases")
    if isinstance(planned, bool) or not isinstance(planned, int) or planned < 1:
        raise MergeError(f"{directory}: planned_cases must be a positive integer")

    cases = _json_lines(directory / "cases.jsonl")
    if len(cases) != planned:
        raise MergeError(f"{directory}: planned_cases={planned} but cases.jsonl has {len(cases)} rows")
    case_by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        pid = _nonempty_text(case.get("probe_id"), f"{directory}: case probe_id")
        _nonempty_text(case.get("prompt_hash"), f"{directory}: {pid} prompt_hash")
        if pid in case_by_id:
            raise MergeError(f"{directory}: duplicate case probe_id {pid}")
        if "status" in case:
            raise MergeError(f"{directory}: case metadata must not contain result status")
        case_by_id[pid] = case

    result_path = directory / "results.jsonl"
    results = _json_lines(result_path) if result_path.exists() else []
    seen: set[str] = set()
    for result in results:
        pid = _nonempty_text(result.get("probe_id"), f"{directory}: result probe_id")
        if pid in seen:
            raise MergeError(f"{directory}: duplicate/overlapping result probe_id {pid}, including mixed error/ok retries")
        seen.add(pid)
        if pid not in case_by_id:
            raise MergeError(f"{directory}: result {pid} was not in cases.jsonl")
        if result.get("status") not in ("ok", "error"):
            raise MergeError(f"{directory}: result {pid} must have status ok or error")
        if result.get("evidence_type") != "pretrained_lm":
            raise MergeError(f"{directory}: result {pid} is not pretrained_lm evidence")
        for key, value in case_by_id[pid].items():
            if key not in result or result[key] != value:
                raise MergeError(f"{directory}: result {pid} disagrees with case metadata field {key}")
        if result["status"] == "ok":
            if type(result.get("correct")) is not int or result["correct"] not in (0, 1):
                raise MergeError(f"{directory}: ok result {pid} needs integer correct=0 or 1")
            if result.get("error") is not None or result.get("error_type") is not None:
                raise MergeError(f"{directory}: result {pid} mixes ok and error fields")
        else:
            if result.get("correct") is not None or result.get("prediction") is not None:
                raise MergeError(f"{directory}: error result {pid} mixes failure and scored outcome")
            _nonempty_text(result.get("error_type"), f"{directory}: error result {pid} error_type")
    if execution["status"] == "completed" and len(results) != planned:
        raise MergeError(f"{directory}: completed status contradicts missing result rows")
    return manifest, cases, results


def _write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def merge_runs(inputs: list[Path | str], output: Path | str) -> dict[str, Any]:
    """Validate then combine shards. Refuse all existing output destinations."""
    if not inputs:
        raise MergeError("at least one input shard is required")
    paths = [Path(path).resolve() for path in inputs]
    if len(set(paths)) != len(paths):
        raise MergeError("the same input directory was supplied more than once")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise MergeError(f"output already exists: {output}")
    manifests: list[dict[str, Any]] = []
    all_cases: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    seen: set[str] = set()
    identity: dict[str, Any] | None = None
    for path in paths:
        # Detect a concurrent writer or modification while the files are read.
        file_paths = [path / name for name in ("manifest.json", "cases.jsonl", "results.jsonl")]
        before = {p.name: _sha256(p) if p.exists() else None for p in file_paths}
        manifest, cases, results = validate_shard(path)
        after = {p.name: _sha256(p) if p.exists() else None for p in file_paths}
        if before != after:
            raise MergeError(f"{path}: source changed during validation; stop its writer and retry")
        current = _identity(manifest)
        if identity is None:
            identity = current
        elif current != identity:
            differing = [name for name in identity if identity[name] != current[name]]
            raise MergeError(f"{path}: incompatible shard fields: {', '.join(differing)}")
        overlaps = seen.intersection(case["probe_id"] for case in cases)
        if overlaps:
            raise MergeError(f"{path}: overlapping planned probe_id: {sorted(overlaps)[0]}")
        seen.update(case["probe_id"] for case in cases)
        manifests.append(manifest)
        all_cases.extend(cases)
        all_results.extend(results)
        source_receipts.append({
            "directory": str(path), "sha256": after,
            "planned_cases": len(cases), "attempted_cases": len(results),
            "execution": manifest["execution"],
            "seed_start": manifest["config"].get("seed_start"),
            "seeds": manifest["config"].get("seeds"),
            "device": manifest["config"].get("device"),
            "runtime": manifest["runtime"],
        })
    assert identity is not None
    counts = Counter(result["status"] for result in all_results)
    complete = all(m["execution"]["status"] == "completed" for m in manifests)
    config = dict(identity["config"])
    # Do not invent a contiguous seed interval for disjoint shards. The source
    # intervals and exact cases are authoritative; merged outputs are not runs
    # that can be resumed by run.py.
    actual_seeds = {case["seed"] for case in all_cases if isinstance(case.get("seed"), int)}
    config.update(seed_start=None, seeds=len(actual_seeds) if actual_seeds else None, device="merged")
    merged = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "resolved": identity["resolved"],
        "code_fingerprint": identity["code_fingerprint"], "runtime": identity["runtime"],
        "backend": identity["backend"], "dry_run": False,
        "loaded_model": {**identity["loaded_model"], "device": "merged"},
        "planned_cases": len(all_cases),
        "execution": {
            "status": "completed" if complete else "partial",
            "attempted_cases": len(all_results), "ok_cases": counts["ok"],
            "error_cases": counts["error"], "missing_cases": len(all_cases)-len(all_results),
        },
        "merge": {
            "version": 1, "source_count": len(paths), "sources": source_receipts,
            "partial_data": not complete, "resumable": False,
            "policy": "No overlapping planned probes; identical resolved checkpoints, code, software runtime, and experiment configuration except seed range and device. Errors are preserved.",
        },
    }
    # Exclusive mkdir is the no-overwrite guard, including races after checking.
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise MergeError(f"output already exists: {output}") from exc
    try:
        _write_json_lines(output / "cases.jsonl", all_cases)
        _write_json_lines(output / "results.jsonl", all_results)
        merged["output_sha256"] = {name: _sha256(output/name) for name in ("cases.jsonl", "results.jsonl")}
        with (output / "manifest.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(merged, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = merge_runs(args.inputs, args.out)
    except (MergeError, OSError) as exc:
        parser.exit(2, f"merge refused: {exc}\n")
    print(json.dumps({"out": str(args.out), **manifest["execution"],
                      "partial_data": manifest["merge"]["partial_data"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
