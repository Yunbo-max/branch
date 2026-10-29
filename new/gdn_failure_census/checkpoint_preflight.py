#!/usr/bin/env python3
"""CPU-only local-checkpoint preflight for the GDN failure census.

Reads config/tokenizer metadata and safetensors HEADERS, not weight tensors.
It does NOT prove a model can execute under the installed CUDA/FLA runtime.
Known FLAME-fork mismatch: model.layers.*.attn.D is not part of stock
FLA 0.5.0's GatedDeltaNet layer and caused unexpected_keys on m-a-p models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

MAX_SAFETENSORS_HEADER = 128 * 1024 * 1024
LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
INCOMPATIBLE_D_PATTERN = re.compile(r"(?:^|\.)attn\.D$")


def safetensors_key_names(path: Path) -> set[str]:
    """Scan the safetensors header with Python stdlib; never allocate weights."""
    with path.open("rb") as fh:
        length_bytes = fh.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path}: truncated safetensors header length")
        (length,) = struct.unpack("<Q", length_bytes)
        if length < 2 or length > MAX_SAFETENSORS_HEADER:
            raise ValueError(f"{path}: implausible safetensors header size {length}")
        header_bytes = fh.read(length)
        if len(header_bytes) != length:
            raise ValueError(f"{path}: truncated safetensors JSON header")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    keys = set(header) - {"__metadata__"}
    if not keys:
        raise ValueError(f"{path}: no weight tensors in safetensors header")
    return keys


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_local_checkpoint(
    model_dir: Path, *,
    expected_weight_sha256: str | None = None,
    check_tokenizer: bool = False,
) -> dict:
    root = model_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"not a local model directory: {root}")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError(f"missing config.json under {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("model config must be JSON object")

    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no .safetensors weight files under {root}")
    all_keys: set[str] = set()
    for path in files:
        keys = safetensors_key_names(path)
        duplicates = all_keys.intersection(keys)
        if duplicates:
            raise ValueError(f"duplicate tensor names across shards: {sorted(duplicates)[:5]}")
        all_keys.update(keys)

    issues: list[str] = []
    warnings: list[str] = []
    model_type = config.get("model_type")
    if model_type != "gated_deltanet":
        issues.append(f"expected model_type=gated_deltanet; found {model_type!r}")
    if config.get("architectures") and "GatedDeltaNetForCausalLM" not in config["architectures"]:
        warnings.append(f"architecture not standard GatedDeltaNetForCausalLM: {config['architectures']}")

    unexpected_d = sorted(k for k in all_keys if INCOMPATIBLE_D_PATTERN.search(k))
    if unexpected_d:
        issues.append(
            f"found {len(unexpected_d)} attn.D tensors; these caused unexpected_keys "
            "with stock FLA 0.5.0 and indicate a FLAME-fork architecture. "
            "Do NOT silently ignore, drop or randomly initialize weights."
        )
    optional_fork_fields = sorted(set(config).intersection({"qk_activation", "qk_norm", "use_beta"}))
    if optional_fork_fields:
        warnings.append(
            "checkpoint has FLAME-specific extra config fields "
            f"{optional_fork_fields}; verify exact implementation compatibility"
        )

    layers = sorted({int(m.group(1)) for k in all_keys if (m := LAYER_PATTERN.search(k))})
    configured_layers = config.get("num_hidden_layers")
    if isinstance(configured_layers, int) and layers != list(range(configured_layers)):
        issues.append(
            f"weight tensors cover {len(layers)} layer indices rather than config "
            f"num_hidden_layers={configured_layers}"
        )

    tokenizer_files = [x.name for x in (
        root / "tokenizer.json", root / "tokenizer.model",
        root / "vocab.json", root / "spiece.model"
    ) if x.is_file()]
    if not tokenizer_files:
        issues.append("missing tokenizer.json / tokenizer.model / vocab.json / spiece.model")

    weight_sha = None
    if expected_weight_sha256:
        expected = expected_weight_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("expected weight SHA-256 must be 64 hexadecimal characters")
        if len(files) != 1:
            raise ValueError("expected-weight-sha256 applies only to one-file safetensors checkpoints")
        weight_sha = sha256(files[0])
        if weight_sha != expected:
            issues.append(f"model weight SHA-256 mismatch: got {weight_sha}, expected {expected}")

    choices = None
    if check_tokenizer and not issues:
        try:
            from transformers import AutoTokenizer
            from backend import single_token_choices
            tok = AutoTokenizer.from_pretrained(
                str(root), local_files_only=True, trust_remote_code=False
            )
            suffix = "\nQuestion: What is the status of record REC-TEST-1234?\nAnswer:"
            choices = single_token_choices(tok, suffix)
        except (ImportError, OSError, ValueError) as exc:
            issues.append("local tokenizer or single-token A/B/C/D scoring failed: "+str(exc))

    result = {
        "schema": "gdn-checkpoint-preflight.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(root),
        "status": "blocked" if issues else "preflight_passed_not_loaded",
        "model_type": model_type,
        "architecture": config.get("architectures"),
        "num_hidden_layers": configured_layers,
        "layers_found": len(layers),
        "hidden_size": config.get("hidden_size"),
        "num_heads": config.get("num_heads"),
        "head_dim": config.get("head_dim"),
        "configured_context": config.get("max_position_embeddings"),
        "weight_shard_count": len(files),
        "weight_tensor_count": len(all_keys),
        "weight_bytes": sum(p.stat().st_size for p in files),
        "weight_sha256": weight_sha,
        "tokenizer_files": tokenizer_files,
        "single_next_token_labels": choices,
        "issues": issues,
        "warnings": warnings,
        "limitations": [
            "CPU/header-only check. This is NOT a successful FLA forward.",
            "Cannot prove tensor shapes or ABI match until strict HF loading.",
            "Does not establish GDN long-context quality or speed."
        ],
    }
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--expected-weight-sha256")
    p.add_argument("--check-tokenizer", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    r = inspect_local_checkpoint(
        a.model_dir,
        expected_weight_sha256=a.expected_weight_sha256,
        check_tokenizer=a.check_tokenizer,
    )
    serialized = json.dumps(r, indent=2, ensure_ascii=False) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite existing preflight evidence without user action.
        with a.out.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized)
    return 0 if not r["issues"] else 2


if __name__ == "__main__":
    sys.exit(main())
