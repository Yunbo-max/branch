"""Exact model/tokenizer identities and fail-closed resume fingerprints."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

# Public HF repository revisions verified 2026-10-28; same repo supplies tokenizer.
MODEL_PINS = {
    "m-a-p/340M-20B-GatedDeltaNet-pure-baseline": "14661a98768e1f5d5493ff7b18f2d6809717a4c8",
    "m-a-p/1.3B-100B-GatedDeltaNet-pure": "930ed6ae4ac629c86cb9855bb3dcb0a0974a29aa",
    "m-a-p/1.3B-100B-GatedDeltaNet-hybrid-6-1": "b2a4ecf3ee629e8ad5538d4ab18e5068b69b341f",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    hashes = {p: sha256_file(root / p) for p in ('run.py', 'backend.py', 'spec.py', 'provenance.py')}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def runtime_versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ('torch', 'transformers', 'flash-linear-attention', 'fla-core', 'triton', 'huggingface-hub'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def resolve_revision(model_id: str, requested: str | None) -> str:
    path = Path(model_id)
    if path.is_dir():
        # Local paths have no Hub commit. Hash all model/tokenizer bytes rather
        # than trusting filenames or mtimes. This can take time for large weights.
        files = sorted(p for p in path.rglob('*') if p.is_file() and '.cache' not in p.parts and '.git' not in p.parts)
        if not files:
            raise ValueError(f"empty local checkpoint: {path}")
        pairs = [(str(p.relative_to(path)), sha256_file(p)) for p in files]
        return 'local-sha256:' + hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
    if requested and re.fullmatch(r'[0-9a-fA-F]{40}', requested):
        return requested.lower()
    if requested is None and model_id in MODEL_PINS:
        return MODEL_PINS[model_id]
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id, revision=requested or 'main').sha


def resolve_inputs(a) -> dict:
    if a.backend == 'mock':
        return {"model_id": "MOCK-NO-MODEL", "tokenizer_id": "test-only-character-tokenizer-v1",
                "model_revision": "mock-v1", "tokenizer_revision": "mock-v1"}
    model_rev = resolve_revision(a.model, a.revision)
    tok_id = a.tokenizer or a.model
    tok_rev = model_rev if tok_id == a.model and a.tokenizer_revision is None else resolve_revision(tok_id, a.tokenizer_revision)
    return {"model_id": a.model, "model_revision": model_rev,
            "tokenizer_id": tok_id, "tokenizer_revision": tok_rev}


def hub_revision(value: str) -> str | None:
    return None if value.startswith('local-sha256:') else value
