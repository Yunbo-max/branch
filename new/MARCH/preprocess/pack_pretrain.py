#!/usr/bin/env python3
"""Tokenize and pack the pretraining corpus into fixed-length uint16 sequences.

Input : the Long-Data-Collections mirror shards (parquet, one `text` column).
Output: ``<out>/shard-XXXXX.npy`` arrays of shape (n_sequences, seq_len), uint16,
        plus ``<out>/manifest.json`` recording the tokenizer, seq_len, per-file token
        counts and the corpus fingerprint.

Documents are concatenated with an EOS separator, then sliced into contiguous
``seq_len`` rows — the standard packing used for pretraining, so no token is dropped
except the final partial row of each input shard.

Resumable: an input shard whose output file already exists is skipped.

    python pack_pretrain.py --out /path/to/packed_16k --seq-len 16384 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

_STATE: dict = {}


def _worker_init(tokenizer_path: str, out_dir: str, seq_len: int, eos_id: int) -> None:
    from transformers import AutoTokenizer

    _STATE["tok"] = AutoTokenizer.from_pretrained(tokenizer_path)
    _STATE["out"] = out_dir
    _STATE["seq_len"] = seq_len
    _STATE["eos"] = eos_id


def _pack_one(src: str) -> dict:
    tok = _STATE["tok"]
    seq_len = _STATE["seq_len"]
    out = _STATE["out"]
    stem = os.path.basename(src).split("-of-")[0].replace("train-", "shard-")
    dst = os.path.join(out, f"{stem}.npy")
    if os.path.exists(dst):
        arr = np.load(dst, mmap_mode="r")
        return {"src": src, "dst": dst, "n_seq": int(arr.shape[0]),
                "n_tokens": int(arr.shape[0] * seq_len), "skipped": True}

    t0 = time.perf_counter()
    texts = pq.read_table(src).column("text").to_pylist()
    # numpy arrays, not Python int lists: a 273 MB shard is ~187 M tokens and a list of
    # Python ints for that costs ~5 GB per worker and stalls on GPFS.
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), 1000):
        batch = [t or "" for t in texts[i:i + 1000]]
        ids = tok(batch, add_special_tokens=False)["input_ids"]
        for e in ids:
            a = np.asarray(e, dtype=np.int32)
            parts.append(np.concatenate([a, np.array([_STATE["eos"]], dtype=np.int32)]))
    del texts
    flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    del parts
    n_seq = len(flat) // seq_len
    if n_seq == 0:
        return {"src": src, "dst": None, "n_seq": 0, "n_tokens": 0, "skipped": False}
    packed = flat[: n_seq * seq_len].reshape(n_seq, seq_len).astype(np.uint16)
    np.save(dst, packed)
    return {"src": src, "dst": dst, "n_seq": int(n_seq),
            "n_tokens": int(packed.size), "seconds": time.perf_counter() - t0,
            "skipped": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="data/pretrain/data/*.parquet")
    ap.add_argument("--tokenizer", default="data/tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=float, default=0.0,
                    help="stop after roughly this many tokens (0 = all)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    vocab = len(tok)
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} does not fit uint16")

    sources = sorted(glob.glob(a.shards))
    if not sources:
        raise SystemExit(f"no shards match {a.shards}")
    print(f"input shards: {len(sources)} | seq_len={a.seq_len} | vocab={vocab} | eos={eos}",
          flush=True)

    started = time.perf_counter()
    records, total = [], 0
    with Pool(a.workers, initializer=_worker_init, maxtasksperchild=4,
              initargs=(a.tokenizer, a.out, a.seq_len, eos)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_pack_one, sources), 1):
            records.append(rec)
            total += rec["n_tokens"]
            if i % 20 == 0 or i == len(sources):
                rate = total / max(time.perf_counter() - started, 1e-9)
                print(f"[{i}/{len(sources)}] {total/1e9:.2f} B tokens | "
                      f"{rate/1e6:.1f} M tok/s | {time.perf_counter()-started:.0f}s", flush=True)
            if a.max_tokens and total >= a.max_tokens:
                print("reached --max-tokens; stopping", flush=True)
                break

    manifest = {
        "seq_len": a.seq_len,
        "tokenizer": a.tokenizer,
        "vocab_size": vocab,
        "eos_token_id": eos,
        "dtype": "uint16",
        "shards": sorted((r for r in records if r["dst"]), key=lambda r: r["dst"]),
        "total_sequences": sum(r["n_seq"] for r in records),
        "total_tokens": total,
        "wall_seconds": time.perf_counter() - started,
        "note": "documents joined with EOS then sliced; final partial row of each input "
                "shard is dropped. Packed at the paper's 16K context; shorter training "
                "contexts can re-slice these rows.",
    }
    path = os.path.join(a.out, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\ntotal {total/1e9:.2f} B tokens in {manifest['total_sequences']:,} sequences")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
