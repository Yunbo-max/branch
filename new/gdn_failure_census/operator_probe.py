"""Controlled Gated DeltaNet operator probes, not trained-model evaluations.

The state convention here is ``[key_dim, value_dim]`` (the transpose of the
paper's convention). Equation (8) of https://arxiv.org/html/2412.06464v1 is
implemented as::

    decayed = alpha * state
    state = decayed + beta * outer(key, value - key @ decayed)
    output = query @ state

Keys in this module are unit length. Gates are externally chosen scalar
controls, not predictions from a pretrained model. Alpha=1 and beta=0/1 are
useful boundary controls for the original open-interval gate parameterization.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Any

import numpy as np


SOURCE_URL = "https://arxiv.org/html/2412.06464v1"
_DTYPES = {"float32": np.dtype("float32"), "float64": np.dtype("float64")}


def _gate(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar in [0, 1]") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite scalar in [0, 1]")
    return result


def _state_dtype(array: np.ndarray) -> np.dtype:
    if array.dtype not in _DTYPES.values():
        raise ValueError("state/dtype must be float32 or float64")
    return array.dtype


def _step(state: np.ndarray, key: np.ndarray, value: np.ndarray,
          alpha: float, beta: float) -> np.ndarray:
    """Unchecked internal recurrence; caller validates shapes and values once."""
    a = np.asarray(alpha, dtype=state.dtype)
    b = np.asarray(beta, dtype=state.dtype)
    decayed = a * state
    return decayed + b * np.outer(key, value - key @ decayed)


def update_state(state: Any, key: Any, value: Any,
                 alpha: float, beta: float) -> np.ndarray:
    """Return one gated-delta update without mutating any input.

    ``state`` must be a finite float32/float64 matrix; key/value must be
    one-dimensional with matching dimensions. A non-unit key is rejected:
    the geometric controls and stability claims in this suite assume unit keys.
    """
    state = np.asarray(state)
    dtype = _state_dtype(state)
    if state.ndim != 2 or min(state.shape) < 1:
        raise ValueError("state must be a nonempty [key_dim, value_dim] matrix")
    key = np.asarray(key, dtype=dtype)
    value = np.asarray(value, dtype=dtype)
    if key.shape != (state.shape[0],) or value.shape != (state.shape[1],):
        raise ValueError("key/value dimensions must match state")
    if not all(np.isfinite(x).all() for x in (state, key, value)):
        raise ValueError("state, key and value must be finite")
    if not np.isclose(np.linalg.norm(key), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("key must have unit L2 norm")
    return _step(state, key, value, _gate(alpha, "alpha"), _gate(beta, "beta"))


def _gate_series(gate: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(gate)
    if array.dtype.kind == "b":
        raise ValueError(f"{name} cannot be boolean")
    if array.ndim == 0:
        return np.full(length, _gate(array.item(), name), dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must be scalar or shape [sequence_length]")
    return np.asarray([_gate(item, name) for item in array], dtype=np.float64)


def reference_sequence(keys: Any, values: Any, alpha: Any = 1.0,
                       beta: Any = 1.0, queries: Any = None,
                       initial_state: Any = None, dtype: Any = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an unbatched sequence; return (outputs, final_state).

    Keys/queries are [T, key_dim], values/outputs are [T, value_dim]. Reads
    happen *after* each corresponding update. Omitted queries equal keys.
    Gates can be scalar or length-T vectors. The default arithmetic is float64,
    or the initial state's dtype when an initial state is supplied.
    """
    if dtype is None:
        dtype = np.asarray(initial_state).dtype if initial_state is not None else np.float64
    dtype = np.dtype(dtype)
    _state_dtype(np.empty(0, dtype=dtype))
    keys = np.asarray(keys, dtype=dtype)
    values = np.asarray(values, dtype=dtype)
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must be [T, D] arrays with equal T")
    if keys.shape[1] < 1 or values.shape[1] < 1:
        raise ValueError("key and value dimensions must be positive")
    queries = keys if queries is None else np.asarray(queries, dtype=dtype)
    if queries.shape != keys.shape:
        raise ValueError("queries must have the same shape as keys")
    if not all(np.isfinite(x).all() for x in (keys, values, queries)):
        raise ValueError("keys, values and queries must be finite")
    if not np.allclose(np.linalg.norm(keys, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("every key must have unit L2 norm")
    shape = (keys.shape[1], values.shape[1])
    state = np.zeros(shape, dtype=dtype) if initial_state is None else np.array(
        initial_state, dtype=dtype, copy=True)
    if state.shape != shape or not np.isfinite(state).all():
        raise ValueError("initial_state has incompatible shape or nonfinite entries")
    alphas = _gate_series(alpha, len(keys), "alpha")
    betas = _gate_series(beta, len(keys), "beta")
    outputs = np.empty((len(keys), values.shape[1]), dtype=dtype)
    for i, (key, value, query) in enumerate(zip(keys, values, queries)):
        state = _step(state, key, value, alphas[i], betas[i])
        outputs[i] = query @ state
    return outputs, state


def _axes(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "dimensions": [16, 64], "value_dim": 16,
        "alphas": [1.0, 0.999, 0.99], "betas": [0.1, 1.0],
        "gaps": [0, 32, 256], "correlations": [0.0, 0.25, 0.5, 0.9, 1.0],
        "loads": [1, 4, 16, 64, 128], "seeds": [0, 1, 2],
        "dtypes": ["float64"],
    }
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown operator config fields: {sorted(unknown)}")
    result = {**defaults, **config}
    for name, minimum in (("dimensions", 2), ("gaps", 0), ("loads", 1), ("seeds", 0)):
        values = result[name]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < minimum
               for x in values):
            raise ValueError(f"{name} must contain integers >= {minimum}")
        result[name] = sorted(set(int(x) for x in values))
    value_dim = result["value_dim"]
    if isinstance(value_dim, bool) or not isinstance(value_dim, (int, np.integer)) or value_dim < 2:
        raise ValueError("value_dim must be an integer >= 2")
    result["value_dim"] = int(value_dim)
    for name in ("alphas", "betas", "correlations"):
        if not isinstance(result[name], (list, tuple)) or not result[name]:
            raise ValueError(f"{name} must be a nonempty list")
        result[name] = list(dict.fromkeys(_gate(x, name) for x in result[name]))
    dtypes = result["dtypes"]
    if not isinstance(dtypes, (list, tuple)) or not dtypes or any(x not in _DTYPES for x in dtypes):
        raise ValueError("dtypes must be a nonempty list of float32/float64")
    result["dtypes"] = list(dict.fromkeys(dtypes))
    return result


def _metrics(prediction: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    """NRMSE = ||prediction-target||_2 / ||target||_2; cosine is null at zero."""
    prediction = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    expected_norm = float(np.linalg.norm(expected))
    prediction_norm = float(np.linalg.norm(prediction))
    if expected_norm == 0:
        raise ValueError("metric target must be nonzero")
    cosine = float(prediction @ expected / (prediction_norm * expected_norm)) if prediction_norm else None
    return {"nrmse": float(np.linalg.norm(prediction - expected) / expected_norm),
            "cosine": None if cosine is None else float(np.clip(cosine, -1, 1)),
            "prediction_norm": prediction_norm}


def _interference_rows(probe: str, dimension: int, value_dim: int, alpha: float,
                       beta: float, rho: float, gaps: list[int], seed: int,
                       dtype_name: str) -> list[dict[str, Any]]:
    dtype = _DTYPES[dtype_name]
    target_key = np.eye(1, dimension, dtype=dtype)[0]
    target_value = np.eye(1, value_dim, dtype=dtype)[0]
    distractor_key = np.zeros(dimension, dtype=dtype)
    distractor_key[0] = rho
    distractor_key[1] = math.sqrt(max(0.0, 1.0 - rho * rho))
    distractor_value = np.zeros(value_dim, dtype=dtype)
    distractor_value[1] = 1.0
    state = np.outer(target_key, target_value)
    rows = []
    elapsed = 0
    for gap in gaps:
        while elapsed < gap:
            state = _step(state, distractor_key, distractor_value, alpha, beta)
            elapsed += 1
        prediction = target_key @ state
        # Repeated use of the SAME distractor direction. This expression is not
        # valid for changing independent distractor directions.
        analytic_retention = alpha ** gap * (1 - rho ** 2 + rho ** 2 * (1 - beta) ** gap)
        transition = alpha * (1 - beta)
        geometric_sum = float(gap) if transition == 1.0 else (1 - transition ** gap) / (1 - transition)
        analytic_interference = rho * beta * geometric_sum
        analytic = analytic_retention * target_value.astype(np.float64)
        analytic += analytic_interference * distractor_value.astype(np.float64)
        rows.append({
            "probe": probe, "dimension": dimension, "value_dim": value_dim,
            "alpha": alpha, "beta": beta, "correlation": rho,
            "gap": gap, "load": 1, "seed": seed, "dtype": dtype_name,
            "gate_source": "fixed_control", "initialization": "exact_preloaded_target",
            "retention_amplitude": float(prediction @ target_value),
            "interference_amplitude": float(prediction @ distractor_value),
            "analytic_retention": analytic_retention,
            "analytic_interference": analytic_interference,
            "max_abs_analytic_error": float(np.max(np.abs(prediction.astype(np.float64) - analytic))),
            **_metrics(prediction, target_value),
        })
    return rows


def _capacity_rows(dimension: int, value_dim: int, alpha: float, beta: float,
                   load: int, gaps: list[int], seed: int, dtype_name: str
                   ) -> list[dict[str, Any]]:
    dtype = _DTYPES[dtype_name]
    # The same keys/values are reused across alpha/beta/dtype/gap conditions.
    rng = np.random.default_rng(np.random.SeedSequence([seed, dimension, load, value_dim]))
    keys = rng.standard_normal((load, dimension))
    values = rng.standard_normal((load, value_dim))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    keys, values = keys.astype(dtype), values.astype(dtype)
    _, state = reference_sequence(keys, values, alpha, beta, dtype=dtype)
    rows = []
    elapsed = 0
    for gap in gaps:
        # Delay-only continuation: no new associations, so this isolates decay
        # from interference. Report the explicitly zero continuation write gate.
        while elapsed < gap:
            state = np.asarray(alpha, dtype=dtype) * state
            elapsed += 1
        predictions = (keys @ state).astype(np.float64)
        targets = values.astype(np.float64)
        norms = np.linalg.norm(predictions, axis=1)
        target_norms = np.linalg.norm(targets, axis=1)
        nonzero = norms > 0.0
        normalized_predictions = np.zeros_like(predictions)
        normalized_predictions[nonzero] = predictions[nonzero] / norms[nonzero, None]
        similarities = normalized_predictions @ (targets / target_norms[:, None]).T
        choices = np.argmax(similarities, axis=1)
        correct = (choices == np.arange(load)) & nonzero
        cosines = np.sum(predictions * targets, axis=1) / np.where(nonzero, norms * target_norms, 1.0)
        nrmse = np.linalg.norm(predictions - targets, axis=1) / target_norms
        quartile_size = max(1, math.ceil(load / 4))
        rows.append({
            "probe": "capacity", "dimension": dimension, "value_dim": value_dim,
            "alpha": alpha, "beta": beta, "gap": gap, "load": load,
            "seed": seed, "dtype": dtype_name, "gate_source": "fixed_control",
            "initialization": "zero_state", "delay_beta": 0.0,
            "classification_accuracy": float(np.mean(correct)),
            "uniform_guess_accuracy": 1.0 / load,
            "candidate_count": load,
            "zero_prediction_fraction": float(np.mean(~nonzero)),
            "mean_nrmse": float(np.mean(nrmse)),
            "mean_cosine_nonzero": float(np.mean(cosines[nonzero])) if np.any(nonzero) else None,
            "oldest_quartile_accuracy": float(np.mean(correct[:quartile_size])),
            "newest_quartile_accuracy": float(np.mean(correct[-quartile_size:])),
            "oldest_quartile_nrmse": float(np.mean(nrmse[:quartile_size])),
            "newest_quartile_nrmse": float(np.mean(nrmse[-quartile_size:])),
            "mean_prediction_norm": float(np.mean(norms)),
        })
    return rows


def run_probe(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a deterministic, CPU-only operator grid and return a JSON report.

    Accepted fields: dimensions, value_dim, alphas, betas, gaps, correlations
    (cosines 0..1), loads, seeds, dtypes. See ``parameters`` in the returned
    report for resolved defaults. Seeds change the capacity draws; analytic
    interference controls have no random components and are emitted once per
    dimension/gate/dtype condition, with seed=0.
    """
    axes = _axes({} if config is None else config)
    rows: list[dict[str, Any]] = []
    for dtype_name, dimension, alpha in product(axes["dtypes"], axes["dimensions"], axes["alphas"]):
        rows.extend(_interference_rows("erase_only", dimension, axes["value_dim"], alpha,
                                       0.0, 0.0, axes["gaps"], 0, dtype_name))
        for beta in axes["betas"]:
            rows.extend(_interference_rows("orthogonal_interference", dimension, axes["value_dim"],
                                           alpha, beta, 0.0, axes["gaps"], 0, dtype_name))
            for rho in axes["correlations"]:
                rows.extend(_interference_rows("correlated_interference", dimension, axes["value_dim"],
                                               alpha, beta, rho, axes["gaps"], 0, dtype_name))
            for load, seed in product(axes["loads"], axes["seeds"]):
                rows.extend(_capacity_rows(dimension, axes["value_dim"], alpha, beta, load,
                                           axes["gaps"], seed, dtype_name))
    analytic_errors = [row["max_abs_analytic_error"] for row in rows if "max_abs_analytic_error" in row]
    return {
        "schema_version": "gdn-operator-probes-v1",
        "evidence_type": "synthetic_operator_diagnostic",
        "trained_model_evaluated": False,
        "source_url": SOURCE_URL,
        "equation": "S_new = alpha*S + beta*outer(k, v - k@(alpha*S)); y = q@S_new",
        "parameters": axes,
        "row_count": len(rows),
        "max_abs_analytic_error": max(analytic_errors, default=0.0),
        "definitions": {
            "nrmse": "||prediction-target||_2 / ||target||_2; magnitude-sensitive",
            "cosine": "Cosine similarity; null for an exactly zero prediction",
            "capacity_classification": "Nearest cosine among this sequence's values; exactly zero predictions abstain and are incorrect",
            "capacity_gap": "Read all stored keys after gap additional decay-only steps (beta=0)",
            "correlated_interference": "Repeated same distractor key rho*target_key+sqrt(1-rho^2)*orthogonal_key and a value orthogonal to target_value",
            "analytic_retention": "alpha^gap * (1-rho^2 + rho^2*(1-beta)^gap), for the repeated same-key construction only",
            "seed_role": "Capacity random draws only; deterministic controls have seed=0",
        },
        "limitations": [
            "Fixed artificial keys, values and gates; no tokenizer, pretrained weights, learned gates, multi-head interaction or output normalization.",
            "These rows test recurrence behavior and numeric correctness, not real-language model failure or benchmark quality.",
            "Decay-only gaps can reduce amplitude without changing nearest-cosine classification; inspect NRMSE and norm as well as accuracy.",
            "Capacity loading is one pass with no training; alpha/beta comparisons are operator controls, not architecture ablations.",
        ],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point, including when copied into another test suite."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="Optional JSON object overriding operator grid axes")
    parser.add_argument("--out", type=Path, required=True, help="New JSON report path (existing files are protected)")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    config = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("Config must contain a JSON object")
    try:
        report = run_probe(config)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "rows": report["row_count"],
                      "trained_model_evaluated": False,
                      "max_abs_analytic_error": report["max_abs_analytic_error"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
