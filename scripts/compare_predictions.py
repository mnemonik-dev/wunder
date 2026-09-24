#!/usr/bin/env python3
"""Compare aligned prediction caches, with paired sequence-bootstrap intervals.

    python scripts/compare_predictions.py old.npz new.npz --out comparison.json

Resample complete sequences, preserving within-sequence dependence. Intervals
describe this validation sample; they do not account for model-selection bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blend import load_aligned, wp


def moments(seq: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Six additive weighted moments per sequence and target."""
    _, groups = np.unique(seq, return_inverse=True)
    y = np.clip(y.astype(np.float32), -2, 2).astype(np.float64)
    p = np.clip(p.astype(np.float32), -2, 2).astype(np.float64)
    w = np.abs(y)
    return np.stack([
        np.stack([np.bincount(groups, weights=value[:, k]) for k in range(2)], axis=1)
        for value in (w, w * y, w * p, w * y * y, w * p * p, w * y * p)
    ], axis=-1)


def score_moments(m: np.ndarray) -> np.ndarray:
    w, sy, sp, syy, spp, syp = np.moveaxis(m, -1, 0)
    safe_w = np.maximum(w, 1e-30)
    vy = np.maximum(syy - sy * sy / safe_w, 0)
    vp = np.maximum(spp - sp * sp / safe_w, 0)
    denominator = np.sqrt(vy * vp)
    valid = (w >= 1e-8) & (vy / safe_w > 1e-16) & (vp / safe_w > 1e-16)
    return np.clip(np.divide(syp - sy * sp / safe_w, denominator,
                             out=np.zeros_like(denominator), where=valid), -1, 1)


def compare(seq, y, old, new, replicates: int, seed: int) -> dict:
    old_score, new_score = wp(y, old), wp(y, new)
    a, b = moments(seq, y, old), moments(seq, y, new)
    # Check our additive implementation against the official direct metric.
    np.testing.assert_allclose(score_moments(a.sum(axis=0)), old_score[:2], atol=1e-10)
    np.testing.assert_allclose(score_moments(b.sum(axis=0)), new_score[:2], atol=1e-10)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        indices = rng.integers(len(a), size=len(a))
        deltas.append(float((score_moments(b[indices].sum(axis=0)) -
                             score_moments(a[indices].sum(axis=0))).mean()))
    return {
        "sequences_with_scored_rows": len(a), "scored_rows": len(seq),
        "baseline": dict(zip(("t0", "t1", "wp"), old_score)),
        "candidate": dict(zip(("t0", "t1", "wp"), new_score)),
        "delta_wp": new_score[2] - old_score[2],
        "delta_wp_bootstrap_95_percent": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": replicates, "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.replicates < 1:
        ap.error("--replicates must be positive")
    seq, y, (old, new) = load_aligned([args.baseline, args.candidate])
    with np.load(args.candidate) as cache:
        if not np.array_equal(y, cache["y"]):
            raise ValueError("prediction caches contain different targets")
    unique = np.unique(seq)
    if len(unique) < 2:
        ap.error("at least two scored sequences are required")
    second = np.isin(seq, unique[len(unique) // 2:])
    result = {
        "full": compare(seq, y, old, new, args.replicates, args.seed),
        "second_half": compare(seq[second], y[second], old[second], new[second],
                               args.replicates, args.seed),
        "limitation": "Sequence-bootstrap intervals exclude model-selection bias and distribution shift; validation is not the hidden competition test.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
