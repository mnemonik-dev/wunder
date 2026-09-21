#!/usr/bin/env python3
"""Evaluate blends of cached validation predictions (see predict_valid.py).

Blend weights are chosen on the first half of the sequences and reported on
the second half as well as on the full set, so the reported gain is honest.

Two models: exhaustive grid over the weight of the second model.
Three or more: per-target weighted least squares of the clipped target on
the predictions (weights |clip(y)|, the metric's weights) on the fit half,
then a local grid refinement of the same objective (WP) around that point.

    python scripts/blend.py datasets/preds_linear.npz datasets/preds_gru.npz
    python scripts/blend.py --names linear,mlp,gru preds_linear.npz preds_mlp.npz preds_gru.npz --out solution/blend.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import weighted_pearson  # noqa: E402


def wp(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    t0 = weighted_pearson(y[:, 0], p[:, 0])
    t1 = weighted_pearson(y[:, 1], p[:, 1])
    return t0, t1, 0.5 * (t0 + t1)


def load_aligned(paths: list[str]):
    ds = [np.load(p) for p in paths]
    key0 = np.stack([ds[0]["seq_ix"], ds[0]["step"]], axis=1)
    for d in ds[1:]:
        key = np.stack([d["seq_ix"], d["step"]], axis=1)
        if key.shape != key0.shape or not np.array_equal(key, key0):
            raise SystemExit("prediction caches cover different rows; regenerate them on the same validation set")
    y = ds[0]["y"].astype(np.float64)
    preds = [d["p"].astype(np.float64) for d in ds]
    return ds[0]["seq_ix"], y, preds


def choose_weights(y: np.ndarray, preds: list[np.ndarray], k: int, grid: int) -> np.ndarray:
    """Weights (one per model) for target k maximising WP on the given rows."""
    P = np.stack([p[:, k] for p in preds], axis=1)
    if len(preds) == 2:
        best_w, best = 0.0, -2.0
        for w in np.linspace(0.0, 1.0, grid):
            s = weighted_pearson(y[:, k], (1 - w) * P[:, 0] + w * P[:, 1])
            if s > best:
                best, best_w = s, w
        return np.array([1 - best_w, best_w])
    yk = np.clip(y[:, k], -2.0, 2.0)
    sw = np.sqrt(np.abs(yk))
    A = np.column_stack([P, np.ones(len(yk))]) * sw[:, None]
    coef, *_ = np.linalg.lstsq(A, yk * sw, rcond=None)
    w = coef[: len(preds)]
    # WP is scale-free: normalise to sum 1 (if positive) and refine locally.
    if w.sum() > 0:
        w = w / w.sum()
    best, best_w = weighted_pearson(yk, P @ w), w.copy()
    for step in (0.1, 0.05, 0.025):
        improved = True
        while improved:
            improved = False
            for i, j in itertools.permutations(range(len(preds)), 2):
                cand = best_w.copy()
                cand[i] += step
                cand[j] -= step
                s = weighted_pearson(yk, P @ cand)
                if s > best + 1e-9:
                    best, best_w, improved = s, cand, True
    return best_w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+", help="npz files from predict_valid.py")
    ap.add_argument("--names", help="comma-separated model names in the same order (default: from file names)")
    ap.add_argument("--grid", type=int, default=41, help="two-model grid resolution")
    ap.add_argument("--out", help="write blend.json for solution.py (keys: weights per model name)")
    ap.add_argument("--onnx", default="baseline.onnx")
    ap.add_argument("--mlp", default="mlp.npz")
    a = ap.parse_args()
    if len(a.caches) < 2:
        raise SystemExit("need at least two caches")

    seq, y, preds = load_aligned(a.caches)
    names = a.names.split(",") if a.names else [Path(c).stem.replace("preds_", "") for c in a.caches]
    assert len(names) == len(preds)
    uniq = np.unique(seq)
    fit_mask = np.isin(seq, uniq[: len(uniq) // 2])
    eval_mask = ~fit_mask

    print(f"{len(uniq)} sequences, {len(seq):,} scored rows; weights chosen on {fit_mask.sum():,} rows, "
          f"reported on the other {eval_mask.sum():,}")
    for n, p in zip(names, preds):
        f = wp(y, p)
        e = wp(y[eval_mask], p[eval_mask])
        print(f"  {n:>8}: full WP {f[2]:.5f} (t0 {f[0]:.5f}, t1 {f[1]:.5f}); eval half {e[2]:.5f}")

    W = np.stack([choose_weights(y[fit_mask], [p[fit_mask] for p in preds], k, a.grid) for k in range(2)], axis=1)  # (models, 2)
    blend = sum(W[i][None, :] * preds[i] for i in range(len(preds)))
    e = wp(y[eval_mask], blend[eval_mask])
    f = wp(y, blend)
    for i, n in enumerate(names):
        print(f"  weight {n:>8}: t0 {W[i, 0]:+.3f}, t1 {W[i, 1]:+.3f}")
    print(f"  blend eval half: WP {e[2]:.5f} (t0 {e[0]:.5f}, t1 {e[1]:.5f})")
    print(f"  blend full     : WP {f[2]:.5f} (t0 {f[0]:.5f}, t1 {f[1]:.5f})")
    if a.out:
        cfg = {"weights": {n: [float(W[i, 0]), float(W[i, 1])] for i, n in enumerate(names)},
               "onnx": a.onnx, "mlp": a.mlp,
               "note": "pred = sum_m weights[m] * pred_m per target; weights chosen on the first half of valid.parquet sequences (scripts/blend.py)",
               "validation": {"full_wp": f[2], "eval_half_wp": e[2]}}
        Path(a.out).write_text(json.dumps(cfg, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
