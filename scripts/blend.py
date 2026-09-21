#!/usr/bin/env python3
"""Evaluate blends of cached validation predictions (see predict_valid.py).

Blend weights are chosen on the first half of the sequences and reported on
the second half as well as on the full set, so the reported gain is honest.

    python scripts/blend.py datasets/preds_linear.npz datasets/preds_gru.npz
"""
from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+", help="npz files from predict_valid.py (first = reference model)")
    ap.add_argument("--grid", type=int, default=41, help="blend-weight grid resolution")
    ap.add_argument("--out", help="write the chosen per-target weights as JSON")
    a = ap.parse_args()
    if len(a.caches) != 2:
        raise SystemExit("this tool blends exactly two models")

    seq, y, (pa, pb) = load_aligned(a.caches)
    uniq = np.unique(seq)
    fit_mask = np.isin(seq, uniq[: len(uniq) // 2])
    eval_mask = ~fit_mask
    names = [Path(c).stem.replace("preds_", "") for c in a.caches]

    print(f"{len(uniq)} sequences, {len(seq):,} scored rows; weights chosen on {fit_mask.sum():,} rows, "
          f"reported on the other {eval_mask.sum():,}")
    for n, p in zip(names, (pa, pb)):
        f = wp(y, p)
        e = wp(y[eval_mask], p[eval_mask])
        print(f"  {n:>8}: full WP {f[2]:.5f} (t0 {f[0]:.5f}, t1 {f[1]:.5f}); eval half {e[2]:.5f}")

    # per-target weight on model B (the second cache): p = (1-w)*A + w*B
    grid = np.linspace(0.0, 1.0, a.grid)
    chosen = []
    for k in range(2):
        best_w, best = 0.0, -2.0
        for w in grid:
            p = (1 - w) * pa[fit_mask, k] + w * pb[fit_mask, k]
            s = weighted_pearson(y[fit_mask, k], p)
            if s > best:
                best, best_w = s, w
        chosen.append(float(best_w))
    w = np.asarray(chosen)
    blend = (1 - w) * pa + w * pb
    e = wp(y[eval_mask], blend[eval_mask])
    f = wp(y, blend)
    print(f"  blend weights on {names[1]}: t0 {w[0]:.3f}, t1 {w[1]:.3f}")
    print(f"  blend eval half: WP {e[2]:.5f} (t0 {e[0]:.5f}, t1 {e[1]:.5f})")
    print(f"  blend full     : WP {f[2]:.5f} (t0 {f[0]:.5f}, t1 {f[1]:.5f})")
    if a.out:
        Path(a.out).write_text(json.dumps({"models": names, "weight_on_second": chosen,
                                           "eval_half_wp": e[2], "full_wp": f[2]}, indent=2))
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
