#!/usr/bin/env python3
"""How much does a candidate model disagree with the ones already in the blend?

A blend gains from a new model only to the extent its *errors* are new. Our
first GRU scored 0.670 alone - better than the MLP - yet replacing the
organisers' 0.617 GRU with it made the blend worse, because its residuals
correlate 0.97 with the MLP's while the organisers' correlate 0.37.

This reports, for a cached prediction file, the standalone WP and the
weighted residual correlation against every other cached model on the same
rows, so a candidate can be judged on diversity before it is trained to
convergence.

    python scripts/diversity_report.py --candidate preds_new.npz \
        --against mlp=preds_mlp.npz,gru=preds_gru.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import weighted_pearson  # noqa: E402


def wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    am = (a * w).sum() / w.sum()
    bm = (b * w).sum() / w.sum()
    ca, cb = a - am, b - bm
    denom = np.sqrt((w * ca * ca).sum() * (w * cb * cb).sum())
    return float((w * ca * cb).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--against", required=True, help="comma-separated name=path.npz")
    a = ap.parse_args()

    cand = np.load(a.candidate)
    key = np.stack([cand["seq_ix"], cand["step"]], axis=1)
    y = np.clip(cand["y"].astype(np.float64), -2.0, 2.0)
    w = np.abs(y)
    p = cand["p"].astype(np.float64)

    t0 = weighted_pearson(cand["y"][:, 0], cand["p"][:, 0])
    t1 = weighted_pearson(cand["y"][:, 1], cand["p"][:, 1])
    print(f"candidate {Path(a.candidate).name}: {len(y):,} rows, "
          f"WP {0.5 * (t0 + t1):.5f} (t0 {t0:.5f}, t1 {t1:.5f})")

    others = []
    for spec in a.against.split(","):
        if not spec:
            continue
        name, path = spec.split("=", 1)
        d = np.load(path)
        k = np.stack([d["seq_ix"], d["step"]], axis=1)
        if k.shape == key.shape and np.array_equal(k, key):
            sel = slice(None)
        else:  # candidate covers a prefix of the reference rows
            n = len(key)
            if len(k) < n or not np.array_equal(k[:n], key):
                raise SystemExit(f"{path}: rows do not align with the candidate")
            sel = slice(0, n)
        others.append((name, d["p"][sel].astype(np.float64)))

    print("\nweighted residual correlation (lower = more diverse; adds more to a blend):")
    print(f"  {'model':>12}  {'t0':>7}  {'t1':>7}   standalone WP")
    for name, q in others:
        r = [wcorr(p[:, k] - y[:, k], q[:, k] - y[:, k], w[:, k]) for k in (0, 1)]
        s0 = weighted_pearson(cand["y"][:, 0], q[:, 0].astype(np.float32))
        s1 = weighted_pearson(cand["y"][:, 1], q[:, 1].astype(np.float32))
        print(f"  {name:>12}  {r[0]:7.4f}  {r[1]:7.4f}   {0.5 * (s0 + s1):.5f}")

    if others:
        worst = max(0.5 * (wcorr(p[:, 0] - y[:, 0], q[:, 0] - y[:, 0], w[:, 0])
                           + wcorr(p[:, 1] - y[:, 1], q[:, 1] - y[:, 1], w[:, 1]))
                    for _, q in others)
        verdict = ("strongly diverse - worth training to convergence" if worst < 0.6 else
                   "moderately diverse - may add to the blend" if worst < 0.85 else
                   "redundant with an existing model - unlikely to help the blend")
        print(f"\nhighest residual correlation with any blend member: {worst:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
