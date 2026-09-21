#!/usr/bin/env python3
"""Cache a solution's validation predictions for offline experiments.

Runs `PredictionModel` from a solution directory over the validation set
(optionally with several worker processes, each owning its own model and a
disjoint range of sequences) and stores, for every *scored* row:
seq_ix, step_in_seq, the two targets and the two raw predictions.

    python scripts/predict_valid.py --solution starterpack/baseline --out datasets/preds_gru.npz --workers 4
    python scripts/predict_valid.py --solution solution --out datasets/preds_linear.npz --workers 4

The cached file lets `scripts/blend.py` evaluate blends of several models
exactly (the metric clips *after* blending, and the cache holds raw values).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, DataPoint, validate_sequence  # noqa: E402


def _load(solution_dir: str):
    path = Path(solution_dir) / "solution.py"
    spec = importlib.util.spec_from_file_location(f"solution_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(solution_dir)))
    spec.loader.exec_module(mod)
    return mod.PredictionModel()


def _run(args):
    solution_dir, dataset, groups = args
    model = _load(solution_dir)
    parquet = pq.ParquetFile(dataset)
    out_seq, out_step, out_y, out_p = [], [], [], []
    for g in groups:
        table = parquet.read_row_group(g, use_threads=False)
        seq, need = validate_sequence(table)
        feats = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        scored = table["is_scored"].to_numpy().astype(bool) & need
        preds = np.full((feats.shape[0], 2), np.nan, dtype=np.float32)
        for step in range(feats.shape[0]):
            v = model.predict(DataPoint(seq, step, bool(need[step]), feats[step]))
            if need[step]:
                preds[step] = np.asarray(v, dtype=np.float32)
        idx = np.flatnonzero(scored)
        out_seq.append(np.full(len(idx), seq, dtype=np.int32))
        out_step.append(idx.astype(np.int32))
        out_y.append(targets[idx])
        out_p.append(preds[idx])
    return (np.concatenate(out_seq), np.concatenate(out_step), np.concatenate(out_y), np.concatenate(out_p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--validation", default=str(ROOT / "datasets" / "valid.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequences", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    n = pq.ParquetFile(a.validation).num_row_groups
    if a.sequences > 0:
        n = min(n, a.sequences)
    groups = list(range(n))
    chunks = [groups[i::a.workers] for i in range(a.workers)]
    t0 = time.perf_counter()
    if a.workers == 1:
        parts = [_run((a.solution, a.validation, groups))]
    else:
        with Pool(a.workers) as pool:
            parts = pool.map(_run, [(a.solution, a.validation, c) for c in chunks if c])
    seq = np.concatenate([p[0] for p in parts])
    step = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts])
    p = np.concatenate([p[3] for p in parts])
    order = np.lexsort((step, seq))
    np.savez_compressed(a.out, seq_ix=seq[order], step=step[order], y=y[order], p=p[order])
    print(f"{a.out}: {len(seq):,} scored rows from {n} sequences in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
