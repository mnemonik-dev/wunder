#!/usr/bin/env python3
"""Rust <-> Python parity: the trainer's predictions must match solution.py.

Runs `neutrino-wunder predict` on a few validation sequences and compares
against PredictionModel from the solution directory, row by row.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import FEATURE_COLUMNS, WARMUP, DataPoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default=str(ROOT / "solution"))
    ap.add_argument("--validation", default=str(ROOT / "datasets" / "valid.parquet"))
    ap.add_argument("--sequences", type=int, default=2)
    ap.add_argument("--binary", default=os.environ.get("NEUTRINO_WUNDER", str(ROOT.parent / "neutrino" / "target" / "release" / "neutrino-wunder")))
    ap.add_argument("--atol", type=float, default=1e-5)
    a = ap.parse_args()

    solution = Path(a.solution)
    out_csv = ROOT / "datasets" / "parity_rust.csv"
    subprocess.run([a.binary, "predict", "--model", str(solution / "model.json"), "--data", a.validation,
                    "--sequences", str(a.sequences), "--out", str(out_csv)], check=True)
    rust = {}
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            rust[(int(row["seq_ix"]), int(row["step_in_seq"]))] = (float(row["p0"]), float(row["p1"]))

    spec = importlib.util.spec_from_file_location("solution", solution / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.PredictionModel(blend=False)  # the Rust trainer knows only the linear part

    parquet = pq.ParquetFile(a.validation)
    worst = 0.0
    n = 0
    for g in range(a.sequences):
        table = parquet.read_row_group(g, use_threads=False)
        seq = int(table["seq_ix"][0].as_py())
        feats = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        for step in range(feats.shape[0]):
            out = model.predict(DataPoint(seq, step, step >= WARMUP, feats[step]))
            if step < WARMUP:
                continue
            r = np.asarray(rust[(seq, step)], dtype=np.float32)
            diff = float(np.max(np.abs(np.asarray(out, dtype=np.float32) - r)))
            worst = max(worst, diff)
            n += 1
    print(f"compared {n} predictions over {a.sequences} sequences; max |rust - python| = {worst:.3e}")
    if worst > a.atol:
        print("PARITY FAILED", file=sys.stderr)
        sys.exit(1)
    print("PARITY OK")


if __name__ == "__main__":
    main()
