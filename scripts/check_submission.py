#!/usr/bin/env python3
"""Pre-flight checks for a submission ZIP, mirroring the organisers' rules.

* solution.py at the archive root, ZIP <= 20 MB
* PredictionModel() constructs without arguments
* returns None on warm-up rows and two finite float32 values otherwise
* identical predictions across two runs (determinism)
* per-row timing extrapolated to the 1,970-sequence test set

Runs from a temporary extraction directory so relative paths inside the ZIP
are exercised exactly as in the scorer container.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, WARMUP, DataPoint  # noqa: E402

MAX_ZIP_BYTES = 20 * 1024 * 1024
TEST_ROWS = 1_970 * SEQUENCE_LENGTH


def run_model(model, sequences: list[tuple[int, np.ndarray]]) -> np.ndarray:
    preds = []
    for seq_ix, feats in sequences:
        for step in range(feats.shape[0]):
            need = step >= WARMUP
            out = model.predict(DataPoint(seq_ix, step, need, feats[step]))
            if not need:
                assert out is None, f"seq {seq_ix} step {step}: warm-up row must return None"
                continue
            out = np.asarray(out, dtype=np.float32)
            assert out.shape == (2,), f"seq {seq_ix} step {step}: shape {out.shape} != (2,)"
            assert np.isfinite(out).all(), f"seq {seq_ix} step {step}: non-finite prediction"
            preds.append(out)
    return np.stack(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", help="submission archive")
    ap.add_argument("--validation", default=str(ROOT / "datasets" / "valid.parquet"))
    ap.add_argument("--sequences", type=int, default=2)
    a = ap.parse_args()

    zpath = Path(a.zip)
    size = zpath.stat().st_size
    print(f"archive: {zpath} ({size / 1e6:.2f} MB)")
    assert size <= MAX_ZIP_BYTES, f"ZIP is larger than 20 MB ({size} bytes)"
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        assert "solution.py" in names, f"solution.py must be at the ZIP root; got {names[:10]}"
        tmp = tempfile.mkdtemp(prefix="wunder-check-")
        z.extractall(tmp)
    print(f"extracted {len(names)} entries to {tmp}")

    cwd = os.getcwd()
    os.chdir(tmp)
    sys.path.insert(0, tmp)
    spec = importlib.util.spec_from_file_location("solution", Path(tmp) / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    parquet = pq.ParquetFile(a.validation)
    sequences = []
    for g in range(min(a.sequences, parquet.num_row_groups)):
        table = parquet.read_row_group(g, use_threads=False)
        feats = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        sequences.append((int(table["seq_ix"][0].as_py()), feats))

    t0 = time.perf_counter()
    model = mod.PredictionModel()
    print(f"PredictionModel() constructed in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    first = run_model(model, sequences)
    elapsed = time.perf_counter() - t0
    rows = sum(f.shape[0] for _, f in sequences)
    us = elapsed / rows * 1e6
    projected_min = TEST_ROWS * elapsed / rows / 60
    print(f"run 1: {rows} rows, {us:.1f} us/row -> projected test-set time {projected_min:.1f} min (limit 60)")
    second = run_model(mod.PredictionModel(), sequences)
    os.chdir(cwd)
    assert np.array_equal(first, second), "predictions differ between two runs (non-deterministic)"
    print("run 2: identical predictions (deterministic)")
    print(f"prediction stats: mean {first.mean(axis=0)}, std {first.std(axis=0)}")
    ok = projected_min < 60
    print("RESULT:", "OK" if ok else "TOO SLOW")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
