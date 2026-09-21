#!/usr/bin/env python3
"""Score a solution directory locally with the organisers' Global WP scorer.

Adds two things `starterpack/utils.ScorerStepByStep` lacks: a sequence
limit/offset for quick iterations, and timing (per-row cost and projected
time for the 1,970-sequence test set against the 60-minute budget).

    python scripts/score_solution.py --solution solution --sequences 50
    python scripts/score_solution.py --solution starterpack/baseline
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    TARGET_COLUMNS,
    WARMUP,
    DataPoint,
    GlobalAccumulator,
    validate_sequence,
)

TEST_SEQUENCES = 1_970
TIME_LIMIT_S = 60 * 60


def load_model_class(solution_dir: Path):
    path = solution_dir / "solution.py"
    spec = importlib.util.spec_from_file_location("solution", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(solution_dir))
    spec.loader.exec_module(mod)
    return mod.PredictionModel


def score(model, dataset: Path, sequences: int, offset: int, quiet: bool = False) -> dict:
    parquet = pq.ParquetFile(dataset)
    n = parquet.num_row_groups
    stop = n if sequences <= 0 else min(n, offset + sequences)
    accumulator = GlobalAccumulator()
    seen = set()
    rows = 0
    t_pred = 0.0
    t_start = time.perf_counter()
    for group in range(offset, stop):
        table = parquet.read_row_group(group, use_threads=False)
        seq, need = validate_sequence(table)
        if seq in seen:
            raise ValueError("duplicate sequence ID")
        seen.add(seq)
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        t0 = time.perf_counter()
        for step in range(SEQUENCE_LENGTH):
            point = DataPoint(seq, step, bool(need[step]), features[step])
            value = model.predict(point)
            if not need[step]:
                if value is not None:
                    raise ValueError("return None during warm-up AFTER updating state")
            else:
                value = np.asarray(value, dtype=np.float32)
                if value.shape != (2,) or not np.isfinite(value).all():
                    raise ValueError("return two finite scores on every required row")
                predictions[step] = value
        t_pred += time.perf_counter() - t0
        rows += SEQUENCE_LENGTH
        mask = table["is_scored"].to_numpy().astype(bool) & need
        accumulator.add(targets, predictions, mask)
        if not quiet and (group - offset + 1) % 10 == 0:
            partial = accumulator.result()["weighted_pearson"]
            print(f"  {group - offset + 1}/{stop - offset} sequences, running WP {partial:.5f}, "
                  f"{t_pred / rows * 1e6:.1f} us/row", file=sys.stderr)
    result = accumulator.result()
    us_per_row = t_pred / rows * 1e6
    projected = TEST_SEQUENCES * SEQUENCE_LENGTH * t_pred / rows
    result.update({
        "sequences": len(seen),
        "rows_seen": rows,
        "predict_us_per_row": round(us_per_row, 2),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "projected_test_minutes": round(projected / 60, 1),
        "within_time_limit": projected < TIME_LIMIT_S,
    })
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", default=str(ROOT / "solution"), help="directory containing solution.py")
    ap.add_argument("--validation", default=str(ROOT / "datasets" / "valid.parquet"))
    ap.add_argument("--sequences", type=int, default=0, help="0 = whole file")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--json", help="write the result to this file")
    a = ap.parse_args()
    model = load_model_class(Path(a.solution))()
    result = score(model, Path(a.validation), a.sequences, a.offset)
    print(json.dumps(result, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
