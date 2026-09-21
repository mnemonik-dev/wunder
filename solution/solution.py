"""Neutrino linear read-out over streaming features - competition callback.

The model is fitted by `neutrino-wunder` (Rust, see the neutrino repository)
and exported to `model.json`; this file only replays it, row by row, using
NumPy on one CPU thread. It has no dependency other than NumPy.

Per row `x` (112 features as float64):

    step 0 : ema_fast = ema_slow = prev = x
    step >0: ema_fast += a_fast * (x - ema_fast)
             ema_slow += a_slow * (x - ema_slow)
    pred    = x @ A - ema_fast @ B - ema_slow @ C - prev @ D + bias
    prev    = x

where A, B, C, D are (112, 2) matrices folded from the exported weights
(zero rows for feature columns the champion does not use), so the whole
per-row cost is four small mat-vecs and a handful of vector updates.
Predictions are deterministic and identical across runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

N_FEATURES = 112
MODEL_PATH = Path(__file__).resolve().parent / "model.json"


def _fold(model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn the exported block weights into four (112, 2) matrices."""
    layout = model["layout"]
    spec = layout["spec"]
    raw_idx = np.asarray(layout["raw_idx"], dtype=np.intp)
    dyn_idx = np.asarray(layout["dyn_idx"], dtype=np.intp)
    weights = np.asarray(model["weights"], dtype=np.float64)  # (2, F)
    if weights.shape[0] != 2:
        raise ValueError("model.json: expected two weight vectors")
    n_raw, n_dyn = len(raw_idx), len(dyn_idx)
    expected = n_raw + n_dyn * (1 + int(spec["use_dslow"]) + int(spec["use_diff1"]))
    if weights.shape[1] != expected:
        raise ValueError(f"model.json: {weights.shape[1]} weights, layout implies {expected}")

    a = np.zeros((N_FEATURES, 2))
    b = np.zeros((N_FEATURES, 2))
    c = np.zeros((N_FEATURES, 2))
    d = np.zeros((N_FEATURES, 2))
    k = 0
    a[raw_idx] += weights[:, k:k + n_raw].T
    k += n_raw
    w_fast = weights[:, k:k + n_dyn].T
    k += n_dyn
    a[dyn_idx] += w_fast
    b[dyn_idx] += w_fast
    if spec["use_dslow"]:
        w_slow = weights[:, k:k + n_dyn].T
        k += n_dyn
        a[dyn_idx] += w_slow
        c[dyn_idx] += w_slow
    if spec["use_diff1"]:
        w_d1 = weights[:, k:k + n_dyn].T
        k += n_dyn
        a[dyn_idx] += w_d1
        d[dyn_idx] += w_d1
    bias = np.asarray(model["bias"], dtype=np.float64)
    return a, b, c, d, bias


class PredictionModel:
    def __init__(self, model_path: str | os.PathLike | None = None):
        with open(model_path or MODEL_PATH) as f:
            model = json.load(f)
        if model.get("format") != "neutrino-wunder-linear-v1":
            raise ValueError(f"unsupported model format {model.get('format')!r}")
        layout = model["layout"]
        self.alpha_fast = float(layout["alpha_fast"])
        self.alpha_slow = float(layout["alpha_slow"])
        self.w_x, self.w_fast, self.w_slow, self.w_prev, self.bias = _fold(model)
        self.use_slow = bool(np.any(self.w_slow))
        self.use_prev = bool(np.any(self.w_prev))

        self.seq_ix = None
        self.step = 0
        self.ema_fast = np.zeros(N_FEATURES)
        self.ema_slow = np.zeros(N_FEATURES)
        self.prev = np.zeros(N_FEATURES)
        self._tmp = np.zeros(N_FEATURES)

    def _reset(self, x: np.ndarray) -> None:
        self.ema_fast[:] = x
        self.ema_slow[:] = x
        self.prev[:] = x

    def predict(self, data_point):
        x = np.asarray(data_point.state, dtype=np.float64)
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.step = 0
        if self.step == 0:
            self._reset(x)
        else:
            tmp = self._tmp
            np.subtract(x, self.ema_fast, out=tmp)
            tmp *= self.alpha_fast
            self.ema_fast += tmp
            np.subtract(x, self.ema_slow, out=tmp)
            tmp *= self.alpha_slow
            self.ema_slow += tmp
        self.step += 1

        if not data_point.need_prediction:
            self.prev[:] = x
            return None

        pred = x @ self.w_x
        pred -= self.ema_fast @ self.w_fast
        if self.use_slow:
            pred -= self.ema_slow @ self.w_slow
        if self.use_prev:
            pred -= self.prev @ self.w_prev
        pred += self.bias
        self.prev[:] = x
        out = pred.astype(np.float32)
        if not np.isfinite(out).all():
            out = np.zeros(2, dtype=np.float32)
        return out


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starterpack"))
    from utils import ScorerStepByStep  # noqa: E402

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))
