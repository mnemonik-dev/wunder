"""Neutrino linear read-out over streaming features - competition callback.

The model is fitted by `neutrino-wunder` (Rust, see the neutrino repository)
and exported to `model.json`; this file only replays it, row by row, using
NumPy on one CPU thread. It has no dependency other than NumPy.

Per row `x` (112 features as float64), `a = 2 / (span + 1)`:

    step 0 : ema_fast = ema_mid = ema_slow = x ; vol = 0 ; hist[*] = x
    step >0: ema_* += a_* * (x - ema_*)
    d_fast  = x - ema_fast
    vol    += a_vol * (|d_fast| - vol)
    scale   = max(vol, floor)  if vol_norm else 1
    lagged  = hist[step mod lag]
    linear  = x @ W_raw + (d_fast/scale) @ W_fast + ((x-ema_mid)/scale) @ W_mid
            + ((x-ema_slow)/scale) @ W_slow + ((x-lagged)/scale) @ W_diff
            + imbalance(x) @ W_imb + bias
    hist[step mod lag] = x

Every W_* is a (112, 2) matrix folded from the exported weights (zero rows
for columns a block does not use), so no fancy indexing happens per row;
blocks the champion did not select are skipped entirely. This mirrors
`neutrino_wunder::features::StreamState::push` operation for operation.
Optionally (``blend.json`` present next to this file) the linear prediction
is blended per target with the organisers' stateful GRU baseline
(``baseline.onnx``, part of the public starter pack, trained only on the
competition data): ``pred = (1 - w) * linear + w * gru``.
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
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.json"
BLEND_PATH = HERE / "blend.json"


class GruBaseline:
    """The organisers' stateful GRU (ONNX), one row per call, one CPU thread.

    Uses two pre-bound I/O bindings that ping-pong the hidden state between
    two buffer pairs, which removes the per-call tensor allocation of
    ``session.run`` (about 30% of the per-row cost). Outputs are bit-identical
    to the plain ``session.run`` loop of the organisers' baseline.
    """

    def __init__(self, path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.x = np.zeros((1, 1, N_FEATURES), dtype=np.float32)
        self.pred = np.zeros((1, 1, 2), dtype=np.float32)
        self.h = [np.zeros((1, 1, 128), dtype=np.float32) for _ in range(4)]  # a0, a1, b0, b1
        self.bindings = [self._bind(0, 2), self._bind(2, 0)]
        self.parity = 0

    def _bind(self, src: int, dst: int):
        b = self.session.io_binding()
        bind = lambda name, arr, inp: (b.bind_input if inp else b.bind_output)(name, "cpu", 0, np.float32, arr.shape, arr.ctypes.data)
        bind("features", self.x, True)
        bind("hidden_0", self.h[src], True)
        bind("hidden_1", self.h[src + 1], True)
        bind("prediction", self.pred, False)
        bind("next_hidden_0", self.h[dst], False)
        bind("next_hidden_1", self.h[dst + 1], False)
        return b

    def reset(self) -> None:
        for h in self.h:
            h.fill(0.0)
        self.parity = 0

    def step(self, state: np.ndarray) -> np.ndarray:
        self.x[0, 0] = state
        self.session.run_with_iobinding(self.bindings[self.parity])
        self.parity ^= 1
        return self.pred[0, 0]


def _fold(model: dict) -> dict:
    """Turn the exported block weights into (112, 2) matrices per block."""
    layout = model["layout"]
    spec = layout["spec"]
    raw_idx = np.asarray(layout["raw_idx"], dtype=np.intp)
    dyn_idx = np.asarray(layout["dyn_idx"], dtype=np.intp)
    imb_pairs = [tuple(p) for p in layout.get("imb_pairs", [])]
    weights = np.asarray(model["weights"], dtype=np.float64)  # (2, F)
    if weights.shape[0] != 2:
        raise ValueError("model.json: expected two weight vectors")
    blocks = [("raw", len(raw_idx)), ("fast", len(dyn_idx))]
    if spec.get("use_dmid"):
        blocks.append(("mid", len(dyn_idx)))
    if spec["use_dslow"]:
        blocks.append(("slow", len(dyn_idx)))
    if spec.get("use_diff", spec.get("use_diff1")):
        blocks.append(("diff", len(dyn_idx)))
    if spec.get("use_imbalance"):
        blocks.append(("imb", len(imb_pairs)))
    expected = sum(n for _, n in blocks)
    if weights.shape[1] != expected:
        raise ValueError(f"model.json: {weights.shape[1]} weights, layout implies {expected}")
    out = {}
    k = 0
    for name, n in blocks:
        w = weights[:, k:k + n].T  # (n, 2)
        k += n
        if name == "imb":
            out[name] = w.copy()
        else:
            full = np.zeros((N_FEATURES, 2))
            full[raw_idx if name == "raw" else dyn_idx] = w
            out[name] = full
    return out


class MlpReadout:
    """Small ReLU MLP on the standardised feature vector (weights from train_mlp.py)."""

    def __init__(self, path: Path):
        d = np.load(path)
        self.mu = d["mu"].astype(np.float64)
        self.inv_sigma = 1.0 / d["sigma"].astype(np.float64)
        self.keep = d["keep"].astype(np.intp) if "keep" in d else None
        n = int(d["n_layers"])
        self.layers = [(d[f"W{i}"].astype(np.float64), d[f"b{i}"].astype(np.float64)) for i in range(n)]

    def __call__(self, phi: np.ndarray) -> np.ndarray:
        if self.keep is not None:
            phi = phi[self.keep]
        h = (phi - self.mu) * self.inv_sigma
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            h = h @ w
            h += b
            if i != last:
                np.maximum(h, 0.0, out=h)
        return h


class PredictionModel:
    def __init__(self, model_path: str | os.PathLike | None = None, blend: bool = True):
        with open(model_path or MODEL_PATH) as f:
            model = json.load(f)
        # Blend: pred = w_lin * linear + w_mlp * mlp + w_gru * gru (per target).
        self.gru = None
        self.mlp = None
        self.w_lin, self.w_mlp, self.w_gru = np.ones(2), np.zeros(2), np.zeros(2)
        if blend and BLEND_PATH.exists():
            with open(BLEND_PATH) as f:
                cfg = json.load(f)
            if "weights" in cfg:
                wts = cfg["weights"]
                self.w_lin = np.asarray(wts.get("linear", [0.0, 0.0]), dtype=np.float64)
                self.w_mlp = np.asarray(wts.get("mlp", [0.0, 0.0]), dtype=np.float64)
                self.w_gru = np.asarray(wts.get("gru", [0.0, 0.0]), dtype=np.float64)
            else:  # step-1 format
                self.w_gru = np.asarray(cfg["weight_on_gru"], dtype=np.float64)
                self.w_lin = 1.0 - self.w_gru
            if np.any(self.w_gru != 0):
                self.gru = GruBaseline(HERE / cfg.get("onnx", "baseline.onnx"))
            if np.any(self.w_mlp != 0):
                self.mlp = MlpReadout(HERE / cfg.get("mlp", "mlp.npz"))
        self.use_lin = bool(np.any(self.w_lin != 0))
        if model.get("format") != "neutrino-wunder-linear-v1":
            raise ValueError(f"unsupported model format {model.get('format')!r}")
        layout = model["layout"]
        spec = layout["spec"]
        self.alpha_fast = float(layout["alpha_fast"])
        self.alpha_mid = float(layout.get("alpha_mid", 0.0))
        self.alpha_slow = float(layout["alpha_slow"])
        self.alpha_vol = float(layout.get("alpha_vol", 0.0))
        self.floor = np.asarray(layout.get("floor", [0.0] * N_FEATURES), dtype=np.float64)
        self.vol_norm = bool(spec.get("vol_norm", False))
        self.lag = int(spec.get("diff_lag", 1)) if spec.get("use_diff", spec.get("use_diff1")) else 1
        self.w = _fold(model)
        self.raw_idx = np.asarray(layout["raw_idx"], dtype=np.intp)
        self.dyn_idx = np.asarray(layout["dyn_idx"], dtype=np.intp)
        self.use_mid = "mid" in self.w
        self.use_slow = "slow" in self.w
        self.use_diff = "diff" in self.w
        self.use_imb = "imb" in self.w
        if self.use_imb:
            pairs = np.asarray(layout["imb_pairs"], dtype=np.intp)
            self.imb_bid, self.imb_ask = pairs[:, 0], pairs[:, 1]
            self.imb_floor = self.floor[self.imb_bid]
        self.bias = np.asarray(model["bias"], dtype=np.float64)

        self.seq_ix = None
        self.step = 0
        self.ema_fast = np.zeros(N_FEATURES)
        self.ema_mid = np.zeros(N_FEATURES)
        self.ema_slow = np.zeros(N_FEATURES)
        self.vol = np.zeros(N_FEATURES)
        self.hist = np.zeros((self.lag, N_FEATURES))
        self._tmp = np.zeros(N_FEATURES)
        self._d = np.zeros(N_FEATURES)

    def _reset(self, x: np.ndarray) -> None:
        self.ema_fast[:] = x
        self.ema_mid[:] = x
        self.ema_slow[:] = x
        self.vol[:] = 0.0
        self.hist[:] = x

    def predict(self, data_point):
        x = np.asarray(data_point.state, dtype=np.float64)
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.step = 0
        tmp = self._tmp
        if self.step == 0:
            self._reset(x)
            if self.gru is not None:
                self.gru.reset()
        else:
            np.subtract(x, self.ema_fast, out=tmp)
            tmp *= self.alpha_fast
            self.ema_fast += tmp
            np.subtract(x, self.ema_mid, out=tmp)
            tmp *= self.alpha_mid
            self.ema_mid += tmp
            np.subtract(x, self.ema_slow, out=tmp)
            tmp *= self.alpha_slow
            self.ema_slow += tmp
        d = self._d
        np.subtract(x, self.ema_fast, out=d)
        np.abs(d, out=tmp)
        tmp -= self.vol
        tmp *= self.alpha_vol
        self.vol += tmp
        slot = self.step % self.lag
        self.step += 1
        gru_pred = self.gru.step(data_point.state) if self.gru is not None else None

        if not data_point.need_prediction:
            self.hist[slot] = x
            return None

        scale = np.maximum(self.vol, self.floor) if self.vol_norm else None
        if scale is not None:
            d /= scale
        blocks = [("raw", x), ("fast", d)]
        if self.use_mid:
            b = x - self.ema_mid
            if scale is not None:
                b /= scale
            blocks.append(("mid", b))
        if self.use_slow:
            b = x - self.ema_slow
            if scale is not None:
                b /= scale
            blocks.append(("slow", b))
        if self.use_diff:
            b = x - self.hist[slot]
            if scale is not None:
                b /= scale
            blocks.append(("diff", b))
        if self.use_imb:
            vb = x[self.imb_bid]
            va = x[self.imb_ask]
            blocks.append(("imb", (vb - va) / (np.abs(vb) + np.abs(va) + self.imb_floor)))
        self.hist[slot] = x

        pred = np.zeros(2)
        if self.use_lin:
            lin = self.bias.copy()
            for name, b in blocks:
                lin += b @ self.w[name]
            pred += self.w_lin * lin
        if self.mlp is not None:
            phi = np.concatenate([b[self.raw_idx] if n == "raw" else (b if n == "imb" else b[self.dyn_idx]) for n, b in blocks])
            pred += self.w_mlp * self.mlp(phi)
        if gru_pred is not None:
            pred += self.w_gru * gru_pred
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
