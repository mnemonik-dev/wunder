#!/usr/bin/env python3
"""A/B candidate feature families against a weighted ridge, in minutes.

Every feature idea currently costs a multi-hour training run to evaluate. This
fits the same closed-form weighted ridge the Rust trainer uses, on a fixed set
of rows, so families can be compared cheaply. Absolute WP on a subset is
biased (a 40-sequence slice once read 0.034 high), so only *paired* deltas on
identical rows are reported, which is what a feature decision needs.

All features are causal: each row sees only itself and earlier rows of its
sequence, matching what `solution.py` can compute at test time.

    python scripts/feature_lab.py --families base,ema_multi,sentinel,vol_norm
    python scripts/feature_lab.py --list
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import weighted_pearson  # noqa: E402

META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")
SENTINEL = -5.199337482452393
N_FEATURES = 112
# Column blocks, from the data overview.
I0_BID_P, I0_ASK_P = slice(0, 11), slice(11, 22)
I0_BID_V, I0_ASK_V = slice(22, 33), slice(33, 44)
I1_BID_P, I1_ASK_P = slice(52, 63), slice(63, 74)
I1_BID_V, I1_ASK_V = slice(74, 85), slice(85, 96)
EXTRA = slice(104, 112)


def ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA down axis 0, seeded with the first row (as StreamState does)."""
    a = 2.0 / (span + 1.0)
    out = lfilter([a], [1.0, -(1.0 - a)], x - x[0], axis=0) + x[0]
    return out.astype(np.float32)


def lag(x: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:k] = x[0]
    out[k:] = x[:-k]
    return out


# Each family maps the raw (T, 112) matrix to extra columns, with a name list.
def f_base(x):
    return [("raw", x)]


def f_ema_fast(x):
    return [("d28", x - ema(x, 28))]


def f_diff42(x):
    return [("lag42", x - lag(x, 42))]


def f_ema_multi(x):
    """The mid/slow residuals the GA switched off; worth +0.002 on the linear read-out."""
    return [("d150", x - ema(x, 150)), ("d1248", x - ema(x, 1248))]


def f_multi_lag(x):
    """Short multi-scale differences: the champion only has a 42-row lag."""
    return [(f"lag{k}", x - lag(x, k)) for k in (1, 4, 16)]


def f_sentinel(x):
    """`a5/a6/a7` use -5.199 to mean zero: a nonlinear fact no EMA can express.

    Emits the indicator and a copy with the spike replaced by the column
    median, so downstream averages are not dragged by it.
    """
    cols = [109, 110, 111]
    ind = (np.abs(x[:, cols] - SENTINEL) < 1e-5).astype(np.float32)
    clean = x[:, cols].copy()
    for j in range(len(cols)):
        m = ind[:, j] > 0
        if m.any():
            clean[m, j] = np.median(clean[~m, j]) if (~m).any() else 0.0
    return [("sent_ind", ind), ("sent_clean", clean),
            ("sent_ind_d28", ind - ema(ind, 28))]


def f_vol_norm(x):
    """Fast residual divided by a running scale - the GA disabled this too."""
    d = x - ema(x, 28)
    vol = ema(np.abs(d), 200)
    return [("dvol", (d / np.maximum(vol, 1e-3)).astype(np.float32))]


def f_cross_group(x):
    """Cross-sectional shape of each book side: level-to-level spread within a group.

    Columns are standardised per-column so absolute differences are not
    economic quantities, but the dispersion across a group still moves with
    book shape.
    """
    out = []
    for nm, sl in (("i0bp", I0_BID_P), ("i0ap", I0_ASK_P), ("i0bv", I0_BID_V), ("i0av", I0_ASK_V),
                   ("i1bp", I1_BID_P), ("i1ap", I1_ASK_P)):
        g = x[:, sl]
        out.append((f"{nm}_stat", np.column_stack([g.mean(1), g.std(1), g.max(1), g.min(1)]).astype(np.float32)))
    return out


def f_imbalance(x):
    """(v_bid - v_ask) / (|v_bid| + |v_ask|) per level, both instruments."""
    out = []
    for nm, b, a in (("i0", I0_BID_V, I0_ASK_V), ("i1", I1_BID_V, I1_ASK_V)):
        vb, va = x[:, b], x[:, a]
        out.append((f"{nm}_imb", (vb - va) / (np.abs(vb) + np.abs(va) + 1e-2)))
    return out


def f_interact(x):
    """Products of the fast price residual with the contemporaneous volume level."""
    d = x - ema(x, 28)
    return [("px_vol", (d[:, I0_BID_P] * x[:, I0_BID_V]).astype(np.float32)),
            ("ax_vol", (d[:, I0_ASK_P] * x[:, I0_ASK_V]).astype(np.float32))]


def f_extra_nl(x):
    """Squares and cross-products of the eight undocumented a0..a7 columns."""
    a = x[:, EXTRA]
    sq = (a ** 2).astype(np.float32)
    cross = np.column_stack([a[:, i] * a[:, j] for i in range(8) for j in range(i + 1, 8)]).astype(np.float32)
    return [("a_sq", sq), ("a_cross", cross)]


FAMILIES = {
    "base": f_base, "ema_fast": f_ema_fast, "diff42": f_diff42,
    "ema_multi": f_ema_multi, "multi_lag": f_multi_lag, "sentinel": f_sentinel,
    "vol_norm": f_vol_norm, "cross_group": f_cross_group, "imbalance": f_imbalance,
    "interact": f_interact, "extra_nl": f_extra_nl,
}
CHAMPION = ["base", "ema_fast", "diff42"]  # what solution.py ships today


def build(path: str, ids: list[int], families: list[str], stride: int):
    pf = pq.ParquetFile(path)
    feat = [c for c in pf.schema_arrow.names if c not in META]
    has_scored = "is_scored" in pf.schema_arrow.names
    Xs, Ys = [], []
    for i in ids:
        cols = feat + ["need_prediction", "t0", "t1"] + (["is_scored"] if has_scored else [])
        tb = pf.read_row_group(i, columns=cols)
        x = np.column_stack([tb.column(c).to_numpy() for c in feat]).astype(np.float32)
        need = tb.column("need_prediction").to_numpy().astype(bool)
        keep = (tb.column("is_scored").to_numpy().astype(bool) & need) if has_scored else need
        sel = np.flatnonzero(keep)[::stride]
        phi = np.concatenate([b for f in families for _, b in FAMILIES[f](x)], axis=1)
        Xs.append(phi[sel])
        Ys.append(np.column_stack([tb.column("t0").to_numpy(), tb.column("t1").to_numpy()]).astype(np.float32)[sel])
    return np.concatenate(Xs), np.concatenate(Ys)


def fit_ridge(X, Y, lam: float, power: float):
    y = np.clip(Y, -2, 2).astype(np.float64)
    w = np.abs(y) ** power
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))]).astype(np.float64)
    W = []
    for k in (0, 1):
        sw = w[:, k]
        A = Z * sw[:, None]
        G = Z.T @ A
        G[np.diag_indices_from(G)] += lam * np.trace(G) / len(G)
        W.append(np.linalg.solve(G, Z.T @ (sw * y[:, k])))
    return mu, sd, np.array(W)


def predict(X, model):
    mu, sd, W = model
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    return Z @ W.T


def wp(Y, P):
    a = weighted_pearson(Y[:, 0], P[:, 0].astype(np.float32))
    b = weighted_pearson(Y[:, 1], P[:, 1].astype(np.float32))
    return 0.5 * (a + b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="datasets/train.parquet")
    ap.add_argument("--valid", default="datasets/valid.parquet")
    ap.add_argument("--train-sequences", type=int, default=60)
    ap.add_argument("--valid-sequences", type=int, default=60)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--lam", type=float, default=3e-4)
    ap.add_argument("--power", type=float, default=0.75)
    ap.add_argument("--families", default="", help="comma list to add on top of the champion set")
    ap.add_argument("--ablate", action="store_true", help="score the champion set plus each family alone")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("champion set:", ",".join(CHAMPION))
        print("available   :", ",".join(FAMILIES))
        return

    tr_ids = list(range(a.train_sequences))
    va_ids = list(range(a.valid_sequences))
    extra = [f for f in a.families.split(",") if f]
    for f in extra:
        if f not in FAMILIES:
            raise SystemExit(f"unknown family {f!r}; --list to see them")

    def score(fams):
        t = time.perf_counter()
        Xtr, Ytr = build(a.train, tr_ids, fams, a.stride)
        Xva, Yva = build(a.valid, va_ids, fams, a.stride)
        m = fit_ridge(Xtr, Ytr, a.lam, a.power)
        s = wp(Yva, predict(Xva, m))
        return s, Xtr.shape[1], time.perf_counter() - t

    base_s, base_d, el = score(CHAMPION)
    print(f"champion features ({'+'.join(CHAMPION)}): {base_d} columns -> WP {base_s:.5f}   [{el:.0f}s]")
    print(f"  (fit on {len(tr_ids)} train sequences, evaluated on {len(va_ids)} valid sequences, "
          f"every {a.stride}th scored row)")
    print()

    todo = [[f] for f in extra] if a.ablate else ([extra] if extra else [])
    for add in todo:
        s, d, el = score(CHAMPION + add)
        print(f"  + {'+'.join(add):<24} {d:>5} cols  WP {s:.5f}  delta {s - base_s:+.5f}  [{el:.0f}s]")


if __name__ == "__main__":
    main()
