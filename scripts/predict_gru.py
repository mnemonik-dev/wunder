#!/usr/bin/env python3
"""Cache a trained GRU's scored validation predictions for `scripts/blend.py`.

Writes the same `seq_ix / step / y / p` archive that `predict_valid.py`
produces for a solution directory, so the output can be blended against the
existing caches. Predictions are produced statefully, one sequence at a time
from a zero hidden state, exactly as `solution.py` runs at test time.

    python scripts/predict_gru.py --checkpoint datasets/experiments/gru-20260922/gru.pt \
        --valid datasets/valid.parquet --out datasets/experiments/gru-20260922/preds_gru_ours.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_gru import GruReadout, SequenceReader, batched, load_batch  # noqa: E402
from utils import weighted_pearson  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid", default="datasets/valid.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--bptt", type=int, default=512)
    ap.add_argument("--sequences", type=int, default=0, help="0 = all")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--align-to", help="existing cache to verify row alignment against")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    ck = torch.load(a.checkpoint, weights_only=False)
    model = GruReadout(ck["hidden"], ck["layers"], ck["mu"], ck["sigma"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']}, monitored valid WP {ck['valid_wp']:.5f}", flush=True)

    va = SequenceReader(a.valid)
    ids = list(range(va.n if a.sequences <= 0 else min(va.n, a.sequences)))
    seq_ids, steps, ys, ps = [], [], [], []
    t = time.perf_counter()
    with torch.no_grad():
        for done, chunk in enumerate(batched(ids, a.batch)):
            x, y, _, scored = load_batch(va, chunk)
            X = torch.from_numpy(x)
            h = None
            out = np.empty_like(y)
            for s in range(0, X.shape[1], a.bptt):
                p, h = model(X[:, s:s + a.bptt], h)
                out[:, s:s + a.bptt] = p.numpy()
            for j, gid in enumerate(chunk):
                m = scored[j]
                tb = va.pf.read_row_group(gid, columns=["seq_ix"])
                seq_ids.append(tb.column("seq_ix").to_numpy()[m])
                steps.append(np.flatnonzero(m).astype(np.int32))
                ys.append(y[j][m])
                ps.append(out[j][m])
            if (done + 1) % 10 == 0:
                seen = min((done + 1) * a.batch, len(ids))
                el = time.perf_counter() - t
                print(f"  {seen}/{len(ids)} sequences, {el:.0f}s, eta {(len(ids) - seen) * el / seen / 60:.1f} min",
                      flush=True)

    seq_ix = np.concatenate(seq_ids).astype(np.int32)
    step = np.concatenate(steps).astype(np.int32)
    y = np.concatenate(ys).astype(np.float32)
    p = np.concatenate(ps).astype(np.float32)

    if a.align_to:
        ref = np.load(a.align_to)
        if not (np.array_equal(ref["seq_ix"], seq_ix) and np.array_equal(ref["step"], step)):
            raise SystemExit(f"row order differs from {a.align_to}; blending would compare mismatched rows")
        if not np.allclose(ref["y"], y, atol=1e-6):
            raise SystemExit(f"targets differ from {a.align_to}")
        print(f"row alignment with {a.align_to} verified on {len(y):,} rows", flush=True)

    np.savez(a.out, seq_ix=seq_ix, step=step, y=y, p=p)
    t0 = weighted_pearson(y[:, 0], p[:, 0])
    t1 = weighted_pearson(y[:, 1], p[:, 1])
    print(f"{len(y):,} scored rows -> {a.out}")
    print(f"GRU alone: WP {0.5 * (t0 + t1):.5f} (t0 {t0:.5f}, t1 {t1:.5f})")


if __name__ == "__main__":
    main()
