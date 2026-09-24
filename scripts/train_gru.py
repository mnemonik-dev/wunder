#!/usr/bin/env python3
"""Train our own stateful GRU read-out on the raw 112 features.

The organisers' baseline GRU scores 0.617 WP alone, yet still earns ~0.18 of
the blend weight next to a 0.668 MLP: it is the only model in the blend that
carries state across rows, so its errors are decorrelated from the feature
read-outs. Replacing it with a GRU trained on this data is therefore worth
more than its standalone score suggests.

No feature export is involved. Each Parquet row group is exactly one sequence
(20,000 rows), so training streams straight from `train.parquet`: decoding a
sequence costs ~7 ms, i.e. ~1.2 min for all 10,607 of them per epoch, against
~45 min of compute. Disk stays untouched.

Training mirrors inference exactly: every sequence starts from a zero hidden
state, windows of `--bptt` steps are processed in order, and the state is
carried (detached) between windows. The loss is the `|clip(y)|^p`-weighted MSE
on `need_prediction` rows that the ridge and MLP read-outs also minimise.

Standardisation is folded into the exported graph, and the ONNX signature
matches `baseline.onnx` (`features`/`hidden_0`/`hidden_1` ->
`prediction`/`next_hidden_0`/`next_hidden_1`), so the model is a drop-in
replacement: `solution.py` keeps its I/O-binding fast path unchanged.

    python scripts/train_gru.py --train datasets/train.parquet \
        --valid datasets/valid.parquet --out solution/gru.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import weighted_pearson  # noqa: E402

N_FEATURES = 112
META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")


class SequenceReader:
    """Row groups of a competition Parquet file, one sequence per group."""

    def __init__(self, path: str):
        self.pf = pq.ParquetFile(path)
        self.feat = [c for c in self.pf.schema_arrow.names if c not in META]
        if len(self.feat) != N_FEATURES:
            raise SystemExit(f"{path}: expected {N_FEATURES} feature columns, found {len(self.feat)}")
        self.has_scored = "is_scored" in self.pf.schema_arrow.names
        self.n = self.pf.metadata.num_row_groups

    def read(self, i: int):
        cols = self.feat + ["need_prediction", "t0", "t1"] + (["is_scored"] if self.has_scored else [])
        tb = self.pf.read_row_group(i, columns=cols + ["step_in_seq"])
        step = tb.column("step_in_seq").to_numpy()
        if step[0] != 0 or not np.all(np.diff(step) == 1):
            raise SystemExit(f"row group {i} is not one contiguous sequence starting at step 0")
        x = np.column_stack([tb.column(c).to_numpy() for c in self.feat]).astype(np.float32)
        y = np.column_stack([tb.column("t0").to_numpy(), tb.column("t1").to_numpy()]).astype(np.float32)
        need = tb.column("need_prediction").to_numpy().astype(bool)
        scored = tb.column("is_scored").to_numpy().astype(bool) if self.has_scored else need
        return x, y, need, scored


class GruReadout(nn.Module):
    """Stateful GRU with standardisation folded in, so the export is self-contained."""

    def __init__(self, hidden: int, layers: int, mu: np.ndarray, sigma: np.ndarray):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.gru = nn.GRU(N_FEATURES, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        self.register_buffer("mu", torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer("inv_sigma", torch.from_numpy((1.0 / sigma).astype(np.float32)))

    def forward(self, x, h=None):
        z = (x - self.mu) * self.inv_sigma
        y, h = self.gru(z, h)
        return self.head(y), h


class ExportWrapper(nn.Module):
    """`baseline.onnx`'s signature: per-layer hidden states in and out."""

    def __init__(self, model: GruReadout):
        super().__init__()
        self.model = model

    def forward(self, features, hidden_0, hidden_1):
        h = torch.cat([hidden_0, hidden_1], dim=0)
        pred, hn = self.model(features, h)
        return pred, hn[0:1], hn[1:2]


def feature_stats(reader: SequenceReader, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of each raw column over `n` sequences sampled across the file."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(reader.n, size=min(n, reader.n), replace=False)
    total = np.zeros(N_FEATURES, dtype=np.float64)
    sq = np.zeros(N_FEATURES, dtype=np.float64)
    rows = 0
    for i in idx:
        x, _, _, _ = reader.read(int(i))
        total += x.sum(axis=0, dtype=np.float64)
        sq += (x.astype(np.float64) ** 2).sum(axis=0)
        rows += len(x)
    mu = total / rows
    var = np.maximum(sq / rows - mu ** 2, 0.0)
    sigma = np.sqrt(var)
    sigma[sigma < 1e-6] = 1.0  # constant columns pass through unscaled
    return mu, sigma


def batched(seq: list[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def load_batch(reader: SequenceReader, ids: list[int]):
    xs, ys, needs, scoreds = [], [], [], []
    for i in ids:
        x, y, need, scored = reader.read(i)
        xs.append(x)
        ys.append(y)
        needs.append(need)
        scoreds.append(scored)
    return (np.stack(xs), np.stack(ys), np.stack(needs), np.stack(scoreds))


@torch.no_grad()
def validate(model: GruReadout, reader: SequenceReader, ids: list[int], batch: int, bptt: int):
    """Global WP over the `is_scored` rows of `ids`, run statefully like inference."""
    model.eval()
    ps, yy = [], []
    for ids_b in batched(ids, batch):
        x, y, _, scored = load_batch(reader, ids_b)
        X = torch.from_numpy(x)
        h = None
        out = np.empty_like(y)
        for s in range(0, X.shape[1], bptt):
            p, h = model(X[:, s:s + bptt], h)
            out[:, s:s + bptt] = p.numpy()
        ps.append(out[scored])
        yy.append(y[scored])
    p = np.concatenate(ps)
    y = np.concatenate(yy)
    t0 = weighted_pearson(y[:, 0], p[:, 0])
    t1 = weighted_pearson(y[:, 1], p[:, 1])
    return t0, t1, 0.5 * (t0 + t1), len(y)


def export_onnx(model: GruReadout, path: Path) -> None:
    model.eval()
    wrapper = ExportWrapper(model)
    args = (
        torch.zeros(1, 1, N_FEATURES),
        torch.zeros(1, 1, model.hidden),
        torch.zeros(1, 1, model.hidden),
    )
    torch.onnx.export(
        wrapper, args, str(path),
        input_names=["features", "hidden_0", "hidden_1"],
        output_names=["prediction", "next_hidden_0", "next_hidden_1"],
        opset_version=17, dynamo=False,
    )


def onnx_parity(model: GruReadout, path: Path, reader: SequenceReader, seq: int, rows: int) -> float:
    """Max |torch - onnxruntime| over `rows` consecutive rows of one sequence."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    x, _, _, _ = reader.read(seq)
    x = x[:rows]
    model.eval()
    with torch.no_grad():
        ref, _ = model(torch.from_numpy(x[None, :, :]))
    ref = ref[0].numpy()
    h0 = np.zeros((1, 1, model.hidden), dtype=np.float32)
    h1 = np.zeros((1, 1, model.hidden), dtype=np.float32)
    worst = 0.0
    for i in range(rows):
        out = sess.run(None, {"features": x[i].reshape(1, 1, -1), "hidden_0": h0, "hidden_1": h1})
        worst = max(worst, float(np.max(np.abs(out[0][0, 0] - ref[i]))))
        h0, h1 = out[1], out[2]
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="datasets/train.parquet")
    ap.add_argument("--valid", default="datasets/valid.parquet")
    ap.add_argument("--out", required=True, help="ONNX path; the torch checkpoint goes next to it")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32, help="sequences trained in parallel")
    ap.add_argument("--bptt", type=int, default=512, help="truncated BPTT window in steps")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--weight-power", type=float, default=0.75)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sequences", type=int, default=0, help="training sequences (0 = all)")
    ap.add_argument("--valid-sequences", type=int, default=300, help="sequences for per-epoch validation")
    ap.add_argument("--stats-sequences", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tr = SequenceReader(a.train)
    va = SequenceReader(a.valid)
    n_train = tr.n if a.sequences <= 0 else min(tr.n, a.sequences)
    val_ids = list(range(min(a.valid_sequences, va.n))) if a.valid_sequences > 0 else list(range(va.n))
    print(f"train {n_train} of {tr.n} sequences ({n_train * 19901 / 1e6:.1f}M predicted rows), "
          f"validation {len(val_ids)} sequences", flush=True)

    t = time.perf_counter()
    mu, sigma = feature_stats(tr, a.stats_sequences, a.seed)
    print(f"feature statistics over {min(a.stats_sequences, tr.n)} sequences in {time.perf_counter() - t:.0f}s "
          f"(sigma min {sigma.min():.4g}, max {sigma.max():.4g})", flush=True)

    model = GruReadout(a.hidden, a.layers, mu, sigma)
    if a.layers != 2:
        print("note: --layers != 2 cannot use the baseline's two-hidden-state ONNX signature", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    windows = (20000 + a.bptt - 1) // a.bptt
    steps = a.epochs * ((n_train + a.batch - 1) // a.batch) * windows
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    print(f"GRU hidden {a.hidden} x {a.layers} layers, {n_params:,} parameters; "
          f"{steps:,} optimiser steps ({windows} windows/sequence)", flush=True)

    rng = np.random.default_rng(a.seed)
    best = (-1.0, None, None)
    history = []
    ckpt = out.with_suffix(".pt")
    for epoch in range(a.epochs):
        model.train()
        order = list(rng.permutation(n_train))
        te = time.perf_counter()
        loss_sum, nb, seen, decode_s = 0.0, 0, 0, 0.0
        for bi, ids in enumerate(batched(order, a.batch)):
            td = time.perf_counter()
            x, y, need, _ = load_batch(tr, [int(i) for i in ids])
            decode_s += time.perf_counter() - td
            X = torch.from_numpy(x)
            Y = torch.from_numpy(np.clip(y, -2.0, 2.0))
            M = torch.from_numpy(need & np.isfinite(y).all(axis=2))
            W = (Y.abs() ** a.weight_power) * M.unsqueeze(-1)
            h = None
            for s in range(0, X.shape[1], a.bptt):
                xb, yb, wb = X[:, s:s + a.bptt], Y[:, s:s + a.bptt], W[:, s:s + a.bptt]
                if wb.sum() <= 0:  # pure warm-up window: advance state only
                    with torch.no_grad():
                        _, h = model(xb, h)
                    continue
                # Normalised per window so every truncated-BPTT step has the same
                # scale regardless of how many scored rows the window holds.
                wsum = wb.sum(dim=(0, 1)).clamp(min=1e-6)
                pred, h = model(xb, h)
                loss = (((pred - yb) ** 2 * wb).sum(dim=(0, 1)) / wsum).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if a.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                sched.step()
                h = h.detach()
                loss_sum += float(loss)
                nb += 1
            seen += len(ids)
            if bi % 20 == 0:
                el = time.perf_counter() - te
                rate = seen * 20000 / max(el, 1e-6)
                print(f"  epoch {epoch + 1} seq {seen}/{n_train}  loss {loss_sum / max(nb, 1):.5f}  "
                      f"{rate:,.0f} rows/s  decode {100 * decode_s / max(el, 1e-6):.0f}%  "
                      f"eta {(n_train - seen) * 20000 / max(rate, 1) / 60:.0f} min", flush=True)
        tv = time.perf_counter()
        v = validate(model, va, val_ids, a.batch, a.bptt)
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(nb, 1),
                        "valid_t0": v[0], "valid_t1": v[1], "valid_wp": v[2], "valid_rows": v[3]})
        print(f"epoch {epoch + 1}/{a.epochs}: loss {loss_sum / max(nb, 1):.5f}  "
              f"valid WP {v[2]:.5f} (t0 {v[0]:.5f}, t1 {v[1]:.5f}) on {v[3]:,} scored rows  "
              f"train {tv - te:.0f}s, valid {time.perf_counter() - tv:.0f}s", flush=True)
        if v[2] > best[0]:
            best = (v[2], epoch + 1, {k: t_.detach().clone() for k, t_ in model.state_dict().items()})
            torch.save({"state_dict": best[2], "hidden": a.hidden, "layers": a.layers,
                        "mu": mu, "sigma": sigma, "epoch": epoch + 1, "valid_wp": v[2]}, ckpt)
            print(f"  new best; checkpoint -> {ckpt}", flush=True)

    model.load_state_dict(best[2])
    export_onnx(model, out)
    worst = onnx_parity(model, out, va, val_ids[0], 2000)
    print(f"exported {out} ({out.stat().st_size / 1e6:.2f} MB); "
          f"torch vs onnxruntime max |delta| over 2,000 rows = {worst:.3g}", flush=True)

    meta = {"train_sequences": n_train, "hidden": a.hidden, "layers": a.layers,
            "parameters": n_params, "epochs": a.epochs, "best_epoch": best[1],
            "best_valid_wp": best[0], "valid_sequences": len(val_ids),
            "bptt": a.bptt, "batch": a.batch, "lr": a.lr, "weight_power": a.weight_power,
            "weight_decay": a.weight_decay, "grad_clip": a.grad_clip, "seed": a.seed,
            "onnx_parity_max_abs": worst, "history": history}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"best epoch {best[1]}, valid WP {best[0]:.5f}")


if __name__ == "__main__":
    main()
