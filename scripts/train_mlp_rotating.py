#!/usr/bin/env python3
"""Train the MLP read-out on more rows than the disk can hold at once.

`train_mlp_streaming.py` keeps one exported chunk resident but still needs
every chunk on disk, which caps training at whatever fits - 13.3M rows here,
while the measured curve is still paying +0.0108 WP per 4x rows. Exporting is
cheap (about 40 s per 3.3M rows) compared with a training pass, so this
regenerates each chunk on demand instead of storing them: export, train,
delete, next. Peak disk is one chunk.

That also makes `--stride` a free parameter: stride 6 over all 10,607
sequences is 35M rows, 2.6x the current training set, with no extra disk and
no change to inference cost.

    python scripts/train_mlp_rotating.py --model solution/model.json \
        --valid datasets/mlp/valid.f32 --out datasets/experiments/mlp_big.npz
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_mlp_streaming import MLP  # noqa: E402
from utils import weighted_pearson  # noqa: E402

NW = ROOT.parent / "neutrino" / "target" / "release" / "neutrino-wunder"


def export_chunk(model: str, data: str, offset: int, sequences: int, stride: int,
                 out: Path, threads: int) -> dict:
    cmd = [str(NW), "--threads", str(threads), "features", "--model", model, "--data", data,
           "--sequences", str(sequences), "--offset", str(offset), "--stride", str(stride),
           "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise SystemExit(f"feature export failed (offset {offset}):\n{r.stderr[-2000:]}")
    return json.load(open(str(out) + ".json"))


def load_matrix(path: Path, side: dict, f: int, mu, sigma, power: float):
    m = np.memmap(path, dtype=np.float32, mode="r", shape=(side["rows"], side["columns"]))
    x = np.array(m[:, :f], dtype=np.float32)
    x -= mu
    x /= sigma
    y = np.clip(np.array(m[:, f:f + 2], dtype=np.float32), -2.0, 2.0)
    w = np.abs(y) ** power if power > 0 else np.ones_like(y)
    del m
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="solution/model.json", help="feature layout to export")
    ap.add_argument("--train", default="datasets/train.parquet")
    ap.add_argument("--valid", required=True, help="pre-exported validation matrix (.f32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default="datasets/mlp/rotate")
    ap.add_argument("--chunk-sequences", type=int, default=1000)
    ap.add_argument("--total-sequences", type=int, default=0, help="0 = whole train file")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--valid-stride", type=int, default=5)
    ap.add_argument("--hidden", default="512,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.75)
    ap.add_argument("--input-noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--keep-chunks", action="store_true", help="do not delete chunks after use")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)

    total = a.total_sequences
    if total <= 0:
        import pyarrow.parquet as pq
        total = pq.ParquetFile(a.train).metadata.num_row_groups
    offsets = list(range(0, total, a.chunk_sequences))
    rows_per_chunk = a.chunk_sequences * 19901 // a.stride
    print(f"{len(offsets)} chunks x ~{rows_per_chunk:,} rows = ~{len(offsets) * rows_per_chunk / 1e6:.1f}M rows "
          f"per epoch (stride {a.stride}, {total} sequences)", flush=True)
    free_gb = shutil.disk_usage(work).free / 1e9
    print(f"peak disk: one chunk at a time; {free_gb:.1f} GB free", flush=True)

    vside = json.load(open(a.valid + ".json"))
    f = vside["features"]
    mu = np.asarray(vside["mu"], dtype=np.float32)
    sigma = np.asarray(vside["sigma"], dtype=np.float32)
    va = np.memmap(a.valid, dtype=np.float32, mode="r", shape=(vside["rows"], vside["columns"]))
    xv = (np.array(va[::a.valid_stride, :f], dtype=np.float32) - mu) / sigma
    yv = np.array(va[::a.valid_stride, f:f + 2], dtype=np.float32)
    XV = torch.from_numpy(np.ascontiguousarray(xv))
    del xv, va
    print(f"validation {len(XV):,} rows x {f} features", flush=True)

    hidden = [int(h) for h in a.hidden.split(",") if h]
    model = MLP(f, hidden, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * len(offsets) * ((rows_per_chunk + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    print(f"hidden {hidden}, {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"~{steps:,} optimiser steps", flush=True)

    def evaluate():
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(XV), 131072):
                outs.append(model(XV[i:i + 131072]).numpy())
        p = np.concatenate(outs)
        s0 = weighted_pearson(yv[:, 0], p[:, 0])
        s1 = weighted_pearson(yv[:, 1], p[:, 1])
        return s0, s1, 0.5 * (s0 + s1)

    rng = np.random.default_rng(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    best = (-1.0, None, None)
    history = []
    mean_w = None
    for epoch in range(a.epochs):
        te = time.perf_counter()
        order = list(rng.permutation(offsets))
        loss_sum, nb, seen, exp_s = 0.0, 0, 0, 0.0
        for ci, off in enumerate(order):
            path = work / f"chunk_{off}.f32"
            t0 = time.perf_counter()
            side = export_chunk(a.model, a.train, int(off), a.chunk_sequences, a.stride, path, a.threads)
            exp_s += time.perf_counter() - t0
            if side["features"] != f:
                raise SystemExit(f"--model exports {side['features']} features, validation has {f}")
            X, Y, W = load_matrix(path, side, f, mu, sigma, a.weight_power)
            if mean_w is None:
                mean_w = W.mean(dim=0)
                print(f"loss normalisation mean|y|^{a.weight_power} = {mean_w.tolist()}", flush=True)
            n = len(X)
            model.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, a.batch):
                idx = perm[i:i + a.batch]
                xb = X[idx]
                if a.input_noise > 0:
                    xb = xb + a.input_noise * torch.randn(xb.shape, generator=g)
                loss = ((((model(xb) - Y[idx]) ** 2 * W[idx]).mean(dim=0)) / mean_w).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += float(loss)
                nb += 1
            seen += n
            del X, Y, W, perm
            if not a.keep_chunks:
                path.unlink(missing_ok=True)
                Path(str(path) + ".json").unlink(missing_ok=True)
            if (ci + 1) % 3 == 0 or ci + 1 == len(order):
                el = time.perf_counter() - te
                print(f"  epoch {epoch + 1} chunk {ci + 1}/{len(order)}  {seen:,} rows  "
                      f"loss {loss_sum / max(nb, 1):.5f}  export {100 * exp_s / max(el, 1e-6):.0f}% of time  "
                      f"eta {(len(order) - ci - 1) * el / (ci + 1) / 60:.0f} min", flush=True)
        v = evaluate()
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(nb, 1), "rows": seen,
                        "valid_t0": v[0], "valid_t1": v[1], "valid_wp": v[2]})
        print(f"epoch {epoch + 1}/{a.epochs}: {seen:,} rows  loss {loss_sum / max(nb, 1):.5f}  "
              f"valid WP {v[2]:.5f} (t0 {v[0]:.5f}, t1 {v[1]:.5f})  {time.perf_counter() - te:.0f}s", flush=True)
        if v[2] > best[0]:
            best = (v[2], epoch + 1, {k: t.detach().clone() for k, t in model.state_dict().items()})
            print(f"  new best", flush=True)

    model.load_state_dict(best[2])
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays = {"mu": mu, "sigma": sigma, "keep": np.arange(f, dtype=np.int64),
              "n_layers": np.int64(len(linears))}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().numpy().T.astype(np.float32).copy()
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32).copy()
    np.savez(a.out, **arrays)
    meta = {"model": a.model, "stride": a.stride, "chunk_sequences": a.chunk_sequences,
            "total_sequences": total, "rows_per_epoch": history[-1]["rows"] if history else 0,
            "features": f, "hidden": hidden, "epochs": a.epochs, "best_epoch": best[1],
            "best_valid_wp": best[0], "weight_power": a.weight_power, "lr": a.lr,
            "dropout": a.dropout, "batch": a.batch, "seed": a.seed, "history": history}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {a.out} (best epoch {best[1]}, valid WP {best[0]:.5f})")


if __name__ == "__main__":
    main()
