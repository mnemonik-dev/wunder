#!/usr/bin/env python3
"""Train the MLP read-out over several feature exports, one chunk in RAM at a time.

`train_mlp.py` loads a single `neutrino-wunder features` export into memory, so
the training-set size is capped by RAM (and, before that, by the free disk the
export needs). This trainer takes a list of exports and keeps only one resident:
an epoch is one pass over every chunk in turn. Training-set size is then bounded
by how many chunks can be produced in rotation, not by how many fit at once.

Chunks must come from the same `model.json` (same feature layout, mu and sigma);
the sidecars are checked. Everything else - the weighted-MSE loss, the feature
standardisation, model selection by Global WP on the validation export, and the
saved weight format - matches `train_mlp.py`, so results are comparable and the
output `.npz` is read by `solution.py` unchanged.

    python scripts/train_mlp_streaming.py \
        --chunks datasets/mlp/train.f32,datasets/mlp/chunks/c2000.f32 \
        --valid datasets/mlp/valid.f32 --out solution/mlp.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
from utils import weighted_pearson  # noqa: E402


def sidecar(path: str) -> dict:
    return json.load(open(path + ".json"))


def memmap(path: str, side: dict) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode="r", shape=(side["rows"], side["columns"]))


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


JOINT = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])  # (u, v) -> (t0, t1) = (v+u, v-u)


def to_targets(raw: torch.Tensor, head: str) -> torch.Tensor:
    """Map the network's two outputs onto (t0, t1)."""
    return raw if head == "plain" else raw @ JOINT.T


def weighted_pearson_loss(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Negative mean weighted Pearson over the batch, per target.

    The competition metric is a correlation, which is invariant to affine
    rescaling of the prediction; squared error is not, and spends capacity on
    a scale the metric ignores.
    """
    total = 0.0
    for k in (0, 1):
        m = w[:, k]
        s = m.sum().clamp(min=1e-6)
        pm = (pred[:, k] * m).sum() / s
        ym = (y[:, k] * m).sum() / s
        cp, cy = pred[:, k] - pm, y[:, k] - ym
        num = (m * cp * cy).sum()
        den = torch.sqrt((m * cp * cp).sum() * (m * cy * cy).sum() + 1e-12)
        total = total + num / den
    return -total / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True, help="comma-separated feature exports (one is resident at a time)")
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid-stride", type=int, default=1, help="evaluate on every N-th scored validation row")
    ap.add_argument("--hidden", default="256,64")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.75)
    ap.add_argument("--input-noise", type=float, default=0.05)
    ap.add_argument("--loss", choices=("mse", "corr"), default="mse",
                    help="mse: |clip(y)|^p-weighted squared error. "
                         "corr: batch weighted Pearson, which is what the metric actually measures")
    ap.add_argument("--head", choices=("plain", "joint"), default="plain",
                    help="plain: two outputs, one per target. "
                         "joint: predict (t0-t1)/2 and (t0+t1)/2, then reconstruct - t0 and t1 "
                         "correlate -0.738, so the rotated pair separates a high-variance "
                         "component from a low-variance one")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    paths = [p for p in a.chunks.split(",") if p]
    sides = [sidecar(p) for p in paths]
    f = sides[0]["features"]
    for p, s in zip(paths[1:], sides[1:]):
        if s["features"] != f or s["feature_names"] != sides[0]["feature_names"]:
            raise SystemExit(f"{p}: feature layout differs from {paths[0]}")
        if s["mu"] != sides[0]["mu"] or s["sigma"] != sides[0]["sigma"]:
            raise SystemExit(f"{p}: mu/sigma differ from {paths[0]} (different model.json?)")
    mu = np.asarray(sides[0]["mu"], dtype=np.float32)
    sigma = np.asarray(sides[0]["sigma"], dtype=np.float32)
    rows = [s["rows"] for s in sides]
    total = sum(rows)
    print(f"{len(paths)} chunk(s), {total:,} training rows x {f} features "
          f"({', '.join(f'{r:,}' for r in rows)})", flush=True)

    vside = sidecar(a.valid)
    if vside["features"] != f:
        raise SystemExit("validation export has a different feature layout")
    t = time.perf_counter()
    va = memmap(a.valid, vside)
    xv = (np.array(va[::a.valid_stride, :f], dtype=np.float32) - mu) / sigma
    yv = np.array(va[::a.valid_stride, f:f + 2], dtype=np.float32)
    XV = torch.from_numpy(np.ascontiguousarray(xv))
    del xv, va
    print(f"validation rows {len(XV):,} (every {a.valid_stride}) loaded in {time.perf_counter() - t:.0f}s", flush=True)

    hidden = [int(h) for h in a.hidden.split(",") if h]
    model = MLP(f, hidden, a.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * sum((r + a.batch - 1) // a.batch for r in rows)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    print(f"hidden {hidden}, {n_params:,} parameters, {steps:,} optimiser steps", flush=True)
    g = torch.Generator().manual_seed(a.seed)

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(XV), 131072):
                outs.append(to_targets(model(XV[i:i + 131072]), a.head).numpy())
        p = np.concatenate(outs)
        s0 = weighted_pearson(yv[:, 0], p[:, 0])
        s1 = weighted_pearson(yv[:, 1], p[:, 1])
        return s0, s1, 0.5 * (s0 + s1)

    def load_chunk(i: int):
        m = memmap(paths[i], sides[i])
        n = sides[i]["rows"]
        x = np.array(m[:, :f], dtype=np.float32)
        x -= mu
        x /= sigma
        y = np.clip(np.array(m[:, f:f + 2], dtype=np.float32), -2.0, 2.0)
        w = np.abs(y) ** a.weight_power if a.weight_power > 0 else np.ones_like(y)
        del m
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w), n

    # Loss scale follows train_mlp.py: per target, weighted MSE divided by the
    # mean sample weight. Fixed from the first chunk so it is epoch-independent.
    Wm_cache = {}
    _, _, w0, _ = load_chunk(0)
    mean_w = w0.mean(dim=0)
    del w0
    print(f"loss normalisation mean|y|^{a.weight_power} = {mean_w.tolist()}", flush=True)

    best = (-1.0, None, None)
    history = []
    for epoch in range(a.epochs):
        te = time.perf_counter()
        seen = 0
        loss_acc = 0.0
        nb = 0
        for ci in range(len(paths)):
            tl = time.perf_counter()
            X, Y, W, n = load_chunk(ci)
            Wm = Y.abs()  # metric weights |clip(y)|, used by the correlation loss
            load_s = time.perf_counter() - tl
            model.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, a.batch):
                idx = perm[i:i + a.batch]
                xb = X[idx]
                if a.input_noise > 0:
                    xb = xb + a.input_noise * torch.randn(xb.shape, generator=g)
                out = to_targets(model(xb), a.head)
                if a.loss == "corr":
                    loss = weighted_pearson_loss(out, Y[idx], Wm[idx])
                else:
                    loss = (((out - Y[idx]) ** 2 * W[idx]).mean(dim=0) / mean_w).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                loss_acc += loss.item()
                nb += 1
            seen += n
            del X, Y, W, perm
            print(f"  epoch {epoch + 1} chunk {ci + 1}/{len(paths)}: {n:,} rows "
                  f"(load {load_s:.0f}s, train {time.perf_counter() - tl - load_s:.0f}s)", flush=True)
        v = evaluate()
        history.append({"epoch": epoch + 1, "loss": loss_acc / max(nb, 1),
                        "valid_t0": v[0], "valid_t1": v[1], "valid_wp": v[2]})
        print(f"epoch {epoch + 1}/{a.epochs}: loss {loss_acc / max(nb, 1):.5f}  "
              f"valid WP {v[2]:.5f} (t0 {v[0]:.5f}, t1 {v[1]:.5f})  "
              f"{seen:,} rows  {time.perf_counter() - te:.0f}s", flush=True)
        if v[2] > best[0]:
            best = (v[2], epoch + 1, {k: t_.detach().clone() for k, t_ in model.state_dict().items()})

    model.load_state_dict(best[2])
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays = {"mu": mu, "sigma": sigma, "keep": np.arange(f, dtype=np.int64),
              "n_layers": np.int64(len(linears))}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().numpy().T.astype(np.float32).copy()  # (in, out)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32).copy()
    np.savez(a.out, **arrays)
    meta = {"chunks": paths, "chunk_rows": rows, "train_rows": total, "features": f,
            "hidden": hidden, "parameters": n_params, "epochs": a.epochs,
            "best_epoch": best[1], "best_valid_wp": best[0],
            "valid_rows": int(len(XV)), "valid_stride": a.valid_stride,
            "weight_power": a.weight_power, "lr": a.lr, "weight_decay": a.weight_decay,
            "dropout": a.dropout, "batch": a.batch, "seed": a.seed,
            "input_noise": a.input_noise, "loss": a.loss, "head": a.head,
            "history": history}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {a.out} (best epoch {best[1]}, valid WP {best[0]:.5f})")


if __name__ == "__main__":
    main()
