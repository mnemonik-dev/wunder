#!/usr/bin/env python3
"""Train a small MLP read-out on the champion's streaming features (step 3).

Input: the float32 matrices written by `neutrino-wunder features`
(rows = [phi, t0, t1, scored]). Features are standardised with the
champion's mu/sigma (from the sidecar JSON), targets are clipped to the
metric range, and the loss is the |clip(y)|^p-weighted MSE that the ridge
read-out also minimises. Model selection is by Global WP on the validation
export. Weights are saved as NumPy arrays for `solution.py` (no torch at
inference time).

    python scripts/train_mlp.py --train datasets/mlp/train.f32 --valid datasets/mlp/valid.f32 --out solution/mlp.npz
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


def load(path: str):
    side = json.load(open(path + ".json"))
    m = np.memmap(path, dtype=np.float32, mode="r", shape=(side["rows"], side["columns"]))
    return m, side


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


def wp_of(model, z: torch.Tensor, y: np.ndarray, batch: int = 65536) -> tuple[float, float, float]:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(z), batch):
            outs.append(model(z[i:i + batch]).numpy())
    p = np.concatenate(outs)
    t0 = weighted_pearson(y[:, 0], p[:, 0])
    t1 = weighted_pearson(y[:, 1], p[:, 1])
    return t0, t1, 0.5 * (t0 + t1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default="256,64")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--weight-power", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=0, help="use only the first N training rows (0 = all)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    tr, side = load(a.train)
    va, vside = load(a.valid)
    f = side["features"]
    assert vside["features"] == f
    mu = np.asarray(side["mu"], dtype=np.float32)
    sigma = np.asarray(side["sigma"], dtype=np.float32)
    n = tr.shape[0] if a.max_rows <= 0 else min(tr.shape[0], a.max_rows)
    t0 = time.perf_counter()
    print(f"loading {n:,} training rows x {f} features ...", flush=True)
    x = np.array(tr[:n, :f], dtype=np.float32)  # copy: memmap is read-only
    x -= mu
    x /= sigma
    y = np.clip(np.array(tr[:n, f:f + 2], dtype=np.float32), -2.0, 2.0)
    w = np.abs(y) ** a.weight_power if a.weight_power > 0 else np.ones_like(y)
    xv = (np.array(va[:, :f], dtype=np.float32) - mu) / sigma
    yv = np.array(va[:, f:f + 2], dtype=np.float32)
    print(f"loaded in {time.perf_counter() - t0:.0f}s; validation rows {len(xv):,} (scored only)", flush=True)

    X = torch.from_numpy(x)
    Y = torch.from_numpy(y)
    W = torch.from_numpy(w.astype(np.float32))
    XV = torch.from_numpy(xv)
    hidden = [int(h) for h in a.hidden.split(",") if h]
    model = MLP(f, hidden, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = (n + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.15)
    wsum = W.sum(dim=0)

    best = (-1.0, None, None)
    history = []
    g = torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        t0 = time.perf_counter()
        loss_sum = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            pred = model(X[idx])
            loss = ((pred - Y[idx]) ** 2 * W[idx]).sum(dim=0) / wsum * (n / len(idx))
            loss = loss.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
        v = wp_of(model, XV, yv)
        history.append({"epoch": epoch + 1, "loss": loss_sum / steps_per_epoch, "valid_t0": v[0], "valid_t1": v[1], "valid_wp": v[2]})
        print(f"epoch {epoch + 1}/{a.epochs}: loss {loss_sum / steps_per_epoch:.5f}  valid WP {v[2]:.5f} "
              f"(t0 {v[0]:.5f}, t1 {v[1]:.5f})  {time.perf_counter() - t0:.0f}s", flush=True)
        if v[2] > best[0]:
            best = (v[2], epoch + 1, {k: t.detach().clone() for k, t in model.state_dict().items()})

    model.load_state_dict(best[2])
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    arrays = {"mu": mu, "sigma": sigma, "n_layers": np.int64(len(linears))}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().numpy().T.astype(np.float32).copy()  # (in, out)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32).copy()
    np.savez(a.out, **arrays)
    meta = {"hidden": hidden, "epochs": a.epochs, "best_epoch": best[1], "best_valid_wp": best[0],
            "train_rows": n, "train_sequences": side["sequences"], "stride": side["stride"],
            "valid_rows": int(len(xv)), "weight_power": a.weight_power, "lr": a.lr,
            "weight_decay": a.weight_decay, "dropout": a.dropout, "batch": a.batch, "seed": a.seed,
            "history": history}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {a.out} (best epoch {best[1]}, valid WP {best[0]:.5f})")


if __name__ == "__main__":
    main()
