#!/usr/bin/env python3
"""Train a diagonal multi-timescale state-space read-out on the raw 112 columns.

Every model tried so far - ridge, MLP, GRU, gradient-boosted trees - agrees
with every other at 0.92-0.97 residual correlation and explains ~18% of target
variance. What they share is short memory: EMA spans of 28, a 42-row lag,
512-step truncated BPTT, over sequences 20,000 rows long. The one long-context
feature that was tested (a span-1248 EMA) helped the linear read-out even
though the target decorrelates by lag ~200.

This layer is built to remove that limitation cheaply:

    h_t = a * h_{t-1} + (1 - a) * (B x_t)      a in (0,1)^N, learned per dim
    y_t = MLP([h_t, x_t])

`a` is diagonal, so the recurrence costs O(N) per row instead of the GRU's
O(N^2) - the reason a 128x2 GRU costs 21 us/row and 256x2 costs 53. Here a
1024-dimensional state costs about 10 us/row, and inference is two elementwise
NumPy operations plus two matvecs, so no ONNX runtime is involved.

Writing it as `a*h + (1-a)*u` makes each dimension an exponential moving
average of a learned projection, with its own span. Spans are initialised
log-spaced from 2 to 20,000 rows, so the model starts holding every timescale
from a few rows to a whole sequence and learns which to keep.

Training uses the fact that `a` is constant in time: the recurrence is then a
convolution with kernel `a^j`, evaluated by FFT over each truncated-BPTT
window rather than by a 512-step Python loop.

    python scripts/train_ssm.py --out datasets/experiments/ssm/ssm.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starterpack"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_gru import SequenceReader, batched, feature_stats, load_batch  # noqa: E402
from utils import weighted_pearson  # noqa: E402

N_FEATURES = 112


def scan_fft(u: torch.Tensor, a: torch.Tensor, h0: torch.Tensor | None) -> torch.Tensor:
    """h_t = a * h_{t-1} + u_t, evaluated as a convolution with kernel a^j.

    u: (batch, T, N); a: (N,); h0: (batch, N) or None. Returns (batch, T, N).
    """
    b, t, n = u.shape
    j = torch.arange(t, device=u.device, dtype=u.dtype)
    k = a.unsqueeze(0) ** j.unsqueeze(1)               # (T, N) kernel a^j
    length = 1
    while length < 2 * t:
        length *= 2
    uf = torch.fft.rfft(u.transpose(1, 2), n=length)    # (batch, N, L/2+1)
    kf = torch.fft.rfft(k.transpose(0, 1), n=length)    # (N, L/2+1)
    h = torch.fft.irfft(uf * kf.unsqueeze(0), n=length)[..., :t].transpose(1, 2)
    if h0 is not None:
        decay = a.unsqueeze(0) ** (j.unsqueeze(1) + 1.0)  # (T, N)
        h = h + h0.unsqueeze(1) * decay.unsqueeze(0)
    return h


class DiagonalSSM(nn.Module):
    def __init__(self, state: int, hidden: list[int], mu: np.ndarray, sigma: np.ndarray,
                 span_min: float = 2.0, span_max: float = 20000.0, dropout: float = 0.0):
        super().__init__()
        self.state = state
        # tau log-spaced over the requested span range; a = exp(-1/tau).
        tau = torch.logspace(math.log10(span_min), math.log10(span_max), state)
        self.log_tau = nn.Parameter(torch.log(tau))
        self.B = nn.Parameter(torch.randn(N_FEATURES, state) / math.sqrt(N_FEATURES))
        layers: list[nn.Module] = []
        d = state + N_FEATURES
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        self.head = nn.Sequential(*layers)
        self.register_buffer("mu", torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer("inv_sigma", torch.from_numpy((1.0 / sigma).astype(np.float32)))

    def decay(self) -> torch.Tensor:
        # tau >= 1 keeps a in (0, 1) and the recurrence contractive.
        return torch.exp(-1.0 / torch.exp(self.log_tau).clamp(min=1.0))

    def forward(self, x, h0=None):
        z = (x - self.mu) * self.inv_sigma
        a = self.decay()
        u = (z @ self.B) * (1.0 - a)
        h = scan_fft(u, a, h0)
        return self.head(torch.cat([h, z], dim=-1)), h[:, -1].detach()


@torch.no_grad()
def validate(model, reader, ids, batch, bptt):
    model.eval()
    ps, ys = [], []
    for chunk in batched(ids, batch):
        x, y, _, scored = load_batch(reader, chunk)
        X = torch.from_numpy(x)
        h = None
        out = np.empty_like(y)
        for s in range(0, X.shape[1], bptt):
            p, h = model(X[:, s:s + bptt], h)
            out[:, s:s + bptt] = p.numpy()
        ps.append(out[scored])
        ys.append(y[scored])
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    t0 = weighted_pearson(y[:, 0], p[:, 0])
    t1 = weighted_pearson(y[:, 1], p[:, 1])
    return t0, t1, 0.5 * (t0 + t1), len(y)


def save_npz(model: DiagonalSSM, path: Path, meta: dict) -> None:
    a = model.decay().detach().numpy().astype(np.float32)
    arrays = {
        "a": a,
        "B": (model.B.detach().numpy() * (1.0 - a)).astype(np.float32),  # fold (1-a) in
        "mu": model.mu.numpy().astype(np.float32),
        "inv_sigma": model.inv_sigma.numpy().astype(np.float32),
    }
    linears = [m for m in model.head if isinstance(m, nn.Linear)]
    arrays["n_layers"] = np.int64(len(linears))
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().numpy().T.astype(np.float32).copy()
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32).copy()
    np.savez(path, **arrays)
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="datasets/train.parquet")
    ap.add_argument("--valid", default="datasets/valid.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", type=int, default=1024)
    ap.add_argument("--hidden", default="128")
    ap.add_argument("--span-max", type=float, default=20000.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bptt", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--weight-power", type=float, default=0.75)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sequences", type=int, default=0)
    ap.add_argument("--valid-sequences", type=int, default=300)
    ap.add_argument("--stats-sequences", type=int, default=60)
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
    val_ids = list(range(min(a.valid_sequences, va.n)))
    print(f"train {n_train} of {tr.n} sequences, validation {len(val_ids)}", flush=True)

    mu, sigma = feature_stats(tr, a.stats_sequences, a.seed)
    hidden = [int(h) for h in a.hidden.split(",") if h]
    model = DiagonalSSM(a.state, hidden, mu, sigma, span_max=a.span_max, dropout=a.dropout)
    n_par = sum(p.numel() for p in model.parameters())
    spans = torch.exp(model.log_tau).detach()
    print(f"state {a.state}, head {hidden}, {n_par:,} parameters; "
          f"initial spans {spans.min():.1f} .. {spans.max():.0f} rows", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    windows = (20000 + a.bptt - 1) // a.bptt
    steps = a.epochs * ((n_train + a.batch - 1) // a.batch) * windows
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)

    rng = np.random.default_rng(a.seed)
    best = (-1.0, None, None)
    history = []
    ckpt = out.with_suffix(".pt")
    for epoch in range(a.epochs):
        model.train()
        order = list(rng.permutation(n_train))
        te = time.perf_counter()
        loss_sum, nb, seen = 0.0, 0, 0
        for bi, ids in enumerate(batched(order, a.batch)):
            x, y, need, _ = load_batch(tr, [int(i) for i in ids])
            X = torch.from_numpy(x)
            Y = torch.from_numpy(np.clip(y, -2.0, 2.0))
            M = torch.from_numpy(need & np.isfinite(y).all(axis=2))
            W = (Y.abs() ** a.weight_power) * M.unsqueeze(-1)
            h = None
            for s in range(0, X.shape[1], a.bptt):
                xb, yb, wb = X[:, s:s + a.bptt], Y[:, s:s + a.bptt], W[:, s:s + a.bptt]
                if wb.sum() <= 0:
                    with torch.no_grad():
                        _, h = model(xb, h)
                    continue
                wsum = wb.sum(dim=(0, 1)).clamp(min=1e-6)
                pred, h = model(xb, h)
                loss = (((pred - yb) ** 2 * wb).sum(dim=(0, 1)) / wsum).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if a.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                sched.step()
                loss_sum += float(loss)
                nb += 1
            seen += len(ids)
            if bi % 20 == 0:
                el = time.perf_counter() - te
                rate = seen * 20000 / max(el, 1e-6)
                print(f"  epoch {epoch + 1} seq {seen}/{n_train}  loss {loss_sum / max(nb,1):.5f}  "
                      f"{rate:,.0f} rows/s  eta {(n_train-seen)*20000/max(rate,1)/60:.0f} min", flush=True)
        v = validate(model, va, val_ids, a.batch, a.bptt)
        sp = torch.exp(model.log_tau).detach()
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(nb, 1), "valid_t0": v[0],
                        "valid_t1": v[1], "valid_wp": v[2], "span_min": float(sp.min()),
                        "span_max": float(sp.max()), "span_median": float(sp.median())})
        print(f"epoch {epoch + 1}/{a.epochs}: loss {loss_sum / max(nb,1):.5f}  valid WP {v[2]:.5f} "
              f"(t0 {v[0]:.5f}, t1 {v[1]:.5f})  spans {sp.min():.1f}/{sp.median():.0f}/{sp.max():.0f}  "
              f"{time.perf_counter() - te:.0f}s", flush=True)
        if v[2] > best[0]:
            best = (v[2], epoch + 1, {k: t.detach().clone() for k, t in model.state_dict().items()})
            torch.save({"state_dict": best[2], "state": a.state, "hidden": hidden,
                        "mu": mu, "sigma": sigma, "epoch": epoch + 1, "valid_wp": v[2]}, ckpt)
            print(f"  new best; checkpoint -> {ckpt}", flush=True)

    model.load_state_dict(best[2])
    save_npz(model, out, {"state": a.state, "hidden": hidden, "parameters": n_par,
                          "train_sequences": n_train, "epochs": a.epochs, "best_epoch": best[1],
                          "best_valid_wp": best[0], "bptt": a.bptt, "batch": a.batch, "lr": a.lr,
                          "weight_power": a.weight_power, "span_max": a.span_max, "seed": a.seed,
                          "history": history})
    print(f"saved {out} (best epoch {best[1]}, valid WP {best[0]:.5f})")


if __name__ == "__main__":
    main()
