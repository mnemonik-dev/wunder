# MLP read-out + 3-way blend — 2026-09-22

## Motivation

The full-data refit (`docs/FULL_DATA_RETRAIN.md`) showed the ridge-linear
read-out is capacity-limited, not data-limited. This experiment adds a
non-linear read-out (PyTorch MLP, exported to NumPy) over the champion's
336 streaming features and blends it with the existing linear and GRU models.

## Training

- Features: `neutrino-wunder features` export of the champion spec
  (`solution/model.json`): train = first 2,000 training sequences at stride 12
  (3,318,000 rows); valid = scored rows only (3,528,273 rows).
  Exports kept at `datasets/mlp/*.f32` (git-ignored).
- `scripts/train_mlp.py --hidden 256,64 --dropout 0.1 --input-noise 0.05
  --epochs 8 --batch 4096 --lr 1e-3 --weight-decay 1e-5 --seed 42`
  (weighted MSE, |clip(y)|^0.75 weights, best-epoch selection by global WP).
- Result: **MLP alone valid WP 0.65555** (best epoch 7) vs shipped linear
  0.63314. Raw prediction cache: `datasets/experiments/full-refit-20260922/preds_mlp.npz`.

## Blending (weights tuned on first half of validation sequences only)

`scripts/blend.py` 3-way over linear + MLP + GRU caches (WLS + local WP
refinement):

| Model | Full WP | Second-half WP |
|---|---:|---:|
| Shipped blend (linear+GRU) | 0.66633 | 0.66800 |
| **3-way blend** | **0.67387** | **0.67468** |
| 4-way (+ GA champion linear) | 0.67369 | 0.67429 (rejected) |

Paired sequence bootstrap vs shipped blend (2,000 replicates, seed 42,
`datasets/experiments/mlp-20260922/3way-vs-shipped.json`): full delta
+0.00754 CI [+0.00534, +0.00927]; second-half delta +0.00669 CI
[+0.00310, +0.00932] — passes the promotion gate.

Final weights (per target t0/t1): linear 0.150/0.070, MLP 0.626/0.688,
GRU 0.224/0.242.

## Verification and packaging

- Candidate dir `datasets/experiments/mlp-20260922/candidate/`; Rust/Python
  parity exact (max |diff| = 0 over 39,802 predictions).
- Re-cached candidate reproduces the offline blend: WP 0.673868, max per-row
  difference 4.8e-07 (float32 rounding).
- MLP non-finite-output scan over all 3,528,273 scored validation rows: 0
  affected rows.
- `submissions/submission-20260922-mlp3way.zip` (1.13 MB, 5 files incl.
  `mlp.npz`), preflight RESULT: OK, deterministic, 43.3 µs/row → ~28 min
  projected test-set time (60-min limit). Receipt alongside.

## Artifacts

- `datasets/experiments/mlp-20260922/`: `mlp.npz` (+ `.json` training meta),
  `blend-3way.json`, `pred/` and `candidate/` inference dirs, `preds_candidate.npz`.
- `datasets/experiments/full-refit-20260922/preds_mlp.npz`, `preds_ga.npz`,
  `preds_linear_shipped.npz`, `preds_3way_blend.npz`.
- GA re-search of the same day (new spec, full-data fit, valid WP 0.63258)
  did not beat the shipped linear; its 4-way contribution was rejected.
