# Neutrino as the solution engine

## Why a genetic algorithm here

Neutrino optimises trading strategies with the `evoforge` GA: a genome is a
vector of feature weights and risk parameters, fitness is a backtest metric.
The Alpha Connectome task is close to that shape — predict a future price move
of `i0` from streaming book/trade features — but its metric (clipped,
`|target|`-weighted, pooled Pearson) is not what a plain least-squares fit
maximises, and the most important choices are discrete: how long the EMA
memory is, whether instrument `i1` helps, whether raw price levels should be
visible to the model at all, how strongly to weight large-move rows.

So the split is:

* **closed form for what is differentiable** — given a feature layout, the
  linear read-out is a weighted ridge regression solved from the Gram matrix
  (`neutrino_wunder::ridge`, Cholesky). One candidate = two streaming passes
  over the GA training slice + an O(F³) solve; ~10 s for 80 sequences.
* **GA for what is not** — `neutrino_optimizer::run_search` (new, typed genes)
  evolves the layout / weighting hyper-parameters, fitness = Global WP of the
  fitted candidate on a held-out validation slice. Deterministic with `--seed`.

## Gene schema (`neutrino_wunder::search::schema`)

| gene | type | range | meaning |
|---|---|---|---|
| `ema_fast` | int | 2–60 | span of the fast EMA |
| `ema_mid` | int | 5–300 | span of the mid EMA |
| `ema_slow` | int | 30–2000 | span of the slow EMA |
| `raw_prices` | bool | | include raw price-like columns (book + trade prices) |
| `use_i1` | bool | | include instrument `i1` in every block |
| `use_dmid` | bool | | add the mid-EMA residual block |
| `use_dslow` | bool | | add the slow-EMA residual block |
| `use_diff` | bool | | add the lagged-difference block |
| `diff_lag` | int | 1–50 | lag of that block (1 = one-step difference) |
| `vol_norm` | bool | | divide the residual blocks by a running EMA of `|x - ema_fast|` |
| `vol_span` | int | 20–2000 | span of that volatility EMA |
| `use_imbalance` | bool | | add `(v_bid - v_ask) / (|v_bid| + |v_ask| + floor)` per book level |
| `log10_lambda` | float | −5–1 | ridge strength (relative to the weight-normalised Gram) |
| `weight_power` | float | 0–2 (steps of ¼) | sample weight `|clip(y)|^p`; 1 = metric weights |

## Feature transform (`neutrino_wunder::features`)

```
step 0 : ema_fast = ema_mid = ema_slow = x ; vol = 0 ; hist[*] = x
step >0: ema_* += a_* * (x - ema_*)                 a = 2 / (span + 1)
d_fast  = x - ema_fast
vol    += a_vol * (|d_fast| - vol)
scale   = max(vol, floor)  if vol_norm else 1        (per column)
lagged  = hist[step mod lag]                          (row step-lag, or row 0 early on)
phi     = [ x[raw_idx], (d_fast/scale)[dyn_idx],
            ((x-ema_mid)/scale)[dyn_idx] if use_dmid, ((x-ema_slow)/scale)[dyn_idx] if use_dslow,
            ((x-lagged)/scale)[dyn_idx] if use_diff,
            (v_bid-v_ask)/(|v_bid|+|v_ask|+floor) per level if use_imbalance ]
hist[step mod lag] = x
```

`raw_idx` always contains the volume-like and additional columns (`a0..a7`),
plus price-like columns when `raw_prices`; `dyn_idx` contains every column of
`i0` (+ `i1` when `use_i1`) plus `a0..a7`. `floor` is 1e-3 × the standard
deviation of each raw column over the first training chunk. Index lists,
alphas, floors and the folded weights are written into `model.json`, so
`solution.py` never re-derives anything. Standardisation (`mu`, `sigma`) is
folded into the exported weights: `pred_k = phi · weights[k] + bias[k]`.

## Training procedure (`neutrino-wunder train`)

1. Load `--ga-train-sequences` from the head of the train file and
   `--ga-holdout-sequences` from the head of `valid.parquet` into memory.
2. GA: population × generations candidates; each is fitted on the train slice
   and scored (with `is_scored` mask) on the hold-out slice. Identical decoded
   candidates are cached.
3. Refit the champion on `--final-train-sequences` (0 = whole file), streaming
   from disk in chunks of 32 sequences (two passes: moments, then Gram).
4. Report Global WP on the whole validation set **and** on the validation
   sequences the GA never saw (`valid_excluding_ga_holdout` in `model.json`).
5. Write `model.json` (+ `--report` with every trial).

`neutrino-wunder score` re-scores a model with the Rust metric;
`neutrino-wunder predict` dumps per-row predictions for the parity test.

## Inference budget

`solution.py` does per row: one `float64` cast, three EMA updates and the
volatility EMA (3–4 NumPy ops each), one `(112,) @ (112, 2)` mat-vec per
active block. Measured 11–27 µs/row on this machine depending on how many
blocks the champion enabled (18 µs for the shipped one) → 7–18 min for the
39.4 M-row test set (limit 60 min, 1 vCPU). The organisers' GRU baseline
needs ≈ 24 µs/row here with I/O binding (40 µs with plain `session.run`), so
the shipped blend runs at ≈ 48 µs/row, ≈ 31 min projected.

## Blending with the GRU baseline (step 1)

The linear read-out and the organisers' GRU are complementary (linear wins
on `t0`, GRU on `t1`). `scripts/predict_valid.py` caches each model's scored
validation predictions, `scripts/blend.py` picks per-target weights on the
first half of the sequences and reports the second half. `solution.py` reads
`blend.json` and mixes the two per target.

## Where to go during the hackathon

* **More data.** The GA slice is 80 sequences; the final refit uses the
  1,000-sequence subset. `FULL_TRAIN=1 scripts/setup_env.sh --all` and
  `FINAL_TRAIN_SEQUENCES=0` use all 10,607 sequences — the fit streams, so
  memory stays flat; only time grows (~1.5 min per 1,000 sequences here).
* **Richer genes.** Add a third EMA, rolling volatility normalisation, or
  per-group on/off flags: extend `FeatureSpec`, `FeatureLayout::new`,
  `StreamState::push`, `search::schema/decode` — and mirror the recurrence in
  `solution.py` (`scripts/parity_test.py` guards the mirror).
* **Non-linear read-out.** Keep the GA + features, replace the ridge with a
  small MLP or a GRU trained in PyTorch, exported to ONNX (the scorer image has
  `onnxruntime`); or blend `solution.py` with the baseline GRU — both are
  stateful and cheap.
* **Validation hygiene.** The public validation mask is only a sample of
  moments (`starterpack/docs/faq.md`). Score with `--offset` past the GA
  hold-out, and try your own masks, before trusting a gain.
