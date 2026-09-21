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
| `ema_slow` | int | 30–2000 | span of the slow EMA |
| `raw_prices` | bool | | include raw price-like columns (book + trade prices) |
| `use_i1` | bool | | include instrument `i1` in every block |
| `use_dslow` | bool | | add the slow-EMA residual block |
| `use_diff1` | bool | | add the one-step difference block |
| `log10_lambda` | float | −5–1 | ridge strength (relative to the weight-normalised Gram) |
| `weight_power` | float | 0–2 (steps of ¼) | sample weight `|clip(y)|^p`; 1 = metric weights |

## Feature transform (`neutrino_wunder::features`)

```
step 0 : ema_fast = ema_slow = prev = x
step >0: ema_fast += a_fast * (x - ema_fast)        a = 2 / (span + 1)
         ema_slow += a_slow * (x - ema_slow)
phi     = [ x[raw_idx], (x - ema_fast)[dyn_idx],
            (x - ema_slow)[dyn_idx] if use_dslow, (x - prev)[dyn_idx] if use_diff1 ]
prev    = x
```

`raw_idx` always contains the volume-like and additional columns (`a0..a7`),
plus price-like columns when `raw_prices`; `dyn_idx` contains every column of
`i0` (+ `i1` when `use_i1`) plus `a0..a7`. Both index lists, the EMA alphas,
and the folded weights are written into `model.json`, so `solution.py` never
re-derives anything. Standardisation (`mu`, `sigma`) is folded into the
exported weights: `pred_k = phi · weights[k] + bias[k]`.

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

`solution.py` does per row: one `float64` cast, two EMA updates (3 NumPy ops
each), four `(112,) @ (112, 2)` mat-vecs, three vector subtractions. Measured
≈ 13 µs/row on this machine → ≈ 8.5 min for the 39.4 M-row test set (limit
60 min, 1 vCPU). The organisers' GRU baseline needs ≈ 42 µs/row here.

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
