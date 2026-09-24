# Runbook: iterating on the solution

Commands are run from `wunder/` with `.venv` installed. `NW` is the Rust
trainer; export it once per shell:

```bash
export NW=../neutrino/target/release/neutrino-wunder
```

Measured reference points (full validation, 3,528,273 scored rows):

| model | WP |
|---|---:|
| organisers' GRU (`baseline.onnx`) | 0.61705 |
| linear read-out, champion spec (`solution/model.json`) | 0.63314 |
| linear read-out, + mid/slow EMA blocks | 0.63511 |
| MLP, 13.3M rows, 512x128 | 0.66743 |
| our GRU, 128x2, 211M rows | 0.67024 |
| 3-way blend (linear + MLP + organisers' GRU) | 0.67742 |
| 4-way blend (+ our GRU) | **0.67971** |

## 0. Screen a feature idea (seconds)

Never spend a training run on an unscreened idea. Add a family to
`FAMILIES` in `scripts/feature_lab.py`, then:

```bash
.venv/bin/python scripts/feature_lab.py --list

# Quick pass. Cheap, but biased against families that add many columns:
# a ridge on few sequences overfits them.
.venv/bin/python scripts/feature_lab.py --ablate \
  --train-sequences 60 --valid-sequences 60 --stride 8 \
  --families vol_norm,multi_lag,sentinel

# Decide on this one. Small-sample overfitting is largely gone by here.
.venv/bin/python scripts/feature_lab.py --ablate \
  --train-sequences 250 --valid-sequences 100 --stride 5 \
  --families vol_norm,multi_lag
```

A family that adds > ~200 columns can still read negative at 250 sequences
and positive at 1,000 (`ema_multi` did: -0.042 -> -0.008 -> +0.002). Confirm
those in step 1 rather than believing the lab.

## 1. Confirm a feature spec at full scale (~5 min)

`--no-ga` fits the spec given on the command line, skipping the search. The
GA's default 60-sequence fitness holdout carries about +-0.03 of noise against
effects worth +-0.003, so confirm any block choice here at 1,000 sequences
rather than trusting the search.

Measured at 1,000 sequences on full validation:

| spec | WP |
|---|---:|
| champion | 0.63314 |
| + mid-EMA only | 0.63421 |
| + mid and slow EMA (**best**) | 0.63511 |
| + vol_norm | 0.60538 |
| + mid, slow and vol_norm | 0.62257 |

`--vol-norm` is strongly negative and the GA was right to reject it; the
feature lab disagreed because it implements a different transform than the
flag. Only the mid/slow blocks are worth adding, and they need >=1,000
training sequences before they stop overfitting the ridge.

```bash
$NW --threads 10 train --no-ga \
  --train datasets/train.parquet --valid datasets/valid.parquet \
  --ga-train-sequences 10 --ga-holdout-sequences 10 \
  --final-train-sequences 1000 --eval-sequences 0 \
  --ema-fast 28 --ema-mid 150 --ema-slow 1248 \
  --raw-prices --dmid --diff-lag 42 \
  --lambda 3.0e-4 --weight-power 0.75 \
  --out solution/model_v2.json
```

The printed `validation WP` is on the whole validation file and is directly
comparable to the table above. Flags: `--dmid` adds the mid-EMA block,
omitting `--no-dslow` keeps the slow one, `--vol-norm` divides residuals by a
running scale, `--imbalance` adds the per-level volume imbalance.

## 2. Free disk before exporting (only for the MLP path)

A 560-feature export is 1.66x the size of the current 336-feature one:
about 7.5 GB per 2,000 sequences at `--stride 12`. Check first:

```bash
df -h .
du -sh datasets/mlp datasets/mlp/chunks
```

Reclaimable elsewhere on this machine, all rebuildable:

```bash
du -sh ~/src/*/*/target 2>/dev/null | sort -rh | head
# e.g. cargo clean --manifest-path ~/src/sessions/monorepo/Cargo.toml
rm -rf ~/Library/Caches/Homebrew ~/Library/Caches/Yarn ~/.cache/uv
```

The GRU path (step 4) reads `train.parquet` directly and needs no disk.

## 3. Export features and retrain the MLP

```bash
# One chunk per 2,000 sequences. Raise --stride to trade rows for disk.
for off in 0 2000 4000 6000; do
  $NW --threads 10 features --model solution/model_v2.json \
    --data datasets/train.parquet --sequences 2000 --offset $off \
    --stride 12 --out datasets/mlp/chunks/v2_$off.f32
done

# Validation export: every scored row, stride 1.
$NW --threads 10 features --model solution/model_v2.json \
  --data datasets/valid.parquet --stride 1 --scored-only \
  --out datasets/mlp/valid_v2.f32

# One chunk is resident at a time, so training size is not capped by RAM.
.venv/bin/python -u scripts/train_mlp_streaming.py \
  --chunks datasets/mlp/chunks/v2_0.f32,datasets/mlp/chunks/v2_2000.f32,datasets/mlp/chunks/v2_4000.f32,datasets/mlp/chunks/v2_6000.f32 \
  --valid datasets/mlp/valid_v2.f32 --valid-stride 5 \
  --epochs 6 --hidden 512,128 --out datasets/experiments/mlp_v2.npz
```

Measured: **+0.0108 WP per 4x rows when the rows are new sequences**, but only
+0.0013 for 2.6x rows obtained by halving `--stride` -- and that gain did not
survive to the public leaderboard. Within-sequence rows are heavily
autocorrelated and add close to nothing, so `--stride 12` over all 10,607
sequences is the sensible setting; lowering it mostly buys overfitting.

Width is flat from 103k to 608k parameters at 3.3M and 13.3M rows (608k
overfits at 13.3M), so scale sequences, not the model.

## 4. Train the GRU (no export, no disk)

Each Parquet row group is one sequence, so training streams from the file:
~211M rows per epoch, ~50 min at 128x2.

```bash
.venv/bin/python -u scripts/train_gru.py \
  --train datasets/train.parquet --valid datasets/valid.parquet \
  --hidden 128 --layers 2 --epochs 6 --batch 32 --bptt 512 \
  --lr 2e-3 --weight-power 0.75 --valid-sequences 300 \
  --out datasets/experiments/gru_v2.onnx
```

`--layers` must stay 2 to match the ONNX signature `solution.py` binds.
`--hidden` may change freely; the width is read from the graph at load.
Epochs 5 and 6 of the 128x2 run agreed to 1e-5, so that width is converged --
more epochs will not help it, but a wider model might.

## 5. Cache predictions, blend, package

```bash
# Full-validation predictions, in the layout scripts/blend.py expects.
# --align-to refuses to write if the rows do not match an existing cache.
.venv/bin/python -u scripts/predict_gru.py \
  --checkpoint datasets/experiments/gru_v2.pt \
  --out datasets/experiments/preds_gru_v2.npz \
  --align-to datasets/experiments/full-refit-20260922/preds_gru.npz

# Is a new model actually adding anything, or repeating an existing one?
.venv/bin/python scripts/diversity_report.py \
  --candidate datasets/experiments/preds_gru_v2.npz \
  --against mlp=<preds_mlp>.npz,gru=<preds_gru>.npz
# NOTE: this metric is confounded by prediction scale and currently
# overstates diversity for models whose output std differs from the target's.
# Rescale predictions to the target's weighted std before trusting it.

# Weights are fitted on the first half of the sequences and reported on the
# second, so the headline number is not the one they were chosen on.
.venv/bin/python scripts/blend.py --names linear,mlp,gru_base,gru_ours \
  <preds_linear>.npz <preds_mlp>.npz <preds_gru_base>.npz <preds_gru_ours>.npz \
  --out datasets/experiments/blend_v2.json

# Assemble: solution.py, model.json, mlp.npz, every .onnx named in the blend,
# and blend.json carrying `gru_models` mapping weight name -> onnx file.
scripts/package_submission.sh datasets/experiments/<candidate> \
  submissions/submission-<date>-<name>.zip \
  --validation datasets/valid.parquet --sequences 2
```

## 5b. Which number to believe

`scripts/blend.py` prints two. **Quote `blend eval half`, never `blend full`.**
Weights are fitted on the first half of the validation sequences, so the full
number includes the half they were tuned on and is optimistic.

| submission | blend full | eval half | public |
|---|---:|---:|---:|
| MLP 3-way | 0.67387 | 0.6747 | 0.6482 |
| MLP 13M | 0.67742 | 0.67841 | 0.6508 |
| mlpbig | 0.67819 | 0.67858 | 0.6496 |

The mlpbig change was +0.00077 on the full set but +0.00017 on the honest
half -- noise -- and it scored *below* the model it replaced. Roughly 70% of
an honest-half gain carries to public, and leaderboard sampling is worth a
couple of ten-thousandths, so **a change needs about +0.003 on the eval half
before it is distinguishable from nothing.**

## 5c. Runtime budget (a timeout scores zero)

The limit is 4200 s on the full test set and the scoring host is **1.6-2.1x
slower than this machine**, bracketed by two submissions:

| projected here | outcome |
|---:|---|
| 33.5 min (51.0 us/row) | scored |
| 44.3 min (67.5 us/row) | **timed out** |

So the ceiling is **~50 us/row / ~33 min projected**, and that is a measured
ceiling, not a guess. Consequences: no 4-way blend (~64 us/row even after the
float32 optimisation), no GRU wider than about 160x2 (128x2 = 20.6 us/row,
192x2 = 33.8, 256x2 = 53.0). Check `projected_test_minutes` from step 6
before every submission and treat anything above 33 min as a coin flip.

Profile before optimising -- `predict` was spending half its budget in the MLP
path on allocations and identity gathers, not arithmetic:

| component | before | after |
|---|---:|---:|
| features | 4.9 | 2.7 |
| linear read-out | ~0 | ~0 |
| MLP | 24.0 | ~19 |
| GRU (128x2) | 21.3 | 21.3 |
| **total** | **49.4** | **43.5** |

## 6. Verify before submitting

Packaging runs a smoke check only. The real gate is whether the shipped
`solution.py` reproduces the offline blend: a packaging mistake is invisible
until the leaderboard.

```bash
.venv/bin/python -u scripts/score_solution.py \
  --solution datasets/experiments/<candidate> \
  --validation datasets/valid.parquet --sequences 200 \
  --json /tmp/verify.json
```

Compare `weighted_pearson` against the same blend computed from the caches on
the same 200 sequences; the two runs so far agreed to 3e-7 and 2.9e-7. Also
check `projected_test_minutes` against the 60-minute limit -- 3-way is ~33
min, 4-way ~44 min, and the scoring host may be slower than this machine.

`divide by zero / overflow / invalid value encountered in matmul` in the
preflight output is a spurious NumPy/BLAS warning triple; predictions were
verified finite over 59,703 rows and inputs are bounded at |x| <= 5.2.

## Local -> public

Three submissions so far: local 0.66633 -> public 0.6423, 0.67387 -> 0.6482,
0.67742 -> 0.6508. The shift is stable at -0.024 to -0.027 and about 73% of a
local gain carries over. Leader is at 0.6646 public, so roughly 0.691 local.
Final ranking is on a private set, so prefer gains that come from data or
features over blend weights tuned on this validation file.


## 7. Objective and weighting: all tested, all negative

The metric is a `|clip(y)|`-weighted correlation over a **non-uniform 9-13%
subset** of predicted rows. Three ways in which training diverges from that
were tested on an MLP (3.7M rows, 512x128, 4 epochs, valid-stride 10):

| change | WP | vs baseline |
|---|---:|---:|
| `--weight-power 1.00` | 0.66155 | **+0.0007** |
| `--weight-power 1.50` | 0.66128 | +0.0005 |
| baseline `--weight-power 0.75` | 0.66082 | - |
| `--head joint` (predict (t0-t1)/2, (t0+t1)/2) | 0.65975 | -0.0011 |
| `--loss corr` (batch weighted Pearson) | 0.65829 | -0.0025 |
| importance weighting by `p(scored\|x)` | 0.66042 | -0.0004 |
| weighting by `p(scored\|x)` alone | 0.64380 | -0.017 |

`--weight-power 1.00` is the metric's own exponent and is a genuine but
sub-threshold gain; fold it in on the next retrain rather than submitting for
it alone.

The scoring mask *is* strongly structured - a classifier predicts `is_scored`
from the 112 raw features at **AUC 0.867**, while the targets predict it at
only 0.674, so `|y|^p` weighting cannot correct for it. Importance weighting
still does not help, because feature-target relationships are sign-stable
across sequences: the function is the same on scored and unscored rows, so
reweighting only shrinks the effective sample. Do not revisit this without a
reason to believe the *relationship*, not just the row density, differs.

The joint head is a linear reparameterisation of the output layer, so a
network with a linear final layer can already express it; only the
optimisation dynamics change, and they change for the worse.
