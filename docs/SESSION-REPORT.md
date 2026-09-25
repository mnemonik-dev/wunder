# Alpha Connectome: what was done, what came of it

Closing report, 2026-09-25. Work spans 2026-09-22 to 2026-09-25.

## 1. Outcome

**The public score did not improve during this work.** The best submission
remains `submission-20260922-mlp13m.zip` at **0.6508 public**, made at the
start of the period. The leader finished at 0.6646, so the gap is ~0.014.

| # | submission | local (blend full) | local (honest half) | public |
|---|---|---:|---:|---:|
| 1 | shipped blend | 0.66633 | - | 0.6423 |
| 2 | mlp3way | 0.67387 | 0.6747 | 0.6482 |
| 3 | **mlp13m** | 0.67742 | 0.67841 | **0.6508** |
| 4 | gru4way | 0.67971 | 0.68068 | **timed out, 0** |
| 5 | mlpbig | 0.67819 | 0.67858 | 0.6496 |

Submissions 4 and 5 are the two failures worth remembering, and both were
process errors rather than modelling ones - see section 8.

What the period did produce: a much better sequence model that could not be
shipped profitably, a 12% faster inference path, tooling that removes two
hard constraints, and a fairly complete map of what does not work on this
problem. That map is the main deliverable and is section 5.

## 2. Where the score actually came from

Component scores on the full validation set (1,873 sequences, 3,528,273
scored rows):

| model | WP |
|---|---:|
| organisers' GRU baseline (`baseline.onnx`) | 0.61705 |
| linear read-out, champion feature spec | 0.63314 |
| linear read-out, + mid/slow EMA blocks | 0.63511 |
| MLP 512x128, 13.3M rows | 0.66743 |
| MLP 512x128, 35M rows | 0.66885 |
| **our GRU 128x2, trained on all 211M rows** | **0.67024** |
| gradient-boosted trees, 400 trees | 0.65511 |
| diagonal state-space model, 1024 state | 0.64546 |

The shipped blend is `linear + MLP + organisers' GRU`. Note the awkward fact
that our own GRU, at 0.670, **blends worse** than the organisers' 0.617 one;
replacing it dropped the blend from 0.67742 to 0.67647. Blend value comes from
decorrelation, not accuracy, and by that measure ours added nothing the MLP
did not already have.

## 3. How the solution works

Not one model: a causal feature transform written in Rust, three read-outs
fitted on top of it, and a blend - replayed at inference by a NumPy script
that processes one row at a time and resets state on every new sequence.

**Stage 1, streaming causal features** (`neutrino-wunder`, Rust). Per row:
three exponential moving averages, a running volatility estimate, and a ring
buffer of past rows. Emits the raw 112 columns, their residual against a fast
EMA, and their difference against a row `lag` steps back - 336 features.
Optional blocks add mid/slow EMA residuals, volatility normalisation, and
per-level book imbalance. Every value uses only the current and earlier rows
of the same sequence.

**Stage 2, weighted ridge read-out**, solved in closed form. Given a feature
layout, the linear read-out is a `|clip(y)|^p`-weighted ridge regression
solved from the Gram matrix by Cholesky - no gradient descent. One candidate
costs two streaming passes plus an O(F^3) solve, about 5 s for 80 sequences.
That speed is what makes the genetic search in section 4 affordable.

**Stage 3, nonlinear and recurrent read-outs** (PyTorch). An MLP on the same
336 features, and a stateful GRU reading the raw 112 columns with truncated
backpropagation through time over 512-row windows. The MLP ships as NumPy
arrays, the GRU as ONNX with the organisers' input signature, so inference
needs no PyTorch.

**Stage 4, per-target blend.** Weights chosen per target by weighted least
squares of the clipped target on the model predictions, then refined on a
local grid against the competition metric itself - fitted on the first half
of the validation sequences and reported on the second.

### Every method used, by stage

Including techniques used only for analysis, and those tried and discarded.

| stage | method | detail |
|---|---|---|
| Data | Parquet row-group streaming | one row group = one 20,000-row sequence |
| Data | Causal EMA transform | three spans + running volatility EMA, seeded at row 0 |
| Data | Ring-buffer lagged difference | `x - x[t-lag]`, searchable lag |
| Data | Strided export | every n-th predicted row to a flat float32 matrix |
| Data | Rotating chunk regeneration | export -> train -> delete; disk holds one chunk |
| Search | **EvoForge genetic algorithm** | 14 typed genes, population 14 x 6 generations, mutation 0.25, seed 42 |
| Search | Metric-as-fitness | Global WP of the fitted candidate, not a proxy loss |
| Search | Candidate caching | identical decoded genomes evaluated once |
| Search | Direct ablation at scale | replaced the GA once its holdout proved too noisy |
| Fitting | **Weighted ridge regression** | closed form, Cholesky on the Gram matrix |
| Fitting | **MLP** | Adam, OneCycle schedule, dropout, Gaussian input noise |
| Fitting | **GRU**, stateful | truncated BPTT over 512 rows, state carried detached, gradient clipping |
| Fitting | **Gradient-boosted trees** | histogram-based, 400 trees, 63 leaves, sample-weighted |
| Fitting | **Diagonal state-space model** | recurrence evaluated by FFT convolution; decays in log space |
| Fitting | Weighted MSE loss | `|clip(y)|^p`, p searched over 0.75 / 1.00 / 1.50 |
| Fitting | Batch weighted-Pearson loss | metric-aligned; tested and rejected |
| Fitting | Joint rotated target head | predict `(t0-t1)/2`, `(t0+t1)/2`; tested and rejected |
| Fitting | Importance weighting | by `p(scored\|x)`; tested and rejected |
| Blending | Weighted least-squares stacking | per target, on the model predictions |
| Blending | Local grid refinement | optimises the competition metric around the WLS point |
| Blending | Honest half/half split | weights fitted on half the sequences, reported on the other |
| Analysis | Sequence-level bootstrap | 400-2,000 replicates for confidence intervals on a delta |
| Analysis | Scale-corrected residual correlation | the diversity metric; calibrates scale before comparing |
| Analysis | PCA of column groups | prices share one factor (70%), volumes do not (25%) |
| Analysis | Target autocorrelation | established the 50-200 step forecast horizon |
| Analysis | Lead-lag cross-correlation | `feature[t-k]` against `target[t]` |
| Analysis | Oracle upper bounds | gives an idea its best case before building it |
| Analysis | Partial-correlation / stacking test | is signal left in the features after the model? |
| Analysis | Covariate-shift classifier | predicts `is_scored` from features; AUC 0.867 |
| Analysis | Learning curves, capacity sweeps | rows and parameters varied independently |
| Deployment | NumPy row-by-row inference | one row per call, state reset on sequence change |
| Deployment | ONNX Runtime with I/O binding | two pre-bound bindings ping-pong the hidden state |
| Deployment | Standardisation folded into layer 1 | removes a subtract and multiply per row |
| Deployment | float32 path, preallocated buffers | matches the training dtype; no per-row allocation |
| Deployment | Three-way parity testing | Rust vs Python, PyTorch vs ONNX, offline blend vs shipped script |
| Deployment | Packaging preflight | determinism, finite output, archive size, projected runtime |

## 4. The genetic search

The project's original premise was to improve the organisers' baseline using
genetic evolution. **EvoForge** is a domain-neutral evolutionary optimisation
core in Rust; `neutrino-optimizer` wraps it with typed genes and drives it
against the competition metric.

The split is deliberate. What is differentiable is solved exactly - the ridge
read-out has a closed form. What is *not* differentiable goes to the GA: how
long the memory should be, whether the second instrument helps, whether raw
price levels should be visible at all, how hard to weight large moves.

| gene | type | range | meaning |
|---|---|---|---|
| `ema_fast` / `ema_mid` / `ema_slow` | int | 2-60 / 5-300 / 30-2000 | spans of three moving averages |
| `use_dmid` / `use_dslow` / `use_diff` | bool | - | which residual blocks to include |
| `diff_lag` | int | 1-50 | lag of the difference block |
| `raw_prices` / `use_i1` | bool | - | show raw price levels; include instrument i1 |
| `vol_norm` / `vol_span` | bool / int | 20-2000 | divide residuals by running volatility |
| `use_imbalance` | bool | - | per-level volume imbalance block |
| `log10_lambda` | float | -5 .. 1 | ridge strength |
| `weight_power` | float | 0-2 | sample weight exponent on \|y\| |

Fitness is the *competition metric itself* - Global Weighted Pearson of the
fitted candidate on a held-out validation slice, not a proxy loss. Population
14, six generations, mutation rate 0.25, seed 42: **68 trials in 440 s**, each
candidate fitted on 80 sequences and scored on 60.

| best fitness after | 10 trials | 20 | 40 | 68 |
|---|---:|---:|---:|---:|
| GA holdout WP | 0.60970 | 0.62314 | 0.62328 | 0.62410 |

The champion refit on 1,000 sequences scored **0.63314** on full validation, a
real gain over the organisers' 0.61705 baseline and the foundation everything
else was built on. It chose a 28-row fast EMA, a 42-row lagged difference,
both instruments, raw prices visible, ridge lambda = 3.0e-4, weight power
0.75 - and switched *off* the mid EMA, the slow EMA, volatility normalisation
and imbalance.

### Auditing its decisions

Late in the work those switched-off genes were re-tested directly, at 1,000
training sequences and scored on the full validation set rather than a
60-sequence slice:

| gene | GA chose | measured at scale | verdict on the GA |
|---|---|---:|---|
| `vol_norm` | off | -0.028 | right |
| `use_imbalance` | off | -0.004 | right |
| `use_dmid` + `use_dslow` | off | **+0.002** | **wrong** |
| `weight_power` | 0.75 | +0.0007 at 1.00 | near miss |

**The search was sound; its measuring instrument was not.** Fitness came from
a 60-sequence holdout, and a slice that size was later measured drifting
+-0.034 from full-validation truth - while the effects being selected between
are worth +-0.003. The signal-to-noise ratio of the fitness function was
roughly one to ten.

The convergence trace shows it: the GA reached 0.62314 by trial 20 and spent
its remaining 48 trials gaining +0.001, well inside its own noise. It had
stopped optimising the objective and started fitting its holdout.

The fix is not a better GA. It is to screen blocks directly at 1,000
sequences, where a +0.002 effect is visible - about five minutes per spec,
and no search at all.

### Why it failed, precisely

The GA spent its compute on **many cheap, noisy evaluations instead of fewer
accurate ones**. Each candidate cost about 5 s - a ridge fit on 80 sequences,
scored on 60 - which bought 68 candidates in 440 s. But a 60-sequence holdout
drifts +-0.034 from truth, and the differences between candidates are worth
+-0.003.

Selection pressure needs a signal to act on. Here it mostly acted on
measurement noise, which is why `use_dmid`/`use_dslow` were switched off: their
true value is +0.002, permanently invisible at that noise level.

The trade was backwards. Scoring a candidate on 1,000 sequences costs ~3 min
instead of 5 s. The same 440-second budget would buy only 2-3 candidates, far
too few - but a 3-hour budget would buy ~60 candidates evaluated *below* the
noise floor of the effects being chosen between. That is the version that
would have worked.

Evolution optimises whatever you actually measure. The machinery was sound;
the measurement was the bug.

Worth stating plainly: the genetic approach *did* deliver the project's
foundation, taking the linear read-out from a hand-guessed configuration to
0.6331 and finding a non-obvious combination - a 42-row lagged difference
alongside a 28-row EMA - that nobody proposed by hand.

## 5. Tooling built

All committed in `29d7180` on branch `claude/magical-fermi-4vcs0z`
(**local only - never pushed**).

| script | what it does |
|---|---|
| `train_gru.py` | trains a stateful GRU streaming straight from `train.parquet` - each Parquet row group is one sequence, so 211M rows/epoch with no export and no disk. Exports ONNX matching the baseline's signature. |
| `predict_gru.py` | caches scored validation predictions for blending, with `--align-to` as a hard guard against row mismatch |
| `train_mlp_streaming.py` | trains over several feature exports with one resident at a time; also carries `--loss {mse,corr}` and `--head {plain,joint}` from the objective experiments |
| `train_mlp_rotating.py` | regenerates each feature chunk on demand instead of storing it, so training size is no longer capped by disk |
| `train_ssm.py` | diagonal multi-timescale state-space layer (see section 7.5) |
| `feature_lab.py` | screens a feature family against a weighted ridge in seconds instead of hours |
| `diversity_report.py` | standalone WP plus residual correlation against existing models. **Its metric is confounded by prediction scale** - rescale to the target's weighted std before trusting it (this bug produced a wrong conclusion once; section 6). |

`solution.py` was also optimised (section 4) and generalised to blend several
GRUs and to read hidden width from the ONNX graph.

## 6. Inference optimisation

Profiling found half the runtime budget going to overhead rather than
arithmetic. `raw_idx`, `dyn_idx` and `keep` were all `arange`, so three fancy
-index operations per row were copying arrays to themselves; `np.concatenate`
allocated a fresh 336-element array per row; `ema_mid`, `ema_slow` and the
volatility EMA were updated every row although the champion spec uses none of
them; and everything ran in float64 although the MLP was trained in float32.

| component | before | after |
|---|---:|---:|
| features | 4.9 | 2.7 |
| linear read-out | ~0 | ~0 |
| MLP | 24.0 | ~19 |
| GRU (128x2) | 21.3 | 21.3 |
| **total** | **49.4** | **43.5** |

Predictions unchanged to six decimals (verified 0.688171 both ways on 200
sequences). Projected runtime 32.5 -> 28.6 min.

## 7. What does not work

Every item below was measured, not assumed.

### 7.1 The central result: five model families converge

| family | inductive bias | WP alone | residual corr. vs MLP |
|---|---|---:|---:|
| ridge on 336 streaming features | linear | 0.6331 | 0.93 |
| MLP 512x128 | smooth nonlinear | 0.6689 | - |
| GRU 128x2 on raw columns | gated recurrent | 0.6702 | 0.964 |
| HistGBM, 400 trees | axis-aligned splits | 0.6551 | 0.962 |
| diagonal SSM, 1024 state | long linear recurrence | 0.6455 | 0.955 |
| organisers' GRU | recurrent, trained by them | 0.6171 | 0.964 |

All explain ~18% of target variance and disagree only in the noise. This is
why every component gain collapsed in the blend: the MLP improving by +0.0118
moved the blend +0.0036; a GRU beating the organisers' by +0.053 moved it
+0.0023; seven engineered feature families bought +0.002 in total.

It is **not** an information ceiling - the leader demonstrably extracted
+0.014 more. Five families sharing a limitation is a better description.

### 7.2 Model capacity

MLP width is flat or negative at every data scale tried:

| params | 3.3M rows | 13.3M rows | 35M rows |
|---|---:|---:|---:|
| 103k (256x64) | 0.65684 | 0.66766 | - |
| 238k (512x128) | 0.65709 | 0.66780 | 0.66905 |
| 608k (1024x256) | 0.65407 | 0.66644 | 0.66205 |

GRU capacity is unaffordable rather than unhelpful: 128x2 costs 20.6 us/row,
192x2 costs 33.8, 256x2 costs 53.0, against a ~50 us/row total budget.

### 7.3 Data scaling

**+0.0108 WP per 4x rows when the rows are new sequences.** But only +0.0013
for 2.6x rows obtained by halving the export stride - and that gain did not
survive to the public leaderboard. Within-sequence rows are heavily
autocorrelated and add close to nothing. All 10,607 sequences were used, so
this lever is exhausted.

### 7.4 Feature engineering

At 1,000 training sequences, scored on full validation:

| spec | WP |
|---|---:|
| champion | 0.63314 |
| + mid-EMA only | 0.63421 |
| **+ mid and slow EMA** | **0.63511** |
| + vol_norm | 0.60538 |
| + mid, slow and vol_norm | 0.62257 |

Seven families were screened in `feature_lab.py`; only the mid/slow EMA blocks
survived, worth +0.002. Notably `vol_norm` is strongly negative - the GA was
right to reject it, and the feature lab disagreed only because it implements a
different transform than the Rust flag.

**Classical microstructure is unrecoverable.** Columns are per-column affine
transformed *including sign flips* (`i0_p10` correlates -0.84 to -0.93 with
its sibling bid prices; volumes go negative; best-ask reads below best-bid).
Microprice, book imbalance and spread need a common scale that preprocessing
destroyed. This is why `imbalance` scored worst of all families (-0.004).

### 7.5 Architecture: the state-space attempt

Designed specifically to break the shared blind spot of short memory. A
diagonal recurrence `h_t = a*h_{t-1} + (1-a)*Bx_t` costs O(N) per row instead
of a GRU's O(N^2), so a 1024-dimensional state costs 11.2 us/row against the
GRU's 21.3 for 256. Spans initialised log-spaced from 2 to 20,000 rows.

It trained well (159k rows/s at state 256, via an FFT evaluation of the
recurrence), reached 0.6455, and learned spans with a median of 406 rows and a
tail to 45,000 - far beyond anything else in the family. And it still landed
at **0.955** residual correlation with the MLP. Long memory was not the blind
spot. Full detail in `SPEC-NEXT-MODEL.md`.

### 7.6 Objective and weighting

| change | WP | vs baseline |
|---|---:|---:|
| `--weight-power 1.00` | 0.66155 | **+0.0007** |
| `--weight-power 1.50` | 0.66128 | +0.0005 |
| baseline `--weight-power 0.75` | 0.66082 | - |
| `--head joint` | 0.65975 | -0.0011 |
| `--loss corr` | 0.65829 | -0.0025 |
| importance weighting by `p(scored\|x)` | 0.66042 | -0.0004 |
| `p(scored\|x)` alone | 0.64380 | -0.017 |

Only `weight-power 1.00` - the metric's own exponent - helps, by a quarter of
what is detectable.

### 7.7 Other dead ends

- **Blend diversity cannot be manufactured.** Training the same GRU with an
  unweighted MSE loss produced 0.94 correlation with the MLP. Architecture,
  input representation and objective were all varied; nothing decorrelated.
- **Per-sequence volatility scaling** loses 0.016 even with an oracle that
  knows each sequence's true target std.
- **Stacking the blend with raw features** loses 0.002 out-of-sample. The
  feature-linear signal is fully extracted; residuals correlate with features
  only because the model explains little variance, not because signal remains.

## 8. Two mistakes that each cost a submission

**Quoting the wrong number.** `scripts/blend.py` prints `blend full` and
`blend eval half`; weights are fitted on the first half of validation, so only
the eval half is honest. A change worth +0.00077 on the full set was worth
+0.00017 on the honest half - noise - and scored *below* the model it
replaced. **Rule: quote the eval half, and require +0.003 before believing a
change**, given ~70% carry-over to public plus leaderboard sampling.

**Submitting into an unproven runtime.** The limit is 4200 s and the scoring
host is 1.6-2.1x slower than the dev machine. A configuration projected at
44.3 min timed out and scored zero, for a +0.0023 expected gain; the
configuration that scored was 33.5 min. **Rule: treat anything above ~33 min
projected as a coin flip.**

A third error was caught internally: the first diversity metric was confounded
by prediction scale and reported the organisers' GRU at 0.37 correlation when
the true figure was 0.96. That produced a wrong strategic conclusion for
several hours until the scale correction was added.

## 9. What was learned about the data

- Columns are per-column affine transformed, including sign flips.
- Price columns share a dominant factor (PC1 = 70%); volume columns do not
  (PC1 = 25%), so the volume side carries more independent information.
- Target autocorrelation decays to zero by lag ~200; the forecast horizon is
  roughly 50-200 steps. A span-1248 EMA still helps, as regime context rather
  than horizon matching.
- `corr(t0, t1) = -0.738`, negative in all 40 sequences measured.
- Target volatility varies 2.7-3.2x across sequences, but feature-target
  relationships are sign-stable (2 sign flips in 40 sequences), so a global
  model is appropriate.
- **The scoring mask is strongly structured.** Only 9-13% of predicted rows
  are scored, they carry 19-26% larger target magnitudes, and a classifier
  predicts `is_scored` from the 112 raw features at **AUC 0.867** - while the
  targets predict it at only 0.674. Correcting for it did not help (7.6), but
  the fact itself is the most surprising thing found.

## 10. State of the repository

Committed in `29d7180`, **local only, not pushed.**

Kept: `datasets/experiments/*/` model weights and training reports (~20 MB),
`submissions/*.zip` (~8 MB, including the best entry).

Deleted as regenerable: `datasets/mlp/valid.f32` (4.5 GB, ~1 min to rebuild),
all `preds_*.npz` caches (~630 MB, rebuild from saved weights), all training
feature exports.

`docs/RUNBOOK.md` carries the commands, measured reference points and the two
rules above. `docs/SPEC-NEXT-MODEL.md` carries the state-space design and its
rejection.

## 11. If anyone returns to this

The honest position is that no promising hypothesis remains. Five
architectures, seven feature families, four training objectives, three data
scales and a covariate-shift correction were all tested and none moved the
score. The gap to the leader is real and unexplained.

Things never tried, in rough order of what I would attempt first:

1. **Much longer training.** Every model here ran 6 epochs with an arbitrary
   learning rate. No systematic hyper-parameter search was done at any point.
   Five families converging may reflect five identically under-tuned recipes.
2. **Ensembling across seeds** rather than across architectures - the one
   averaging axis not tested.
3. **Predicting the targets at multiple horizons** as auxiliary tasks, if the
   two targets are horizons of one process.
4. **The volume side specifically** - it carries the independent information
   (PC1 25% vs 70%) and no feature work targeted it directly.

What I would not repeat: architecture search, classical microstructure
features, blend-weight tuning, and anything justified by a gain below +0.003
on the honest half.
