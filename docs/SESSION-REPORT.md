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
process errors rather than modelling ones - see section 9.

What the period did produce: a much better sequence model that could not be
shipped profitably, a 12% faster inference path, tooling that removes two
hard constraints, and a fairly complete map of what does not work on this
problem. That map is the main deliverable and is section 6.

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

## 5. Iteration log: why each method, when

The rest of this report says *what* was measured. This section says *why that
method, at that moment* - the decision chain, with the machine-learning idea
behind each step. Sources are in section 13.

### Iteration 0 - baseline: GA-searched features + ridge, blended with the GRU

**Situation.** The organisers ship a stateful GRU scoring 0.617. A competitor
needs something better.

**Decision.** Build a causal feature transform, fit a *linear* read-out on it,
and let a genetic algorithm choose the transform's shape (section 4).

**Why a linear read-out first.** Not because linear is expected to win, but
because it has a closed-form solution. That makes one candidate cost seconds
instead of minutes, which is the only reason a search over feature designs is
affordable at all. The generic lesson: *pick the model class that makes your
search loop cheap, then upgrade the read-out later.*

> **Concept - ridge regression.** Least squares with an L2 penalty on the
> coefficients, `(X'WX + lambda I)^-1 X'Wy`. The penalty is what allows more
> features than the data can cleanly support: it trades a little bias for a
> large drop in variance. Closed form means no learning rate, no epochs, no
> seed sensitivity. See ESL ch. 3.

**Result.** 0.6331 linear; blended with the organisers' GRU, 0.6663 -> public
0.6423.

### Iteration 1 - swap the read-out for an MLP

**Situation.** The features were designed by the GA, but the read-out on top
was linear.

**Decision.** Keep the same 336 features, replace the ridge with a small MLP.

**Why.** The feature transform is fixed and causal; the only thing the linear
read-out cannot express is *interaction between features* and any nonlinear
response. An MLP adds exactly that for a few microseconds per row. This is the
cheapest possible upgrade because it reuses all the upstream work.

**Result.** MLP alone 0.6556, three-way blend 0.6739 -> public 0.6482.

### Iteration 2 - diagnose before optimising: learning curve and capacity sweep

**Situation.** The obvious next moves were "ensemble a few MLP seeds" or "make
the MLP bigger", with guessed gains of +0.001 to +0.005.

**Decision.** Measure first. Train the same MLP at 414k / 829k / 1.66M / 3.3M
rows, and separately at 103k / 238k / 608k parameters.

**Why this specific pair of experiments.** They separate the two reasons a
model underperforms. If performance climbs with more *data* but is flat in
*parameters*, the model is data-limited and you should buy data. If the
reverse, it is capacity-limited. Guessing which one you are in is the single
most common way to waste a week.

> **Concept - learning curves.** Plot validation score against training-set
> size at fixed capacity, and against capacity at fixed data. The shapes tell
> you which resource is binding. This is the oldest diagnostic in applied ML
> and still the highest-value one. The modern large-scale version is the
> scaling-law literature (Kaplan et al. 2020).

**Result.** +0.0108 WP per 4x rows, no saturation; capacity flat from 103k to
608k parameters and *negative* at the top. Unambiguously data-limited - and
the earlier guess of "+0.002 to +0.005 for 2-3x rows" was off by about 4x.

### Iteration 3 - remove the constraint the diagnosis exposed

**Situation.** Data was the lever, but training data came from disk exports and
there was not enough disk.

**Decision.** Write a chunk-rotating trainer: export a chunk, train on it,
delete it, regenerate the next.

**Why.** The constraint was never information, it was storage. Recomputing is
cheap (about 40 s per 3.3M rows) relative to a training pass, so trading
compute for disk is obviously right once you notice the trade exists.

> **Concept - out-of-core learning.** When the dataset does not fit in RAM or
> on disk, stream it in shards and take gradient steps as it arrives. SGD does
> not care whether it sees the data from memory or from a pipe, as long as the
> shard order is shuffled between epochs.

**Result.** 13.3M rows -> MLP 0.6678, blend 0.6774 -> public **0.6508**, still
the best result of the whole project.

### Iteration 4 - build our own sequence model

**Situation.** Two observations pointed the same way. First, the organisers'
GRU scores only 0.617 yet still earns 0.18 of the blend weight - so a sequence
model contributes something the feature read-outs do not. Second, the MLP's
data lever was capped by the export stride, while a recurrent model reading
the Parquet file directly has no stride at all: 211M rows per epoch instead of
13M.

**Decision.** Train a GRU from scratch on the raw 112 columns.

**Why a GRU specifically.** Inference is row-by-row with strict causality, so
the model must carry O(1) state. That rules out attention over a window
(cost grows with context) and favours a recurrent cell. The GRU also matched
the organisers' ONNX signature exactly, making it a drop-in.

> **Concept - truncated backpropagation through time.** A 20,000-step sequence
> cannot be differentiated end-to-end. Process it in windows (512 rows here),
> backpropagate within a window, then carry the hidden state forward
> *detached* so the next window inherits the state but not the gradient. The
> state is exact; only the gradient path is truncated.

**Result.** 0.670 - better than the MLP, and +0.053 over the organisers' GRU
with the identical 128x2 architecture. The entire difference was training data
and a metric-aligned loss.

### Iteration 5 - the blend refuses the better model

**Situation.** Our GRU (0.670) was supposed to replace theirs (0.617).
Substituting it made the blend *worse*: 0.67742 -> 0.67647. Keeping both was
better than either.

**Decision.** Stop and measure *why*, rather than tuning blend weights.

**Why.** This is the moment the project's central question appeared. An
ensemble does not gain from accuracy, it gains from *error decorrelation* - so
a better-but-redundant model can be worth less than a worse-but-independent
one.

> **Concept - why ensembles work.** The squared error of an average decomposes
> into average individual error minus average *disagreement* between members
> (the ambiguity decomposition). If members agree, the second term vanishes and
> averaging buys nothing. Diversity is not a nice-to-have; it is the entire
> mechanism.

**Result, eventually.** Every model we owned sat at 0.93-0.96 residual
correlation with every other. There was no diversity to exploit. (The first
attempt at this measurement was *wrong* - see section 9 - and briefly
suggested the organisers' GRU was highly diverse at 0.37.)

### Iteration 6 - is the ceiling in the features?

**Situation.** If all read-outs agree, perhaps the 336 features are the limit.

**Decision.** Build a feature lab that fits a weighted ridge over a candidate
feature family in seconds, and ablate seven families.

**Why a ridge as the screen.** It is the fastest thing that can answer "is
there signal in these columns at all". A family that cannot help a linear
model is unlikely to help a nonlinear one; the converse is not guaranteed, but
it is a sound first filter for *cheap*.

> **Concept - ablation.** Change one component, hold everything else fixed,
> measure. The discipline that matters is comparing on *identical rows*, which
> makes the paired difference far less noisy than either absolute score.

**Result.** +0.002 total, all from restoring the mid and slow EMA blocks.
Classical microstructure features (microprice, book imbalance, spread) turned
out to be *structurally unrecoverable*: the columns are per-column affine
transformed including sign flips, so cross-column arithmetic is meaningless.

### Iteration 7 - a genuinely different inductive bias

**Situation.** Linear, MLP and GRU all agreed. All three are smooth function
approximators trained by gradient descent.

**Decision.** Fit gradient-boosted trees.

**Why.** Trees are the standard counter-example to neural networks on tabular
data: axis-aligned splits, automatic interactions, no smoothness assumption,
robust to uninformative features. If anything was going to disagree with the
MLP, this was the best candidate - and it doubles as the recognised strong
baseline for this data type.

> **Concept - inductive bias.** Every learner has built-in assumptions about
> what functions are plausible. MLPs prefer smooth functions; trees prefer
> piecewise-constant ones; recurrent nets prefer temporally local structure.
> Two models with different biases making the *same* errors is strong evidence
> the errors come from the data, not the model. See Grinsztajn et al. (2022)
> on why trees still win on much tabular data.

**Result.** 0.655 alone, residual correlation 0.962 with the MLP. No
disagreement.

### Iteration 8 - attack the one property all models shared

**Situation.** Four families now agreed. What did they have in common? Short
memory: EMA spans of 28, a 42-row lag, 512-step BPTT - over 20,000-row
sequences. And a span-1248 EMA *had* measurably helped the linear model.

**Decision.** Build a diagonal state-space layer with learned per-dimension
decay rates initialised across spans from 2 to 20,000 rows.

**Why this architecture.** A GRU's recurrence is a matrix multiply on the
state - O(N^2), which is why 128 state costs 21 us/row and 256 costs 53. A
*diagonal* recurrence `h <- a*h + (1-a)*Bx` is elementwise - O(N) - so a
1024-dimensional state costs 11 us/row. The cost asymmetry is what makes long,
multi-scale memory affordable at all. Each dimension is literally an EMA with
its own learned span: the hand-built feature, generalised and made learnable.

> **Concept - state-space models.** A linear recurrence with a diagonal
> transition, plus a nonlinear read-out. Because the transition is constant in
> time, the recurrence is a convolution with an exponential kernel and can be
> evaluated in parallel during training (by FFT here), while still running as
> an O(1) recurrence at inference. This is the core trick behind S4 and Mamba.

**Result.** Trained well, learned a median span of 406 rows with a tail to
45,000 - and landed at 0.955 correlation with the MLP. Long memory was not the
blind spot. Killed at the pre-declared gate after 40 minutes rather than the
planned 6.5 hours.

### Iteration 9 - stop changing the model, change the objective

**Situation.** Five architectures had converged. What was left was *what we
were optimising*, which every model shared.

**Decision.** Test three mismatches between the training objective and the
scoring rule: the weighting exponent, a correlation loss, and the row
distribution.

**Why each.** The metric is a `|y|`-weighted *correlation* over a non-uniform
9-13% subset of rows. We trained a `|y|^0.75`-weighted *squared error* over
100% of rows. Each difference is a candidate explanation for a systematic gap.

> **Concept - loss/metric mismatch.** Squared error penalises scale error;
> correlation is invariant to affine rescaling. A model can be badly
> calibrated and perfectly correlated - the organisers' GRU predicts with 2.4x
> the target's standard deviation and still scores 0.617.

> **Concept - covariate shift and importance weighting.** When training and
> evaluation draw inputs from different distributions but share the same
> conditional `p(y|x)`, the standard correction is to weight each training row
> by `p_eval(x)/p_train(x)`, estimated by training a classifier to tell the two
> apart. It helps when the model is misspecified or the relationship varies;
> it costs effective sample size always.

**Result.** The mask *is* strongly structured - a classifier predicts
`is_scored` from the features at AUC 0.867, while the targets predict it at
only 0.674, so `|y|^p` weighting cannot possibly correct for it. But the
correction did not help (-0.0004), because feature-target relationships are
sign-stable across sequences: the function is the same on scored and unscored
rows, so reweighting only shrinks the sample. Correlation loss -0.0025, joint
target head -0.0011. Only `weight_power 1.00` helped, by +0.0007.

### Iteration 10 - optimise the binding constraint

**Situation.** A submission had timed out. Runtime, not accuracy, was now what
limited which models could ship.

**Decision.** Profile `solution.py` before touching it.

**Why profile first.** The intuition was that the GRU dominated. It did not:
the MLP path cost 24 us/row against roughly 10 us of actual matrix arithmetic,
with the rest going to per-row allocation and three fancy-index operations
that were copying arrays *to themselves*.

**Result.** 49.4 -> 43.5 us/row, predictions identical to six decimals.

## 6. Tooling built

All committed in `29d7180` on branch `claude/magical-fermi-4vcs0z`
(**local only - never pushed**).

| script | what it does |
|---|---|
| `train_gru.py` | trains a stateful GRU streaming straight from `train.parquet` - each Parquet row group is one sequence, so 211M rows/epoch with no export and no disk. Exports ONNX matching the baseline's signature. |
| `predict_gru.py` | caches scored validation predictions for blending, with `--align-to` as a hard guard against row mismatch |
| `train_mlp_streaming.py` | trains over several feature exports with one resident at a time; also carries `--loss {mse,corr}` and `--head {plain,joint}` from the objective experiments |
| `train_mlp_rotating.py` | regenerates each feature chunk on demand instead of storing it, so training size is no longer capped by disk |
| `train_ssm.py` | diagonal multi-timescale state-space layer (see section 8.5) |
| `feature_lab.py` | screens a feature family against a weighted ridge in seconds instead of hours |
| `diversity_report.py` | standalone WP plus residual correlation against existing models. **Its metric is confounded by prediction scale** - rescale to the target's weighted std before trusting it (this bug produced a wrong conclusion once; section 6). |

`solution.py` was also optimised (section 4) and generalised to blend several
GRUs and to read hidden width from the ONNX graph.

## 7. Inference optimisation

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

## 8. What does not work

Every item below was measured, not assumed.

### 8.1 The central result: five model families converge

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

### 8.2 Model capacity

MLP width is flat or negative at every data scale tried:

| params | 3.3M rows | 13.3M rows | 35M rows |
|---|---:|---:|---:|
| 103k (256x64) | 0.65684 | 0.66766 | - |
| 238k (512x128) | 0.65709 | 0.66780 | 0.66905 |
| 608k (1024x256) | 0.65407 | 0.66644 | 0.66205 |

GRU capacity is unaffordable rather than unhelpful: 128x2 costs 20.6 us/row,
192x2 costs 33.8, 256x2 costs 53.0, against a ~50 us/row total budget.

### 8.3 Data scaling

**+0.0108 WP per 4x rows when the rows are new sequences.** But only +0.0013
for 2.6x rows obtained by halving the export stride - and that gain did not
survive to the public leaderboard. Within-sequence rows are heavily
autocorrelated and add close to nothing. All 10,607 sequences were used, so
this lever is exhausted.

### 8.4 Feature engineering

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

### 8.5 Architecture: the state-space attempt

Designed specifically to break the shared blind spot of short memory. A
diagonal recurrence `h_t = a*h_{t-1} + (1-a)*Bx_t` costs O(N) per row instead
of a GRU's O(N^2), so a 1024-dimensional state costs 11.2 us/row against the
GRU's 21.3 for 256. Spans initialised log-spaced from 2 to 20,000 rows.

It trained well (159k rows/s at state 256, via an FFT evaluation of the
recurrence), reached 0.6455, and learned spans with a median of 406 rows and a
tail to 45,000 - far beyond anything else in the family. And it still landed
at **0.955** residual correlation with the MLP. Long memory was not the blind
spot. Full detail in `SPEC-NEXT-MODEL.md`.

### 8.6 Objective and weighting

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

### 8.7 Other dead ends

- **Blend diversity cannot be manufactured.** Training the same GRU with an
  unweighted MSE loss produced 0.94 correlation with the MLP. Architecture,
  input representation and objective were all varied; nothing decorrelated.
- **Per-sequence volatility scaling** loses 0.016 even with an oracle that
  knows each sequence's true target std.
- **Stacking the blend with raw features** loses 0.002 out-of-sample. The
  feature-linear signal is fully extracted; residuals correlate with features
  only because the model explains little variance, not because signal remains.

## 9. Two mistakes that each cost a submission

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

## 10. What was learned about the data

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
  targets predict it at only 0.674. Correcting for it did not help (8.6), but
  the fact itself is the most surprising thing found.

## 11. State of the repository

Committed in `29d7180`, **local only, not pushed.**

Kept: `datasets/experiments/*/` model weights and training reports (~20 MB),
`submissions/*.zip` (~8 MB, including the best entry).

Deleted as regenerable: `datasets/mlp/valid.f32` (4.5 GB, ~1 min to rebuild),
all `preds_*.npz` caches (~630 MB, rebuild from saved weights), all training
feature exports.

`docs/RUNBOOK.md` carries the commands, measured reference points and the two
rules above. `docs/SPEC-NEXT-MODEL.md` carries the state-space design and its
rejection.

## 12. If anyone returns to this

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

## 13. Concepts and sources

Grouped by where they appear in section 5. Links are to the canonical
reference where one exists; the rest are cited by name.

### Fundamentals

- **Ridge regression, bias-variance, learning curves, bagging, boosting** -
  Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*,
  free PDF at <https://hastie.su.domains/ElemStatLearn/>. Chapters 3 (linear
  methods), 7 (model assessment) and 10 (boosting) cover most of what this
  project used.
- **Adam optimiser** - Kingma & Ba, 2014. <https://arxiv.org/abs/1412.6980>
- **One-cycle learning-rate schedule** - Smith & Topin, *Super-Convergence*.
  <https://arxiv.org/abs/1708.07120>. Explains the mid-training dip that
  appeared in nearly every run here and recovers during annealing - which I
  twice misread as overfitting.

### Diagnosing what limits a model

- **Scaling laws** - Kaplan et al., 2020. <https://arxiv.org/abs/2001.08361>.
  The large-scale formalisation of the learning-curve diagnostic: performance
  as a power law in data, parameters and compute, and how to tell which one
  binds.
- **Bootstrap confidence intervals** - Efron & Tibshirani, *An Introduction to
  the Bootstrap*. Resampling *whole sequences* rather than rows is what makes
  the interval honest when observations inside a sequence are correlated.

### Sequence models

- **Understanding LSTMs / GRUs** - Chris Olah's explainer remains the clearest
  introduction. <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
- **Structured state spaces (S4)** - Gu, Goel & Re, 2021.
  <https://arxiv.org/abs/2111.00396>. The diagonal-recurrence idea in
  iteration 8.
- **The Annotated S4** - Sasha Rush's line-by-line implementation walkthrough.
  <https://srush.github.io/annotated-s4/>. The practical companion to the
  paper; closest thing to what `scripts/train_ssm.py` does.
- **Mamba** - Gu & Dao, 2023. <https://arxiv.org/abs/2312.00752>. The
  selective-state successor; input-dependent decay rates, which this project
  did *not* try.

### Tabular data and inductive bias

- **Why do tree-based models still outperform deep learning on typical tabular
  data?** - Grinsztajn, Oyallon & Varoquaux, 2022.
  <https://arxiv.org/abs/2207.08815>. Directly motivates iteration 7, and
  explains why a GBM was the strongest candidate for disagreeing with the MLP.

### Ensembles and diversity

- **Stacked generalization** - Wolpert, 1992. The origin of fitting a model on
  the outputs of other models, which is what `scripts/blend.py` does.
- **Ambiguity decomposition** - Krogh & Vedelsby, 1995. The result that an
  ensemble's error equals average member error minus average disagreement;
  the formal reason iteration 5 failed.
- **Diversity creation methods: a survey and categorisation** - Brown et al.,
  2005. Survey of the ways people try to manufacture diversity - most of which
  were tried here without success.

### Distribution shift

- **Dataset Shift in Machine Learning** - Quinonero-Candela et al. (eds.), MIT
  Press, 2009. The standard reference for covariate shift and importance
  weighting, used in iteration 9.
- **Classifier two-sample test** - training a classifier to distinguish two
  distributions and reading its AUC as a measure of how different they are.
  That is exactly the `is_scored` experiment: AUC 0.867 means the scoring mask
  is far from random.

### Things worth knowing that this project learned the hard way

- **Paired comparison beats absolute measurement.** Absolute WP on a
  60-sequence slice drifts +-0.034 from truth; the *difference* between two
  models on the same rows is far more stable. Always compare on identical
  rows.
- **A held-out set used for many decisions stops being held out.** Blend
  weights, checkpoint selection, feature specs and configuration choices were
  all made against one validation file. The honest-half discipline in section 9
  is the minimum defence; a genuinely untouched split is better.
- **Profile before optimising.** The intuition about where runtime went was
  wrong by a factor of two.
- **Define the kill criterion before running the experiment.** The state-space
  model in iteration 8 had a pre-declared diversity gate, which is why it cost
  40 minutes instead of 6.5 hours.
