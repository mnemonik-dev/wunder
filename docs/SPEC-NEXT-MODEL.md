# Spec: a model outside the current family

Status: **implemented and rejected**, 2026-09-24. The layer was built
(`scripts/train_ssm.py`), trained, and failed the milestone-3 diversity gate.
Kept as a record of what was tried and what it ruled out; see section 9.

## 1. Why a new model rather than more tuning

Four independent model families were fitted to this data and all of them
converge to the same function:

| family | inductive bias | WP alone | residual corr. vs MLP |
|---|---|---:|---:|
| ridge on 336 streaming features | linear | 0.6331 | 0.93 |
| MLP 512x128 on the same features | smooth nonlinear | 0.6689 | - |
| GRU 128x2 on the raw 112 columns | recurrent, learned features | 0.6702 | 0.964 |
| HistGBM, 400 trees | axis-aligned splits, interactions | 0.6551 | 0.962 |
| organisers' GRU (trained by them) | recurrent | 0.6171 | 0.964 |

They explain ~18% of target variance and disagree only in the noise. That is
why every large component gain collapsed in the blend: the MLP improving by
+0.0118 moved the blend +0.0036; a GRU beating the organisers' by +0.053
moved it +0.0023; seven engineered feature families bought +0.002 in total.

The tempting conclusion is an information ceiling. **It is not one**: the
leader sits at 0.6646 public against our 0.6508, so roughly +0.014 more is
demonstrably extractable. Our four families are not hitting a property of the
data, they are sharing a blind spot.

## 2. What the four models have in common

Everything below is true of all of them, and is therefore a candidate for the
blind spot:

1. **Short effective memory.** The features carry EMA spans of 28 and a
   42-row lag; the GRU is trained with 512-step truncated BPTT. Sequences are
   20,000 steps. Nothing in the family can hold state over thousands of steps,
   yet the slow (span 1248) EMA block measurably helped the linear read-out
   (+0.002) even though the target decorrelates by lag ~200 - evidence that
   slow context matters for reasons other than horizon matching.
2. **Independent targets.** `t0` and `t1` are predicted by two output units
   over a shared trunk, but they correlate **-0.738** (negative in all 40
   sequences measured, range -0.937 to -0.427). No model is told this.
3. **Contemporaneous `i1`.** The second instrument enters only at the current
   row, through the same transform as `i0`. Lead-lag between instruments is
   never represented.
4. **One objective.** All are fitted to `|clip(y)|^0.75`-weighted MSE. Metric
   is a *correlation*, which is invariant to affine rescaling; MSE is not.

## 3. Design

### 3.1 Primary proposal: diagonal multi-timescale state

Replace the GRU with a **diagonal linear recurrence** (a state-space /
linear-recurrent-unit layer) feeding a small MLP:

```
h_t = a * h_{t-1} + B x_t          a in (0,1)^N, learned, elementwise
z_t = [h_t, x_t, |h_t|]            state, current row, and a scale summary
y_t = MLP(z_t)
```

The point is the cost asymmetry. A GRU's recurrence is `O(N^2)` -- that is why
128x2 costs 21 us/row and 256x2 costs 53. A diagonal recurrence is `O(N)`:
one elementwise multiply-add per state dimension. The expensive part becomes
the input projection `B`, which is a single matvec.

Budget for `N = 1024` (see section 4 for the hard limit):

| term | MACs | est. us/row |
|---|---:|---:|
| input projection `B` (112 x 1024) | 114,688 | ~4.6 |
| recurrence `a * h + Bx` | 2,048 | ~0.3 |
| read-out MLP (1024 -> 128 -> 2) | 131,328 | ~5.3 |
| **total** | | **~10** |

That is **half the GRU's 21 us/row for an eight times larger state**, and it
needs no ONNX: the recurrence is two elementwise NumPy operations, so it runs
directly in `solution.py` with no runtime overhead.

Initialise `a` spread logarithmically over spans 2 to 20,000, i.e.
`a_i = exp(-1/tau_i)` with `tau` log-spaced. The model then *starts* holding
every timescale from a few rows to a whole sequence and learns which to keep.
This addresses blind spot 1 directly, and cheaply enough to also address 3:
with a 1024-dim multi-timescale state over both instruments, `i1` lead-lag at
any horizon is representable.

Training uses a chunked associative scan (within a chunk, the recurrence is a
matrix product over a small window; state carries across chunks) so a 512-step
window costs one pass, not 512 Python iterations. Fall back to sequential
scan if the scan proves fiddly -- the GRU trained at 70k rows/s with a
sequential inner loop, which was fast enough.

### 3.2 Joint target head

Predict the rotated pair instead of the targets directly:

```
u = (t0 - t1) / 2        the anti-correlated component, high variance
v = (t0 + t1) / 2        the common component, low variance
```

then emit `t0 = v + u`, `t1 = v - u`. With `corr(t0,t1) = -0.738`, `u` carries
most of the signal and `v` most of the noise, so the head can spend capacity
where the information is, and errors in `v` are shared rather than
independent. This is a free change -- it reparameterises the output layer
only. Verify it beats the plain two-output head before keeping it; a linear
reparameterisation cannot help a linear model, so any gain must come from
optimisation dynamics.

### 3.3 Metric-aligned loss

Train on batch weighted Pearson directly rather than weighted MSE:

```
L = -0.5 * (wcorr(y_t0, p_t0) + wcorr(y_t1, p_t1))     weights |clip(y)|
```

computed per batch over a window. MSE penalises scale error that the metric
ignores -- note the organisers' GRU predicts with std 2.43 against a target
std of 1.01 and still scores 0.617. Risk: correlation loss is less stable
than MSE; if it will not train, keep MSE and treat this as optional.

## 4. Hard constraints

**Runtime is the binding constraint, not accuracy.** The limit is 4200 s on
the full test set and the scoring host is 1.6-2.1x slower than the dev
machine. One submission timed out and scored zero at 44.3 min projected; the
configuration that scored was 33.5 min. Therefore:

- **ceiling: ~50 us/row, ~33 min projected.** Treat anything above 33 min as
  a coin flip.
- current shipped path: 43.5 us/row (features 2.7, MLP ~19, GRU 21.3).
- replacing the GRU with the ~10 us/row recurrence leaves **~32 us/row**,
  room for a 1024-state model *and* the existing MLP, or a much larger
  recurrence alone.

Submission size 20 MB; `N=1024` weights are ~1 MB in float32. Not binding.

## 5. Validation protocol

The two mistakes from the last session, both of which cost a submission:

1. **Quote `blend eval half`, never `blend full`.** Blend weights are fitted
   on the first half of validation sequences. A change worth +0.00077 on the
   full set was worth +0.00017 on the honest half, and scored *below* the
   model it replaced.
2. **A gain must clear +0.003 on the honest half** to be distinguishable from
   noise, given ~70% carry-over to public and leaderboard sampling.

Additionally: validate on sequences never used for any selection, and prefer
gains that come from the model over gains that come from blend weights -- the
final ranking is on a private set and we have made many decisions against
this one validation file.

## 6. Milestones and kill criteria

| # | milestone | pass | kill |
|---|---|---|---|
| 1 | Recurrence layer trains at all; 1 epoch, 2,000 sequences | WP > 0.60 | fails to beat the linear read-out's 0.633 after 3 epochs |
| 2 | Full training, all 10,607 sequences, 6 epochs | **WP >= 0.67** alone (matches our GRU) | < 0.66 |
| 3 | Residual correlation vs the MLP | **< 0.90** | >= 0.95 -- it has joined the family, stop |
| 4 | Blend, honest half | **>= +0.003** over 0.67858 | below threshold |
| 5 | Packaged, verified, projected < 33 min | exact reproduction, within budget | over budget |

Milestone 3 is the real test and should be checked as soon as milestone 2
passes, before any blend work. Every model so far has landed at 0.96; if this
one does too, the architecture is not the blind spot and the remaining
hypotheses (targets, `i1` lead-lag, objective) should be tested separately
rather than together.

## 7. Estimated effort and odds

| stage | effort |
|---|---|
| implement layer + training loop (reuse `train_gru.py` scaffolding) | ~4 h |
| milestone 1-2 training runs | ~6 h machine |
| diversity check, blend, package, verify | ~2 h |

Honest odds: the strongest evidence *against* is that our GRU already learns
its own representation from raw columns and still landed at 0.964. The
strongest evidence *for* is that no model in the family can see past a few
hundred steps, the one long-context feature we tested did help, and the
leader's +0.014 proves the headroom exists. Call it **40%** to clear
milestone 3, and conditional on that, good odds on milestone 4 -- a genuinely
decorrelated model at even 0.65 alone would be worth more to the blend than
anything we have added since the first GRU.

## 8. What not to spend time on

Measured dead ends, with numbers in [RUNBOOK.md](RUNBOOK.md): MLP width (flat
or negative at 3.3M, 13.3M, 35M rows); `vol_norm` (-0.028); classical
microstructure features -- microprice, book imbalance, spread -- which are
unrecoverable because columns are per-column affine transformed *including
sign flips*; per-sequence volatility scaling (-0.016 even with an oracle);
stacking the blend with raw features (-0.002, the feature-linear signal is
fully extracted); stride reduction (+0.0013 local, did not survive to
public); training-objective diversity (0.94 correlation).


## 9. Outcome

Implemented in `scripts/train_ssm.py` and trained at state 1024 / head 256 on
2,000 sequences for 2 epochs.

| milestone | target | result |
|---|---|---|
| 1. trains, beats the linear read-out | > 0.60 | **0.6455** pass |
| 3. residual correlation vs the MLP | < 0.90 | **0.9556 / 0.9532** fail |

Milestone 2 (full training) was never run: milestone 3 was checked first, on
the cheap model, precisely so that a failure would cost 40 minutes instead of
6.5 hours. It failed, so training stopped there.

What this rules out: **long memory is not the shared blind spot.** The layer
held a 1024-dimensional state whose learned spans settled at a median of 406
rows with a tail to 45,000 - far beyond anything the GRU (512-step BPTT) or
the feature set (spans 28 and 1248) can represent - and still produced the
same function as everything else.

Five independent families now agree at 0.93-0.96 residual correlation while
explaining ~18% of variance: linear, smooth-nonlinear (MLP), gated-recurrent
(GRU), axis-aligned trees (GBM), and long-memory linear-recurrent (this).
Architecture is not the lever.

Worth keeping from the attempt, if the model is ever revisited:

- The diagonal recurrence really is cheap: **11.2 us/row at state 1024**
  against the GRU's 21.3 at state 256, and inference is pure NumPy with no
  `onnxruntime` dependency. If a future model needs state, this is how to buy
  it.
- Training exploits `a` being constant in time, so the recurrence is a
  convolution with kernel `a^j` evaluated by FFT: **159k rows/s at state 256,
  52k at state 1024**, versus 70k for the GRU. No sequential Python loop.
- Training/inference parity verified at max |delta| 7.7e-07, correlation
  1.000000000000 over 4,000 rows.
- `lr 2e-3` moves the log-space decay parameters much further than it moves
  `B` or the head; spans travelled from 2/200/20,000 to 1/406/45,178 in one
  epoch. A separate, lower learning rate for `log_tau` would be the first
  thing to change.

Still untested from section 3, and now the only remaining ideas in this
document: the joint target head (3.2) and the correlation loss (3.3). Both
are output-side changes, independent of architecture, and cheap to try - but
with five architectures converging, neither is likely to be worth +0.014.
