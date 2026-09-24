# Wunder Improvement and Recovery Plan

Last checked: 2026-09-22, approximately 01:28 Europe/Moscow.

## Objective and order

Improve the competition submission through controlled experiments. Complete
each stage and record its outcome before changing the next factor:

1. Download and verify the full official training data.
2. Refit the shipped feature configuration on all training sequences.
3. Compare the fixed blend, then tune blend weights on the designated fit half.
4. Integrate warm starts and representative training-sequence sampling.
5. Evaluate adaptive mutation and restarts separately under equal search budgets.
6. Integrate grammar-generated causal features and their Python inference mirror.

The generation-statistics fix is already integrated. Regularized MLP experiments
are a separate unfinished research branch; do not mix them into the full-data
refit comparison.

## Current checkpoint

- [x] Python 3.10 environment with pinned inference dependencies installed.
- [x] Release trainer built with `RUSTFLAGS='-C target-cpu=native'`.
- [x] Relevant Rust tests passed.
- [x] Validation data and mask downloaded and archive CRC-verified.
- [x] Shipped linear predictions passed Rust/Python parity: 39,802 predictions,
  maximum absolute difference zero on the checked two sequences.
- [x] Full validation scores and GRU/shipped prediction caches reproduced.
- [ ] Training download complete: last observation approximately 65%, 18 GiB
  written, 27 GiB free. These figures are historical; check them again.
- [ ] Full-data refit complete.
- [ ] Refit parity, fixed-blend comparison, retuned-blend comparison and preflight.
- [ ] Warm starts and representative sampling integrated and evaluated.
- [ ] Adaptive mutation/restarts evaluated.
- [ ] Grammar feature pipeline integrated and evaluated.

At the last check, downloader PID 9739 was active. A waiting shell, PID 17270,
was queued to start `refit.sh` once `datasets/train.parquet` appeared. PIDs can
be reused: verify command lines before acting. Do not start duplicate jobs or
stop another session's work.

## Paths and preserved evidence

Run the commands below from the `wunder/` repository, not its parent workspace.

```sh
cd /Users/syi/src/wunder/wunder
```

Experiment directory: `datasets/experiments/full-refit-20260922/`.
It contains `manifest.json`, `refit-command.json`, `refit.sh`, the original
assets in `shipped/`, candidate assets in `refit/`, logs and prediction caches.
**This directory is Git-ignored.** Preserve it separately when moving machines;
this plan is outside the ignored directory.

The candidate directory initially contains copies of the old model. Its
existence alone does not prove retraining finished. Check `refit.log`, the new
report and `model.json.train_sequences` against training Parquet metadata.

Baseline source revisions recorded by the experiment:

- Wunder: `4d99689a749b1ec45fed715d07300a75a575b3f1`
- Neutrino: `5108673f8d59c0d1a0cf8e85205ac5eed38081b2`
- EvoForge: `edbc30c53253e923826b68c4e96be30c2f60f7de`

Local downloader recovery and comparison-script changes are additional to
these revisions. Preserve uncommitted changes; do not reset repositories.

## Restart procedure

First inspect state without changing it:

```sh
df -h .
ls -lh datasets/train.parquet*
tail -c 1200 datasets/download-train.log
ps -axo pid,etime,command | rg 'fetch_starterpack.py|neutrino-wunder|refit.sh'
```

Full training data occupies 29,240,219,011 bytes; validation plus its mask
occupies about 5.7 GB. Check remaining capacity before downloading or exporting
large feature matrices. The downloader inflates directly into a `.part` file,
so a second complete compressed archive is unnecessary.

If the downloader is still active, let it finish. If it has stopped and there
is no verified final file, rerun only after confirming no writer remains:

```sh
.venv/bin/python scripts/fetch_starterpack.py --what train > datasets/download-train.log 2>&1
```

Interrupted HTTP connections recover within the running process. A process
restart cannot resume the DEFLATE state from the inflated `.part` file: this
command restarts that member from the beginning. Never rename `.part` to
`.parquet` manually. Successful download checks size and CRC before renaming.

If the experiment directory was lost, first copy the still-shipped solution
into a new experiment directory and record its hashes/revisions. Do not assume
that today's `solution/` still matches this plan's baseline.

## Full-data refit

If the queued training shell still exists, do not start a second trainer.
Otherwise, after the dataset is verified, use the saved command:

```sh
sh datasets/experiments/full-refit-20260922/refit.sh > datasets/experiments/full-refit-20260922/refit.log 2>&1
```

`refit.sh` currently contains absolute paths. If the workspace moved, adjust
those paths or use this equivalent command, creating the output directory first:

```sh
../neutrino/target/release/neutrino-wunder --threads 10 train \
  --train datasets/train.parquet --valid datasets/valid.parquet \
  --no-ga --seed 42 --final-train-sequences 0 \
  --ema-fast 28 --ema-mid 16 --ema-slow 1248 --diff-lag 42 \
  --vol-span 1236 --lambda 0.0003021582483841608 --weight-power 0.75 \
  --raw-prices --no-dslow \
  --out datasets/experiments/full-refit-20260922/refit/model.json \
  --report datasets/experiments/full-refit-20260922/refit/train_report.json
```

This holds the feature configuration fixed, fits on all training sequences,
and does not run a new GA. The organizer's GRU remains unchanged.

## Evaluation and promotion gate

Verified baselines on 1,873 validation sequences / 3,528,273 scored rows:

| Model | Full WP | Second-half WP |
| --- | ---: | ---: |
| Organizer GRU | 0.617052289 | 0.616726638 |
| Shipped linear | 0.633136431 | Not separately recorded here |
| Shipped blend | 0.666325197 | 0.667989544 |

1. Run `scripts/parity_test.py --solution <candidate-dir>` with `.venv/bin/python`.
2. Cache the candidate's fixed blend with `scripts/predict_valid.py` and compare
   it against `preds_shipped.npz` using `scripts/compare_predictions.py`.
3. To cache raw linear predictions, copy candidate `solution.py` and `model.json`
   into a separate directory without `blend.json`. Do not remove the candidate's
   configuration or modify the preserved baseline.
4. Run `scripts/blend.py <linear-cache> <gru-cache> --names linear,gru --out
   <candidate-dir>/blend.json`. Use the existing `preds_gru.npz` cache.
5. Re-cache the actual retuned candidate and compare against the same shipped
   cache. Record per-target/full/second-half WP and paired sequence-bootstrap
   intervals. Keep the fixed-blend result too.
6. Run the candidate's 50-sequence smoke score and
   `scripts/package_submission.sh <candidate-dir> <candidate-zip>`.
7. Promote only a candidate with a supported second-half improvement, acceptable
   full-validation behavior, parity and preflight success. If uncertain, keep
   the shipped solution and document the outcome. Record final model/configuration
   hashes, commands, source revisions and reports before moving to the next stage.

Blend weights use the first half of sorted validation sequence IDs; report the
other half separately. That other half has already been inspected historically:
it is not a fresh hidden test. Bootstrap intervals exclude selection bias and
distribution shift. A 50-sequence score is a smoke check, not comparable to the
full-validation score. No leaderboard claim exists without an actual submission.

## EvoForge integration stages

**Warm starts and sampling:** preserve default behavior; add an opt-in champion
seed to `neutrino-optimizer::run_search` and `--warm-start model.json` to the
Wunder trainer. Test feature-parameter/genome conversion and schema mismatch
handling. Introduce reproducible sampling across training sequences, recording
the selected IDs. Compare sampling first, then seeded versus random initialization
on identical samples, seeds and fitness-evaluation budgets.

**Adaptive mutation and restarts:** expose independent opt-in settings. Compare
ordinary search, adaptation alone, restarts alone, and their combination using
the same samples and budgets over several seeds. Record unique evaluations,
runtime and validation outcomes. Keep default seeded behavior compatible.

**Grammar-generated features:** integrate the existing EvoForge DSL into a new
Wunder training path; serialize causal expressions and their state requirements.
Implement the matching Python evaluator, sequence resets, numerical guards and
Rust/Python parity. Start with small feature/depth budgets and check CPU inference
cost before larger searches. Library support alone does not constitute a working
submission or a demonstrated score improvement.

After every stage, update this plan's checkpoint and add an outcome report in
`docs/` so a future session can resume without relying on chat history.
