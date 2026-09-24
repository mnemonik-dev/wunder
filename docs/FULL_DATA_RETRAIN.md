# Full-data retraining experiment — 2026-09-22

## Question and protocol

Does fitting the existing champion on all official training sequences improve
the shipped linear + GRU solution? The original linear model used 1,000
training sequences. Keep its feature configuration, ridge strength, and sample
weight exponent fixed for the first comparison, so the effect of additional
training data can be measured separately from architecture and search changes.

Use the official global clipped Weighted Pearson (WP) metric on identical
validation rows. Tune blend weights only on the first 936 validation sequence
IDs; report the other 937 sequences separately. This second half has been
inspected in previous experiments and is not a fresh hidden test set.

## Reproduced baseline

| Model | Full validation WP | Second-half WP |
|---|---:|---:|
| Organizer GRU | 0.617052289 | 0.616726638 |
| Shipped linear model | 0.633136431 | — |
| Shipped linear + GRU | 0.666325197 | 0.667989544 |

Validation contains 1,873 sequences, 37,460,000 total rows, and 3,528,273 scored
rows. The shipped blend improves on the GRU by 0.049272908 WP. A paired
sequence bootstrap (2,000 replicates, seed 42) gives a 95% interval of
[0.045287262, 0.053125724] for that difference on full validation. This interval
does not account for model selection or hidden-test distribution differences.

Rust/Python parity is exact over 39,802 predictions from two sequences. A
50-sequence smoke run reproduces WP 0.711106922 and takes 28.69 microseconds per
row on the local machine. The slice score is a regression check, not a
full-validation improvement. Timing on the competition CPU may differ.

## Refit result (2026-09-22)

The full-data refit (`refit.sh`: same feature spec, ridge lambda, weight power
and seed 42 as the champion, `--no-ga`, all 10,607 training sequences /
211,089,907 rows, 666s) changed nothing:

| Model | Full validation WP | Second-half WP |
|---|---:|---:|
| Refit linear (10,607 seqs) | 0.632926715 | 0.63310 |
| Refit linear + GRU blend | 0.665910969 | 0.666235102 |
| Shipped linear + GRU blend | 0.666325197 | 0.667989544 |

Blend weights for the refit were re-tuned on the first half of validation
sequences only (`scripts/blend.py --grid 41`); the grid selected the same
weights as shipped (linear 0.725/0.675, GRU 0.275/0.325). Paired sequence
bootstrap of refit blend vs shipped blend (2,000 replicates, seed 42,
`refit-vs-shipped.json`): full-validation delta -0.000414 WP, 95% CI
[-0.002760, +0.000855]; second-half delta -0.001754 WP, 95% CI
[-0.006184, +0.000646]. The refit fails the pre-registered decision rule
(needs a second-half win with a CI excluding 0), so **the shipped solution is
kept** and no repackaging was done. Ten times more training data bought no
measurable improvement on this linear read-out; the model appears
capacity/architecture-limited, not data-limited. Rust/Python parity for the
refit `model.json` is exact (max |diff| = 0 over 39,802 predictions).

## Artifacts and reproducibility

Local experiment files live in
`datasets/experiments/full-refit-20260922/` (ignored by Git):

- `manifest.json`: source revisions and original model metrics.
- `shipped/`: preserved original submission assets.
- `refit-command.json` and `refit.sh`: exact full-data fit command.
- `refit/`: candidate assets and training report.
- `preds_*.npz`: aligned raw predictions for scored validation rows.
- `preds_refit_blend.npz`: blended refit+GRU predictions (weights 0.725/0.675
  linear, 0.275/0.325 GRU) used for the paired comparison.
- `shipped-vs-gru.json`: baseline comparison and bootstrap intervals.
- `refit-vs-shipped.json`: refit blend vs shipped blend comparison.

Run the comparison utility from the repository root:

```bash
.venv/bin/python scripts/compare_predictions.py old.npz new.npz --out comparison.json
```

The downloader verifies archive CRC and extracted byte count and reconnects
interrupted transfers from the last received compressed byte within the same
process. Restarting the process restarts an incomplete file's transfer.
