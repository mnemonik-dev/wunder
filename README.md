# wunder — Wunder Fund "Alpha Connectome" hackathon workspace

Development environment and solution workspace for the
[Wunder Fund RNN challenge / Alpha Connectome ML competition](https://wundernn.io/connectome/docs/quick_start),
prepared by following the official quick start. The solution engine is
**Neutrino** (`../neutrino`, Rust): its `evoforge` genetic algorithm searches
the feature hyper-parameters directly on the competition metric, and the
resulting champion is replayed by a tiny NumPy `solution.py`.

## The challenge in one paragraph

Each sequence is 20,000 rows of 112 anonymised order-book / trade features for
two instruments (`i0`, `i1`). Rows 0–98 are warm-up; from row 99 on the model
must return two finite predictions `[t0, t1]` for the future price move of
`i0`, processing rows strictly in order and resetting state when `seq_ix`
changes. Score: **Global Weighted Pearson (WP)** — pooled over all scored rows,
targets and predictions clipped to `[-2, 2]`, weights `|target|`, averaged over
the two targets. Inference runs offline on **1 vCPU, 16 GB RAM, 60 minutes for
1,970 sequences (39.4 M rows)**, ZIP ≤ 20 MB, 5 submissions/day. Timeline:
starts **Sep 11 2026**, final submissions **Nov 15 2026**, winners **Dec 1 2026**.
Full details: [starterpack/docs](starterpack/docs) (data overview, submission
guide, rules, FAQ, prizes) and [starterpack/METRIC.md](starterpack/METRIC.md).

## Layout

| Path | What |
|---|---|
| `scripts/setup_env.sh` | venv + `requirements.txt` + starter pack docs/baseline (quick start steps 1–2) |
| `scripts/fetch_starterpack.py` | **partial** downloader for the 33.8 GB archive: HTTP-Range reads of single ZIP members, incl. an N-sequence train subset |
| `scripts/score_solution.py` | organisers' scorer + sequence limit + µs/row timing and projected test-set time |
| `scripts/check_submission.py` | ZIP pre-flight: `solution.py` at root, ≤ 20 MB, warm-up `None`, finite shape `(2,)`, determinism, timing |
| `scripts/package_submission.sh` | `solution/` → `submission.zip` + pre-flight |
| `scripts/train_neutrino.sh` | build `neutrino-wunder`, train `solution/model.json`, parity test, quick score |
| `scripts/parity_test.py` | Rust trainer predictions == Python `solution.py` predictions |
| `solution/` | the submission: `solution.py`, `model.json`, `MODEL.md` |
| `starterpack/` | organisers' docs, `utils.py` (scorer), `baseline/` (stateful GRU, ONNX) |
| `datasets/` | git-ignored Parquet data |
| `docker/Dockerfile.scorer` | replica of the scoring image from the submission guide |
| `docs/NEUTRINO_SOLUTION.md` | how the Neutrino solution works and how to iterate on it |

## Setup

Prerequisites: Python ≥ 3.10 (scorer image is `python:3.11-slim-bookworm`),
and for training the Rust toolchain plus the `neutrino` and `evoforge`
repositories checked out **next to** this one (`../neutrino`, `../evoforge`;
`neutrino` already depends on `../../../evoforge` by path).

```bash
scripts/setup_env.sh            # venv, packages, starter pack docs + baseline (~2 MB)
scripts/setup_env.sh --valid    # + valid.parquet & valid_mask.parquet (5.6 GB)
scripts/setup_env.sh --all      # + first 1000 training sequences (~4 GB on disk)
FULL_TRAIN=1 scripts/setup_env.sh --all   # full 28 GB train.parquet instead
```

The official archive is a single 33.8 GB ZIP. `fetch_starterpack.py` reads
its central directory with HTTP Range requests and pulls only the members
you ask for; the train subset streams the whole 28 GB deflate stream once but
keeps only the first N row groups (one row group = one sequence) plus the
Parquet footer, then rewrites them as a clean `datasets/train_subset.parquet`.
`make list-archive` shows the members without downloading anything.

## Workflow

```bash
make score-baseline SEQ=10   # organisers' GRU baseline: ~0.645 WP on 10 seqs, ~42 µs/row
make train                   # GA search + refit + parity + quick score  (see below)
make score SEQ=100           # solution/ on 100 validation sequences
make score-full              # solution/ on all 1,873 validation sequences
make package                 # submission.zip + pre-flight checks
make docker-scorer           # build the scoring image; then run the check inside it
make test-rust               # unit tests of neutrino-wunder / neutrino-optimizer
```

Training knobs are environment variables of `scripts/train_neutrino.sh`
(`GA_TRAIN_SEQUENCES`, `GA_HOLDOUT_SEQUENCES`, `POPULATION`, `GA_GENERATIONS`,
`FINAL_TRAIN_SEQUENCES`, `SEED`, `EXTRA="--no-ga --ema-fast 8 ..."`).
The trainer binary itself: `../neutrino/target/release/neutrino-wunder --help`.

## The Neutrino solution

`crates/neutrino-wunder` in the neutrino repository. A causal streaming
feature transform (raw features + residuals against a fast and a slow EMA +
one-step differences) feeds a linear read-out fitted by weighted ridge
regression in closed form. The non-differentiable choices — EMA spans, which
feature groups to include, the sample-weight exponent, the ridge strength —
are searched by the `evoforge` GA (`neutrino_optimizer::run_search`) with the
fitness being the *competition metric on a held-out validation slice*. The
champion is exported as `solution/model.json`; `solution/solution.py` replays
it with four `(112, 2)` mat-vecs per row.

Results on this machine (1,000-sequence train subset; see
`solution/train_report.json` for every GA trial):

| Model | Validation WP, all 1,873 seqs | WP, 50-seq slice | µs/row (1 thread) | projected test time |
|---|---:|---:|---:|---:|
| Organisers' GRU baseline (ONNX) | 0.6171 (reported) | 0.6564 | 40.1 | 26 min |
| **Neutrino linear read-out** (`solution/model.json`) | **0.6161** (0.6157 on the 1,813 seqs the GA never saw) | **0.6580** | **10.7** | **7 min** |

GA run: population 14 × 6 generations, seed 42, 80 training / 60 hold-out
sequences per candidate (≈ 1–9 s each), champion refit on 1,000 sequences
(19.9 M rows, 120 features) in 46 s, full validation scoring in 102 s.
Champion: fast EMA span 57, slow EMA span 606, raw prices on, instrument `i1`
off, no slow-residual / one-step-difference blocks, ridge λ = 2.2e-3,
sample-weight power 0.75. Rust vs Python parity: max |Δ| = 0 over 39,802
predictions.

Design, gene schema and ideas for the hackathon are in
[docs/NEUTRINO_SOLUTION.md](docs/NEUTRINO_SOLUTION.md).

## Hackathon checklist

- [ ] `scripts/setup_env.sh --all` on the hackathon machine (needs ~10 GB disk for the subset, 34 GB for everything)
- [ ] `make score-baseline` reproduces ≈ 0.617 WP on the full validation set
- [ ] `make train` → `solution/model.json`; `make parity` is exact
- [ ] `make package` → `submission.zip` passes pre-flight; upload at https://wundernn.io/connectome/submit
- [ ] Keep `solution/train_report.json` and the seed: winners must hand over reproducible training code (see `starterpack/docs/prizes.md`)
