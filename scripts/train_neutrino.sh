#!/usr/bin/env bash
# Build the neutrino-wunder trainer from the sibling neutrino checkout and
# train a champion into solution/model.json, then run parity + local scoring.
#
#   scripts/train_neutrino.sh                # defaults (see below)
#   GA_GENERATIONS=8 POPULATION=16 scripts/train_neutrino.sh
#   EXTRA="--no-ga --ema-fast 8" scripts/train_neutrino.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEUTRINO="${NEUTRINO_DIR:-$ROOT/../neutrino}"
TRAIN="${TRAIN_PARQUET:-$ROOT/datasets/train_subset.parquet}"
VALID="${VALID_PARQUET:-$ROOT/datasets/valid.parquet}"
OUT="${MODEL_OUT:-$ROOT/solution/model.json}"

test -f "$TRAIN" || { echo "missing $TRAIN (run: make fetch-train-subset)" >&2; exit 1; }
test -f "$VALID" || { echo "missing $VALID (run: make fetch-valid)" >&2; exit 1; }

echo "== building neutrino-wunder (release)"
RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" cargo build --release --manifest-path "$NEUTRINO/Cargo.toml" -p neutrino-wunder
BIN="$NEUTRINO/target/release/neutrino-wunder"

echo "== training"
"$BIN" train \
  --train "$TRAIN" --valid "$VALID" \
  --ga-train-sequences "${GA_TRAIN_SEQUENCES:-80}" \
  --ga-holdout-sequences "${GA_HOLDOUT_SEQUENCES:-60}" \
  --final-train-sequences "${FINAL_TRAIN_SEQUENCES:-0}" \
  --eval-sequences "${EVAL_SEQUENCES:-0}" \
  --population "${POPULATION:-12}" --generations "${GA_GENERATIONS:-5}" --seed "${SEED:-42}" \
  --out "$OUT" --report "$ROOT/solution/train_report.json" ${EXTRA:-}

echo "== Rust/Python parity"
NEUTRINO_WUNDER="$BIN" "$ROOT/.venv/bin/python" "$ROOT/scripts/parity_test.py" --validation "$VALID"

echo "== quick local score (50 sequences, official scorer + timing)"
"$ROOT/.venv/bin/python" "$ROOT/scripts/score_solution.py" --validation "$VALID" --sequences 50
