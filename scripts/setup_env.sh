#!/usr/bin/env bash
# One-shot development environment for the Wunder Fund Alpha Connectome
# challenge, following https://wundernn.io/connectome/docs/quick_start.
#
#   scripts/setup_env.sh            # venv + packages + starter pack docs/baseline
#   scripts/setup_env.sh --valid    # ... plus datasets/valid.parquet (5.6 GB)
#   scripts/setup_env.sh --all      # ... plus a 1000-sequence train subset
#   FULL_TRAIN=1 scripts/setup_env.sh --all   # full 28 GB train.parquet instead
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
assert sys.version_info >= (3, 10), f"Python 3.10+ required, found {sys.version}"
print(f"python {sys.version.split()[0]} ok (scorer uses python:3.11-slim-bookworm)")
PY

if [ ! -x .venv/bin/python ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "venv ready: $(.venv/bin/python -c 'import numpy, pyarrow, onnxruntime; print("numpy", numpy.__version__, "pyarrow", pyarrow.__version__, "onnxruntime", onnxruntime.__version__)')"

.venv/bin/python scripts/fetch_starterpack.py --what small

case "${1:-}" in
  --valid) .venv/bin/python scripts/fetch_starterpack.py --what valid ;;
  --all)
    .venv/bin/python scripts/fetch_starterpack.py --what valid
    if [ "${FULL_TRAIN:-0}" = "1" ]; then
      .venv/bin/python scripts/fetch_starterpack.py --what train
    else
      .venv/bin/python scripts/fetch_starterpack.py --what train-subset --train-sequences "${TRAIN_SEQUENCES:-1000}"
    fi ;;
  "") ;;
  *) echo "unknown option $1" >&2; exit 2 ;;
esac

if command -v cargo >/dev/null 2>&1 && [ -d ../neutrino ] && [ -d ../evoforge ]; then
  echo "rust toolchain: $(cargo --version); neutrino + evoforge checkouts found next to this repo"
else
  echo "note: cargo, ../neutrino or ../evoforge missing - the neutrino trainer needs all three (see README)"
fi
echo "done. Next: make score-baseline (needs datasets/valid.parquet) or make train"
