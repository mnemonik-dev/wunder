#!/usr/bin/env bash
# Package a solution directory, run preflight and save a checksum receipt.
#   scripts/package_submission.sh [solution_dir] [output_zip] [check args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
exec "$PY" "$ROOT/scripts/package_submission.py" "$@"
