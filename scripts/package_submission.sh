#!/usr/bin/env bash
# Package a solution directory into submission.zip (solution.py at the root)
# and run the pre-flight checks.
#   scripts/package_submission.sh [solution_dir] [output_zip] [check args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$(cd "${1:-$ROOT/solution}" && pwd)"
OUT="${2:-$ROOT/submission.zip}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
PY="${PYTHON:-$ROOT/.venv/bin/python}"

test -f "$SRC/solution.py" || { echo "no solution.py in $SRC" >&2; exit 1; }
rm -f "$OUT"
"$PY" - "$SRC" "$OUT" <<'PYZ'
import sys, zipfile
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
skip = {"__pycache__", "train_report.json"}
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or any(part in skip or part.startswith(".") for part in rel.parts) or p.suffix == ".pyc":
            continue
        z.write(p, str(rel))
        print(f"  + {rel} ({p.stat().st_size:,} bytes)")
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
PYZ
"$PY" "$ROOT/scripts/check_submission.py" "$OUT" "${@:3}"
