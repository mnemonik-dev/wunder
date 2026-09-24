#!/usr/bin/env python3
"""Build and preflight a submission, retaining a JSON receipt beside the ZIP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
ASSET_SUFFIXES = {".py", ".json", ".onnx", ".npz", ".npy", ".bin"}


def build_archive(source: Path, output: Path) -> list[dict]:
    """Snapshot runtime assets with stable ordering, timestamps and permissions."""
    if not (source / "solution.py").is_file():
        raise ValueError(f"missing {source / 'solution.py'}")
    files = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.suffix not in ASSET_SUFFIXES or path.name == "train_report.json":
                continue
            if path.is_symlink():
                raise ValueError(f"copy symlinked assets into the solution directory first: {path}")
            if not path.is_file():
                continue
            data = path.read_bytes()
            entry = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
            files.append({"path": relative.as_posix(), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    return files


def publish(archive: Path, receipt: Path, output: Path, receipt_output: Path):
    """Exclusive publication: never replace an earlier iteration's artifacts."""
    created = []
    try:
        for src, dst in ((receipt, receipt_output), (archive, output)):
            os.link(src, dst)
            created.append(dst)
    except OSError:
        for path in created:
            path.unlink()
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("solution", nargs="?", default=str(ROOT / "solution"))
    ap.add_argument("output", nargs="?", help="ZIP path; defaults to a dated file in submissions/")
    ap.add_argument("--validation", default=str(ROOT / "datasets" / "valid.parquet"))
    ap.add_argument("--sequences", type=int, default=2, help="preflight sequences (default: 2)")
    args = ap.parse_args()
    if args.sequences < 2:
        ap.error("use at least two sequences to exercise sequence transitions")
    source = Path(args.solution).resolve()
    validation = Path(args.validation).resolve()
    now = datetime.now(timezone.utc)
    output = (Path(args.output) if args.output else ROOT / "submissions" /
              f"submission-{now.strftime('%Y%m%dT%H%M%S%fZ')}.zip").resolve()
    receipt_output = output.with_suffix(output.suffix + ".json")
    if output.suffix != ".zip":
        ap.error("output must have a .zip suffix")
    if not validation.is_file():
        ap.error(f"validation file is missing: {validation}")
    if output.exists() or receipt_output.exists():
        ap.error("output or receipt already exists; choose a new filename")
    if source == output.parent or source in output.parents:
        ap.error("write the archive outside the solution directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".package-", dir=output.parent) as tmp:
        staged = Path(tmp) / "submission.zip"
        files = build_archive(source, staged)
        if staged.stat().st_size > 20 * 1024 * 1024:
            ap.error("archive exceeds the preflight checker's 20 MiB limit")
        command = [sys.executable, str(ROOT / "scripts" / "check_submission.py"),
                   str(staged), "--validation", str(validation),
                   "--sequences", str(args.sequences)]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode:
            raise SystemExit("Preflight failed; no final ZIP or receipt was published.")
        with zipfile.ZipFile(staged) as archive:
            model = json.loads(archive.read("model.json")) if "model.json" in archive.namelist() else {}
            blend = json.loads(archive.read("blend.json")) if "blend.json" in archive.namelist() else None
        receipt = {
            "created_utc": now.isoformat(), "source_directory": str(source),
            "archive": output.name, "archive_bytes": staged.stat().st_size,
            "archive_sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
            "files": files, "python": sys.version,
            "model_summary": {k: model[k] for k in ("format", "train_sequences", "train_rows", "metrics") if k in model},
            "blend": blend,
            "preflight": {"passed": True, "validation": str(validation),
                          "requested_sequences": args.sequences, "stdout": result.stdout,
                          "stderr": result.stderr},
            "note": "Preflight is a smoke check. Embedded model/blend metrics are copied metadata, not newly measured full-validation scores.",
        }
        staged_receipt = Path(tmp) / "receipt.json"
        staged_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        publish(staged, staged_receipt, output, receipt_output)
    print(f"Submission: {output}\nReceipt: {receipt_output}")


if __name__ == "__main__":
    main()
