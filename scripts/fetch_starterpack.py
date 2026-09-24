#!/usr/bin/env python3
"""Partial fetcher for the Wunder Fund "Alpha Connectome" starter pack.

The official archive (https://files.wundernn.io/wnn_connectome_starterpack.zip)
is ~33.8 GB, almost all of it `datasets/train.parquet` (28 GB deflated).
The file server honours HTTP Range requests, and a ZIP central directory lives
at the end of the archive, so this script can pull individual members without
downloading everything:

  * ``--what small``         docs, baseline, utils.py, METRIC.md, requirements
  * ``--what valid``         datasets/valid.parquet + datasets/valid_mask.parquet
  * ``--what train-subset``  the first N sequences of datasets/train.parquet
                             (streams the whole deflate stream but keeps only
                             the head and the Parquet footer on disk)
  * ``--what all``           everything above

Every ZIP member is deflate-compressed, so a train subset still costs the full
28 GB of download bandwidth, but only ``--train-sequences`` worth of disk.

Only the Python standard library is needed for ``small`` and ``valid``;
``train-subset`` additionally needs ``pyarrow`` (in requirements.txt).
"""
from __future__ import annotations

import argparse
import http.client
import io
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import deque
from pathlib import Path

URL = "https://files.wundernn.io/wnn_connectome_starterpack.zip"
PREFIX = "wnn_connectome_starterpack/"
SMALL = [
    "METRIC.md",
    "README.md",
    "requirements.txt",
    "utils.py",
    "baseline/README.md",
    "baseline/solution.py",
    "baseline/baseline.onnx",
    "baseline/baseline_submission.zip",
    "docs/submission_guide.md",
    "docs/prizes.md",
    "docs/faq.md",
    "docs/timeline.md",
    "docs/data_overview.md",
    "docs/rules.md",
    "docs/quick_start.md",
    "docs/get_help.md",
]
CHUNK = 8 << 20
# files.wundernn.io sits behind Cloudflare, which answers 403 to the default
# "Python-urllib" user agent; any browser/curl-like UA is accepted.
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) wunder-fetch/1.0"}


def http_len(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def http_range(url: str, start: int, end: int):
    """Open a streaming response for bytes [start, end] (inclusive)."""
    req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={start}-{end}"})
    r = urllib.request.urlopen(req, timeout=60)
    if r.status != 206:
        r.close()
        raise RuntimeError(f"server ignored Range header (HTTP {r.status})")
    if not r.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
        r.close()
        raise RuntimeError("server returned an unexpected byte range")
    return r


def read_central_directory(url: str, total: int) -> dict[str, dict]:
    tail_len = min(total, 1 << 20)
    with http_range(url, total - tail_len, total - 1) as r:
        d = r.read()
    base = total - len(d)
    j = d.rfind(b"PK\x06\x06")  # ZIP64 end of central directory
    if j >= 0:
        (_, _, _, _, _, _, _, n, _, cd_off) = struct.unpack("<IQHHIIQQQQ", d[j : j + 56])
    else:
        i = d.rfind(b"PK\x05\x06")
        (_, _, _, _, n, _, cd_off, _) = struct.unpack("<IHHHHIIH", d[i : i + 22])
    if cd_off < base:
        raise RuntimeError("central directory larger than 1 MiB; enlarge tail_len")
    p = cd_off - base
    entries: dict[str, dict] = {}
    while p < len(d) and d[p : p + 4] == b"PK\x01\x02":
        (_, _, _, _, meth, _, _, crc, csz, usz, fnl, exl, cml, _, _, _, lho) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", d[p : p + 46]
        )
        name = d[p + 46 : p + 46 + fnl].decode()
        extra = d[p + 46 + fnl : p + 46 + fnl + exl]
        q = 0
        while q < len(extra):
            hid, hsz = struct.unpack("<HH", extra[q : q + 4])
            if hid == 1:  # ZIP64 extended information
                vals = list(struct.unpack("<" + "Q" * (hsz // 8), extra[q + 4 : q + 4 + hsz]))
                if usz == 0xFFFFFFFF:
                    usz = vals.pop(0)
                if csz == 0xFFFFFFFF:
                    csz = vals.pop(0)
                if lho == 0xFFFFFFFF:
                    lho = vals.pop(0)
            q += 4 + hsz
        entries[name] = dict(method=meth, csize=csz, usize=usz, lho=lho, crc=crc)
        p += 46 + fnl + exl + cml
    return entries


def member_data_offset(url: str, e: dict) -> int:
    """Skip the local file header (its name/extra lengths can differ from the CD)."""
    with http_range(url, e["lho"], e["lho"] + 29) as r:
        h = r.read(30)
    if h[:4] != b"PK\x03\x04":
        raise RuntimeError("bad local file header")
    fnl, exl = struct.unpack("<HH", h[26:30])
    return e["lho"] + 30 + fnl + exl


def iter_member(url: str, e: dict, log=lambda s: None):
    """Yield inflated bytes of one member, streaming."""
    start = member_data_offset(url, e)
    end = start + e["csize"] - 1
    dec = zlib.decompressobj(-15) if e["method"] == 8 else None
    done = 0
    failures = 0
    while done < e["csize"]:
        before = done
        try:
            with http_range(url, start + done, end) as r:
                while done < e["csize"]:
                    buf = r.read(min(CHUNK, e["csize"] - done))
                    if not buf:
                        raise OSError("archive response ended early")
                    out = dec.decompress(buf) if dec else buf
                    done += len(buf)
                    if out:
                        yield out
                    log(done)
        except (OSError, http.client.HTTPException, urllib.error.URLError) as exc:
            failures = 1 if done > before else failures + 1
            if failures > 5:
                raise
            print(f"\n  retrying at compressed byte {done}: {exc}", file=sys.stderr)
            time.sleep(min(2 ** failures, 30))
    if dec:
        if not dec.eof:
            raise RuntimeError("incomplete compressed archive member")
        out = dec.flush()
        if out:
            yield out


def progress(name: str, total: int):
    last = [-1]

    def log(done: int):
        pct = done * 100 // max(total, 1)
        if pct != last[0] and (pct % 5 == 0 or done == total):
            last[0] = pct
            sys.stderr.write(f"\r  {name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct}%)")
            sys.stderr.flush()

    return log


def fetch_file(url: str, e: dict, dest: Path, name: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == e["usize"]:
        print(f"  {name}: already present, skipping")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    crc = 0
    with open(tmp, "wb") as f:
        for out in iter_member(url, e, progress(name, e["csize"])):
            f.write(out)
            crc = zlib.crc32(out, crc)
    sys.stderr.write("\n")
    if tmp.stat().st_size != e["usize"] or crc & 0xFFFFFFFF != e["crc"]:
        tmp.unlink()
        raise RuntimeError(f"CRC mismatch for {name}")
    os.replace(tmp, dest)
    print(f"  {name}: ok ({e['usize'] / 1e6:.1f} MB)")


def fetch_train_subset(url: str, e: dict, dest: Path, n_sequences: int):
    """Keep the first `n_sequences` row groups of train.parquet.

    Each Parquet row group is one 20,000-row sequence (~2.75 MB deflated in the
    archive, ~2.75 MB on disk). We stream the whole inflate stream, keeping the
    head on disk and the footer (file metadata) in a rolling in-memory buffer,
    stitch them into a sparse-but-valid Parquet file, then rewrite only the
    complete row groups into a clean file with pyarrow.
    """
    import pyarrow.parquet as pq

    if dest.exists():
        print(f"  {dest.name}: already present, skipping")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # ~2.75 MB per sequence measured on the real file; keep a generous margin.
    keep_bytes = int(n_sequences * 3.2e6) + (16 << 20)
    tail_keep = 96 << 20
    sparse = dest.with_suffix(".sparse.parquet")
    tail: deque[bytes] = deque()
    tail_len = 0
    written = 0
    with open(sparse, "wb") as f:
        for out in iter_member(url, e, progress("train.parquet (stream)", e["csize"])):
            if written < keep_bytes:
                take = out[: keep_bytes - written]
                f.write(take)
                written += len(take)
            tail.append(out)
            tail_len += len(out)
            while tail and tail_len - len(tail[0]) >= tail_keep:
                tail_len -= len(tail.popleft())
        sys.stderr.write("\n")
        blob = b"".join(tail)
        if blob[-4:] != b"PAR1":
            raise RuntimeError("stream did not end with a Parquet magic")
        footer_len = struct.unpack("<I", blob[-8:-4])[0]
        footer = blob[-(footer_len + 8) :]
        if len(footer) != footer_len + 8:
            raise RuntimeError("footer larger than the retained tail; raise tail_keep")
        f.write(footer)
    pf = pq.ParquetFile(sparse)
    md = pf.metadata
    keep = []
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        end = 0
        for c in range(rg.num_columns):
            col = rg.column(c)
            off = col.dictionary_page_offset or col.data_page_offset
            end = max(end, off + col.total_compressed_size)
        if end <= written:
            keep.append(i)
        else:
            break
    keep = keep[:n_sequences]
    print(f"  train.parquet: {md.num_row_groups} sequences in full file, keeping {len(keep)}")
    with pq.ParquetWriter(dest, pf.schema_arrow, compression="snappy") as w:
        for i in keep:
            w.write_table(pf.read_row_group(i))
    sparse.unlink()
    print(f"  {dest.name}: ok ({dest.stat().st_size / 1e6:.1f} MB, {len(keep)} sequences)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent), help="repo root (default: parent of scripts/)")
    ap.add_argument("--what", choices=["small", "valid", "train-subset", "train", "all"], default="small")
    ap.add_argument("--train-sequences", type=int, default=1000, help="sequences to keep for train-subset")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--list", action="store_true", help="only list archive members")
    a = ap.parse_args()

    root = Path(a.dest)
    total = http_len(a.url)
    entries = read_central_directory(a.url, total)
    if a.list:
        for k, v in entries.items():
            print(f"{k:55s} deflated={v['csize']:>13,} raw={v['usize']:>13,}")
        return

    def ent(rel: str) -> dict:
        return entries[PREFIX + rel]

    if a.what in ("small", "all"):
        print("Fetching starter pack docs, baseline and helpers -> starterpack/")
        for rel in SMALL:
            fetch_file(a.url, ent(rel), root / "starterpack" / rel, rel)
    if a.what in ("valid", "all"):
        print("Fetching validation set -> datasets/")
        for rel in ("datasets/valid_mask.parquet", "datasets/valid.parquet"):
            fetch_file(a.url, ent(rel), root / rel, rel)
    if a.what == "train":
        print("Fetching FULL training set (28 GB) -> datasets/")
        fetch_file(a.url, ent("datasets/train.parquet"), root / "datasets/train.parquet", "datasets/train.parquet")
    if a.what in ("train-subset", "all"):
        print(f"Fetching first {a.train_sequences} training sequences -> datasets/train_subset.parquet")
        fetch_train_subset(a.url, ent("datasets/train.parquet"), root / "datasets/train_subset.parquet", a.train_sequences)


if __name__ == "__main__":
    main()
