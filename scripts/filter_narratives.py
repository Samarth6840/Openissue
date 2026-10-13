"""Filter the CFPB FOIA narratives dump down to rows with a non-empty narrative.

Streaming (no full load into memory). Keeps every column; only drops rows where
"Consumer complaint narrative" is missing/blank. Does not dedupe or alter text.

Usage:
    python3 scripts/filter_narratives.py
    python3 scripts/filter_narratives.py --input /path/to/file.csv --output data/raw/foo.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
from pathlib import Path

NARRATIVE_COL = "Consumer complaint narrative"
DEFAULT_INPUT = "/Users/samarthjoshi/Downloads/complaints_narratives.csv"
DEFAULT_OUTPUT = "data/raw/cfpb_narratives_nonempty.csv"


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KB.
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter rows with a non-empty Consumer complaint narrative.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    kept_count = 0
    dropped_count = 0

    with inp.open("r", newline="", encoding="utf-8") as src, outp.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        header = next(reader)
        if NARRATIVE_COL not in header:
            print(f"ERROR: column {NARRATIVE_COL!r} not found. Header: {header}", file=sys.stderr)
            return 1
        narrative_idx = header.index(NARRATIVE_COL)
        writer.writerow(header)
        for row in reader:
            read_count += 1
            if narrative_idx >= len(row):
                dropped_count += 1
                continue
            if (row[narrative_idx] or "").strip():
                writer.writerow(row)
                kept_count += 1
            else:
                dropped_count += 1

    size = outp.stat().st_size
    print(f"Rows read:   {read_count}")
    print(f"Rows kept:   {kept_count}")
    print(f"Rows dropped: {dropped_count}")
    print(f"File size (bytes): {size}")
    print(f"Peak RSS (MB): {_peak_rss_mb():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
