"""Benchmark the OpenIssue NLP pipeline at several sample sizes.

Each sample size runs in a *fresh subprocess* (so peak RSS is not inflated
cumulatively across sizes). Every child prints a JSON array of per-stage
records to stdout; this module consolidates them into outputs/reports/benchmark.csv.

Usage:
    python3 scripts/benchmark.py
    python3 scripts/benchmark.py --sizes 1000 5000 20000
    python3 scripts/benchmark.py --worker --n 5000 --input data/raw/cfpb_narratives_nonempty.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.openissue import ingest, validate, preprocess  # noqa: E402

TEXT_COL = "Consumer complaint narrative"
DATE_COL = "Date received"
CAT_COL = "Product"
NARRATIVE_IDX = None
DEFAULT_SIZES = [1000, 5000, 20000]
DEFAULT_INPUT = "data/raw/cfpb_narratives_nonempty.csv"
DEFAULT_OUTPUT = "outputs/reports/benchmark.csv"


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    return ru / 1024.0


def _reservoir_sample(path: str, n: int, seed: int) -> tuple[list[str], list[list[str]], list[str]]:
    """Stream the CSV, keep n rows via reservoir sampling, return header, rows, narratives."""
    rng = random.Random(seed)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        narr_idx = header.index(TEXT_COL)
        sample_rows: list[list[str]] = []
        total = 0
        for row in reader:
            total += 1
            if len(sample_rows) < n:
                sample_rows.append(row)
            else:
                j = rng.randint(0, total - 1)
                if j < n:
                    sample_rows[j] = row
    narrs = [r[narr_idx] for r in sample_rows]
    return header, sample_rows, narrs


def run_worker(n: int, input_path: str, n_clusters: int, max_features: int, min_df: int, seed: int, clean_masked: bool = False) -> list[dict]:
    _clean = clean_masked
    # Baseline RSS/peak right after imports (logged as the reference; before data work).
    baseline_rss = _peak_rss_mb()
    records: list[dict] = []

    def log(stage: str, seconds: float, tr_peak_mb: float, avg_chars: float) -> None:
        records.append(
            {
                "N": n,
                "stage": stage,
                "seconds": round(seconds, 6),
                "tracemalloc_peak_mb": round(tr_peak_mb, 4),
                "rss_growth_over_baseline_mb": round(_peak_rss_mb() - baseline_rss, 4),
                "avg_chars_per_row": round(avg_chars, 4),
            }
        )

    avg_chars = 0.0

    # Stage 0 selection (reservoir sample) — captures avg_chars_per_row; not timed.
    header, rows, narrs = _reservoir_sample(input_path, n, seed)
    avg_chars = statistics.fmean(len(str(x)) for x in narrs) if narrs else 0.0
    # Build a mini CSV text for the ingest stage
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(header)
    for r in rows:
        csv.writer(buf, lineterminator="\n").writerow(r)
    sample_csv = buf.getvalue()

    # ---- ingest ----
    tracemalloc.start()
    t0 = time.perf_counter()
    df = ingest.load_csv_string(sample_csv)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log("ingest", elapsed, peak / (1024 * 1024), avg_chars)

    # ---- validate ----
    tracemalloc.start()
    t0 = time.perf_counter()
    df_valid, report = validate.validate(df, text_col=TEXT_COL, date_col=DATE_COL, deduplicate=False)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log("validate", elapsed, peak / (1024 * 1024), avg_chars)

    # ---- preprocess ----
    tracemalloc.start()
    t0 = time.perf_counter()
    norm = preprocess.normalise_series(df_valid[TEXT_COL], clean_masked=_clean)
    texts = norm.tolist()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log("preprocess", elapsed, peak / (1024 * 1024), avg_chars)

    # ---- tfidf ----
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=min_df)
    tracemalloc.start()
    t0 = time.perf_counter()
    X = vectorizer.fit_transform(texts)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log("tfidf", elapsed, peak / (1024 * 1024), avg_chars)

    # ---- minibatch_kmeans ----
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
    tracemalloc.start()
    t0 = time.perf_counter()
    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log("minibatch_kmeans", elapsed, peak / (1024 * 1024), avg_chars)

    # ---- silhouette (sampled, sparse-safe) ----
    tracemalloc.start()
    t0 = time.perf_counter()
    from src.openissue.evaluate import evaluate as _eval
    from numpy import array
    report = _eval(labels=labels, matrix=X, n_clusters_requested=n_clusters, empty_clusters=[], random_seed=seed)
    _ = report.silhouette
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log("silhouette", elapsed, peak / (1024 * 1024), avg_chars)

    # keep X alive until here
    del X, labels, model, vectorizer, texts, norm, df_valid, df
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenIssue stages.")
    parser.add_argument("--worker", action="store_true", help="Child worker mode (single size).")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-masked", dest="clean_masked", action="store_true", help="Apply masked-token cleaning in preprocess stage.")
    args = parser.parse_args()

    if args.worker:
        if args.n is None:
            print("worker requires --n", file=sys.stderr)
            return 2
        try:
            recs = run_worker(args.n, args.input, args.n_clusters, args.max_features, args.min_df, args.seed, clean_masked=args.clean_masked)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1
        print(json.dumps(recs))
        return 0

    # Parent mode
    all_records: list[dict] = []
    for sz in args.sizes:
        size_mb_before = _peak_rss_mb()
        print(f"=== benchmarking N={sz} in fresh subprocess ===", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--n",
            str(sz),
            "--input",
            args.input,
            "--n-clusters",
            str(args.n_clusters),
            "--max-features",
            str(args.max_features),
            "--min-df",
            str(args.min_df),
            "--seed",
            str(args.seed),
        ] + (["--clean-masked"] if args.clean_masked else [])

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"subprocess failed for N={sz} (code {proc.returncode})", file=sys.stderr)
            print("--- stderr ---", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            print("--- stdout ---", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            return proc.returncode
        recs = json.loads(proc.stdout)
        all_records.extend(recs)
        print(proc.stderr if False else "", end="")
    # write CSV
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["N", "stage", "seconds", "tracemalloc_peak_mb", "rss_growth_over_baseline_mb", "avg_chars_per_row"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in all_records:
            w.writerow(r)
    print(f"Wrote {len(all_records)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
