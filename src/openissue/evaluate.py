"""
evaluate.py — Cluster quality evaluation.

Key design decisions
---------------------
* Silhouette scoring uses COSINE distance, which is more appropriate for
  TF-IDF vectors than Euclidean distance (TF-IDF vectors live on a
  high-dimensional unit simplex where cosine similarity is the natural metric).

* The matrix is kept sparse until sampling. Only the sampled rows are
  converted to dense for sklearn's silhouette_score. This avoids allocating
  a large dense matrix for the full dataset.

* Sampling is bounded (default 2 000 rows) and uses the configured random
  seed so results are reproducible.

* A silhouette score is a geometric signal, NOT proof of semantic
  coherence. It measures how well-separated clusters are in the TF-IDF
  feature space, which does not directly correspond to how meaningful
  the discovered themes are. Always inspect representative complaints.

* No threshold (e.g. "score > 0.3 means good") is applied here. Such
  thresholds are dataset-dependent and would be misleading.

Responsibilities:
  - Compute silhouette score safely on a bounded sample.
  - Report cluster sizes, empty clusters, and coverage.
  - Identify very small clusters that may be noise or outliers.
  - Return a structured EvaluationReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics import silhouette_score


_MAX_SILHOUETTE_SAMPLE = 2_000   # hard ceiling to avoid OOM on dense conversion


@dataclass
class EvaluationReport:
    """Cluster quality metrics for one ClusterResult."""
    n_clusters_requested: int = 0
    n_clusters_found: int = 0         # clusters actually containing documents
    empty_clusters: list[int] = field(default_factory=list)
    small_clusters: list[int] = field(default_factory=list)  # size < small_threshold
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    silhouette: Optional[float] = None
    silhouette_sample_size: Optional[int] = None  # how many rows were sampled
    coverage_pct: float = 0.0          # % of docs in non-empty clusters (always 100% for KMeans)
    small_threshold: int = 3           # clusters below this size are flagged


def evaluate(
    labels: np.ndarray,
    matrix: spmatrix,
    n_clusters_requested: int,
    empty_clusters: list[int],
    random_seed: int = 42,
    small_threshold: int = 3,
    max_sample: int = _MAX_SILHOUETTE_SAMPLE,
) -> EvaluationReport:
    """
    Compute cluster quality metrics.

    Args:
        labels:               Cluster assignment per document.
        matrix:               Sparse TF-IDF matrix (n_docs × n_features).
        n_clusters_requested: The k value passed to MiniBatchKMeans.
        empty_clusters:       Cluster IDs with zero assigned documents.
        random_seed:          Seed for reproducible sampling.
        small_threshold:      Flag clusters with fewer than this many docs.
        max_sample:           Max documents to use for silhouette scoring.

    Returns:
        EvaluationReport
    """
    report = EvaluationReport(
        n_clusters_requested=n_clusters_requested,
        empty_clusters=empty_clusters,
        small_threshold=small_threshold,
    )

    unique, counts = np.unique(labels, return_counts=True)
    report.cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
    report.n_clusters_found = len(unique)
    report.small_clusters = [
        int(k) for k, v in report.cluster_sizes.items() if v < small_threshold
    ]
    report.coverage_pct = 100.0  # MiniBatchKMeans assigns every document

    # ── Silhouette ─────────────────────────────────────────────────────
    # Requires at least 2 distinct labels and enough documents.
    n_docs = len(labels)
    n_unique = len(unique)

    if n_unique < 2:
        report.silhouette = None
        return report

    if n_docs < 2 * n_unique:
        # Not enough docs to compute a meaningful score.
        report.silhouette = None
        return report

    rng = np.random.default_rng(random_seed)

    if n_docs > max_sample:
        idx = rng.choice(n_docs, size=max_sample, replace=False)
        sample_matrix = matrix[idx]             # kept sparse
        sample_labels = labels[idx]
        report.silhouette_sample_size = max_sample
    else:
        sample_matrix = matrix                  # kept sparse
        sample_labels = labels
        report.silhouette_sample_size = n_docs

    # Guard: if the sample happens to collapse to 1 unique label, skip.
    if len(np.unique(sample_labels)) < 2:
        report.silhouette = None
        return report

    report.silhouette = float(
        silhouette_score(sample_matrix, sample_labels, metric="cosine")
    )
    return report
