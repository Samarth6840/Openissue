"""
cluster.py — TF-IDF vectorisation and MiniBatchKMeans clustering.

Architectural notes
-------------------
* TF-IDF matrices are kept SPARSE throughout. X.toarray() is never called
  inside this module. Callers that need a dense matrix for silhouette scoring
  must convert a bounded sample themselves (see evaluate.py).

* MiniBatchKMeans alone does NOT make the pipeline distributed or suitable
  for unlimited data. It processes mini-batches locally. For very large
  datasets (millions of rows) you would need Spark MLlib or similar
  infrastructure. Do not claim scalability without a benchmark.

* The baseline is TF-IDF + MiniBatchKMeans. Do not replace it with another
  model without running a comparison on the same dataset.

Responsibilities:
  - Fit TF-IDF vectoriser on normalised complaint texts.
  - Guard against empty vocabularies (min_df too high, tiny corpus).
  - Fit MiniBatchKMeans and return sparse feature matrix + labels.
  - Extract top centroid terms per cluster for human interpretation.
  - Guard against n_clusters exceeding the number of usable documents.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


class ClusterResult:
    """Holds everything produced by a single clustering run."""

    def __init__(
        self,
        labels: np.ndarray,
        vectorizer: TfidfVectorizer,
        matrix: spmatrix,          # sparse — intentionally kept sparse
        model: MiniBatchKMeans,
        top_terms: dict[int, str],
        n_features: int,
    ):
        self.labels = labels
        self.vectorizer = vectorizer
        self.matrix = matrix       # scipy sparse matrix
        self.model = model
        self.top_terms = top_terms # {cluster_id: "term1, term2, ..."}
        self.n_features = n_features

    @property
    def n_clusters(self) -> int:
        return len(np.unique(self.labels))

    @property
    def empty_clusters(self) -> list[int]:
        """Cluster IDs present in the model but assigned to zero documents."""
        assigned = set(self.labels.tolist())
        all_ids = set(range(self.model.n_clusters))
        return sorted(all_ids - assigned)


def fit(
    texts: list[str],
    n_clusters: int,
    max_features: int = 3000,
    min_df: int = 2,
    n_top_terms: int = 8,
    random_seed: int = 42,
    max_df: float = 1.0,
    sublinear_tf: bool = False,
    ngram_range: tuple = (1, 1),
    fit_on_unique: bool = False,
) -> ClusterResult:
    """
    Vectorise texts with TF-IDF and cluster with MiniBatchKMeans.

    Args:
        texts:        List of normalised complaint strings.
        n_clusters:   Number of clusters to discover.
        max_features: Maximum TF-IDF vocabulary size.
        min_df:       Minimum document frequency for a token to be kept.
        n_top_terms:  Number of top centroid terms to extract per cluster.
        random_seed:  Seed for reproducibility.

    Returns:
        ClusterResult

    Raises:
        ValueError: If the corpus is too small, the vocabulary is empty,
                    or n_clusters exceeds the number of usable documents.
    """
    if not texts:
        raise ValueError("No texts provided for clustering.")

    n_docs = len(texts)
    unique_texts = None
    if fit_on_unique:
        unique_texts = list(dict.fromkeys(texts))
        if len(unique_texts) < n_clusters:
            raise ValueError(
                f"fit_on_unique needs at least n_clusters unique texts "
                f"({len(unique_texts)} < {n_clusters})."
            )

    fit_corpus = unique_texts if unique_texts is not None else texts

    # ── Vectorise ──────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        ngram_range=ngram_range,
    )
    try:
        if fit_on_unique:
            X_fit = vectorizer.fit_transform(fit_corpus)
            X_all = vectorizer.transform(texts)
        else:
            X_fit = X_all = vectorizer.fit_transform(texts)
    except ValueError as exc:
        # sklearn raises ValueError when the vocabulary is empty
        raise ValueError(
            f"TF-IDF produced an empty vocabulary. "
            f"Try reducing min_df (currently {min_df}) or adding more text. "
            f"Original error: {exc}"
        ) from exc

    if X_all.shape[1] == 0:
        raise ValueError(
            "TF-IDF vocabulary is empty after filtering. "
            f"min_df={min_df} may be too high for {n_docs} documents."
        )

    # ── Guard n_clusters ───────────────────────────────────────────────
    # MiniBatchKMeans requires n_clusters <= number of fitted documents.
    n_fit = len(fit_corpus)
    if n_clusters > n_fit:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed the number of "
            f"fitted documents ({n_fit}). Reduce the cluster count."
        )
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    # ── Cluster ────────────────────────────────────────────────────────
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(random_seed),
        n_init=10,
    )
    model.fit(X_fit)
    labels = model.predict(X_all)

    # ── Extract top terms ──────────────────────────────────────────────
    top_terms = _top_terms(vectorizer, model, n_top_terms)

    return ClusterResult(
        labels=labels,
        vectorizer=vectorizer,
        matrix=X_all,          # sparse
        model=model,
        top_terms=top_terms,
        n_features=X_all.shape[1],
    )


def _top_terms(
    vectorizer: TfidfVectorizer,
    model: MiniBatchKMeans,
    n_terms: int,
) -> dict[int, str]:
    """
    Return top TF-IDF centroid terms per cluster.

    Limitation: centroid terms summarise the average TF-IDF direction of
    each cluster. They do NOT guarantee semantic coherence; always inspect
    representative complaints alongside the terms.
    """
    feature_names = vectorizer.get_feature_names_out()
    centroids = model.cluster_centers_   # shape: (n_clusters, n_features)
    result: dict[int, str] = {}
    for i, centroid in enumerate(centroids):
        sorted_idx = centroid.argsort()[::-1]
        top_idx = [idx for idx in sorted_idx if centroid[idx] > 0.0 and str(feature_names[idx]) != "money_amount"][:n_terms]
        result[i] = ", ".join(feature_names[top_idx])
    return result
