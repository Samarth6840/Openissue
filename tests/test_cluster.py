import pytest
import numpy as np
from scipy.sparse import issparse
from openissue.cluster import fit, ClusterResult


SAMPLE_TEXTS = [
    "battery drains quickly after update",
    "phone battery life is very poor",
    "battery drops from 100 to 20 fast",
    "screen flickers and displays black lines",
    "display cracked easily after a short fall",
    "broken glass on screen",
    "package arrived 2 weeks late",
    "delivery delayed and package lost in transit",
    "shipping took forever to deliver item",
    "customer support never answered my phone call",
    "support agent was unhelpful and rude",
    "no response from customer support team",
]


def test_fit_success_sparse_matrix():
    result = fit(
        SAMPLE_TEXTS,
        n_clusters=3,
        max_features=500,
        min_df=1,
        n_top_terms=4,
        random_seed=42,
    )
    assert isinstance(result, ClusterResult)
    assert len(result.labels) == len(SAMPLE_TEXTS)
    assert issparse(result.matrix)  # Kept sparse
    assert result.matrix.shape[0] == len(SAMPLE_TEXTS)
    assert len(result.top_terms) == 3
    for tid, terms in result.top_terms.items():
        assert isinstance(terms, str)
        assert len(terms) > 0


def test_fit_reproducible_seed():
    r1 = fit(SAMPLE_TEXTS, n_clusters=3, min_df=1, random_seed=99)
    r2 = fit(SAMPLE_TEXTS, n_clusters=3, min_df=1, random_seed=99)
    np.testing.assert_array_equal(r1.labels, r2.labels)


def test_fit_invalid_cluster_counts():
    with pytest.raises(ValueError, match="cannot exceed"):
        fit(SAMPLE_TEXTS[:3], n_clusters=5, min_df=1)

    with pytest.raises(ValueError, match="at least 2"):
        fit(SAMPLE_TEXTS, n_clusters=1, min_df=1)


def test_fit_empty_texts():
    with pytest.raises(ValueError, match="No texts provided"):
        fit([], n_clusters=2)


def test_fit_empty_vocabulary():
    # min_df set higher than total docs so no words meet threshold
    with pytest.raises(ValueError, match="empty vocabulary"):
        fit(SAMPLE_TEXTS, n_clusters=2, min_df=100)


def test_top_terms_excludes_zero_centroid_weights():
    # Distinct documents with isolated vocabularies
    texts = [
        "battery battery battery",
        "battery battery",
        "delivery package shipment",
        "delivery package",
    ]
    # Requesting n_top_terms=10 when each cluster has far fewer terms
    result = fit(texts, n_clusters=2, min_df=1, n_top_terms=10, random_seed=42)
    feature_names = result.vectorizer.get_feature_names_out()
    
    for cluster_id in range(2):
        terms = [t.strip() for t in result.top_terms[cluster_id].split(",") if t.strip()]
        centroid = result.model.cluster_centers_[cluster_id]
        for term in terms:
            term_idx = list(feature_names).index(term)
            assert centroid[term_idx] > 0.0, f"Term '{term}' has non-positive weight {centroid[term_idx]}"
        # Ensure no term with 0.0 weight was included
        assert len(terms) < len(feature_names)


def test_fit_on_unique_preserves_row_count():
    from src.openissue.cluster import fit
    texts = ["alpha alpha", "alpha alpha", "beta beta", "gamma gamma", "beta beta"]
    result = fit(texts=texts, n_clusters=2, max_features=100, min_df=1, random_seed=42, fit_on_unique=True)
    assert len(result.labels) == len(texts)
    assert result.matrix.shape[0] == len(texts)
    # template shares can be computed
    from collections import Counter
    counts = Counter(result.labels.tolist())
    assert sum(counts.values()) == len(texts)
