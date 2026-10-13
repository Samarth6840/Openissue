import numpy as np
from scipy.sparse import csr_matrix
from openissue.evaluate import evaluate, EvaluationReport
from openissue.cluster import fit


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


def test_evaluate_basic():
    res = fit(SAMPLE_TEXTS, n_clusters=3, min_df=1, random_seed=42)
    eval_rep = evaluate(
        labels=res.labels,
        matrix=res.matrix,
        n_clusters_requested=3,
        empty_clusters=res.empty_clusters,
        random_seed=42,
        small_threshold=2,
    )
    assert isinstance(eval_rep, EvaluationReport)
    assert eval_rep.n_clusters_found <= 3
    assert eval_rep.silhouette is not None
    assert -1.0 <= eval_rep.silhouette <= 1.0
    assert eval_rep.silhouette_sample_size == len(SAMPLE_TEXTS)


def test_evaluate_bounded_sampling():
    res = fit(SAMPLE_TEXTS, n_clusters=3, min_df=1, random_seed=42)
    eval_rep = evaluate(
        labels=res.labels,
        matrix=res.matrix,
        n_clusters_requested=3,
        empty_clusters=res.empty_clusters,
        random_seed=42,
        max_sample=6,
    )
    assert eval_rep.silhouette_sample_size == 6


def test_evaluate_undefined_single_cluster():
    labels = np.zeros(10, dtype=int)
    matrix = csr_matrix((10, 5))
    eval_rep = evaluate(
        labels=labels,
        matrix=matrix,
        n_clusters_requested=2,
        empty_clusters=[1],
    )
    # When all points have the same label, silhouette is undefined
    assert eval_rep.silhouette is None
    assert eval_rep.n_clusters_found == 1
