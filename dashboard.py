"""
OpenIssue — Complaint Intelligence Dashboard

Caveman + Ponytail style: smallest working solution, no new dependencies.

This file handles PRESENTATION and USER INTERACTION only.
Core NLP logic lives in src/openissue/*.py for testability and reuse.
"""

import sys
import os

# Make the src package importable when running: streamlit run dashboard.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from openissue.ingest import load_csv_bytes, load_csv_string
from openissue.validate import validate, ValidationReport
from openissue.preprocess import normalise_series
from openissue import cluster as _cluster
from openissue.evaluate import evaluate as _evaluate
from openissue.trends import compute_trends, flag_emerging

try:
    from sentence_transformers import SentenceTransformer
    _EMBED_AVAILABLE = True
except Exception:  # optional extra may not be installed
    SentenceTransformer = None
    _EMBED_AVAILABLE = False


# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────

from openissue.sample import SAMPLE_CSV


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(page_title="OpenIssue Dashboard", layout="wide")
st.title("🔍 OpenIssue — Complaint Intelligence Dashboard")
st.caption(
    "Discover recurring issue themes in user complaints. "
    "TF-IDF + MiniBatchKMeans baseline. Results are exploratory — "
    "always inspect representative complaints before acting on flags."
)


# ─────────────────────────────────────────────
# SIDEBAR SETTINGS (all configurable — no hardcoded params)
# ─────────────────────────────────────────────

# sidebar rendered after validation (see below)
# ─────────────────────────────────────────────
# SECTION 1: DATA LOAD
# ─────────────────────────────────────────────

st.header("1. Data")

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
with col2:
    use_sample = st.button("Use sample data (20 complaints)")

# Use session_state to keep sample data loaded across reruns.
if use_sample:
    st.session_state["_data_source"] = "sample"
if uploaded is not None:
    st.session_state["_data_source"] = "upload"
    st.session_state["_uploaded_bytes"] = uploaded.getvalue()

df_raw = None
load_error = None

source = st.session_state.get("_data_source")
if source == "sample":
    try:
        df_raw = load_csv_string(SAMPLE_CSV)
        st.success("Loaded sample dataset (20 complaints).")
    except ValueError as exc:
        load_error = str(exc)
elif source == "upload":
    raw_bytes = st.session_state.get("_uploaded_bytes")
    if raw_bytes:
        try:
            df_raw = load_csv_bytes(raw_bytes)
            st.success(f"Loaded {len(df_raw):,} rows from uploaded file.")
        except ValueError as exc:
            load_error = str(exc)

if load_error:
    st.error(f"Could not load data: {load_error}")
    st.stop()

if df_raw is None:
    st.info("Upload a CSV file or click **Use sample data** to begin.")
    st.stop()

st.dataframe(df_raw.head(10), use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 2: COLUMN MAPPING
# ─────────────────────────────────────────────

st.header("2. Column Mapping")
cols = df_raw.columns.tolist()

default_text = cols.index("text") if "text" in cols else 0
text_col = st.selectbox("Text column (complaints)", cols, index=default_text)
date_col = st.selectbox("Date column (optional, for trends)", ["(none)"] + cols)
cat_col  = st.selectbox("Product / service column (optional)", ["(none)"] + cols)

_date_col_arg = date_col if date_col != "(none)" else None



# sidebar will be rendered below after safe defaults
# sidebar auto-tuned after validation (rendered below)
# ─────────────────────────────────────────────
# SECTION 3: DATA QUALITY
# ─────────────────────────────────────────────

deduplicate = bool(st.session_state.get("deduplicate", False))
n_clusters = int(st.session_state.get("n_clusters", 5))
n_top_terms = int(st.session_state.get("n_top_terms", 8))

st.header("3. Data Quality")

try:
    df_valid, report = validate(
        df_raw,
        text_col=text_col,
        date_col=_date_col_arg,
        deduplicate=deduplicate,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total rows", f"{report.total_rows:,}")
c2.metric("Missing text", f"{report.missing_text:,}",
          help="Rows where the text column was null/NaN — excluded.")
c3.metric("Blank text", f"{report.blank_text:,}",
          help="Rows where text was whitespace only — excluded.")
c4.metric("Exact duplicates",
          f"{report.exact_duplicates:,}",
          help="Count of exact text duplicates. " +
               ("Removed." if deduplicate else "Kept (deduplication is OFF)."))
c5.metric("Retained rows", f"{report.retained_rows:,}")

if report.non_string_coerced > 0:
    st.info(
        f"{report.non_string_coerced:,} non-string value(s) in the text column "
        "were safely coerced to strings."
    )

if _date_col_arg and report.invalid_dates > 0:
    st.warning(
        f"{report.invalid_dates:,} row(s) had unparseable dates and will be "
        "excluded from trend charts. Complaint-text records are still kept."
    )

if report.categorical_warning:
    st.warning(report.categorical_warning)

deduplicate = bool(st.session_state.get("deduplicate", False))
n_clusters = int(st.session_state.get("n_clusters", 5))
n_top_terms = int(st.session_state.get("n_top_terms", 8))

auto_rows = int(report.retained_rows)
if st.session_state.get("_auto_tuned_rows") != auto_rows:
    rec_clusters = min(20, max(2, int(round(auto_rows ** 0.5 / 10.0))))
    rec_max = min(8000, max(1000, (auto_rows // 10)))
    rec_max = int(round(rec_max / 500.0) * 500) if rec_max >= 500 else rec_max
    st.session_state["n_clusters"] = rec_clusters
    st.session_state["max_features"] = rec_max
    st.session_state["min_df"] = 1 if auto_rows < 2000 else 2
    st.session_state["_auto_tuned_rows"] = auto_rows
    st.rerun()

if report.retained_rows == 0:
    st.error("No usable rows remain after validation. Check your data and column mapping.")
    st.stop()

if report.retained_rows < n_clusters * 2:
    st.error(
        f"Only {report.retained_rows:,} usable rows — too few for {n_clusters} clusters. "
        "Reduce the cluster count in the sidebar or provide more data."
    )
    st.stop()


with st.sidebar:
    st.header("⚙️ Settings")

    if "n_clusters" not in st.session_state:
        st.session_state["n_clusters"] = 5
    if "random_seed" not in st.session_state:
        st.session_state["random_seed"] = 42
    if "clean_masked" not in st.session_state:
        st.session_state["clean_masked"] = True
    if "use_embeddings" not in st.session_state:
        st.session_state["use_embeddings"] = False
    if "deduplicate" not in st.session_state:
        st.session_state["deduplicate"] = False

    st.subheader("Ease of use")
    n_clusters = st.slider("Number of themes", 2, 20, key="n_clusters",
                            help="Auto-tuned after you load and validate a file.")
    random_seed = st.number_input("Random seed", min_value=0, key="random_seed", step=1)

    st.subheader("Text cleaning")
    clean_masked = st.checkbox("Strip masked CFPB tokens (Variant B)", key="clean_masked",
                              help="Remove XXXX, formatted dates and dollar amounts before TF-IDF (recommended).")

    st.subheader("Representation")
    use_embeddings = (st.checkbox("Use semantic embeddings (all-MiniLM-L6-v2)",
                                  key="use_embeddings",
                                  help="Optional. Truncates at 256 tokens. Runs on CPU unless you have a GPU.")
                      if _EMBED_AVAILABLE else False)
    if not _EMBED_AVAILABLE:
        st.caption("Embeddings unavailable. Install sentence-transformers for this option.")

    st.subheader("Duplicate handling")
    deduplicate = st.checkbox("Remove exact duplicates", key="deduplicate")

    advanced = st.expander("Advanced parameters", expanded=False)
    with advanced:
        max_features = st.slider("Min document frequency", 500, 10000, step=500, key="max_features")
        min_df = st.slider("Max vocabulary", 1, 10, key="min_df")
        n_top_terms = st.slider("Top terms per theme", 3, 20, key="n_top_terms",
                                value=8 if "n_top_terms" not in st.session_state else st.session_state["n_top_terms"])

    st.subheader("Trend settings")
    min_vol = st.number_input("Min volume for a trend flag", min_value=1, value=3)
    growth = st.number_input(
        "Growth factor threshold", min_value=1.1, value=2.0, step=0.1,
        help="A theme is flagged when its newest-week count doubles over the prior week."
    )

    st.caption("MiniBatchKMeans and validation run locally, so this is single-machine scale.")

# ─────────────────────────────────────────────
# SECTION 4: THEME DISCOVERY
# ─────────────────────────────────────────────

st.header("4. Theme Discovery")

df_valid["_text"] = normalise_series(df_valid[text_col])

with st.spinner("Running theme discovery …"):
    try:
        if use_embeddings:
            st.warning(
                "Embeddings are truncated at 256 tokens per model limit. "
                "About 38.6% of the 5k benchmark narratives exceeded this."
            )
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = model.encode(
                df_valid["_text"].tolist(),
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            km = KMeans(n_clusters=int(n_clusters), random_state=int(random_seed), n_init=10)
            labels = km.fit_predict(emb)
            result = _cluster.ClusterResult(
                labels=labels,
                vectorizer=TfidfVectorizer(max_features=2000, stop_words="english", min_df=2).fit(df_valid["_text"].tolist()),
                matrix=None,
                model=type('o',(),{'n_clusters':int(n_clusters)})(),
                top_terms={},
                n_features=int(len(df_valid["_text"].tolist())),
            )
            # top terms via average tfidf
            vectorizer = result.vectorizer
            T = vectorizer.transform(df_valid["_text"].tolist())
            names = vectorizer.get_feature_names_out()
            top_terms = {}
            for tid in sorted(set(labels.tolist())):
                idx = [i for i, lab in enumerate(labels) if lab == tid]
                mean_vec = T[idx].mean(axis=0)
                arr = (mean_vec.A1 if hasattr(mean_vec, "A1") else __import__("numpy").asarray(mean_vec).ravel())
                order = __import__("numpy").argsort(arr)[::-1]
                top_terms[tid] = ", ".join(names[order[:int(n_top_terms)]])
            result.top_terms = top_terms
            result.matrix = T
        else:
            result = _cluster.fit(
                texts=df_valid["_text"].tolist(),
                n_clusters=int(n_clusters),
                max_features=int(max_features),
                min_df=int(min_df),
                n_top_terms=int(n_top_terms),
                random_seed=int(random_seed),
            )
    except Exception as exc:
        st.error(f"Theme discovery failed: {exc}")
        st.stop()

df_valid["theme_id"] = result.labels
df_valid["theme_label"] = df_valid["theme_id"].map(
    lambda i: f"Theme {i}: {result.top_terms[i]}"
)

# ── Evaluation ────────────────────────────────
eval_report = _evaluate(
    labels=result.labels,
    matrix=result.matrix,
    n_clusters_requested=int(n_clusters),
    empty_clusters=result.empty_clusters,
    random_seed=int(random_seed),
)

# ── Theme summary table ───────────────────────
summary_rows = []
for tid in sorted(df_valid["theme_id"].unique()):
    subset = df_valid[df_valid["theme_id"] == tid]
    template_share = 1 - (subset["_text"].nunique() / len(subset)) if len(subset) else 0.0
    row = {
        "Theme ID": tid,
        "Top Terms": result.top_terms[tid],
        "Count": len(subset),
        "Template share": round(template_share, 3),
    }
    if cat_col != "(none)" and cat_col in df_valid.columns:
        mode_vals = subset[cat_col].mode()
        row["Top Product"] = mode_vals.iloc[0] if len(mode_vals) > 0 else "-"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

st.subheader("Theme Summary")
st.caption(
    "Top terms show the highest TF-IDF centroid weights. "
    "They do not guarantee semantic coherence — inspect representative "
    "complaints to judge whether a theme is meaningful."
)
st.dataframe(summary_df, use_container_width=True)

# ── Empty and small clusters ──────────────────
if result.empty_clusters:
    st.warning(
        f"Empty clusters (no documents assigned): {result.empty_clusters}. "
        "Consider reducing the cluster count."
    )
if eval_report.small_clusters:
    st.info(
        f"Very small clusters (< {eval_report.small_threshold} documents): "
        f"{eval_report.small_clusters}. These may be noise or outliers."
    )

# ── Silhouette score ──────────────────────────
if eval_report.silhouette is not None:
    st.metric(
        "Silhouette Score (cosine)",
        f"{eval_report.silhouette:.3f}",
        help=(
            f"Computed on a sample of {eval_report.silhouette_sample_size:,} documents "
            "using cosine distance (appropriate for TF-IDF). "
            "Range: −1 to +1. Higher = better-separated clusters in feature space. "
            "This is a geometric signal only — not proof of semantic coherence. "
            "No threshold is meaningful without baseline comparison on your dataset."
        ),
    )
else:
    st.info("Silhouette score not computed (too few documents or only one cluster found).")

# ── Bar chart ─────────────────────────────────
st.subheader("Complaint Volume by Theme")
st.bar_chart(summary_df.set_index("Top Terms")["Count"])

# ── Representative complaint explorer ─────────
st.subheader("Explore Representative Complaints")
selected_theme = st.selectbox(
    "Pick a theme to inspect",
    options=sorted(df_valid["theme_id"].unique()),
    format_func=lambda i: f"Theme {i}: {result.top_terms[i]}",
)
show_cols = [text_col] + ([cat_col] if cat_col != "(none)" else [])
show_cols = list(dict.fromkeys(show_cols))
positions = []
for wanted in show_cols:
    for i, name in enumerate(df_valid.columns):
        if name == wanted and i not in positions:
            positions.append(i)
            break
selected_mask = df_valid["theme_id"] == selected_theme
examples = df_valid.loc[selected_mask].iloc[:, positions].head(10)

st.dataframe(examples, use_container_width=True)


# ─────────────────────────────────────────────
# SECTION 5: TRENDS
# ─────────────────────────────────────────────

if _date_col_arg and _date_col_arg in df_valid.columns:
    st.header("5. Trends Over Time")

    trend = compute_trends(df_valid, theme_col="theme_id", date_col=_date_col_arg)

    if trend.rows_excluded_bad_date > 0:
        st.info(
            f"{trend.rows_excluded_bad_date:,} row(s) excluded from trends "
            "due to missing or unparseable dates."
        )

    if trend.pivot.empty:
        st.warning("No valid dates found — trend chart unavailable.")
    else:
        chart_df = trend.pivot.copy()
        chart_df.columns = [f"Theme {c}" for c in chart_df.columns]
        st.line_chart(chart_df)

        st.subheader("Emerging Issue Flags")
        st.caption(
            "A theme is flagged when its complaint count in the most recent "
            "period meets the minimum volume AND is at least the configured "
            "growth factor times the previous period. "
            "This is a heuristic review trigger — not a statistical test. "
            "Inspect the counts and the complaints before acting."
        )

        flags = flag_emerging(
            trend=trend,
            top_terms=result.top_terms,
            min_volume=int(min_vol),
            growth_factor=float(growth),
        )

        if flags:
            for f in flags:
                gf_str = "∞ (new)" if f.growth_factor == float("inf") else f"{f.growth_factor:.1f}×"
                st.warning(
                    f"⚠️ **Theme {f.cluster_id}** ({f.top_terms}) — "
                    f"{f.last_period_label}: **{f.last_period_count}** complaints "
                    f"vs {f.prev_period_label}: {f.prev_period_count} "
                    f"(growth: {gf_str}). Review it manually."
                )
        else:
            st.success("No emerging issues flagged with current settings.")


# ─────────────────────────────────────────────
# SECTION 6: EXPORT
# ─────────────────────────────────────────────

st.header("6. Export")

export_cols = (
    [text_col, "theme_id", "theme_label"]
    + ([_date_col_arg] if _date_col_arg else [])
    + ([cat_col] if cat_col != "(none)" else [])
)
# Keep only columns that exist (safety against column mapping changes)
export_cols = [c for c in export_cols if c in df_valid.columns]
export_df = df_valid[export_cols]
csv_out = export_df.to_csv(index=False)

st.download_button(
    "⬇️ Download labelled results (CSV)",
    data=csv_out,
    file_name="openissue_results.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "OpenIssue — modular NLP pipeline. "
    "Core logic: src/openissue/. Tests: tests/. "
    "Run tests: python -m pytest -q"
)
