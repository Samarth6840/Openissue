# OpenIssue — General-Purpose Complaint Intelligence

> **Project status:** Working, testable baseline implementation, with optional masked-clean flag, TF-IDF optional params, semantic candidate (5k), and stability tests. Includes a modular Python pipeline (`src/openissue`), comprehensive test suite (`tests/`), and interactive Streamlit dashboard (`dashboard.py`).

## Quickstart

### 1. Requirements & Installation
```bash
python3 -m pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
python3 -m streamlit run dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. You can click **"Use sample data"** to test with 20 pre-loaded complaints or upload your own CSV file.

### 3. Run Automated Tests
```bash
python3 -m py_compile dashboard.py src/openissue/*.py
python3 -m pytest -q
```

## 1. Project Overview

OpenIssue is a working NLP system with implemented baseline clustering for discovering recurring and emerging issues in large collections of user complaints, reviews, support tickets, and feedback.

Most simple complaint-analysis projects depend on sentiment labels or a fixed list of categories such as *delivery*, *payment*, or *battery*. That approach can miss new problems and may not transfer well between domains. OpenIssue aims to discover issue themes from the text itself, then show how those themes change over time.

The system should accept feedback from different product or service domains without requiring a separate manually written category list for every domain. A source may include a product name or category, but those fields are optional context—not a hardcoded taxonomy that determines every issue.

### Example

A dataset might contain these complaints:

- “Phone loses charge even when I am not using it.”
- “Laptop battery drops from 50% to 10% in minutes.”
- “The app drains my device battery in the background.”

OpenIssue should be able to discover that these texts share a possible *rapid battery drain* theme, while retaining each complaint's original product or service context. Whether these belong in one cross-domain theme or separate product-specific themes is a design decision to test, not assume.

## 2. Problem Statement

Organizations receive large volumes of unstructured feedback. Manually reading, categorizing, deduplicating, and tracking every complaint becomes costly as volume grows. Fixed keyword rules are brittle, while broad sentiment scores do not explain what is going wrong.

**Problem:** Design and evaluate a scalable NLP pipeline that groups semantically related complaints into interpretable issue themes, surfaces themes that are increasing over time, and helps a human investigate the evidence behind each theme.

## 3. Goals

1. Ingest complaint or feedback records from a documented data source.
2. Validate records, normalize text, identify duplicates, and handle missing values.
3. Discover issue themes without requiring a predefined category for every possible problem.
4. Generate understandable descriptions and representative examples for each theme.
5. Track theme frequency over time and flag possible emerging issues.
6. Compare at least two text-representation or clustering approaches using measurable criteria.
7. Make the process reproducible and explain key engineering decisions.
8. Design processing stages so larger datasets can be distributed across workers when justified by volume.

## 4. Non-Goals

- Automatically deciding refunds, rejecting claims, or making other high-impact decisions.
- Treating sentiment analysis as a complete solution to complaint understanding.
- Claiming that a cluster is a real-world issue without checking its examples.
- Promising real-time processing before a batch pipeline has been measured.
- Claiming “big data” performance without recording dataset size, runtime, memory use, and hardware.
- Using a large language model to generate an answer for every record when simpler NLP methods can do the job.

## 5. Proposed Architecture

The diagram is organized around the three top-level blocks used in the initial draw.io design.

```mermaid
flowchart LR
    A[CSV / Public Dataset / Future API] --> B[1. Data Ingestion and Management]
    B --> B1[Schema Validation]
    B1 --> B2[Cleaning and Deduplication]
    B2 --> B3[Raw and Clean Data Storage]
    B3 --> C[2. NLP and Theme Discovery]
    C --> C1[Text Preprocessing]
    C1 --> C2[Text Representation]
    C2 --> C3[Clustering / Topic Discovery]
    C3 --> C4[Theme Labels and Examples]
    C4 --> D[3. Issue Intelligence and Evaluation]
    D --> D1[Theme Counts and Trends]
    D1 --> D2[Emerging-Issue Flags]
    D2 --> D3[Evaluation and Human Review]
    D3 --> E[Report / Dashboard / Export]
```

### Layer 1 — Data Ingestion and Management

**Purpose:** Turn source files into consistent, traceable records before modeling.

Components:

- **Source adapter:** Reads a CSV or other selected source. Keep source-specific loading code separate from NLP code.
- **Schema validator:** Checks required fields, field types, empty complaint text, invalid dates, and duplicate record IDs.
- **Cleaning and deduplication:** Standardizes whitespace and encoding; detects exact duplicates and, if useful, near-duplicates. Do not delete records silently—log counts and rules.
- **Optional metadata:** Keeps product/service name, source, date, region, or existing category when available. Metadata helps filter or compare themes but should not determine discovered themes by itself.
- **Storage:** Preserve a raw copy and a cleaned version. For a prototype, local CSV/Parquet files may be enough. A database or distributed storage can be considered after requirements are clear.
- **Data-quality report:** Records read, records rejected, missing fields, duplicates removed, and final row count.

**Output:** A cleaned table containing stable record IDs, complaint text, timestamps when available, and optional metadata.

### Layer 2 — NLP and Theme Discovery

**Purpose:** Represent complaint text and group records that describe similar problems.

Components:

1. **Text preprocessing:** Minimal normalization, such as trimming whitespace and handling missing text. Test whether lowercasing, stop-word removal, stemming, or lemmatization helps. Do not assume heavy cleaning always improves semantic models; preserve negation and meaningful technical terms.
2. **Text representation:** Convert each complaint into a numerical representation suitable for similarity or clustering.
3. **Theme discovery:** Group similar complaints, while allowing the system to mark uncertain or unmatched records where the selected method supports it.
4. **Theme interpretation:** Produce top terms and representative complaints so a person can understand why records were grouped together.
5. **Model artifacts:** Save preprocessing settings, model names/versions, clustering parameters, and discovered labels so results can be reproduced.

#### Candidate methods to compare

| Method | What it captures | Strengths | Limitations | Role in experiment |
|---|---|---|---|---|
| Bag of Words + K-Means | Word occurrence counts | Simple and easy to explain | Large sparse vectors; does not understand word meaning | Optional basic baseline |
| TF-IDF + MiniBatchKMeans | Words that distinguish documents | Fast, interpretable, memory-conscious for many records | Similar wording matters more than deeper semantic similarity; requires choosing cluster count | Primary classical baseline |
| Sentence embeddings + clustering | Contextual semantic similarity | Can group differently worded complaints about similar issues | More compute and model dependencies; clusters still need validation | Semantic candidate |
| BERTopic | Topic discovery using embeddings, clustering, and class-based TF-IDF topic descriptions | Convenient topic-analysis pipeline with interpretable topic terms | Multiple stages and parameters; results depend on data and configuration | Candidate end-to-end topic model |

These approaches overlap in places. BERTopic commonly combines sentence embeddings, dimensionality reduction, clustering, and class-based TF-IDF. The experiment should compare clearly defined pipelines rather than presenting overlapping components as unrelated algorithms.

**Starting recommendation:** Build TF-IDF + MiniBatchKMeans first as a transparent baseline. Then compare it with one semantic approach, such as Sentence Transformers with a documented clustering method or BERTopic. Choose the final approach using results, not hype or assumptions that the newest model must be best.

### Layer 3 — Issue Intelligence and Evaluation

**Purpose:** Turn discovered themes into evidence that helps people investigate problems.

Components:

- **Theme summary:** Theme ID, human-reviewed name, record count, top terms, representative examples, and optional product/service breakdown.
- **Trend aggregation:** Count records per theme by day or week when timestamps are available. Show both raw counts and normalized rates when overall complaint volume varies.
- **Emerging-issue flag:** Flag themes with a notable increase against an explicit baseline, minimum-volume rule, and time window. Treat a flag as a review suggestion, not proof of an incident.
- **Filtering:** Allow review by date, source, product/service, or discovered theme where those fields exist.
- **Evaluation:** Compare theme quality, interpretability, coverage, stability, runtime, and memory usage.
- **Output:** Export a CSV/JSON report or present findings in a small dashboard. A dashboard is optional until the core pipeline works.

## 6. End-to-End Processing Flow

1. Load source data and record its origin, licence, retrieval date, and dataset version.
2. Validate schema and produce a data-quality summary.
3. Normalize text and handle duplicates using documented rules.
4. Split or order data by time if temporal evaluation is planned. Avoid using future records to claim that the system predicted past events.
5. Build the classical baseline: TF-IDF features plus MiniBatchKMeans.
6. Inspect cluster examples and tune a small, documented set of parameters.
7. Build one semantic candidate on the same records and comparable hardware.
8. Compare the methods using the evaluation plan below.
9. Select or recommend a method, documenting trade-offs and failure cases.
10. Aggregate selected theme assignments over time and flag candidate emerging issues.
11. Review sample records from large, small, mixed, and noisy clusters.
12. Export results and write conclusions supported by actual measurements.

## 7. Data Requirements and Proposed Schema

The dataset is selected and in use: the January–February 2026 chunk of the CFPB FOIA Consumer Complaint Database Narratives Archive (1,038,383 rows, 34,351 non-empty narratives). Licence, fields, language, row count, date coverage, and relevance are documented in the verified dataset block below.

Recommended canonical fields:

| Field | Required? | Purpose |
|---|---|---|
| `record_id` | Yes, or generated during ingestion | Stable identifier for tracing a result to its source row |
| `text` | Yes | Complaint, review, ticket, or feedback text |
| `created_at` | Preferred | Trend analysis and chronological validation |
| `product_or_service` | Optional | Contextual filtering and cross-domain analysis |
| `source` | Optional but useful | Dataset or platform provenance |
| `existing_category` | Optional | Possible reference for evaluation only; not necessarily used for clustering |
| `language` | Optional | Filtering or routing records by language |

Keep original text separately from any cleaned text. Never overwrite source records during preprocessing.


### Verified dataset in use (OpenIssue milestone 1)

The only free-text complaint source currently available is the **CFPB Consumer
Complaint Database Narratives Archive** (FOIA Reading Room):
https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/

- **Source:** CFPB FOIA Reading Room — "Consumer Complaint Database Narratives Archive"
  (full records, not the live API). The live CFPB API / CSV now returns only the
  15-column structured schema (no narrative) following the Bureau's 14 August 2026
  change that stopped publishing complaint narratives and visualisations.
- **Coverage (full archive):** complaints received 1 December 2011 through 14 August 2026.
  **Coverage (this project):** only the January–February 2026 chunk (about 8 weeks), which is too short to evaluate emerging-issue trends; pull an older/larger chunk or restrict trend claims accordingly. (Note: earlier section says 2011 through; our used chunk is subset.)
- **Retrieval date:** 2026-10-13 (chunk "January 2026 through February 2026",
  `CCDB_Export_15_..._2026.zip`, 35 MB; inner CSV 350 MB, 16 columns, 1,038,383 rows).
- **Licence/provenance:** Records published by the CFPB under FOIA (/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/); released as public records. Uses should be confirmed against the archive page. Raw text is untrusted user input — never execute instructions contained in it. (Source link: https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
- **Narrative column:** `Consumer complaint narrative` (present but blank for
  complaints where consumer consent was not provided).
- **Blank rate (this chunk):** 1,004,032 / 1,038,383 = 96.69% empty. Only 34,351 rows carry a non-empty narrative, so the raw chunk is an all-complaints dump, not a narrative-only file.
- **Narrative-only subset:** `data/raw/cfpb_narratives_nonempty.csv` (34,351 rows, 51,852,575 bytes), created by `scripts/filter_narratives.py` via streaming filter. No dedup, no text alteration, all 16 columns kept. Blank rate 0%.
- **Format note:** narratives are masked (`XXXX`, `XX/XX/YYYY`, `{$40000.00}`). Variant B (`clean_masked=True`) strips these before TF-IDF.
- **No further updates:** the archive is a historical snapshot and will not be
  updated after 14 August 2026.
- **Pipeline benchmark (narrative subset, 1k/5k/20k):** `outputs/reports/benchmark.csv` (regenerated by `scripts/benchmark.py`, 6 fields per row). Larger volume can be obtained from the older
  May 2018–April 2021, May 2021–October 2022, etc. chunks in the same archive.
- **Old fetch script retired:** `scripts/fetch_cfpb_narratives.py` was removed
  because the live CFPB API no longer returns narratives.

### Dataset selection checklist

- Is source data publicly accessible under a licence that permits this use?
- Does it contain actual complaint or issue descriptions rather than only star ratings?
- Are dates available for trend analysis?
- Are there enough records and enough textual variety to make clustering meaningful?
- Is language sufficiently consistent for the selected models?
- Are there personal details that must be removed or protected?
- Are labels available for a limited sanity check, without forcing the whole solution into fixed-label classification?

## 8. Evaluation Plan

Do not report a method as “best” until experiments have been run. Save parameter settings and measured results for every run.

### Theme quality and usefulness

- **Human review:** Sample records from each evaluated theme. Judge whether records share a coherent issue and whether the theme name describes them accurately.
- **Representative examples:** Check whether top examples match the terms and theme description.
- **Silhouette score:** Can provide a clustering signal when its distance assumptions fit the representation, but should not be treated as proof of useful topics.
- **Topic coherence or keyword inspection:** Useful for interpreting themes, but may not capture all semantic quality.
- **Coverage/noise:** Measure the share of records assigned to usable themes and, where supported, the share marked unassigned/noise.
- **Stability:** Re-run with controlled parameter or sample changes and check whether major themes remain recognizable.

If a reliable set of human labels is available, use it for a carefully defined secondary comparison. Do not invent labels or apply classification accuracy to an unsupervised setup without explaining how ground truth was obtained.

### Engineering performance

Record, at minimum:

- Total records and total text volume.
- Hardware and software/model versions.
- Data-loading and preprocessing time.
- Feature-generation or embedding time.
- Clustering/topic-discovery time.
- Peak memory usage when measurable.
- Total pipeline runtime.
- Output coverage and number of discovered themes.

Run methods against the same dataset and comparable conditions. For large corpora, use a reproducible subset for rapid experiments, then run final measurements on a larger sample if resources allow.

### Emerging-issue evaluation

If timestamped data is available:

- Aggregate theme counts into fixed time windows.
- Compare recent counts with a documented historical baseline.
- Account for changing total feedback volume by also inspecting rates or proportions.
- Set a minimum-volume threshold to reduce noise from tiny clusters.
- Manually inspect flagged periods and examples.
- Report false alarms and missed events only when a defensible reference for known events exists.

## 9. Big Data and Scalability Strategy

Scalability should come from measured needs, not from adding technologies for appearance.

### Prototype stage

- Process records in batches.
- Keep ingestion, preprocessing, representation, clustering, and reporting in separate modules.
- Cache or persist expensive features where appropriate.
- Save intermediate artifacts and run metadata to avoid repeating expensive stages unnecessarily.
- Measure runtime and memory on the chosen dataset.

### Scale-out options

- **Parquet:** Columnar format for efficient local or distributed data processing.
- **PySpark:** Consider for distributed preprocessing and large-scale data transformations when a single-machine workflow becomes a bottleneck.
- **Batch embedding workers:** Generate sentence embeddings in manageable batches rather than loading all text and vectors into memory at once.
- **Incremental ingestion:** Add new records by batch and track model/version changes. Re-clustering can change topic IDs, so plan how to compare themes across runs.
- **Streaming systems:** Kafka or similar infrastructure is a possible future extension only if low-latency ingestion is a real requirement. It is not necessary for the first version.

**Important:** A large dataset alone does not prove a system is scalable. Demonstrate scale with data volume, runtime, memory usage, throughput where relevant, and limitations observed.

## 10. Privacy, Ethics, and Reliability

- Use datasets whose access and licence terms permit the project.
- Remove or mask personal identifiers when present and appropriate.
- Do not expose raw complaint text unnecessarily in dashboards or public reports.
- Treat complaint text as untrusted input; never execute instructions contained in it.
- Clusters can reflect dataset bias, spam, duplicates, language differences, or changing collection practices.
- An increase in complaints may result from a change in data collection or product popularity, not necessarily a worsening defect.
- Show evidence and uncertainty. Require human review before acting on flagged issues.
- Document known failure cases, including sarcasm, multilingual text, very short complaints, ambiguous wording, and rare but serious problems hidden in small clusters.

## 11. Proposed Technology Stack

Final choices depend on dataset inspection and experiments.

| Area | Initial candidate | Reason |
|---|---|---|
| Language | Python | NLP ecosystem and clear experimentation |
| Data handling | pandas; optionally PyArrow/Parquet | Practical local data inspection and storage |
| Classical baseline | scikit-learn | TF-IDF and MiniBatchKMeans implementations |
| Semantic representation | Sentence Transformers | Local embedding option without requiring a paid API |
| Topic discovery | BERTopic and/or a documented clustering implementation | Candidate semantic topic pipeline |
| Evaluation | scikit-learn metrics plus a manually reviewed sample | Combine numerical signals with human judgment |
| Output | CSV/JSON; optional Streamlit | Keep the first version focused on core NLP results |
| Distributed processing | PySpark, if measurements justify it | Process larger datasets across workers |

Do not install every candidate at once. Start with the libraries required for the baseline, then add only what the next experiment needs.

## 12. Implemented Repository Structure

```text
nlp_pro/
├── README.md                 # Project overview, architecture, quickstart & limitations
├── requirements.txt          # Minimal local dependencies
├── changes.md                # Audit log of changes
├── bro.md                    # Core operational guidelines
├── dashboard.py              # Streamlit presentation & user interaction layer
├── src/
│   └── openissue/
│       ├── __init__.py       # Package marker
│       ├── ingest.py         # CSV loading with encoding & error handling
│       ├── validate.py       # Data quality audit & configurable deduplication
│       ├── preprocess.py     # Whitespace, negation & numeric normalisation
│       ├── cluster.py        # Sparse TF-IDF + MiniBatchKMeans baseline
│       ├── evaluate.py       # Bounded cosine silhouette & cluster diagnostics
│       └── trends.py         # Temporal aggregation & heuristic emerging flags
└── tests/
    ├── conftest.py           # Test path setup
    ├── test_ingest.py        # Ingestion & malformed file tests
    ├── test_validate.py      # Quality reporting & date/duplicate checks
    ├── test_preprocess.py    # Text normalisation tests
    ├── test_cluster.py       # Sparse matrix, empty vocab & k-limit tests
    ├── test_evaluate.py      # Cosine silhouette & bounded sampling tests
    └── test_trends.py        # Trend aggregations & emerging flag tests
```


## 18. Dataset Card (summary)

- Source: CFPB FOIA Reading Room Consumer Complaint Database Narratives Archive
- URL: https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/
- Retrieved: 2026-10-13, chunk January–February 2026
- Raw chunk: 1,038,383 rows, 16 columns, 96.69% empty narrative
- Derived non-empty subset: 34,351 rows, 51,852,575 bytes
- Licence: published by CFPB under FOIA; confirm reuse terms on archive page
- Date range used: 2025-12-29 through 2026-02-23 (about 9 weekly bins)
- Text is redacted/masked (`XXXX`, dates, currency), no personal identifiers after masking
- Duplicates: 3,078 exact duplicates retained in baseline

## 19. Experiment Table (5,000-row sample unless noted)

| Measure | TF-IDF-B (n_init=10, B2) | Embeddings all-MiniLM-L6-v2 |
|---|---:|---:|
| Time | ~4.9 s / 5k (extrap. ~33 s / 34k; bench 1k–20k in benchmark.csv) | 15.225 s / 5k (extrap. ~104.6 s / 34k) |
| Peak RSS | ~193 MB / 5k estimate | ~1.34 GB / 5k |
| ARI across seeds | 0.3662 full-rate (after n_init=10); 0.3990 dedup | 0.6572 (3 seeds) on 5k |
| Coherent clusters | 6 / 15 | 10 / 15 |
| Brand-driven clusters | 4 (Chime, Wells, Chase, Cash App) | 0 clear brand monoliths |
| Catch-all share | ~2.3% by one broad row (4600/5000 sample: 92%) at full vocab; 1 clone ~30.4% two-row total | 0 single catch-all |
| Company-majority clusters | 3 / 15 | 1 / 15 |
| Truncation risk | none (full vocabulary) | 1,931/5,000 = 0.3862 over 256 tokens |

## 20. Limitations

- Masked text and truncated dates reduce lexicon quality.
- Only an 8- to 9-week window is available in the chosen chunk; trend claims are weak.
- All results run on one laptop; embeddings require ~1.34 GB peak RSS and about 105 s / 34k rows.
- Matrix, random seed, and one-run-per-benchmark-size; no variability estimates outside the multi-seed runs.
- NMI is only a rough sanity check against Issue/Product labels, not ground truth.
- No GPU was used.

### How to Interpret Clustering & Trend Results

1. **Top Terms:** Centroid terms indicate highest-weighted TF-IDF tokens for a cluster. They are a directional hint, **not** proof of semantic coherence. Always review the representative complaints in Section 4 before naming or acting on an issue.
2. **Silhouette Score:** Evaluated using **cosine distance** on a bounded sample (up to 2,000 documents). Values range between −1 and +1; higher indicates geometric separation in TF-IDF space, but no arbitrary threshold proves a cluster is useful.
3. **Emerging-Issue Flags:** Flags indicate a heuristic review suggestion (e.g. week-over-week doubling with minimum volume). They are **not** statistical proof of a systemic defect. Verify with real complaint volume and external release schedules.
4. **MiniBatchKMeans & Scalability:** MiniBatchKMeans computes mini-batch gradient updates locally on a single machine. It does **not** make the application distributed. For large-scale multi-million record processing, distributed frameworks (e.g. PySpark MLlib) would be required and must be validated with empirical benchmarks (runtime, peak memory, throughput).

## 13. Research Questions

1. Do semantic embeddings produce more coherent issue groups than TF-IDF on the selected dataset?
2. What trade-off exists between theme coherence, coverage, runtime, and memory usage?
3. How much does text preprocessing affect each method?
4. Can theme frequency changes identify potentially emerging issues, and how many false alarms do simple thresholds produce?
5. How well do discovered themes transfer across product/service groups without using a fixed issue taxonomy?

### Example hypotheses to test

- **H1:** Semantic representations may group differently worded complaints more coherently than TF-IDF, at a higher computational cost.
- **H2:** TF-IDF + MiniBatchKMeans may run faster and yield easier-to-explain keywords, but may split semantically similar complaints that use different vocabulary.
- **H3:** Trend flags become less noisy when minimum-volume rules and normalized rates are used.

These are hypotheses, not results. The final report must retain them only if they match the actual experiments—or explain evidence that contradicts them.

## 14. Milestones

1. **Architecture and scope:** Confirm use case, three layers, inputs, outputs, and non-goals.
2. **Dataset audit:** Select source, read licence, inspect rows/columns, quantify missing data, and document limitations.
3. **Baseline pipeline:** Build ingestion, cleaning, TF-IDF, and MiniBatchKMeans end to end.
4. **Baseline inspection:** Review example clusters; tune parameters with a written reason.
5. **Semantic comparison:** Add one candidate and run it on the same data.
6. **Evaluation:** Compare quality, interpretability, coverage, runtime, and memory.
7. **Issue trends:** Aggregate themes over time if trustworthy dates exist.
8. **Final presentation:** Explain the problem, architecture, design choices, experiments, evidence, errors, limitations, and future work.

## 15. What Makes This More Than a One-Prompt Project?

The value lies in the decisions and evidence produced during development:

- Inspecting and justifying the dataset.
- Designing schema validation and repeatable preprocessing.
- Building a transparent baseline before adding complex methods.
- Forming hypotheses and comparing alternatives fairly.
- Inspecting wrong or incoherent clusters and analyzing why they failed.
- Measuring runtime and memory rather than making unsupported scale claims.
- Deciding what counts as an emerging issue and evaluating false alarms.
- Recording trade-offs, limitations, and reproducible settings.

Maintain `docs/decisions.md` with entries such as:

```text
Decision:
Options considered:
Evidence collected:
Reason for choice:
Trade-offs:
What would make us reconsider:
```

Use this log to record your own reasoning as the project evolves. Do not claim a decision or experiment you did not personally carry out.

## 16. Expected Deliverables

- Source code for a reproducible NLP pipeline.
- Dataset card documenting source, licence, fields, date range, and limitations.
- Architecture diagram matching the implemented system.
- Experiment table comparing at least a classical baseline and a semantic candidate.
- Output examples showing discovered themes and representative complaints.
- Trend report if timestamped data supports it.
- Final report and presentation containing measured results, failure analysis, and future improvements.

## 17. Current Limitations and Open Decisions

Current limitations and open decisions:

- **Language:** English only.
- **Dataset window:** only January–February 2026, so trending claims are limited; extend with older chunks.
- **Cross-domain vs domain-specific:** still being tested, since clusters sometimes split by company/topic.
- **Semantic comparison:** candidate runs on a 5k sample only; full-scale comparison remains.
- **Emerging-issue validation:** week-over-week flagging is heuristic; needs external event ground truth.
- **Scale:** no distributed processing yet; single-machine bounds sufficient.

These choices should be made after inspecting candidate datasets and running small experiments. The project should remain honest about what has and has not been tested.

---

**Working principle:** Build the simplest complete pipeline first. Measure it. Inspect its failures. Add complexity only when evidence shows what the simpler version cannot handle.
