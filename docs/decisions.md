# Decision Log — OpenIssue

## Decision: Optional masked-token cleaning + TF-IDF/B variant as comparison baseline

Decision:
  Add an optional masked-token cleaning step in `preprocess.normalise_text(..., clean_masked=False)`
  and TF-IDF option flags `max_df`, `sublinear_tf`, `ngram_range` in
  `cluster.fit(...)` (defaults preserving the original behaviour). Treat the clean-only
  unigram variant (B) as the preferred exploratory default on evidence below.

Options considered:
  A — current settings (no cleaning)
  B — cleaning + max_df=1.0 + sublinear_tf=False + unigrams
  C — B + max_df=0.5 + sublinear_tf=True
  D — C + ngrams (1,2) + max_features=20000

Evidence collected:
  Controlled comparison on the same seeded 10,000-row sample (seed 42), k=15:
  - A: silhouette 0.0743, sizes min/median/max 154/477/1576, masked-token-in-top-terms YES,
       clusters with single-product majority >=50% 9/15, NMI vs Issue 0.2492, vs Product 0.2618
  - B: silhouette 0.0746, 161/527/2403, masked-token NO,
       12/15 single-product majority, NMI Issue 0.3228, Product 0.3435
  - C: silhouette 0.0575, 154/608/1245, masked-token NO,
       6/15 majority, NMI Issue 0.2731, Product 0.2614
  - D: silhouette 0.0615, 30/672/1550, masked-token NO,
       10/15 majority, NMI Issue 0.2847, Product 0.2883
  Masked tokens (XXXX) appeared in top-term lists for clusters 0,1,2,3 under A.
  Full-corpus B runs (k=10/15/20/30, 34,351 rows): no masked tokens in any top-term
  list; runtimes 1.33–1.39 s; silhouettes 0.0607/0.0679/0.0699/0.0805.
  k=10 produced one 1-document singleton cluster; k=30 created 30 clusters but no new
  semantic gain beyond k=15 in NMI terms (NMI Issue fell from 0.3109 at k=15 to 0.3024 at k=30).
  Exact-duplicate share per cluster on B k=15: clusters 1 and 5 are dominated by repeated
  narratives (98.1% and 67.2% excess duplicate share respectively); the other 13 clusters
  are <= 22.7%.

Reason for choice:
  Variant B cleared masked-token noise from the top terms, maximised NMI against both
  Issue (0.3228) and Product (0.3435) on the 10k sanity check, and kept the best balance
  of cluster sizes (no singletons on full corpus). NMI is reported only as a sanity check,
  not as ground-truth accuracy.

Trade-offs:
  Cleaning changes vocabulary counts and shifted masked tokens into a generic money_amount
  token; NMI still depends on Issue/Product labels that are not ground truth.
  k=15 favours mid-granularity themes; larger k increases silhouette but fragments small
  clusters and raises duplicate-heavy clusters.
  Exact-duplicate narratives are retained (no deduplication yet), so two clusters are still
  inflated by templated complaints.

What would make us reconsider:
  If a future run with a different corpus/tokeniser raises NMI for C/D, if masked-token
  cleaning is shown to merge distinct issues, or if duplicate-heavy clusters dominate downstream
  reporting after deduplication is evaluated.


## Entry 2 (2026-10-13) — files, dedupe experiment

Decision:
  Wrote variant_comparison.md and clusters_B_k15.md from actual runs; benchmark.csv retained.
  Ran B k=15 without dedupe: 34,351 rows, dup count 3,078, silhouette 0.06789057, sizes 395/2679/5833.
  Ran B k=15 with validate(..., deduplicate=True): 31,273 rows, removed 3,078, silhouette 0.03154970, sizes 612/1758/5635.
  Cluster correspondence measured by shared Complaint IDs; multiple dedup clusters shrank >30%.

Skipped:
  Semantic/embedding models not installed; dashboard export not re-tested; row-overlap mapping is not one-to-one.


## Entry 3 (2026-10-13) — Task 6 items

Decision:
  Append results from task items 1-5:
  - regenerated benchmark.csv with --clean-masked --n-clusters 15; 19 rows (incl. header), each row 6 fields.
  - profiled validate memory; top python line <=2.3 MB in blocks.py; no obvious single unnecessary full-DataFrame copy identified; validate left unchanged.
  - stability B k=15: ARI mean 0.4075 (min .3296, max .5215) on full; mean 0.4221 (.3426, .5150) on dedup.
  - fit_on_unique flag added (default False) and tested; full B template shares reported.
  - semantic candidate all-MiniLM-L6-v2 embedded; packages installed: sentence-transformers 6.1.0, transformers 5.14.1, torch 2.13.0, scikit-learn 1.8.0, tokenizers 0.22.2, huggingface-hub 1.23.0, numpy 2.3.5. 5k embedding 16.49 s, 38.62% truncation (>256 tokens); compared vs TF-IDF-B fit_on_unique (see tfidf_vs_semantic_5k.md).

Skipped:
  - packaging requirements.txt was not updated with sentence-transformers (not requested as a lock file change).
  - model inference remained CPU; no GPU.
  - Semantic candidate ran on 5k subset only; full 34,351 not embedded.
  - ARI/silhouette across representation comparison intentionally not done (different spaces).


## Entry 4 (2026-10-13) — Phase 2 B2 + method review decision

Decision:
  Move money_amount out of displayed term lists (B2), set n_init=10 in
  MiniBatchKMeans, keep semantic comparison optional and CPU-only.

  Method decision: use embeddings (all-MiniLM-L6-v2) behind the --embeddings flag
  as the primary semantic comparison; retain TF-IDF-B as a cheap transparent baseline.
  Earnings: embeddings 10/15 coherent versus TF-IDF 6/15; TF-IDF has 4 brand-driven
  clusters and 1 catch-all (14.1% of rows); embeddings have 1 generic catch-all-like
  mixture but no single cluster dominates. ARI is higher for embeddings (0.6572 vs 0.3662)
  on 5k/full samples after n_init=10. Cost (~105 s / 34k rows, 1.34 GB peak RSS) is
  acceptable for batch processing. No BERTopic run on this data.

Evidence collected:
  - human_review.md tallies and representative samples for both methods.
  - semantic_cost.md: ~0.003045 s/row embedding; extrapolated 104.6 s / 34k rows.
  - stability_B_k15.md: TF-IDF ARI mean 0.3662 full / 0.3990 dedup after n_init=10.
  - tfidf_vs_semantic_5k.md: TF-IDF sizes, NMI, Company share, ARI, samples.
  - truncation: 1,931/5,000 (0.3862) narratives exceed 256 tokens.

Trade-offs:
  - Embeddings cost ~100x more runtime and 1.34 GB more RSS versus TF-IDF-B.
  - 38.62% of narratives are truncated at 256 tokens; head+tail ablation remains optional.
  - Embedding boundaries (brands, credit card, loan) are slightly cleaner, but topic
    names still require human names. No method removed the templating problem (2 clusters
    are still dominated by repeated FCRA templates on TF-IDF, plus a high value on embeddings).
  - Small sample (5k, same seed) limits convergence certainty.

What would make us reconsider:
  - If NMI or coherence dropped on a second chunk of data.
  - If truncating to 256 tokens removed the decisive cluster identity.
  - If a cloud/batch cost budget required TF-IDF-only mode.
  - If an LLM or BERTopic naming/experiment budget freed up.

B2 variant note:
  - B2 keeps the full token vocabulary but suppresses "money_amount" in displayed top terms.
