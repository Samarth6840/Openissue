# Semantic candidate — all-MiniLM-L6-v2 on cleaned dedup narratives
package: sentence-transformers 6.1.0; transformers 5.14.1; torch 2.13.0; scikit-learn 1.8.0; tokenizers 0.22.2; huggingface-hub 1.23.0; numpy 2.3.5
model: all-MiniLM-L6-v2, max_seq_length=256, dim=384
sample: 5000 dedup cleaned texts, avg chars=1285.98
truncation >256 tokens: count=1931 share=0.3862 median_tokens=209 max_tokens=3425
model load seconds=7.8670 peak_rss_delta_mb=0.00
embedding 500 rows seconds=1.8717 (per-row 0.003743); extrapolated 5000=18.72s
embedding 500 peak RSS MB=1354.80 growth over idle 0.00
embedding 5000 seconds=14.7150 per-row=0.002943; peak RSS 1354.80 MB, growth over model-loaded 0.00 MB
KMeans k=15 seeds [0, 1, 42]: ARI pairwise=[0.6987, 0.6018, 0.6711] mean=0.6572 min=0.6018 max=0.6987
