# ngram_pipeline

## Overview
A 4-stage pipeline for discovering semantic themes in text corpora via n-gram extraction, embedding-based clustering, and document scoring. Designed to be corpus- and domain-agnostic.

## Architecture
- `utils.py` — Shared utilities (config loading, NLTK, n-gram extraction, embedding abstraction)
- `01_extract_text.py` — Stage 1: Raw documents → cleaned text files
- `02_cluster_ngrams.py` — Stage 2: Corpus-level n-gram extraction → embedding → PCA → spherical KMeans → Leiden community detection
- `03_extract_doc_ngrams.py` — Stage 3: Per-document n-gram extraction + master mapping build (joins clustering output with user-labeled communities)
- `04_score_documents.py` — Stage 4: Cosine-weighted document scoring → category/subcategory probability CSVs

## Data Flow
```
input texts → 01 → output/<run>/texts/
           → 02 → output/<run>/clusters/ (community_labels_k{K}.csv ← user labels this)
           → 03 → output/<run>/doc_ngrams/ + master_ngram_mapping.csv
           → 04 → output/<run>/scores/ (wide CSVs with probability distributions)
```

## Key Design Decisions
- All config is in a single `config.yaml` (YAML). No hardcoded paths, API keys, or domain-specific stop words.
- N-gram extraction logic exists in one place (`utils.extract_ngrams_from_text`), imported by stages 2 and 3.
- Embedding provider is configurable: `"openai"` (reads `OPENAI_API_KEY` env var) or `"sentence-transformers"` (local, free).
- The `community_id_k{K}` column name is dynamic from config `k_value`, not hardcoded.
- Subcategory detection is automatic — any category with subcategories in the labeled CSV gets subcategory scoring.
- Stage 1 optional features (HTML cleaning, section extraction) are off by default for plain text corpora.

## Running
```bash
python 01_extract_text.py    --config config.yaml
python 02_cluster_ngrams.py  --config config.yaml
# Manual step: label communities in the output CSV
python 03_extract_doc_ngrams.py --config config.yaml
python 04_score_documents.py    --config config.yaml
```

## Origin
Refactored from the research pipeline in `../n_gram_generation/`. Source files:
- `ngram_clustering_chatgpt_api_v3.py` → `02_cluster_ngrams.py` + `utils.py`
- `sga_extract_n_gram_mapping_parallel.py` → `03_extract_doc_ngrams.py` + `utils.py`
- `ngram_community_ngram_scoring_v3.py` → `04_score_documents.py`
- `01_EXTRACT_ITEM7_V3.py` → `01_extract_text.py`

## Testing
No formal test suite. Verify by running all 4 stages on the sample data in `examples/intangible_investment/` and checking that score CSVs have one row per document with probabilities summing to ~1.0.

## Dependencies
See `requirements.txt`. Core: numpy, pandas, scikit-learn, nltk, python-igraph, leidenalg, sentence-transformers, torch, pyyaml.
