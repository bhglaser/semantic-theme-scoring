# ngram_pipeline

Discover semantic themes in any text corpus using n-gram extraction, embedding-based clustering, and document scoring.

Given a collection of documents, this pipeline:
1. Extracts and cleans text from raw documents
2. (Optional) Uses an LLM to extract domain-relevant text for dictionary construction
3. Extracts part of speach (POS)-filtered noun phrases (bigrams and trigrams) and clusters them into semantic communities via embeddings + Leiden community detection
4. Lets you label those communities with your own categories
5. Scores every document against the labeled communities, producing category probability distributions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run, or manually)
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('words')"

# Copy and edit the config
cp config.yaml my_config.yaml
# Edit my_config.yaml: set run_name, input paths, embedding provider, etc.

# Run the pipeline
python 01_extract_text.py       --config my_config.yaml
python 01b_llm_extract.py      --config my_config.yaml   # optional: LLM-based text filtering
python 02_cluster_ngrams.py     --config my_config.yaml   # auto-detects LLM extracts if present

# >>> Manual step: open output/<run>/clusters/community_results/community_labels_k500.csv
# >>> Add 'category' and 'subcategory' columns, then save
# >>> Update doc_ngrams.community_labels_csv in your config if you saved it elsewhere

python 03_extract_doc_ngrams.py --config my_config.yaml
python 04_score_documents.py    --config my_config.yaml
```

## Pipeline Stages

### Stage 1: Text Extraction (`01_extract_text.py`)

Reads raw documents and writes cleaned text files. For plain text corpora, this is a pass-through with whitespace normalization. Optional features (off by default):
- **HTML tag stripping** for web-scraped content
- **Section extraction** between configurable regex patterns (e.g., extract only a specific section from structured documents)

### Stage 1b: LLM Text Extraction (`01b_llm_extract.py`) — Optional

When enabled (`llm_extract.enabled: true`), this stage uses an LLM to extract domain-relevant text from each document before dictionary construction. For each document:
1. Finds context blocks around configured expense keywords
2. Sends context to an LLM that extracts definition, business driver, and change driver quotes
3. Writes filtered text to `output/<run>/llm_extracts/`

Stage 2 will automatically detect and use the LLM extracts if present, building the n-gram dictionary from focused text rather than full documents.

**LLM providers:** Ollama (local, free) or OpenAI API (requires `OPENAI_API_KEY`).

### Stage 2: N-Gram Clustering (`02_cluster_ngrams.py`)

The core of the pipeline:
1. Extracts POS-filtered bigrams/trigrams (noun-noun patterns) from all documents (or from LLM extracts if Stage 1b was run)
2. Keeps the top N by corpus frequency (default: 10,000)
3. Embeds them using OpenAI API or a local sentence-transformers model
4. PCA-whitens and L2-normalizes the embeddings
5. Runs spherical KMeans to create micro-clusters
6. Builds a cosine similarity graph over centroids
7. Applies Leiden community detection to find ~20-50 semantic communities
8. Outputs `community_labels_k{K}.csv` with representative n-grams for each community

### Manual Labeling Step

After Stage 2, you must open the `community_labels_k{K}.csv` file and add two columns:
- **`category`**: A high-level label (e.g., "technology", "marketing", "operations")
- **`subcategory`** (optional): A finer-grained label within the category

Review the `representatives` column to understand what each community captures. See `examples/intangible_investment/labeled_communities.csv` for a worked example.

### Stage 3: Document N-Gram Extraction (`03_extract_doc_ngrams.py`)

1. Joins clustering results with your labeled communities to create a master mapping
2. Re-extracts n-grams from each document individually (parallel)
3. Writes one JSON per document with n-gram counts

### Stage 4: Document Scoring (`04_score_documents.py`)

1. Embeds all whitelist n-grams and computes community centroids
2. For each document: computes cosine similarity of its n-grams to their community centroids
3. Accumulates category/subcategory scores with cosine weighting
4. Normalizes to probability distributions
5. Outputs wide CSVs: `scores_category_prob_embedding.csv`, etc.

## Embedding Providers

The pipeline supports two embedding backends, configurable independently for clustering (Stage 2) and scoring (Stage 4):

| Provider | Config value | Pros | Cons |
|----------|-------------|------|------|
| sentence-transformers | `"sentence-transformers"` | Free, local, fast | Lower dimensionality (768) |
| OpenAI API | `"openai"` | High quality (3072-dim) | Costs money, needs API key |

To use OpenAI, set the environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

The clustering and scoring stages use different embedding spaces by design. The community assignments from clustering are the bridge between stages, not the embedding vectors.

## Configuration

All parameters live in a single `config.yaml`. Key sections:

- **`extract`**: Input directory, format, optional HTML cleaning and section extraction
- **`llm_extract`**: Optional LLM-based text filtering (provider, model, expense keywords)
- **`ngrams`**: Top N count, n-gram range, POS patterns, custom stop words
- **`embedding`**: Provider choice and model settings for clustering
- **`clustering`**: PCA, KMeans, Leiden parameters
- **`doc_ngrams`**: Path to labeled communities, optional keyword windowing
- **`scoring`**: Embedding provider for scoring, similarity threshold, normalization

See `config.yaml` for the full annotated schema.

## Output Structure

```
output/<run_name>/
    texts/                          # Stage 1 output
    llm_extracts/                   # Stage 1b output (if LLM extraction enabled)
    clusters/
        ngram_master_list.csv       # Top N n-grams
        embeddings/                 # N-gram and community centroid embeddings
        k_level_results/            # Micro-cluster assignments
        community_results/
            communities_k{K}.csv    # N-gram -> community assignments
            community_labels_k{K}.csv  # For manual labeling
    master_ngram_mapping.csv        # Stage 3: n-gram -> category mapping
    doc_ngrams/                     # Stage 3: per-document n-gram JSONs
    scores/                         # Stage 4: probability distribution CSVs
        scores_category_prob_embedding.csv
        scores_category_prob_counts.csv
        scores_subcategory_prob_embedding.csv
        scores_subcategory_prob_counts.csv
```

## Example: Intangible Investment from 10-K Filings

The `examples/intangible_investment/` directory contains a complete worked example replicating the measurement of intangible investment intensity from SEC 10-K filings. See its README for details.

## Citation

If you use this pipeline in your research, please cite:

> Eisfeldt, Andrea L., Barney Hartman-Glaser, Edward T. Kim, and Ki Beom Lee. "Intangible Intensity." Working Paper, 2025.

BibTeX:
```bibtex
@unpublished{eisfeldt2025intangible,
  title={Intangible Intensity},
  author={Eisfeldt, Andrea L. and Hartman-Glaser, Barney and Kim, Edward T. and Lee, Ki Beom},
  year={2025},
  note={Working Paper}
}
```
