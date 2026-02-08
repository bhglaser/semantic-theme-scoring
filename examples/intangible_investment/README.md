# Example: Measuring Intangible Investment from 10-K Filings

This example replicates the measurement of intangible investment intensity from SEC 10-K filings using the n-gram pipeline.

## Background

Firms report Selling, General & Administrative (SG&A) expenses in their financial statements, but these aggregate numbers mix routine operating costs with investments in intangible capital (R&D, brand building, organizational processes). This pipeline analyzes the text of 10-K filings to decompose SG&A into meaningful categories.

## What's Included

- **`config_intangible.yaml`** — Pre-filled pipeline configuration with:
  - Item 7 (MD&A) section extraction patterns
  - Financial domain stop words
  - SG&A keyword windowing patterns
  - Clustering parameters matching the paper

- **`labeled_communities.csv`** — 231 hand-labeled n-gram communities from the full-corpus analysis. Each community is classified as:
  - `Intangible investment` with subcategories: `brand or customer capital`, `knowledge capital`, `organization capital`
  - `Not intangible investment`
  - `unknown`

- **`sample_10k_texts/`** — Sample cleaned Item 7 text files (add your own here)

## Running the Example

```bash
cd ngram_pipeline/

# Stage 1: Extract text (applies section extraction for Item 7)
python 01_extract_text.py --config examples/intangible_investment/config_intangible.yaml

# Stage 2: Cluster n-grams
python 02_cluster_ngrams.py --config examples/intangible_investment/config_intangible.yaml

# The labeled_communities.csv is already provided, so skip the manual labeling step.

# Stage 3: Extract per-document n-grams and build master mapping
python 03_extract_doc_ngrams.py --config examples/intangible_investment/config_intangible.yaml

# Stage 4: Score documents
python 04_score_documents.py --config examples/intangible_investment/config_intangible.yaml
```

## Output

The final scores appear in `output/intangible_investment/scores/`:
- `scores_category_prob_embedding.csv` — Per-document probability of each category
- `scores_subcategory_prob_embedding.csv` — Per-document probability of each intangible investment subcategory

## Adapting to Your Own Corpus

1. Replace `sample_10k_texts/` with your own documents
2. Modify `config_intangible.yaml`:
   - Update `extract.input_dir` to point to your documents
   - Adjust `section_extraction` patterns (or disable if using full documents)
   - Update `custom_stop_words` for your domain
   - Adjust `keyword_window.patterns` (or disable for full-document n-gram extraction)
3. After Stage 2, label the communities with your own categories
4. Run Stages 3 and 4 with your labeled communities
