# Example: Measuring Intangible Investment from 10-K Filings

This example replicates the measurement of intangible investment intensity from SEC 10-K filings using the n-gram pipeline.

## Background

Firms report Selling, General & Administrative (SG&A) expenses in their financial statements, but these aggregate numbers mix routine operating costs with investments in intangible capital (R&D, brand building, organizational processes). This pipeline analyzes the text of 10-K filings to decompose SG&A into meaningful categories.

## What's Included

- **`config_intangible.yaml`** — Pre-filled pipeline configuration with:
  - Item 7 (MD&A) section extraction patterns
  - LLM-based SG&A text extraction keywords (for dictionary construction)
  - Financial domain stop words
  - Clustering parameters matching the paper

- **`labeled_communities_reference.csv`** — A reference showing what the completed labeling step looks like, from the full-corpus analysis (231 communities labeled across thousands of filings). **This is not used as a pipeline input** — your own Stage 2 run will produce different communities that you need to label yourself. Use this file as a guide for how to structure your labels.

- **`sample_10k_texts/`** — 10 cleaned Item 7 (MD&A) text excerpts from public EDGAR filings

## Running the Example

```bash
cd ngram_pipeline/

# Stage 1: Extract text (applies section extraction for Item 7)
python 01_extract_text.py --config examples/intangible_investment/config_intangible.yaml

# Stage 1b: LLM-based SG&A text extraction (requires Ollama or OpenAI API key)
# This extracts expense-relevant quotes from each document for dictionary construction.
python 01b_llm_extract.py --config examples/intangible_investment/config_intangible.yaml

# Stage 2: Cluster n-grams (auto-detects LLM extracts if present)
python 02_cluster_ngrams.py --config examples/intangible_investment/config_intangible.yaml

# >>> Manual step: open output/intangible_investment/clusters/community_results/community_labels_k500.csv
# >>> Add 'category' and 'subcategory' columns (see labeled_communities_reference.csv for an example)
# >>> Then set doc_ngrams.community_labels_csv in config_intangible.yaml to point to your labeled file

# Stage 3: Extract per-document n-grams from full Item 7 text and build master mapping
python 03_extract_doc_ngrams.py --config examples/intangible_investment/config_intangible.yaml

# Stage 4: Score documents
python 04_score_documents.py --config examples/intangible_investment/config_intangible.yaml
```

**Note on Stage 1b:** The paper uses an LLM to extract SG&A-focused text (definition quotes, business driver quotes, and change driver quotes) before building the n-gram dictionary. This ensures the dictionary captures expense-relevant language. Stage 2 reads from these LLM extracts rather than full documents. Stage 3 then scores each document using the full Item 7 text against this focused dictionary.

## Labeling Guide

After Stage 2, open the `community_labels_k500.csv` file. Each row is a community with representative n-grams. Add two columns:

| category | subcategory |
|----------|------------|
| `Intangible investment` | `knowledge capital` |
| `Intangible investment` | `brand or customer capital` |
| `Intangible investment` | `organization capital` |
| `Not intangible investment` | *(leave blank)* |
| `unknown` | *(leave blank)* |

Review the `representatives` column to decide which category fits. For example:
- `function research development, research development work` → **knowledge capital**
- `advertising promotion expense, advertising expense cost` → **brand or customer capital**
- `cost office rent, rent expense office` → **Not intangible investment**

## Output

The final scores appear in `output/intangible_investment/scores/`:
- `scores_category_prob_embedding.csv` — Per-document probability of each category
- `scores_subcategory_prob_embedding.csv` — Per-document probability of each intangible investment subcategory

## Adapting to Your Own Corpus

1. Replace `sample_10k_texts/` with your own documents
2. Modify `config_intangible.yaml`:
   - Update `extract.input_dir` to point to your documents
   - Adjust `section_extraction` patterns (or disable if using full documents)
   - Update `llm_extract.expense_keywords` for your domain (or disable LLM extraction)
   - Update `custom_stop_words` for your domain
3. After Stage 2, label the communities with your own categories
4. Run Stages 3 and 4 with your labeled communities
