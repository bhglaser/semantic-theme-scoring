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

- **`sample_10k_texts/`** — 12 pre-extracted Item 7 (MD&A) sections from public EDGAR filings, already cleaned. **Note:** If you want to run the pipeline on raw 10-K HTML filings from EDGAR, enable `html_cleaning` and `section_extraction` in the config. For this example, they're disabled since the sample data is already processed.

## Prerequisites

Before running the example, you need to set up Ollama for Stage 1b (LLM text extraction). **These are one-time setup steps:**

1. **Install Ollama** (one-time):
   ```bash
   brew install ollama  # macOS
   # Or download from https://ollama.com
   ```

2. **Pull the model** used in the example (one-time, downloads ~2GB):
   ```bash
   ollama pull gemma3n:e2b
   ```

3. **Install the Python ollama client** (one-time per Python environment):
   ```bash
   pip3 install ollama
   ```

4. **Ensure the Ollama server is running** (usually runs automatically after install):
   ```bash
   ollama serve  # Only needed if you get connection errors
   ```
   If you see "address already in use", the server is already running and you're good to go.

5. **Download NLTK data** (one-time, required for Stage 2 n-gram extraction):
   ```bash
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('words')"
   ```

**Alternative:** If you want to skip Stage 1b, set `llm_extract.enabled: false` in `config_intangible.yaml` and skip straight to Stage 2. The pipeline will build the dictionary from full Item 7 text instead of LLM-filtered quotes.

## Running the Example

```bash
cd semantic-theme-scoring/

# Stage 1: Extract text (sample files are already pre-extracted Item 7, so this is a pass-through)
python3 01_extract_text.py --config examples/intangible_investment/config_intangible.yaml

# Stage 1b: LLM-based SG&A text extraction (requires Ollama or OpenAI API key)
# This extracts expense-relevant quotes from each document for dictionary construction.
python3 01b_llm_extract.py --config examples/intangible_investment/config_intangible.yaml

# Stage 2: Cluster n-grams (auto-detects LLM extracts if present)
python3 02_cluster_ngrams.py --config examples/intangible_investment/config_intangible.yaml

# >>> Manual step: open output/intangible_investment/clusters/community_results/community_labels_k20.csv
# >>> (k20 for this small sample; the full corpus uses k500)
# >>> Add 'category' and 'subcategory' columns (see labeled_communities_reference.csv for an example)
# >>> Then set doc_ngrams.community_labels_csv in config_intangible.yaml to point to your labeled file

# Stage 3: Extract per-document n-grams from full Item 7 text and build master mapping
python3 03_extract_doc_ngrams.py --config examples/intangible_investment/config_intangible.yaml

# Stage 4: Score documents
python3 04_score_documents.py --config examples/intangible_investment/config_intangible.yaml
```

**Note on Stage 1b:** The paper uses an LLM to extract SG&A-focused text (definition quotes, business driver quotes, and change driver quotes) before building the n-gram dictionary. This ensures the dictionary captures expense-relevant language. Stage 2 reads from these LLM extracts rather than full documents. Stage 3 then scores each document using the full Item 7 text against this focused dictionary.

## Labeling Guide

After Stage 2, open the `community_labels_k{K}.csv` file (k20 for this sample, k500 for the full corpus). Each row is a community with representative n-grams. Add two columns:

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
