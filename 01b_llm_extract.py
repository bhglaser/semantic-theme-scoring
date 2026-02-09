#!/usr/bin/env python3
"""
Stage 1b: LLM-based focused text extraction.

Reads cleaned text files (Stage 1 output) and uses an LLM to extract
relevant quotes around configured expense keywords. This produces a
filtered text corpus for Stage 2 dictionary construction.

In the intangible investment use case, this step finds SG&A-related
context in each document's Item 7 section, then asks an LLM to extract
definition, business driver, and change driver quotes.

Usage:
    python 01b_llm_extract.py --config config.yaml
"""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from utils import load_config, get_run_dir, call_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("01b_llm_extract")


# ---------------------------------------------------------------------------
# Prompt template (matches Eisfeldt et al. 2025 methodology)
# ---------------------------------------------------------------------------

QUOTE_EXTRACTION_PROMPT = """
You are a precise, rule-following financial analyst. Your task is to analyze text from a company's 10-K and extract the most relevant sentences that explain a specific expense category.

**EXPENSE CATEGORY TO ANALYZE:**
{component_name}

**CONTEXT TO ANALYZE:**
The following text snippets have been pre-selected because they likely mention "{component_name}" and discuss financial results.
---
{context_blocks}
---

**CRITICAL INSTRUCTIONS (MUST BE FOLLOWED):**

1.  **Extract Three Types of Information:** Your goal is to extract direct quotes that fall into three specific categories.

    *   **1. Definition Quotes (The "What"):**
        *   Extract sentences that define the composition of the expense. These sentences describe what types of costs make up the expense category (e.g., personnel, marketing, facilities).

    *   **2. Business Driver Quotes (The "Ongoing Why"):**
        *   Extract sentences that explain the fundamental business purpose of the expense. These sentences describe the core activities the expense supports and why the company incurs these costs as a part of its strategy.

    *   **3. Change Driver Quotes (The "Why it Changed"):**
        *   Extract sentences that explain the specific reasons for an increase or decrease in the expense during the reporting period. These sentences often compare the current period to a prior period and mention specific causes for the variance.

2.  **Extract Direct Quotes:** You MUST extract the information as direct quotes from the "CONTEXT TO ANALYZE". Each quote MUST be a complete, full sentence from its beginning to its end (e.g., to the period). Do not extract partial phrases or fragments. Do not paraphrase.

3.  **No Data Fabrication (Most Important Rule):** If you cannot find any relevant quotes for a specific category in the provided context, you MUST return an empty list for that category (e.g., "change_driver_quotes": []). Do not invent information or use any text from these instructions.

**OUTPUT SCHEMA:**
Your entire response MUST be a single, valid JSON object that follows the structure below. Do not add any text before or after the JSON object.

{{
  "component_name": "{component_name}",
  "definition_quotes": [
  ],
  "business_driver_quotes": [
  ],
  "change_driver_quotes": [
  ]
}}
"""


# ---------------------------------------------------------------------------
# Context block extraction
# ---------------------------------------------------------------------------

def preprocess_line(line: str) -> str:
    """Normalize a line for keyword matching."""
    processed = line.lower()
    processed = re.sub(r"[^a-z0-9\s.,$%:;-]", "", processed)
    processed = re.sub(r"\s+", " ", processed).strip()
    return processed


def find_context_blocks(text: str, keywords: list[str],
                        lines_before: int = 0, lines_after: int = 10) -> list[str]:
    """
    Find text blocks around keyword matches in a document.

    Splits the text into lines, finds lines matching any keyword,
    and extracts context windows around each match. Overlapping
    windows are merged.
    """
    normalized = text.replace("\r\n", "\n")

    # Ensure we have line-based structure
    if normalized.count("\n") < 5:
        normalized = re.sub(r"(?<=[.!?])\s+", r"\g<0>\n", normalized)

    all_lines = normalized.splitlines()
    original_lines = [line.strip() for line in all_lines if line and line.strip()]
    if not original_lines:
        return []

    processed_lines = [preprocess_line(line) for line in original_lines]
    processed_keywords = [k.lower() for k in keywords]

    found_chunks = set()
    for keyword in processed_keywords:
        matching_indices = [
            i for i, processed_line in enumerate(processed_lines)
            if re.search(r"\b" + re.escape(keyword) + r"\b", processed_line)
        ]
        for idx in matching_indices:
            start = max(0, idx - lines_before)
            end = min(len(original_lines), idx + lines_after + 1)
            chunk = "\n".join(original_lines[start:end])
            found_chunks.add(chunk)

    return list(found_chunks)


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def process_one_document(filepath: str, llm_cfg: dict, expense_keywords: dict,
                         lines_before: int, lines_after: int,
                         out_dir: str) -> tuple[str, bool, str]:
    """
    Extract LLM-filtered text from one document.

    For each expense keyword category, finds context blocks and asks the
    LLM to extract relevant quotes. All quotes across categories are
    combined into a single output text file.

    Returns (filename, success, message).
    """
    name = os.path.basename(filepath)
    out_path = Path(out_dir) / Path(name).with_suffix(".txt").name

    # Skip if already processed
    if out_path.exists() and out_path.stat().st_size > 0:
        return name, True, "already exists"

    try:
        text = Path(filepath).read_text(encoding="utf-8")
    except Exception as e:
        return name, False, f"read error: {e}"

    if not text.strip():
        return name, False, "empty file"

    all_quotes = []

    for category_key, keywords in expense_keywords.items():
        context_blocks = find_context_blocks(text, keywords, lines_before, lines_after)
        if not context_blocks:
            continue

        context_for_llm = "\n\n---\n\n".join(context_blocks)
        component_name = category_key.replace("_", " ").title()
        prompt = QUOTE_EXTRACTION_PROMPT.format(
            component_name=component_name,
            context_blocks=context_for_llm,
        )

        response = call_llm(prompt, llm_cfg)
        if not response:
            continue

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            log.warning(f"Invalid JSON from LLM for {name} [{category_key}]")
            continue

        # Collect all quotes from this category
        for quote_key in ["definition_quotes", "business_driver_quotes", "change_driver_quotes"]:
            quotes = data.get(quote_key, [])
            if isinstance(quotes, list):
                all_quotes.extend(q for q in quotes if isinstance(q, str) and q.strip())

    if not all_quotes:
        return name, False, "no quotes extracted"

    # Deduplicate while preserving order
    seen = set()
    unique_quotes = []
    for q in all_quotes:
        q_stripped = q.strip()
        if q_stripped not in seen:
            seen.add(q_stripped)
            unique_quotes.append(q_stripped)

    # Write combined quotes as plain text (one sentence per line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(unique_quotes), encoding="utf-8")
    return name, True, f"{len(unique_quotes)} quotes"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1b: LLM-based text extraction")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm_cfg = cfg.get("llm_extract", {})

    if not llm_cfg.get("enabled", False):
        log.info("LLM extraction is disabled in config (llm_extract.enabled: false). Skipping.")
        return

    expense_keywords = llm_cfg.get("expense_keywords", {})
    if not expense_keywords:
        log.error("No expense_keywords configured in llm_extract section. Nothing to extract.")
        return

    lines_before = llm_cfg.get("context_lines_before", 0)
    lines_after = llm_cfg.get("context_lines_after", 10)
    max_workers = llm_cfg.get("max_workers", 1)

    run_dir = get_run_dir(cfg)

    # Input: Stage 1 text output
    text_dir = run_dir / "texts"
    if not text_dir.is_dir() or not any(text_dir.iterdir()):
        text_dir = Path(cfg.get("extract", {}).get("input_dir", "./raw_documents"))
        log.info(f"No Stage 1 output found; reading directly from {text_dir}")

    out_dir = run_dir / "llm_extracts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather input files
    files = [
        str(p) for p in sorted(text_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json"}
    ]
    if not files:
        log.error(f"No input files found in {text_dir}")
        return

    log.info(f"Processing {len(files)} documents with LLM extraction "
             f"(provider={llm_cfg.get('provider', 'ollama')}, workers={max_workers}) ...")

    success, skipped, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                process_one_document, f, llm_cfg, expense_keywords,
                lines_before, lines_after, str(out_dir),
            ): f
            for f in files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM extraction"):
            name, ok, msg = fut.result()
            if ok and msg == "already exists":
                skipped += 1
            elif ok:
                success += 1
            else:
                failed += 1
                log.debug(f"{name}: {msg}")

    log.info(f"Stage 1b complete: {success} extracted, {skipped} skipped (existing), {failed} failed")
    log.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
