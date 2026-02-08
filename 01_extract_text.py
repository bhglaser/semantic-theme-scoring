#!/usr/bin/env python3
"""
Stage 1: Text extraction and cleaning.

Reads raw documents from the input directory, applies optional HTML cleaning
and section extraction, then writes cleaned text files to the output directory.

For a plain-text corpus this is largely a pass-through with whitespace
normalization. Enable html_cleaning and/or section_extraction in the config
for structured documents (e.g., SEC filings, HTML pages).

Usage:
    python 01_extract_text.py --config config.yaml
"""

import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from utils import load_config, get_run_dir, read_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("01_extract_text")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def strip_html_tags(text: str, tags: list[str]) -> str:
    """Remove specified HTML opening+closing tags (keeps content between them)."""
    for tag in tags:
        text = re.sub(rf"<{tag}[^>]*>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(rf"</{tag}>", "", text, flags=re.IGNORECASE)
    # Remove any remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize various whitespace characters to clean spaces and newlines."""
    text = text.replace("\r\n", "\n")
    text = text.replace("\xa0", " ")       # non-breaking space
    text = text.replace("\t", " ")
    text = text.replace("\v", "")
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"(\n\s*){3,}", "\n\n", text)
    # Remove HTML entities
    text = re.sub(r"&#?\w{1,8};", " ", text)
    return text.strip()


def extract_section(text: str, start_patterns: list[str], end_patterns: list[str],
                    min_words: int = 250) -> str | None:
    """
    Extract text between start and end regex patterns.

    Returns the longest qualifying section, or None if nothing found.
    """
    lower = text.lower()
    starts = []
    for pat in start_patterns:
        for m in re.finditer(pat, lower):
            starts.append(m.start())
    starts = sorted(set(starts))

    ends = []
    for pat in end_patterns:
        for m in re.finditer(pat, lower):
            ends.append(m.start())
    ends = sorted(set(ends))

    if not starts:
        return None

    sections = []
    for s in starts:
        # Find earliest end after this start
        e = next((ep for ep in ends if ep > s), len(text))
        section = text[s:e]
        if len(section.split()) >= min_words:
            sections.append(section)

    if not sections:
        return None

    # Return all qualifying sections joined
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_one_file(filepath: str, cfg_extract: dict) -> tuple[str, str | None]:
    """
    Process a single file: read, clean, optionally extract section.
    Returns (filename, cleaned_text_or_None).
    """
    name = os.path.basename(filepath)
    input_format = cfg_extract.get("input_format", "text")

    try:
        text = read_document(filepath, input_format)
    except Exception as e:
        log.warning(f"Could not read {name}: {e}")
        return name, None

    # Optional HTML cleaning
    html_cfg = cfg_extract.get("html_cleaning", {})
    if html_cfg.get("enabled", False):
        tags = html_cfg.get("strip_tags", ["div", "tr", "td", "font", "p", "table"])
        text = strip_html_tags(text, tags)

    # Whitespace normalization (always)
    text = normalize_whitespace(text)

    # Optional section extraction
    sec_cfg = cfg_extract.get("section_extraction", {})
    if sec_cfg.get("enabled", False):
        section = extract_section(
            text,
            sec_cfg.get("start_patterns", []),
            sec_cfg.get("end_patterns", []),
            sec_cfg.get("min_words", 250),
        )
        if section is None:
            log.info(f"No matching section found in {name}, skipping")
            return name, None
        text = section

    if not text.strip():
        return name, None

    return name, text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Text extraction")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_extract = cfg.get("extract", {})
    input_dir = cfg_extract.get("input_dir", "./raw_documents")
    max_workers = cfg.get("max_workers", 8)

    run_dir = get_run_dir(cfg)
    out_dir = run_dir / "texts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather input files
    exts = {".txt", ".html", ".htm", ".json", ".md"}
    files = [
        str(p) for p in sorted(Path(input_dir).iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    if not files:
        log.error(f"No input files found in {input_dir}")
        return

    log.info(f"Processing {len(files)} files from {input_dir} ...")

    processed, skipped = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_file, f, cfg_extract): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting text"):
            name, text = fut.result()
            if text is None:
                skipped += 1
                continue
            out_path = out_dir / Path(name).with_suffix(".txt").name
            out_path.write_text(text, encoding="utf-8")
            processed += 1

    log.info(f"Stage 1 complete: {processed} files written, {skipped} skipped")
    log.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
