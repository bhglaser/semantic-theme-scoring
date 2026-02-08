#!/usr/bin/env python3
"""
Stage 3: Per-document n-gram extraction + master mapping build.

1. Joins the clustering output (communities_k{K}.csv) with the user-labeled
   community_labels CSV to produce master_ngram_mapping.csv.
2. Re-extracts n-grams from each document and writes one JSON per doc.

Usage:
    python 03_extract_doc_ngrams.py --config config.yaml
"""

import argparse
import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    load_config, get_run_dir, setup_nltk,
    get_stop_words, get_english_words, build_pos_patterns,
    extract_ngrams_from_text, read_document,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("03_extract_doc_ngrams")


# ---------------------------------------------------------------------------
# Keyword windowing (optional)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [s.strip() for s in sents if s.strip()]


def extract_keyword_windows(text: str, patterns: list[str],
                            n_before: int = 1, n_after: int = 10) -> list[str]:
    """Extract text windows around keyword matches."""
    regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
    sents = split_sentences(text)
    if not sents:
        return []
    hits = [i for i, s in enumerate(sents) if regex.search(s)]
    if not hits:
        return []
    windows = []
    for i in hits:
        a = max(0, i - max(0, n_before))
        b = min(len(sents), i + 1 + max(0, n_after))
        if windows and a <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], b)
        else:
            windows.append([a, b])
    blocks = [" ".join(sents[a:b]) for a, b in windows]
    return list(dict.fromkeys(blocks))


# ---------------------------------------------------------------------------
# Master mapping
# ---------------------------------------------------------------------------

def build_master_mapping(clusters_csv: str, labels_csv: str, k: int) -> pd.DataFrame:
    """
    Join clusters (ngram -> community) with user-labeled communities
    (community -> category/subcategory) to produce the master mapping.
    """
    df_c = pd.read_csv(clusters_csv)
    df_l = pd.read_csv(labels_csv)
    df_c.columns = [str(c).strip() for c in df_c.columns]
    df_l.columns = [str(c).strip() for c in df_l.columns]

    comm_col = f"community_id_k{k}"

    # The clusters CSV has the community assignment column
    if comm_col not in df_c.columns:
        raise ValueError(f"Clusters CSV missing column: {comm_col}")

    # The labels CSV might use "community_id" or the k-specific name
    if comm_col in df_l.columns:
        label_join_col = comm_col
    elif "community_id" in df_l.columns:
        label_join_col = "community_id"
        df_l = df_l.rename(columns={"community_id": comm_col})
    else:
        raise ValueError(f"Labels CSV missing 'community_id' or '{comm_col}' column")

    if "category" not in df_l.columns:
        raise ValueError("Labels CSV missing 'category' column. Did you complete the manual labeling step?")

    merged = df_c.merge(df_l, how="inner", on=comm_col, suffixes=("", "_y"))
    merged = merged.dropna(subset=["category"])

    keep = ["ngram", comm_col]
    for col in ["category", "subcategory", "frequency"]:
        if col in merged.columns:
            keep.append(col)

    out = merged[keep].copy()
    out["ngram"] = out["ngram"].astype(str).str.strip().str.lower()
    if "category" in out.columns:
        out["category"] = out["category"].astype(str).str.strip().str.lower()
    if "subcategory" in out.columns:
        out["subcategory"] = out["subcategory"].astype(str).str.strip().str.lower()

    # Deduplicate by ngram
    out = out.drop_duplicates(subset=["ngram"], keep="first").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Per-document extraction (process-pool worker)
# ---------------------------------------------------------------------------

# Module-level state set by _init_worker (for ProcessPoolExecutor)
_worker_cfg = {}


def _init_worker(cfg_dict):
    """Initializer for worker processes — sets up NLTK and shared config."""
    global _worker_cfg
    _worker_cfg = cfg_dict
    setup_nltk()


def _process_one_doc(args):
    """Extract n-grams from one document and write JSON."""
    filepath, out_dir = args
    cfg = _worker_cfg
    p = Path(filepath)

    try:
        text = read_document(filepath, cfg.get("input_format", "text"))

        # Optional keyword windowing
        kw_cfg = cfg.get("keyword_window", {})
        if kw_cfg.get("enabled", False) and kw_cfg.get("patterns"):
            windows = extract_keyword_windows(
                text, kw_cfg["patterns"],
                kw_cfg.get("sentences_before", 1),
                kw_cfg.get("sentences_after", 10),
            )
            text_chunks = windows if windows else [text]
        else:
            text_chunks = [text]

        stop_words = get_stop_words(cfg.get("custom_stop_words", []))
        english_words = get_english_words() if cfg.get("require_english", True) else set()
        ngram_range = tuple(cfg.get("ngram_range", [2, 3]))
        pos_patterns = build_pos_patterns(cfg.get("pos_patterns", {2: [["NN", "NN"]], 3: [["NN", "NN", "NN"]]}))
        min_word_len = cfg.get("min_word_length", 3)

        total = Counter()
        for chunk in text_chunks:
            total.update(extract_ngrams_from_text(
                chunk, stop_words, english_words,
                ngram_range, pos_patterns, min_word_len,
                cfg.get("require_english", True),
            ))

        out = {
            "doc_id": p.name,
            "n_windows": len(text_chunks),
            "ngrams": {
                " ".join(k) if isinstance(k, (tuple, list)) else str(k): int(v)
                for k, v in total.items()
            },
        }
        out_path = Path(out_dir) / f"{p.stem}.ngrams.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception as e:
        log.error(f"Failed on {p.name}: {e}")
        fail = {"doc_id": p.name, "n_windows": 0, "ngrams": {}, "error": str(e)}
        (Path(out_dir) / f"{p.stem}.ngrams.json").write_text(
            json.dumps(fail, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return p.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Per-document n-gram extraction + master mapping")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_nltk()

    run_dir = get_run_dir(cfg)
    dn_cfg = cfg.get("doc_ngrams", {})
    ng_cfg = cfg.get("ngrams", {})
    k = dn_cfg.get("k_value", 500)

    # --- Build master mapping ---
    clusters_csv = str(run_dir / "clusters" / "community_results" / f"communities_k{k}.csv")
    labels_csv = dn_cfg.get("community_labels_csv", "")
    if not labels_csv:
        # Default: look for the labels file in the clustering output
        labels_csv = str(run_dir / "clusters" / "community_results" / f"community_labels_k{k}.csv")
        log.info(f"No community_labels_csv specified; using default: {labels_csv}")

    log.info("Building master mapping ...")
    mapping = build_master_mapping(clusters_csv, labels_csv, k)
    mapping_path = run_dir / "master_ngram_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    log.info(f"Master mapping: {len(mapping):,} n-grams -> {mapping_path}")

    # --- Per-document extraction ---
    text_dir = run_dir / "texts"
    if not text_dir.is_dir() or not any(text_dir.iterdir()):
        text_dir = Path(cfg.get("extract", {}).get("input_dir", "./raw_documents"))

    out_dir = run_dir / "doc_ngrams"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_paths = [
        str(p) for p in sorted(text_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json"}
    ]
    log.info(f"Extracting n-grams from {len(doc_paths)} documents ...")

    # Build worker config (serializable dict)
    worker_cfg = {
        "input_format": cfg.get("extract", {}).get("input_format", "text"),
        "keyword_window": dn_cfg.get("keyword_window", {}),
        "custom_stop_words": ng_cfg.get("custom_stop_words", []),
        "require_english": ng_cfg.get("require_english_words", True),
        "ngram_range": ng_cfg.get("ngram_range", [2, 3]),
        "pos_patterns": ng_cfg.get("pos_patterns", {2: [["NN", "NN"]], 3: [["NN", "NN", "NN"]]}),
        "min_word_length": ng_cfg.get("min_word_length", 3),
    }

    jobs = cfg.get("max_workers", os.cpu_count() or 4)
    work_args = [(p, str(out_dir)) for p in doc_paths]

    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(worker_cfg,)) as ex:
        list(tqdm(
            ex.map(_process_one_doc, work_args),
            total=len(work_args),
            desc="Extracting doc n-grams",
        ))

    log.info(f"Stage 3 complete. Doc n-grams: {out_dir}")


if __name__ == "__main__":
    main()
