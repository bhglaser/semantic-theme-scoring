#!/usr/bin/env python3
"""
Stage 4: Score documents against labeled communities.

For each document's n-grams:
  1. Filter to the whitelist (master mapping).
  2. Compute cosine similarity of each n-gram to its community centroid.
  3. Accumulate weighted scores per category/subcategory.
  4. Normalize to probability distributions.

Outputs wide CSVs with one row per document and one column per category.

Usage:
    python 04_score_documents.py --config config.yaml
"""

import argparse
import csv
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_config, get_run_dir, get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("04_score_documents")

EPS = 1e-12


# ---------------------------------------------------------------------------
# Mapping + embeddings
# ---------------------------------------------------------------------------

def load_master_mapping(path: Path, comm_col: str) -> pd.DataFrame:
    """Load and clean the master ngram mapping."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["ngram", comm_col, "category"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Master mapping missing column: {c}")
    if "subcategory" not in df.columns:
        df["subcategory"] = np.nan

    df["ngram"] = df["ngram"].astype(str).str.strip().str.lower()
    df[comm_col] = df[comm_col].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["subcategory"] = df["subcategory"].astype(str).str.strip().str.lower()

    df = df[df["category"].notna() & (df["category"].str.len() > 0)]
    df = df.drop_duplicates(subset=["ngram"], keep="first").reset_index(drop=True)
    return df


def embed_whitelist(texts: list[str], emb_cfg: dict, normalize: bool = True) -> dict[str, np.ndarray]:
    """Embed all whitelist n-grams and return {text: vector} dict."""
    uniq = list(dict.fromkeys(texts))
    if not uniq:
        return {}
    vecs = get_embeddings(uniq, emb_cfg)
    out = {}
    for t, v in zip(uniq, vecs):
        v = np.asarray(v, dtype=np.float32)
        if normalize:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        out[t] = v
    return out


def build_community_centroids(
    mapping: pd.DataFrame, emb_map: dict, comm_col: str, do_normalize: bool = True,
) -> tuple:
    """
    Compute community centroids and build lookup dicts.

    Returns: (centroids, ng2comm, ng2cat, ng2sub,
              all_categories, all_subcategories, cat_wsize, sub_wsize)
    """
    def _norm(v):
        if not do_normalize:
            return v
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    centroids = {}
    for name, group in mapping.groupby(comm_col):
        cid = str(name)
        member_vecs = [_norm(emb_map[ng]) for ng in group["ngram"] if ng in emb_map]
        if member_vecs:
            centroids[cid] = _norm(np.mean(member_vecs, axis=0))

    ng2comm = dict(mapping[["ngram", comm_col]].astype(str).values)
    ng2cat = dict(mapping[["ngram", "category"]].values)
    ng2sub = dict(mapping[["ngram", "subcategory"]].values)

    all_cats = sorted(mapping["category"].dropna().unique().tolist())

    # Auto-detect which categories have subcategories
    sub_mask = mapping["subcategory"].notna() & (mapping["subcategory"].str.len() > 0) & (mapping["subcategory"] != "nan")
    all_subs = sorted(mapping.loc[sub_mask, "subcategory"].unique().tolist()) if sub_mask.any() else []
    cats_with_subs = set(mapping.loc[sub_mask, "category"].unique().tolist()) if sub_mask.any() else set()

    cat_wsize = mapping.groupby("category")["ngram"].nunique().to_dict()
    sub_wsize = mapping.loc[sub_mask].groupby("subcategory")["ngram"].nunique().to_dict() if sub_mask.any() else {}

    return centroids, ng2comm, ng2cat, ng2sub, all_cats, all_subs, cats_with_subs, cat_wsize, sub_wsize


# ---------------------------------------------------------------------------
# Per-document scoring
# ---------------------------------------------------------------------------

def score_one_doc(
    doc_path: Path,
    emb_map: dict,
    comm_centroids: dict,
    ng2comm: dict, ng2cat: dict, ng2sub: dict,
    all_cats: list, all_subs: list, cats_with_subs: set,
    whitelist: set,
    cat_wsize: dict, sub_wsize: dict,
    sim_threshold: float | None,
    relative_weight: bool,
    excluded_cats: set,
) -> tuple:
    """Score a single document. Returns (doc_id, cat_prob_emb, cat_prob_cnt, sub_prob_emb, sub_prob_cnt, debug_rows)."""

    data = json.loads(doc_path.read_text(encoding="utf-8"))
    doc_id = doc_path.stem
    raw_map = data.get("ngrams", {}) or {}

    # Whitelist filter
    doc_ngrams = {}
    for k, v in raw_map.items():
        ng = str(k).strip().lower()
        if ng in whitelist:
            try:
                doc_ngrams[ng] = doc_ngrams.get(ng, 0) + int(v)
            except Exception:
                continue

    raw_cat_emb = Counter()
    raw_cat_cnt = Counter()
    raw_sub_emb = Counter()
    raw_sub_cnt = Counter()
    debug_rows = []

    for ng, cnt in doc_ngrams.items():
        v = emb_map.get(ng)
        if v is None:
            debug_rows.append({"ngram": ng, "count": cnt, "kept": False, "reason": "no_embedding"})
            continue

        comm = str(ng2comm.get(ng))
        cat = ng2cat.get(ng)
        sub = ng2sub.get(ng)

        if comm not in comm_centroids or cat is None:
            debug_rows.append({"ngram": ng, "count": cnt, "kept": False, "reason": "no_centroid"})
            continue

        cos = float(np.dot(v, comm_centroids[comm]))

        if cos <= 0:
            debug_rows.append({"ngram": ng, "count": cnt, "cosine": cos, "kept": False, "reason": "cos<=0"})
            continue
        if sim_threshold is not None and cos < sim_threshold:
            debug_rows.append({"ngram": ng, "count": cnt, "cosine": cos, "kept": False, "reason": f"cos<{sim_threshold}"})
            continue

        raw_cat_emb[cat] += cnt * cos
        raw_cat_cnt[cat] += cnt
        if cat in cats_with_subs and isinstance(sub, str) and sub != "nan" and len(sub) > 0:
            raw_sub_emb[sub] += cnt * cos
            raw_sub_cnt[sub] += cnt
        debug_rows.append({"ngram": ng, "count": cnt, "cosine": cos, "kept": True, "reason": "kept"})

    # --- Normalize ---
    norm_cats = [c for c in all_cats if c not in excluded_cats]

    def _normalize(raw: Counter, keys: list, wsize: dict) -> dict:
        adj = {}
        for k in keys:
            denom = float(wsize.get(k, 0)) or EPS if relative_weight else 1
            adj[k] = raw.get(k, 0.0) / denom
        Z = sum(adj.values())
        return {k: (adj[k] / Z) if Z > 0 else 0.0 for k in keys}

    cat_prob_emb = _normalize(raw_cat_emb, norm_cats, cat_wsize)
    cat_prob_cnt = _normalize(raw_cat_cnt, norm_cats, cat_wsize)
    sub_prob_emb = _normalize(raw_sub_emb, all_subs, sub_wsize)
    sub_prob_cnt = _normalize(raw_sub_cnt, all_subs, sub_wsize)

    return doc_id, cat_prob_emb, cat_prob_cnt, sub_prob_emb, sub_prob_cnt, debug_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Document scoring")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sc_cfg = cfg.get("scoring", {})
    k = cfg.get("doc_ngrams", {}).get("k_value", 500)
    comm_col = f"community_id_k{k}"

    run_dir = get_run_dir(cfg)
    mapping_path = run_dir / "master_ngram_mapping.csv"

    # --- Load mapping ---
    mapping = load_master_mapping(mapping_path, comm_col)
    log.info(f"Master mapping: {len(mapping):,} n-grams")

    # --- Embed whitelist ---
    emb_cfg = sc_cfg.get("embedding", cfg.get("embedding", {}))
    do_normalize = sc_cfg.get("normalize_vectors", True)
    whitelist = sorted(mapping["ngram"].unique().tolist())
    log.info(f"Embedding {len(whitelist):,} whitelist n-grams ...")
    emb_map = embed_whitelist(whitelist, emb_cfg, normalize=do_normalize)

    # --- Build centroids ---
    (comm_centroids, ng2comm, ng2cat, ng2sub,
     all_cats, all_subs, cats_with_subs,
     cat_wsize, sub_wsize) = build_community_centroids(mapping, emb_map, comm_col, do_normalize)
    log.info(f"Communities: {len(comm_centroids)} | Categories: {len(all_cats)} | Subcategories: {len(all_subs)}")

    whitelist_set = set(whitelist)
    sim_threshold = sc_cfg.get("similarity_threshold", 0.0)
    relative_weight = sc_cfg.get("relative_weight", False)
    excluded_cats = set(sc_cfg.get("excluded_categories", []))
    write_debug = sc_cfg.get("write_debug", False)

    # --- Score documents ---
    json_dir = run_dir / "doc_ngrams"
    doc_paths = sorted(p for p in json_dir.iterdir() if p.suffix == ".json")
    log.info(f"Scoring {len(doc_paths)} documents ...")

    out_dir = run_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_debug:
        (out_dir / "debug").mkdir(exist_ok=True)

    rows_cat_emb, rows_cat_cnt = [], []
    rows_sub_emb, rows_sub_cnt = [], []

    import os as _os
    num_workers = min(8, _os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(
                score_one_doc, p, emb_map, comm_centroids,
                ng2comm, ng2cat, ng2sub,
                all_cats, all_subs, cats_with_subs,
                whitelist_set, cat_wsize, sub_wsize,
                sim_threshold, relative_weight, excluded_cats,
            ): p
            for p in doc_paths
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            try:
                doc_id, cat_emb, cat_cnt, sub_emb, sub_cnt, dbg = fut.result()
                rows_cat_emb.append({"doc_id": doc_id, **cat_emb})
                rows_cat_cnt.append({"doc_id": doc_id, **cat_cnt})
                rows_sub_emb.append({"doc_id": doc_id, **sub_emb})
                rows_sub_cnt.append({"doc_id": doc_id, **sub_cnt})

                if write_debug and dbg:
                    dbg_path = out_dir / "debug" / f"{doc_id}__contribs.csv"
                    with dbg_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["ngram", "count", "cosine", "kept", "reason"])
                        writer.writeheader()
                        for r in dbg:
                            writer.writerow({k: r.get(k) for k in ["ngram", "count", "cosine", "kept", "reason"]})

            except Exception as e:
                log.exception(f"Failed on {futures[fut].name}: {e}")

    # --- Save wide CSVs ---
    def save_wide(rows, cols, filename):
        if not rows:
            return
        df = pd.DataFrame(rows)
        norm_cols = [c for c in cols if c not in excluded_cats]
        for c in norm_cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[["doc_id"] + norm_cols].fillna(0.0)
        df.to_csv(out_dir / filename, index=False)

    save_wide(rows_cat_emb, all_cats, "scores_category_prob_embedding.csv")
    save_wide(rows_cat_cnt, all_cats, "scores_category_prob_counts.csv")
    save_wide(rows_sub_emb, all_subs, "scores_subcategory_prob_embedding.csv")
    save_wide(rows_sub_cnt, all_subs, "scores_subcategory_prob_counts.csv")

    log.info(f"Stage 4 complete. Scores saved to {out_dir}")


if __name__ == "__main__":
    main()
