#!/usr/bin/env python3
"""
Stage 2: N-gram extraction, embedding, and clustering.

1. Extracts POS-filtered bigrams/trigrams from all documents in the corpus.
2. Keeps the top N by frequency.
3. Embeds them (OpenAI API or sentence-transformers).
4. PCA-whitens and L2-normalizes.
5. Runs spherical KMeans, builds a centroid graph, then Leiden community detection.
6. Outputs community_labels_k{K}.csv for manual labeling.

Usage:
    python 02_cluster_ngrams.py --config config.yaml
"""

import argparse
import glob
import json
import logging
import os

import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import igraph as ig
import leidenalg

from utils import (
    load_config, get_run_dir, setup_nltk,
    get_stop_words, get_english_words, build_pos_patterns,
    extract_ngrams_from_text, read_document, get_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("02_cluster_ngrams")


# ---------------------------------------------------------------------------
# Corpus-level n-gram extraction
# ---------------------------------------------------------------------------

def _process_one_file(args):
    """Worker: extract n-grams from a single file. Returns a Counter."""
    filepath, input_format, stop_words, english_words, ngram_range, pos_patterns, min_word_len, require_english = args
    try:
        text = read_document(filepath, input_format)
        return extract_ngrams_from_text(
            text, stop_words, english_words,
            ngram_range, pos_patterns, min_word_len, require_english,
        )
    except Exception as e:
        log.warning(f"Could not process {filepath}: {e}")
        return Counter()


def build_master_ngram_list(cfg: dict, text_dir: str, output_path: str) -> pd.DataFrame:
    """
    Scan all documents, extract n-grams, filter bigrams subsumed by trigrams,
    keep top N, and save to CSV.
    """
    ngram_cfg = cfg.get("ngrams", {})
    top_n = ngram_cfg.get("top_n", 10000)
    ngram_range = tuple(ngram_cfg.get("ngram_range", [2, 3]))
    min_word_len = ngram_cfg.get("min_word_length", 3)
    require_english = ngram_cfg.get("require_english_words", True)
    pos_patterns = build_pos_patterns(ngram_cfg.get("pos_patterns", {2: [["NN", "NN"]], 3: [["NN", "NN", "NN"]]}))
    stop_words = get_stop_words(ngram_cfg.get("custom_stop_words", []))
    english_words = get_english_words() if require_english else set()
    max_workers = cfg.get("max_workers", 8)

    input_format = cfg.get("extract", {}).get("input_format", "text")

    # Gather files
    exts = ("*.txt", "*.json", "*.md")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(text_dir, ext)))
    if not files:
        log.error(f"No text files found in {text_dir}")
        return pd.DataFrame()

    log.info(f"Extracting n-grams from {len(files)} files (workers={max_workers}) ...")

    # Parallel extraction
    work_args = [
        (f, input_format, stop_words, english_words, ngram_range, pos_patterns, min_word_len, require_english)
        for f in files
    ]
    master_counter = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one_file, a): a[0] for a in work_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting n-grams"):
            try:
                master_counter.update(fut.result())
            except Exception as e:
                log.error(f"Worker exception: {e}")

    log.info(f"Unique POS-filtered n-grams: {len(master_counter):,}")
    if not master_counter:
        log.error("No n-grams extracted. Check your input files and config.")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(master_counter.items(), columns=["ngram", "frequency"])

    # Filter bigrams that are subparts of extracted trigrams
    if ngram_range[1] >= 3:
        df["length"] = df["ngram"].apply(len)
        trigrams = df[df["length"] == 3]
        bigrams = df[df["length"] == 2]
        if not trigrams.empty:
            tri_parts = set()
            for tri in trigrams["ngram"]:
                tri_parts.add(tri[:2])
                tri_parts.add(tri[1:])
            bigrams = bigrams[~bigrams["ngram"].isin(tri_parts)]
            df = pd.concat([trigrams, bigrams])
        df = df.drop(columns=["length"])

    # Keep top N
    df = df.nlargest(top_n, "frequency")
    df["ngram_str"] = df["ngram"].apply(lambda x: " ".join(x))
    out = df[["ngram_str", "frequency"]].copy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info(f"Saved top {len(out):,} n-grams to {output_path}")
    return out


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def pca_project(X: np.ndarray, n_components: int, whiten: bool, seed: int):
    """Mean-center, PCA-project, L2-normalize."""
    n_components = min(n_components, X.shape[1] - 1)
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, whiten=whiten, random_state=seed)
    Z = pca.fit_transform(Xc)
    return normalize(Z), pca


def spherical_kmeans(X: np.ndarray, k: int, n_init: int, max_iter: int, seed: int):
    """Run KMeans and return labels + L2-normalized centroids."""
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(X)
    return labels, normalize(km.cluster_centers_)


def build_centroid_graph(centroids: np.ndarray, knn: int, min_cos: float) -> ig.Graph:
    """Build igraph from centroid cosine similarities above threshold."""
    n = len(centroids)
    if n == 0:
        return ig.Graph(n=0)
    nn = NearestNeighbors(n_neighbors=min(knn + 1, n), metric="cosine").fit(centroids)
    dist, ind = nn.kneighbors(centroids)

    edges, weights = {}, {}
    for i in range(n):
        for idx in range(1, ind.shape[1]):
            j = int(ind[i, idx])
            cos_sim = 1.0 - float(dist[i, idx])
            if cos_sim >= min_cos:
                edge = tuple(sorted((i, j)))
                if edge not in weights or cos_sim > weights[edge]:
                    edges[edge] = True
                    weights[edge] = cos_sim

    edge_list = list(edges.keys())
    weight_list = [weights[e] for e in edge_list]
    g = ig.Graph(n=n, edges=edge_list, directed=False)
    g.es["weight"] = weight_list
    return g


def sweep_resolution(g: ig.Graph, target_min: int, target_max: int,
                     res_grid: list, seed: int):
    """Sweep Leiden resolution to find community count in target range."""
    best_labels, best_res, best_diff = None, None, 1e9
    for res in res_grid:
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=float(res), seed=seed,
        )
        labels = np.array(part.membership)
        n_comm = len(set(labels))
        if target_min <= n_comm <= target_max:
            return labels, float(res)
        diff = min(abs(n_comm - target_min), abs(n_comm - target_max))
        if diff < best_diff:
            best_labels, best_res, best_diff = labels, float(res), diff
    fallback_res = best_res if best_res is not None else float(res_grid[len(res_grid) // 2])
    return best_labels, fallback_res


def representative_phrases(X: np.ndarray, phrases: list, labels: np.ndarray, topk: int = 6) -> dict:
    """For each community, find the topk n-grams closest to its centroid."""
    reps = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        centroid = normalize(X[idx].mean(axis=0, keepdims=True))
        sims = (X[idx] @ centroid.T).ravel()
        reps[c] = [phrases[idx[i]] for i in np.argsort(-sims)[: min(topk, len(idx))]]
    return reps


# ---------------------------------------------------------------------------
# Main clustering pipeline
# ---------------------------------------------------------------------------

def run_clustering(cfg: dict, master_df: pd.DataFrame, run_dir: Path):
    """Embed n-grams, cluster, detect communities, save results."""
    cl_cfg = cfg.get("clustering", {})
    seed = cfg.get("random_seed", 42)

    ngrams_list = master_df["ngram_str"].tolist()
    freqs = master_df["frequency"].tolist()

    # --- Embed ---
    emb_cfg = cfg.get("embedding", {})
    log.info(f"Embedding {len(ngrams_list):,} n-grams (provider={emb_cfg.get('provider', 'sentence-transformers')})")
    X_raw = get_embeddings(ngrams_list, emb_cfg)

    # --- PCA ---
    pca_comp = cl_cfg.get("pca_components", 100)
    whiten = cl_cfg.get("pca_whiten", True)
    log.info(f"PCA projection: components={pca_comp}, whiten={whiten}")
    Xw, _ = pca_project(X_raw, pca_comp, whiten, seed)

    # --- Save embeddings ---
    emb_dir = run_dir / "clusters" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ngram": ngrams_list,
        "frequency": freqs,
        "embedding_pca": [json.dumps(v.tolist()) for v in Xw],
    }).to_csv(emb_dir / "ngram_embeddings.csv", index=False)

    # --- Per-K clustering ---
    k_dir = run_dir / "clusters" / "k_level_results"
    c_dir = run_dir / "clusters" / "community_results"
    k_dir.mkdir(parents=True, exist_ok=True)
    c_dir.mkdir(parents=True, exist_ok=True)

    k_values = cl_cfg.get("k_values", [500])
    knn = cl_cfg.get("centroid_knn", 80)
    min_cos = cl_cfg.get("edge_min_cosine", 0.65)
    target_min, target_max = cl_cfg.get("leiden_target_communities", [20, 50])
    res_grid = cl_cfg.get("leiden_resolution_grid", [0.1])
    min_cluster_sz = cl_cfg.get("min_cluster_size", 10)
    min_sim = cl_cfg.get("min_similarity_score", 0.3)

    for k in k_values:
        log.info(f"--- K={k} ---")

        # Spherical KMeans
        micro_labels, centroids = spherical_kmeans(
            Xw, k, cl_cfg.get("kmeans_n_init", 10), cl_cfg.get("kmeans_max_iter", 300), seed,
        )
        cos_to_micro = np.array([Xw[i] @ centroids[micro_labels[i]] for i in range(len(Xw))])
        pd.DataFrame({
            "ngram": ngrams_list, "frequency": freqs,
            f"cluster_id_k{k}": micro_labels,
            f"distance_to_centroid_k{k}": 1.0 - cos_to_micro,
        }).to_csv(k_dir / f"clusters_k{k}.csv", index=False)

        # Leiden
        g = build_centroid_graph(centroids, knn, min_cos)
        centroid_comm, best_res = sweep_resolution(g, target_min, target_max, res_grid, seed)
        n_comm = len(set(centroid_comm))
        log.info(f"[K={k}] Leiden found {n_comm} communities (resolution={best_res:.3f})")

        point_comm = centroid_comm[micro_labels]
        _, point_comm_re = np.unique(point_comm, return_inverse=True)

        # Community centroids + similarity
        mean_sims = {}
        comm_centroids = {}
        for cid in sorted(set(point_comm_re)):
            idx = np.where(point_comm_re == cid)[0]
            vecs = Xw[idx]
            if len(vecs) > 0:
                cent = normalize(vecs.mean(axis=0, keepdims=True))[0]
                comm_centroids[cid] = cent
                mean_sims[cid] = float(np.mean(vecs @ cent)) if len(vecs) > 1 else 0.0

        # Save community centroids
        pd.DataFrame([
            {f"community_id_k{k}": int(cid),
             "centroid_embedding": json.dumps(vec.tolist()),
             "size": int((point_comm_re == cid).sum()),
             "mean_similarity": float(mean_sims.get(cid, 0.0))}
            for cid, vec in comm_centroids.items()
        ]).to_csv(emb_dir / f"community_centroids_k{k}.csv", index=False)

        # Save community assignments
        pd.DataFrame({
            "ngram": ngrams_list, "frequency": freqs,
            f"community_id_k{k}": point_comm_re,
        }).to_csv(c_dir / f"communities_k{k}.csv", index=False)

        # Community labels (for user to hand-label)
        reps = representative_phrases(Xw, ngrams_list, point_comm_re)
        rows = []
        for cid in sorted(set(point_comm_re)):
            sz = int((point_comm_re == cid).sum())
            rows.append({
                "community_id": cid,
                "size": sz,
                "mean_similarity": mean_sims.get(cid, 0.0),
                "representatives": ", ".join(reps.get(cid, [])),
            })
        labels_df = pd.DataFrame(rows)
        labels_df = labels_df[
            (labels_df["size"] >= min_cluster_sz) &
            (labels_df["mean_similarity"] >= min_sim)
        ].sort_values("size", ascending=False)
        log.info(f"[K={k}] {len(labels_df)} communities pass size/similarity filters (pruned {len(rows) - len(labels_df)})")

        labels_path = c_dir / f"community_labels_k{k}.csv"
        labels_df.to_csv(labels_path, index=False)
        log.info(f"Saved community labels to {labels_path}")
        log.info(f">>> NEXT STEP: Open {labels_path} and add 'category' and 'subcategory' columns <<<")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2: N-gram clustering")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_nltk()

    run_dir = get_run_dir(cfg)

    # Text source priority: LLM extracts → Stage 1 texts → raw input_dir
    llm_dir = str(run_dir / "llm_extracts")
    text_dir = str(run_dir / "texts")
    if os.path.isdir(llm_dir) and os.listdir(llm_dir):
        text_dir = llm_dir
        log.info(f"Using LLM-extracted text for dictionary construction from {text_dir}")
    elif not os.path.isdir(text_dir) or not os.listdir(text_dir):
        text_dir = cfg.get("extract", {}).get("input_dir", "./raw_documents")
        log.info(f"No Stage 1 output found; reading directly from {text_dir}")

    master_path = str(run_dir / "clusters" / "ngram_master_list.csv")
    master_df = build_master_ngram_list(cfg, text_dir, master_path)

    if master_df.empty:
        log.error("Empty master n-gram list. Cannot proceed.")
        return

    run_clustering(cfg, master_df, run_dir)
    log.info("Stage 2 complete.")


if __name__ == "__main__":
    main()
