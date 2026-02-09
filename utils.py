"""
Shared utilities for the n-gram pipeline.

Provides config loading, NLTK setup, n-gram extraction, document reading,
and a unified embedding interface (OpenAI API or local sentence-transformers).
"""

import os
import re
import json
import logging
import yaml
import numpy as np
from pathlib import Path
from collections import Counter

import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk import ngrams as nltk_ngrams, word_tokenize, pos_tag

logger = logging.getLogger("ngram_pipeline")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config, validate required keys, return dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    required = ["run_name", "output_root"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Config missing required key: '{k}'")
    return cfg


def get_run_dir(cfg: dict) -> Path:
    """Return <output_root>/<run_name>, creating it if needed."""
    p = Path(cfg["output_root"]) / cfg["run_name"]
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# NLTK
# ---------------------------------------------------------------------------

def setup_nltk():
    """Download required NLTK data if missing."""
    for resource in ["averaged_perceptron_tagger",
                     "averaged_perceptron_tagger_eng",
                     "stopwords", "punkt", "punkt_tab",
                     "wordnet", "words"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"tokenizers/{resource}")
                except LookupError:
                    nltk.download(resource, quiet=True)


def get_stop_words(custom: list | None = None) -> set:
    """Return NLTK English stopwords merged with any user-provided extras."""
    sw = set(stopwords.words("english"))
    if custom:
        sw.update(w.lower() for w in custom)
    return sw


def get_english_words() -> set:
    """Return the NLTK English word corpus as a set."""
    return set(words.words())


# ---------------------------------------------------------------------------
# POS patterns
# ---------------------------------------------------------------------------

def build_pos_patterns(raw: dict) -> dict:
    """
    Convert config-style POS patterns to tuple lookup.

    Config format:  {2: [["NN","NN"]], 3: [["NN","NN","NN"]]}
    Returns:        {2: [("NN","NN")],  3: [("NN","NN","NN")]}
    """
    out = {}
    for n, patterns in raw.items():
        out[int(n)] = [tuple(p) for p in patterns]
    return out


def has_valid_pos_pattern(tags: tuple, length: int, allowed: dict) -> bool:
    """Check if a POS tag sequence matches any allowed pattern for its length."""
    if length not in allowed:
        return False
    for pattern in allowed[length]:
        if all(tag.startswith(pt) for tag, pt in zip(tags, pattern)):
            return True
    return False


def is_valid_ngram(word_tuple: tuple, tag_tuple: tuple,
                   allowed: dict, min_word_len: int = 3) -> bool:
    """Validate an n-gram: no digits, min word length, no duplicate words, valid POS."""
    if any(w.isdigit() or len(w) < min_word_len for w in word_tuple):
        return False
    if len(set(word_tuple)) != len(word_tuple):
        return False
    return has_valid_pos_pattern(tag_tuple, len(word_tuple), allowed)


# ---------------------------------------------------------------------------
# N-gram extraction (single canonical implementation)
# ---------------------------------------------------------------------------

def extract_ngrams_from_text(
    text: str,
    stop_words: set,
    english_words: set,
    ngram_range: tuple = (2, 3),
    pos_patterns: dict | None = None,
    min_word_len: int = 3,
    require_english: bool = True,
) -> Counter:
    """
    Extract POS-filtered n-grams from a text string.

    Returns Counter mapping word tuples -> count.
    """
    if pos_patterns is None:
        pos_patterns = {2: [("NN", "NN")], 3: [("NN", "NN", "NN")]}

    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = [lemmatizer.lemmatize(w) for w in word_tokenize(text)]
    filtered = [
        t for t in tokens
        if t not in stop_words
        and (not require_english or t in english_words)
        and not t.isdigit()
        and len(t) >= min_word_len
    ]
    tagged = pos_tag(filtered)
    counter = Counter()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for gram in nltk_ngrams(tagged, n):
            ws = tuple(item[0] for item in gram)
            ts = tuple(item[1].upper() for item in gram)
            if is_valid_ngram(ws, ts, pos_patterns, min_word_len):
                counter[ws] += 1
    return counter


# ---------------------------------------------------------------------------
# Document reading
# ---------------------------------------------------------------------------

def read_document(filepath: str | Path, input_format: str = "text") -> str:
    """
    Read a document and return its text content.

    Formats:
      - "text": plain text file
      - "json": expects {"text": "..."} or treats whole content as text
      - "html": reads raw HTML (caller should clean separately)
    """
    p = Path(filepath)
    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = p.read_text(errors="ignore")

    if input_format == "json":
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "text" in data:
                return str(data["text"])
        except json.JSONDecodeError:
            pass

    return raw


# ---------------------------------------------------------------------------
# Embeddings (OpenAI API or sentence-transformers)
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str], embedding_cfg: dict) -> np.ndarray:
    """
    Embed a list of texts using the configured provider.

    embedding_cfg should contain:
      - provider: "openai" or "sentence-transformers"
      - For OpenAI: openai_model, openai_batch_size
      - For ST:     st_model, st_batch_size

    Returns (N, D) float32 array.
    """
    provider = embedding_cfg.get("provider", "sentence-transformers")

    if provider == "openai":
        return _embed_openai(texts, embedding_cfg)
    else:
        return _embed_sentence_transformers(texts, embedding_cfg)


def _embed_openai(texts: list[str], cfg: dict) -> np.ndarray:
    """Fetch embeddings from OpenAI API in batches."""
    import openai
    import time
    from tqdm import tqdm

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Run: export OPENAI_API_KEY='your-key-here'"
        )

    client = openai.OpenAI(api_key=api_key)
    model = cfg.get("openai_model", "text-embedding-3-large")
    batch_size = cfg.get("openai_batch_size", 512)

    all_embeddings = []
    embed_dim = None

    for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI embeddings"):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            vecs = [item.embedding for item in resp.data]
            if embed_dim is None and vecs:
                embed_dim = len(vecs[0])
            all_embeddings.extend(vecs)
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"OpenAI API error on batch at index {i}: {e}")
            if embed_dim is None:
                embed_dim = 3072 if "large" in model else 1536
            all_embeddings.extend([np.zeros(embed_dim).tolist()] * len(batch))

    return np.array(all_embeddings, dtype=np.float32)


def _embed_sentence_transformers(texts: list[str], cfg: dict) -> np.ndarray:
    """Embed using a local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    model_name = cfg.get("st_model", "sentence-transformers/all-mpnet-base-v2")
    batch_size = cfg.get("st_batch_size", 256)

    # Device selection
    device = "cpu"
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    if cfg.get("force_cpu", False):
        device = "cpu"

    logger.info(f"Loading sentence-transformers model: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# LLM calls (Ollama or OpenAI)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, llm_cfg: dict) -> str | None:
    """
    Send a prompt to an LLM and return the response text.

    llm_cfg should contain:
      - provider: "ollama" or "openai"
      - For Ollama: ollama_model
      - For OpenAI: openai_model

    Returns response text, or None on failure.
    """
    provider = llm_cfg.get("provider", "ollama")

    if provider == "openai":
        return _llm_openai(prompt, llm_cfg)
    else:
        return _llm_ollama(prompt, llm_cfg)


def _llm_ollama(prompt: str, cfg: dict) -> str | None:
    """Call a local Ollama model."""
    import ollama

    model = cfg.get("ollama_model", "gemma3n:e2b")
    try:
        response = ollama.chat(
            model=model,
            format="json",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama LLM error: {e}")
        return None


def _llm_openai(prompt: str, cfg: dict) -> str | None:
    """Call the OpenAI chat completions API."""
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Run: export OPENAI_API_KEY='your-key-here'"
        )

    client = openai.OpenAI(api_key=api_key)
    model = cfg.get("openai_model", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI LLM error: {e}")
        return None
