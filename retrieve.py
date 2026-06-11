"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PAGE_TEXTS_NAME = "page_texts.pkl"

_cross_encoder: CrossEncoder | None = None
_page_texts: dict | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def get_page_texts(root: Path) -> dict:
    global _page_texts
    if _page_texts is None:
        with open(root / PAGE_TEXTS_NAME, "rb") as f:
            _page_texts = pickle.load(f)
    return _page_texts


def expand_query(query):
    return query + " " + query 


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    root = artifacts_dir or ARTIFACTS_DIR
    corpus_vectors, page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(([expand_query(q) for q in queries]))
    if query_vectors.size == 0:
        return [[] for _ in queries]

    page_texts = get_page_texts(root)
    cross_encoder = get_cross_encoder()

    scores = query_vectors @ corpus_vectors.T

    candidates_k = 10  

    ranked = []

    for query, row in zip(queries, scores):

        # Mean aggregation
        page_chunks_scores = {}
        for idx, score in enumerate(row):
            pid = page_ids[idx]
            if pid not in page_chunks_scores:
                page_chunks_scores[pid] = []
            page_chunks_scores[pid].append(float(score))

        emb_page_scores = {
            pid: np.mean(chunk_scores)
            for pid, chunk_scores in page_chunks_scores.items()
        }

        # top candidates from dense retrieval
        top_candidates = sorted(
            emb_page_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:candidates_k]

        # rerank with cross encoder
        pairs = [
            (query, page_texts[pid][:1500])
            for pid, _ in top_candidates
            if pid in page_texts
        ]
        ce_scores = cross_encoder.predict(pairs)

        reranked = sorted(
            zip([pid for pid, _ in top_candidates if pid in page_texts], ce_scores),
            key=lambda x: x[1],
            reverse=True
        )

        ranked.append([pid for pid, _ in reranked[:top_k]])

    return ranked