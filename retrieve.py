"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """
    Return ranked page_id lists (best first) for each query.

    Default: brute-force dot product on L2-normalized vectors.
    Replace with FAISS / reranking as needed.
    """
    corpus_vectors, page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    scores = query_vectors @ corpus_vectors.T
    ranked = []

    for row in scores:
        page_chunks_scores = {}

        for idx, score in enumerate(row):
            pid = page_ids[idx]
            if pid not in page_chunks_scores:
                page_chunks_scores[pid] = []
            page_chunks_scores[pid].append(float(score))

        page_scores = {}
        for pid, chunk_scores in page_chunks_scores.items():
            page_scores[pid] = np.mean(chunk_scores)

        ordered = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
        ranked.append([pid for pid, _ in ordered[:top_k]])

    return ranked