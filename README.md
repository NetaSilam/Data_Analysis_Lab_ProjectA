# Section B - Wikipedia Retrieval Pipeline

This repository contains our Section B solution for Project A. The system builds an offline retrieval index over the Wikipedia-style corpus and exposes the required autograder API:

```python
def run(queries: list[str]) -> list[list[int]]:
```

`run()` receives all evaluation queries in one batch and returns, for each query, a ranked list of page IDs. Only the first 10 IDs are scored with NDCG@10.

## Team and Links

- Team members: TODO
- Presentation video: TODO
- Public GitHub repository: TODO


## Project Structure

- `main.py` - required entry point. The autograder imports and calls `run(queries)`.
- `chunk.py` - splits each page into overlapping text chunks.
- `embed.py` - loads `sentence-transformers/all-MiniLM-L6-v2` and creates normalized embeddings.
- `index.py` - builds and loads the offline index artifacts.
- `retrieve.py` - embeds queries, scores chunk vectors, aggregates by page ID, and reranks candidates.
- `utils.py` - shared paths, corpus loading, and helper functions.
- `eval.py` - read-only NDCG@10 evaluation utilities from the course bundle.
- `scripts/build_index.py` - read-only helper for building artifacts locally.
- `scripts/eval_public.py` - read-only public self-test script.
- `requirements.txt` - allowed Python dependencies.
- `artifacts/` - required prebuilt index files for grading.

## Retrieval Method

1. **Chunking:** each entry is represented as `title + content` and split into 200-word chunks with 30-word overlap.
2. **Embedding:** all chunks and queries are embedded with `sentence-transformers/all-MiniLM-L6-v2`; vectors are L2-normalized.
3. **Offline index:** chunk vectors and metadata are saved under `artifacts/`.
4. **Query-time retrieval:** each query vector is compared with all chunk vectors by dot product. Chunk scores are grouped by `page_id` and averaged to produce page-level candidates.
5. **Reranking:** the top page candidates are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`, using the query and the first 1500 characters of each candidate page.

## Setup

Use Python 3.10+ from the repository root:

```bash
pip install -r requirements.txt
```

The course corpus must be available at:

```text
data/Wikipedia Entries/
```

Public queries must be available at:

```text
data/public_queries.json
```

## Build Artifacts Locally

Build the offline index once on your own machine:

```bash
python scripts/build_index.py
```

This creates the files required by `run()` inside `artifacts/`. The grader does not run this script, so these generated files must be included in the public GitHub repository.

## Required Artifacts

Commit the following files before submission:

| Path | Format | Purpose |
| --- | --- | --- |
| `artifacts/index_vectors.npy` | NumPy `float32` array, shape `(num_chunks, 384)` | L2-normalized MiniLM embedding for each text chunk. |
| `artifacts/index_meta.json` | JSON | Stores `page_ids`, `chunk_ids`, embedding model name, and vector count. |
| `artifacts/page_texts.pkl` | Python pickle | Maps each integer `page_id` to its full page text for reranking. |

If any of these files are missing or incompatible, `python scripts/eval_public.py` will fail on a fresh clone and the functional component may receive 0.

## Public Self-Test

After artifacts are built, run:

```bash
python scripts/eval_public.py
```

Expected output includes:

```text
public_queries=50
mean_ndcg@10=...
query_phase_time=...
```

