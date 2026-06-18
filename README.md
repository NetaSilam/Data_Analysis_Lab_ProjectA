# Section B - Wikipedia Retrieval Pipeline

This repository contains our Section B solution for Project A. The system builds an offline retrieval index over the Wikipedia-style corpus and exposes the required autograder API:

```python
def run(queries: list[str]) -> list[list[int]]:
```

`run()` receives all evaluation queries in one batch and returns, for each query, a ranked list of page IDs. Only the first 10 IDs are scored with NDCG@10.

## Team and Links

- Team members: Liz Melamed, Neta Silam
- Presentation video: https://drive.google.com/file/d/1oUaP4GDRvoUsuwS-XM7Nsx7L7Tk0N_ax/view?usp=sharing 
- Public GitHub repository: https://github.com/NetaSilam/Data_Analysis_Lab_ProjectA.git 

## Project Structure

- `main.py` - required entry point. The autograder imports and calls `run(queries)`.
- `chunk.py` - splits each page into overlapping text chunks.
- `embed.py` - loads `sentence-transformers/all-MiniLM-L6-v2` and creates normalized embeddings.
- `index.py` - builds and loads the offline index artifacts.
- `retrieve.py` - expands and embeds queries, scores chunk vectors, aggregates by page ID, and reranks candidates.
- `utils.py` - shared paths, corpus loading, and helper functions.
- `eval.py` - read-only NDCG@10 evaluation utilities from the course bundle.
- `scripts/build_index.py` - read-only helper for building artifacts locally.
- `scripts/eval_public.py` - read-only public self-test script.
- `requirements.txt` - allowed Python dependencies.
- `artifacts/` - prebuilt index files required at grading time.

## Retrieval Method

1. **Chunking:** each page (`title + content`) is split into 200-word chunks with 30-word overlap.
2. **Embedding:** all chunks are embedded offline with `sentence-transformers/all-MiniLM-L6-v2`; vectors are L2-normalized and stored in `artifacts/`.
3. **Query expansion:** at query time, each query string is duplicated before embedding to amplify the query signal.
4. **Dense retrieval:** each expanded query vector is compared with all chunk vectors by dot product. Chunk scores are grouped by `page_id` and averaged (Mean aggregation) to produce page-level scores.
5. **Reranking:** the top 10 page candidates are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`, using the original query and the first 1500 characters of each candidate page. The cross-encoder is loaded automatically from Hugging Face Hub at query time.

## Setup

Requires Python 3.10+. From the repository root:

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

## Artifacts

The following prebuilt files are already included in the repository under `artifacts/`. The autograder does not rebuild them — `run()` loads them directly from disk.

| Path | Format | Description |
| --- | --- | --- |
| `artifacts/index_vectors.npy` | NumPy `float32`, shape `(348786, 384)` | L2-normalized MiniLM embeddings for all text chunks. |
| `artifacts/index_meta.json` | JSON | Stores `page_ids`, `chunk_ids`, embedding model name, and vector count. |
| `artifacts/page_texts.pkl` | Python pickle | Maps each `page_id` (int) to its full page text, used for reranking. |

To rebuild artifacts locally (optional):

```bash
python scripts/build_index.py
```

## Public Self-Test

With dependencies installed and artifacts present, run:

```bash
python scripts/eval_public.py
```

Expected output:

```text
public_queries=29
mean_ndcg@10=...
query_phase_time=...
```

This command should succeed on a fresh clone without rebuilding the index.
