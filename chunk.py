"""Preprocessing and chunking for the retrieval corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into 200-word chunks with 30-word overlap.

    The title is included through utils.entry_text(), so each page's first chunk
    carries both title and content signals.
    """
    page_id = int(record["page_id"])
    text = entry_text(record)

    words = text.split()

    chunk_size = 200
    overlap = 30

    chunks = []
    chunk_id = 0

    start = 0
    while start < len(words):
        end = start + chunk_size

        chunk_text = " ".join(words[start:end])

        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=chunk_id,
                text=chunk_text,
            )
        )

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """Chunk all corpus records into retrieval units."""
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
