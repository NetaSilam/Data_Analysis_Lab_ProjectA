"""Corpus preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


CHUNK_SIZE = 140
CHUNK_OVERLAP = 30
MIN_FINAL_CHUNK_SIZE = 40


@dataclass(frozen=True)
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the original textual content.
    """
    return " ".join(str(text).split())


def _format_chunk(title: str, content: str) -> str:
    """
    Add the page title to every chunk.

    The title is often highly informative for retrieval and helps connect
    a local passage to the entity or topic described by the page.
    """
    if title and content:
        return f"{title}. {content}"

    return title or content


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one Wikipedia-style page into overlapping retrieval chunks.

    Each content chunk contains at most CHUNK_SIZE words. The title is
    prepended to every chunk but is not counted toward CHUNK_SIZE.

    Consecutive chunks overlap by CHUNK_OVERLAP words.
    """
    page_id = int(record["page_id"])
    title = _normalize_text(record.get("title", ""))
    content = _normalize_text(record.get("content", ""))

    if not title and not content:
        return []

    words = content.split()

    # A page containing only a title should still be searchable.
    if not words:
        return [
            Chunk(
                page_id=page_id,
                chunk_id=0,
                text=title,
            )
        ]

    step = CHUNK_SIZE - CHUNK_OVERLAP

    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_words = words[start:end]

        # Avoid a very small final chunk when almost all of its words
        # already appeared in the preceding chunk.
        remaining_words = len(words) - end

        if 0 < remaining_words < MIN_FINAL_CHUNK_SIZE:
            end = len(words)
            chunk_words = words[start:end]

        chunk_content = " ".join(chunk_words)
        chunk_text = _format_chunk(title, chunk_content)

        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=chunk_id,
                text=chunk_text,
            )
        )

        chunk_id += 1

        if end >= len(words):
            break

        start += step

    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    """
    Chunk every page in the corpus while preserving corpus order.
    """
    return [
        chunk
        for record in records
        for chunk in chunk_entry(record)
    ]