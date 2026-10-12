# chunker.py
# Responsibilities:
#   - Accept cleaned page text and its page number
#   - Split text into paragraph-level chunks using the "\n\n" boundary
#     produced by text_cleaner.py
#   - Keep each paragraph as one chunk by default
#   - Split oversized paragraphs at sentence boundaries when a max_chars
#     limit is configured, preserving readability and page metadata
#   - Return a list of Chunk objects ready for embedding and storage

from __future__ import annotations

import re
from typing import Optional

from backend.models.schemas import Chunk

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Sentence boundary: a period / exclamation / question mark followed by
# one or more whitespace characters and an uppercase letter or digit.
# This is intentionally simple — a full sentence splitter is not needed here.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")

# Default maximum characters per chunk.  None means no limit.
_DEFAULT_MAX_CHARS: Optional[int] = None


def _split_large_paragraph(text: str, max_chars: int) -> list[str]:
    """Split *text* into sub-chunks of at most *max_chars* characters.

    Splitting is attempted at sentence boundaries first.  If a single
    sentence already exceeds *max_chars* it is emitted as-is to avoid
    silently discarding content.

    Args:
        text: A single paragraph string that exceeds *max_chars*.
        max_chars: Maximum character budget per sub-chunk.

    Returns:
        A list of non-empty sub-chunk strings.
    """
    sentences = _SENTENCE_END.split(text)
    sub_chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate_len = len(sentence)

        if current_len == 0:
            # First sentence in a new sub-chunk — always add it, even if it
            # exceeds the budget on its own.
            current_parts.append(sentence)
            current_len = candidate_len
        elif current_len + 1 + candidate_len <= max_chars:
            # Fits within the budget: accumulate.
            current_parts.append(sentence)
            current_len += 1 + candidate_len
        else:
            # Flush the current sub-chunk and start a new one.
            sub_chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_len = candidate_len

    if current_parts:
        sub_chunks.append(" ".join(current_parts))

    return sub_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_page(
    cleaned_text: str,
    page_number: int,
    doc_id: str,
    max_chars: Optional[int] = _DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Split the cleaned text of one PDF page into paragraph-level chunks.

    The text_cleaner module normalises paragraph boundaries to exactly
    ``"\\n\\n"``.  This function splits on that separator so that each
    paragraph becomes one :class:`~backend.models.schemas.Chunk` by default.

    When *max_chars* is set and a paragraph exceeds that limit, the paragraph
    is further split at sentence boundaries.  Normal-sized paragraphs are
    never split regardless of *max_chars*.

    Args:
        cleaned_text: Cleaned text for a single page, as produced by
            :func:`~backend.services.text_cleaner.clean_page_text`.
        page_number: 1-based page number from which the text was extracted.
        doc_id: Identifier of the parent document, used to construct
            globally unique chunk IDs.
        max_chars: Optional character budget per chunk.  Paragraphs that
            exceed this limit are split at sentence boundaries.  Pass
            ``None`` (the default) to disable splitting — every paragraph
            becomes exactly one chunk regardless of its length.

    Returns:
        A list of :class:`~backend.models.schemas.Chunk` objects in the order
        they appear in the page.  Returns an empty list if *cleaned_text* is
        empty, ``None``, or contains only whitespace.
    """
    if not cleaned_text or not cleaned_text.strip():
        return []

    paragraphs = cleaned_text.split("\n\n")
    chunks: list[Chunk] = []
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            # Skip blank paragraphs that arise from leading/trailing "\n\n"
            # or consecutive separators.
            continue

        # Determine which texts to turn into chunks for this paragraph.
        if max_chars is not None and len(paragraph) > max_chars:
            texts = _split_large_paragraph(paragraph, max_chars)
        else:
            texts = [paragraph]

        for text in texts:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page_number}-c{chunk_index}",
                    doc_id=doc_id,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
            chunk_index += 1

    return chunks
